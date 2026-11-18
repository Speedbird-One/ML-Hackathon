"""Experiment harness for PathSnap.

Provides:
- Grouped K-Fold cross-validation (GroupKFold grouped by trip_id to prevent leakage)
- Standardized evaluation on 80 train trips and 10 student-test trips
- Comprehensive logging to results/logs/experiments.csv
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import ARTIFACTS_DIR, RESULTS_DIR, SEED
from .metrics import (
    _labelled_only,
    build_adjacency,
    candidate_metrics,
    candidate_miss_rate,
    route_continuity,
    top1_accuracy,
)

# Road-graph adjacency is expensive to build (up to ~165k arcs per trip); build once per trip.
_ADJ_CACHE: Dict[str, Dict[int, List[int]]] = {}


def _adjacency_for(trip_id: str, trip: Any) -> Dict[int, List[int]]:
    if trip_id not in _ADJ_CACHE:
        _ADJ_CACHE[trip_id] = build_adjacency(trip.arcs)
    return _ADJ_CACHE[trip_id]


@dataclass
class ExperimentResult:
    model_name: str
    family: str
    key_config: str
    seed: int
    # Cross-validation metrics (80 train trips, grouped by trip_id)
    cv_macro_f1: float
    cv_top1_acc: float
    cv_log_loss: float
    cv_binary_f1: float
    cv_accuracy: float
    cv_route_continuity: float
    # Student-test metrics (10 test trips)
    test_macro_f1: float
    test_top1_acc: float
    test_log_loss: float
    test_binary_f1: float
    test_accuracy: float
    test_route_continuity: float
    # Computational efficiency
    train_time_sec: float
    inference_time_ms_per_point: float
    timestamp: str
    # Added for the report's overfitting / stability / per-class evidence (NaN if not computed)
    cv_macro_f1_std: float = float("nan")
    cv_top1_std: float = float("nan")
    cv_log_loss_std: float = float("nan")
    test_precision: float = float("nan")
    test_recall: float = float("nan")
    test_candidate_miss_rate: float = float("nan")  # on candidates present in the table only
    train_macro_f1: float = float("nan")
    train_top1_acc: float = float("nan")
    train_log_loss: float = float("nan")


class ExperimentLogger:
    """Manages experiment records in results/logs/experiments.csv."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or (RESULTS_DIR / "logs" / "experiments.csv")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self):
        if not self.log_path.exists():
            pd.DataFrame(columns=list(ExperimentResult.__dataclass_fields__)).to_csv(self.log_path, index=False)

    def log(self, result: ExperimentResult):
        """Append one row. Rewrites the (tiny) file so older logs with fewer columns stay valid."""
        new = pd.DataFrame([asdict(result)])
        if self.log_path.exists() and self.log_path.stat().st_size > 0:
            old = pd.read_csv(self.log_path)
            new = pd.concat([old, new], ignore_index=True) if len(old) else new
        new.to_csv(self.log_path, index=False)
        print(f"[Logger] Logged {result.model_name} ({result.family}) to {self.log_path}")

    def load_table(self) -> pd.DataFrame:
        """Load experiment history as DataFrame."""
        if self.log_path.exists():
            return pd.read_csv(self.log_path)
        return pd.DataFrame()


def score_model(model: Any, X: np.ndarray) -> np.ndarray:
    """P(candidate is the reference arc) for each row (probability, else sigmoid of margin, else hard label)."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
    if hasattr(model, "decision_function"):
        return 1.0 / (1.0 + np.exp(-model.decision_function(X)))
    return model.predict(X).astype(float)


def save_run_outputs(save_name: str, model: Any, test_df: pd.DataFrame, scores: np.ndarray) -> None:
    """Persist what other components need: fitted model, student-test scores, MLP loss history."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / f"{save_name}.joblib")

    scores_dir = RESULTS_DIR / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    out = test_df[["trip_id", "point_idx", "arc_id"]].copy()
    out["score"] = scores
    out.to_parquet(scores_dir / f"{save_name}_student_test.parquet", index=False)

    # Real epoch-wise history, only for estimators that record one (e.g. sklearn MLP). Never fabricated.
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if getattr(est, "loss_curve_", None) is not None:
        hist = {"loss_curve": [float(v) for v in est.loss_curve_]}
        if getattr(est, "validation_scores_", None) is not None:
            hist["validation_score"] = [float(v) for v in est.validation_scores_]
        (RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "logs" / f"{save_name}_history.json").write_text(json.dumps(hist))


def _save_confusion(save_name: str, model_name: str, y_true: np.ndarray, scores: np.ndarray) -> None:
    """Save the student-test confusion matrix (threshold 0.5) as JSON and as a figure."""
    from .metrics import candidate_confusion_matrix
    from .plots import plot_confusion_matrix_heatmap

    cm = candidate_confusion_matrix(y_true, (scores >= 0.5).astype(int))
    (RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "logs" / f"{save_name}_confusion.json").write_text(json.dumps({"labels": [0, 1], "matrix": cm.tolist()}))
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix_heatmap(cm, model_name, RESULTS_DIR / "figures" / f"confusion_{save_name}.png")


def evaluate_split(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_df: pd.DataFrame,
    trips_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Evaluate a trained model on a given validation candidate slice."""
    # Predict probabilities if supported, else decision function or hard labels
    y_prob = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)
        y_prob = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
    elif hasattr(model, "decision_function"):
        dfunc = model.decision_function(X_val)
        # Sigmoid calibration stand-in for margin
        y_prob = 1.0 / (1.0 + np.exp(-dfunc))

    if y_prob is not None:
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_val).astype(int)

    cand_metrics = candidate_metrics(y_val, y_pred, y_prob)

    # Observation-level Top-1 Accuracy
    eval_df = val_df[["trip_id", "point_idx", "arc_id"]].copy()
    eval_df["score"] = y_prob if y_prob is not None else y_pred
    eval_df["label"] = y_val
    top1 = top1_accuracy(eval_df, score_col="score")

    # Route continuity if road network trips are provided
    r_cont = float("nan")
    if trips_dict:
        # Pick top candidate per point
        idx_top = eval_df.groupby(["trip_id", "point_idx"])["score"].idxmax()
        top_picks = eval_df.loc[idx_top]
        conts = []
        for trip_id, group in top_picks.groupby("trip_id"):
            trip = trips_dict.get(trip_id)
            if trip is not None and hasattr(trip, "arcs"):
                pred_seq = group.sort_values("point_idx")["arc_id"].to_numpy()
                conts.append(route_continuity(pred_seq, trip.arcs, _adjacency_for(trip_id, trip)))
        if conts:
            r_cont = float(np.mean(conts))

    return {
        "macro_f1": cand_metrics["macro_f1"],
        "binary_f1": cand_metrics["binary_f1"],
        "accuracy": cand_metrics["accuracy"],
        "precision": cand_metrics["precision"],
        "recall": cand_metrics["recall"],
        "log_loss": cand_metrics["log_loss"],
        "top1_acc": top1,
        "route_continuity": r_cont,
    }


def cross_validate_model(
    model_fn: Callable[[], Any],
    train_df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
    seed: int = SEED,
    trips_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], float]:
    """Run GroupKFold cross-validation grouped strictly by trip_id.

    Only points with a reference arc (has_ref) are used; others carry no valid label.
    """
    train_df = _labelled_only(train_df).reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_splits)
    groups = train_df["trip_id"].to_numpy()
    X = train_df[feature_cols].to_numpy()
    y = train_df["label"].to_numpy()

    fold_metrics = []
    fit_times = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        model = model_fn()
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_val = X[val_idx], y[val_idx]
        val_sub_df = train_df.iloc[val_idx]

        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_times.append(time.perf_counter() - t0)

        metrics = evaluate_split(model, X_va, y_val, val_sub_df, trips_dict)
        fold_metrics.append(metrics)

    # Average metrics across folds
    mean_metrics = {}
    for key in fold_metrics[0].keys():
        vals = [fm[key] for fm in fold_metrics if not np.isnan(fm[key])]
        mean_metrics[key] = float(np.mean(vals)) if vals else float("nan")
        mean_metrics[key + "_std"] = float(np.std(vals)) if vals else float("nan")

    avg_fit_time = float(np.mean(fit_times))
    return mean_metrics, avg_fit_time


def run_experiment(
    model_name: str,
    family: str,
    key_config: str,
    model_fn: Callable[[], Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    seed: int = SEED,
    n_splits: int = 5,
    train_trips: Optional[Dict[str, Any]] = None,
    test_trips: Optional[Dict[str, Any]] = None,
    logger: Optional[ExperimentLogger] = None,
    save_name: Optional[str] = None,
    compute_train_metrics: bool = True,
) -> ExperimentResult:
    """Execute full training, CV evaluation, student-test evaluation, and logging."""
    logger = logger or ExperimentLogger()
    # Points without a reference arc are excluded from training and from every metric.
    train_df = _labelled_only(train_df).reset_index(drop=True)
    test_df = _labelled_only(test_df).reset_index(drop=True)

    print(f"\n=======================================================")
    print(f"Running Experiment: {model_name} ({family})")
    print(f"Config: {key_config}")
    print(f"=======================================================")

    # 1. Grouped Cross-Validation on Train
    print("[1/3] Running GroupKFold Cross-Validation on train trips...")
    cv_res, cv_fit_time = cross_validate_model(
        model_fn=model_fn,
        train_df=train_df,
        feature_cols=feature_cols,
        n_splits=n_splits,
        seed=seed,
        trips_dict=train_trips,
    )
    print(f"  CV Macro F1: {cv_res['macro_f1']:.4f} | Top-1 Acc: {cv_res['top1_acc']:.4f}")

    # 2. Fit on full train data and evaluate on student-test
    print("[2/3] Fitting on full training data (80 trips)...")
    full_model = model_fn()
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label"].to_numpy()

    t_start = time.perf_counter()
    full_model.fit(X_train, y_train)
    total_train_time = time.perf_counter() - t_start

    print("[3/3] Evaluating on Student-Test partition (10 trips)...")
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["label"].to_numpy()

    # Inference time: time ONLY the model's scoring call (not metric computation), per GPS point.
    t_inf_start = time.perf_counter()
    test_scores = score_model(full_model, X_test)
    inf_duration = time.perf_counter() - t_inf_start
    n_points = max(1, test_df.groupby(["trip_id", "point_idx"]).ngroups)
    ms_per_point = inf_duration / n_points * 1000.0
    test_res = evaluate_split(full_model, X_test, y_test, test_df, test_trips)
    if save_name:
        save_run_outputs(save_name, full_model, test_df, test_scores)
        _save_confusion(save_name, model_name, y_test, test_scores)
    miss_rate = candidate_miss_rate(test_df)

    # Training-set metrics: the evidence for the report's "overfitting or underfitting?" question.
    train_res = {"macro_f1": float("nan"), "top1_acc": float("nan"), "log_loss": float("nan")}
    if compute_train_metrics:
        train_res = evaluate_split(full_model, X_train, y_train, train_df, None)

    print(f"  Test Macro F1: {test_res['macro_f1']:.4f} | Test Top-1 Acc: {test_res['top1_acc']:.4f}")
    print(f"  Train Time: {total_train_time:.2f}s | Inf Speed: {ms_per_point:.4f} ms/point")

    result = ExperimentResult(
        model_name=model_name,
        family=family,
        key_config=key_config,
        seed=seed,
        cv_macro_f1=cv_res["macro_f1"],
        cv_top1_acc=cv_res["top1_acc"],
        cv_log_loss=cv_res["log_loss"],
        cv_binary_f1=cv_res["binary_f1"],
        cv_accuracy=cv_res["accuracy"],
        cv_route_continuity=cv_res["route_continuity"],
        test_macro_f1=test_res["macro_f1"],
        test_top1_acc=test_res["top1_acc"],
        test_log_loss=test_res["log_loss"],
        test_binary_f1=test_res["binary_f1"],
        test_accuracy=test_res["accuracy"],
        test_route_continuity=test_res["route_continuity"],
        train_time_sec=total_train_time,
        inference_time_ms_per_point=ms_per_point,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        cv_macro_f1_std=cv_res.get("macro_f1_std", float("nan")),
        cv_top1_std=cv_res.get("top1_acc_std", float("nan")),
        cv_log_loss_std=cv_res.get("log_loss_std", float("nan")),
        test_precision=test_res["precision"],
        test_recall=test_res["recall"],
        test_candidate_miss_rate=miss_rate,
        train_macro_f1=train_res["macro_f1"],
        train_top1_acc=train_res["top1_acc"],
        train_log_loss=train_res["log_loss"],
    )

    logger.log(result)
    return result
