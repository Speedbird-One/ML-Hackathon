"""Train LightGBM through the shared harness. Usage: python scripts/train_lgbm.py"""
import sys
from pathlib import Path

# Add repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from pathsnap.config import ARTIFACTS_DIR, CACHE_DIR, RESULTS_DIR, SEED
from pathsnap.data import list_trips, load_trip
from pathsnap.experiment import ExperimentLogger, run_experiment
from pathsnap.metrics import candidate_confusion_matrix, candidate_metrics

FEATURES = [
    "dist_m",
    "along_frac",
    "arc_len_m",
    "speed_mps",
    "cos_heading",
    "dist_rank",
    "dist_minus_min",
    "n_cand",
]


def load_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        import duckdb
        return duckdb.read_parquet(str(path)).df()


def main():
    print("Loading candidate tables...")
    train_df = load_parquet(CACHE_DIR / "candidates_train.parquet")
    test_df = load_parquet(CACHE_DIR / "candidates_student_test.parquet")
    print(f"Loaded train ({len(train_df):,} rows), student_test ({len(test_df):,} rows).")

    # Load road networks for route continuity
    print("Loading road networks for route continuity calculation...")
    train_trips = {t: load_trip("train", t, with_route=False) for t in list_trips("train")}
    test_trips = {t: load_trip("student_test", t, with_route=False) for t in list_trips("student_test")}

    def make_model():
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )

    res = run_experiment(
        model_name="LightGBM",
        family="Boosting",
        key_config="n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=SEED",
        model_fn=make_model,
        train_df=train_df,
        test_df=test_df,
        feature_cols=FEATURES,
        seed=SEED,
        n_splits=5,
        train_trips=train_trips,
        test_trips=test_trips,
        logger=ExperimentLogger(RESULTS_DIR / "logs" / "experiments_lgbm.csv"),
        save_name="lgbm",
    )

    # -------------------------------------------------------------
    # Deliverables & Technical Report Diagnostics (Section 4.5)
    # -------------------------------------------------------------
    print("\n--- Diagnostic checks for Technical Report ---")
    saved_model_path = ARTIFACTS_DIR / "lgbm.joblib"
    model = joblib.load(saved_model_path)
    model_size_mb = saved_model_path.stat().st_size / (1024 * 1024)
    print(f"Artifact size: {model_size_mb:.2f} MB (limit < 50 MB)")

    # 1. Training set score (overfitting check on 200k sample)
    sample_train = train_df[train_df.has_ref].sample(min(200000, len(train_df)), random_state=SEED)
    X_sample = sample_train[FEATURES].to_numpy()
    y_sample = sample_train["label"].to_numpy()
    p_train = model.predict_proba(X_sample)[:, 1]
    train_diag = candidate_metrics(y_sample, (p_train >= 0.5).astype(int), p_train)
    print(f"Train sample Macro F1: {train_diag['macro_f1']:.4f} | Binary F1: {train_diag['binary_f1']:.4f} | Log Loss: {train_diag['log_loss']:.4f}")

    # 2. Student-test diagnostics (Precision, Recall, Confusion Matrix)
    test_sub = test_df[test_df.has_ref].reset_index(drop=True)
    X_test = test_sub[FEATURES].to_numpy()
    y_test = test_sub["label"].to_numpy()
    p_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (p_test >= 0.5).astype(int)

    test_diag = candidate_metrics(y_test, y_pred_test, p_test)
    cm = candidate_confusion_matrix(y_test, y_pred_test)
    print(f"Student-test Precision: {test_diag['precision']:.4f} | Recall: {test_diag['recall']:.4f}")
    print(f"Confusion Matrix [[TN, FP], [FN, TP]]:\n{cm}")

    scores_path = RESULTS_DIR / "scores" / "lgbm_student_test.parquet"
    print(f"\n[Done] Scores file generated for Teammate 2: {scores_path}")
    print(f"File exists: {scores_path.exists()} ({scores_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
