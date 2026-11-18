"""Evaluation metrics for PathSnap map-matching and candidate classification.

Includes:
1. Candidate-level metrics:
   - accuracy, precision, recall, macro F1, log loss, confusion matrix
2. Observation-level metrics:
   - top-1 accuracy (per point, highest-scoring candidate compared to ref_arc)
   - candidate miss rate (points where ref_arc is missing from candidate set)
3. Sequence and route-level metrics:
   - route continuity (fraction of consecutive predicted arcs connected in road graph)
   - sequence-level recovery (fraction of reference route arcs recovered)
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)


# -------------------------------------------------------------------------
# Candidate-level metrics (Binary classification: is candidate arc the true arc?)
# -------------------------------------------------------------------------

def candidate_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Dict[str, float]:
    """Compute standard candidate-level classification metrics.
    
    Args:
        y_true: Binary ground truth (1 if candidate is true arc, else 0).
        y_pred: Predicted binary label (0 or 1).
        y_prob: Predicted probability score for class 1 (optional, for log_loss).
        
    Returns:
        Dict of metrics: accuracy, precision, recall, macro_f1, f1_binary, log_loss.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }

    if y_prob is not None:
        y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1.0 - 1e-15)
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception:
            metrics["log_loss"] = float("nan")
    else:
        metrics["log_loss"] = float("nan")

    return metrics


def candidate_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> np.ndarray:
    """Return 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


# -------------------------------------------------------------------------
# Observation-level metrics (Per GPS point)
# -------------------------------------------------------------------------

def _labelled_only(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows of GPS points that have a reference arc (has_ref True).

    Points without a reference (label -1 from pathsnap.labels) must not count as
    all-negative rows or as 'misses'.
    """
    if "has_ref" in candidates_df.columns:
        return candidates_df[candidates_df["has_ref"].to_numpy(dtype=bool)]
    return candidates_df


def top1_accuracy(
    candidates_df: pd.DataFrame,
    ref_arcs: Optional[Union[np.ndarray, pd.Series, Dict[Tuple[str, int], int]]] = None,
    score_col: str = "score",
) -> float:
    """Observation-level Top-1 Accuracy.

    For each labelled GPS point (grouped by trip_id, point_idx) pick the candidate with the
    highest score and check it is the reference arc. Points whose reference arc is not among
    the candidates count as WRONG (a miss is a failure of the whole pipeline). Points with no
    reference (has_ref False / ref_arc -1) are excluded. Points with no candidates at all do
    not appear in the table and are not counted here; report candidate_miss_rate with
    n_labelled_points to account for them.

    Reference source, in order: `ref_arcs` dict {(trip_id, point_idx): arc}, a `ref_arc`
    column, or the `label` column (1 = reference arc).
    """
    df = _labelled_only(candidates_df)
    if len(df) == 0:
        return 0.0

    grp = df.groupby(["trip_id", "point_idx"])[score_col]
    idx_top = grp.idxmax().to_numpy()
    top = df.loc[idx_top, ["trip_id", "point_idx", "arc_id"]].copy()

    if isinstance(ref_arcs, dict) and len(ref_arcs) > 0:
        top["ref_arc"] = [ref_arcs.get((r.trip_id, r.point_idx), -1) for r in top.itertuples()]
        valid = top[top["ref_arc"] >= 0]
        return float((valid["arc_id"] == valid["ref_arc"]).mean()) if len(valid) else 0.0
    if "ref_arc" in df.columns:
        top["ref_arc"] = df.loc[idx_top, "ref_arc"].to_numpy()
        valid = top[top["ref_arc"] >= 0]
        return float((valid["arc_id"] == valid["ref_arc"]).mean()) if len(valid) else 0.0
    if "label" in df.columns:
        # every point in `df` is labelled; it is correct only if the top pick is the reference arc
        return float((df.loc[idx_top, "label"].to_numpy() == 1).mean())
    raise ValueError("Must provide ref_arcs dict or include 'ref_arc'/'label' in candidates_df")


def candidate_miss_rate(
    candidates_df: pd.DataFrame,
    ref_arcs: Optional[Dict[Tuple[str, int], int]] = None,
    n_labelled_points: Optional[int] = None,
) -> float:
    """Share of labelled GPS points whose reference arc is NOT among the candidates.

    Modes:
      * `ref_arcs` dict {(trip_id, point_idx): arc_id} (non-empty): exact, also counts points
        that have no candidate rows at all.
      * otherwise uses the `label` column (1 = reference arc). Only points that appear in the
        table can be seen, so pass `n_labelled_points` (the true number of labelled points,
        e.g. from pathsnap.labels) to include points with no candidates at all in the
        denominator and count them as misses. Without it, the rate is an underestimate.
    """
    if isinstance(ref_arcs, dict) and len(ref_arcs) > 0:
        grouped = candidates_df.groupby(["trip_id", "point_idx"])["arc_id"].apply(set)
        total_valid = missed = 0
        for (trip_id, pt_idx), true_arc in ref_arcs.items():
            if true_arc < 0:
                continue
            total_valid += 1
            if true_arc not in grouped.get((trip_id, pt_idx), set()):
                missed += 1
        return float(missed / total_valid) if total_valid > 0 else 0.0

    if "label" not in candidates_df.columns:
        raise ValueError("Must provide a non-empty ref_arcs dict or a 'label' column in candidates_df")

    df = _labelled_only(candidates_df)
    has_hit = df.groupby(["trip_id", "point_idx"])["label"].max()
    n_seen = len(has_hit)
    n_hit = int((has_hit == 1).sum())
    total = max(n_labelled_points, n_seen) if n_labelled_points is not None else n_seen
    return float((total - n_hit) / total) if total > 0 else 0.0


# -------------------------------------------------------------------------
# Route & Sequence-level metrics
# -------------------------------------------------------------------------

def build_adjacency(arcs: np.ndarray) -> Dict[int, List[int]]:
    """Build directed adjacency dict: arc_id -> list of successor arc_ids.
    
    arcs: (n_arcs, 2) where row index is arc_id and columns are [from_node, to_node].
    Arc u connects to arc v if u's to_node == v's from_node.
    """
    node_to_incoming = {}
    node_to_outgoing = {}
    for arc_id, (u, v) in enumerate(arcs):
        node_to_outgoing.setdefault(u, []).append(arc_id)
        node_to_incoming.setdefault(v, []).append(arc_id)

    adj = {}
    for arc_id, (u, v) in enumerate(arcs):
        # Successors are arcs whose from_node is v
        adj[arc_id] = node_to_outgoing.get(v, [])
    return adj


def route_continuity(
    predicted_arcs: Union[List[int], np.ndarray],
    arcs: np.ndarray,
    adjacency: Optional[Dict[int, List[int]]] = None,
) -> float:
    """Compute Route Continuity.
    
    Fraction of consecutive predicted arcs (a_t, a_{t+1}) that are legally
    connected in the road network:
    Either a_{t+1} == a_t (vehicle remains on same segment), or
    a_{t+1} is directly reachable from a_t (a_t.to_node == a_{t+1}.from_node).
    
    Args:
        predicted_arcs: Sequence of predicted arc_ids per GPS point.
        arcs: (n_arcs, 2) network arcs array.
        adjacency: Optional precomputed adjacency dict from build_adjacency(arcs).
        
    Returns:
        Float in [0.0, 1.0]. Returns 1.0 if sequence length < 2.
    """
    preds = np.asarray(predicted_arcs, dtype=int)
    if len(preds) < 2:
        return 1.0

    if adjacency is None:
        adjacency = build_adjacency(arcs)

    # Filter out negative / invalid sentinel arc IDs (-1)
    connected_steps = 0
    total_steps = 0

    for i in range(len(preds) - 1):
        a1, a2 = preds[i], preds[i + 1]
        if a1 < 0 or a2 < 0:
            total_steps += 1
            continue
        total_steps += 1
        if a1 == a2:
            connected_steps += 1
        elif a2 in adjacency.get(a1, []):
            connected_steps += 1

    return float(connected_steps / total_steps) if total_steps > 0 else 1.0


def sequence_recovery(
    predicted_arcs: Union[List[int], np.ndarray],
    reference_route: Union[List[int], np.ndarray],
) -> Dict[str, float]:
    """Evaluate sequence-level recovery against the ground truth reference route.
    
    Args:
        predicted_arcs: Sequence of predicted arc IDs (may have consecutive duplicates).
        reference_route: Chronological sequence of unique road segments traversed.
        
    Returns:
        Dict containing:
        - 'route_recall': fraction of reference route arcs present in unique predicted arcs
        - 'route_precision': fraction of unique predicted arcs present in reference route
        - 'route_f1': harmonic mean of route recall and route precision
    """
    ref = np.asarray(reference_route, dtype=int)
    ref_valid = ref[ref >= 0]
    preds = np.asarray(predicted_arcs, dtype=int)
    preds_valid = preds[preds >= 0]

    # Compact consecutive identical arcs into ordered list of segments
    compact_preds = []
    for arc in preds_valid:
        if not compact_preds or compact_preds[-1] != arc:
            compact_preds.append(arc)

    ref_set = set(ref_valid)
    pred_set = set(compact_preds)

    if not ref_set:
        return {"route_recall": 0.0, "route_precision": 0.0, "route_f1": 0.0}

    hits = len(ref_set.intersection(pred_set))
    rec = float(hits / len(ref_set)) if len(ref_set) > 0 else 0.0
    prec = float(hits / len(pred_set)) if len(pred_set) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "route_recall": rec,
        "route_precision": prec,
        "route_f1": f1,
    }


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    candidates_df: pd.DataFrame,
    predicted_arcs_by_trip: Optional[Dict[str, np.ndarray]] = None,
    trips_dict: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Comprehensive evaluation combining candidate, observation, and route metrics."""
    res = candidate_metrics(y_true, y_pred, y_prob)
    
    # Top-1 observation accuracy
    if "score" in candidates_df.columns:
        res["top1_accuracy"] = top1_accuracy(candidates_df, score_col="score")
    if "label" in candidates_df.columns:
        res["candidate_miss_rate"] = candidate_miss_rate(candidates_df)
        
    # Route continuity if trip road networks are provided
    if predicted_arcs_by_trip and trips_dict:
        continuities = []
        for trip_id, pred_arcs in predicted_arcs_by_trip.items():
            trip = trips_dict.get(trip_id)
            if trip is not None:
                continuities.append(route_continuity(pred_arcs, trip.arcs))
        if continuities:
            res["mean_route_continuity"] = float(np.mean(continuities))

    return res
