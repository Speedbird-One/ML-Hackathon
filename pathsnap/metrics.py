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

def top1_accuracy(
    candidates_df: pd.DataFrame,
    ref_arcs: Optional[Union[np.ndarray, pd.Series, Dict[Tuple[str, int], int]]] = None,
    score_col: str = "score",
) -> float:
    """Compute observation-level Top-1 Accuracy.
    
    For each GPS point (grouped by trip_id, point_idx), pick the candidate
    with the maximum score. Compare that candidate's arc_id with ref_arc.
    Points with ref_arc == -1 (unmatched/no reference) are excluded.
    
    Args:
        candidates_df: DataFrame with ['trip_id', 'point_idx', 'arc_id', score_col]
        ref_arcs: Optional mapping (trip_id, point_idx) -> ref_arc.
                  If not provided, uses 'ref_arc' column or evaluates via 'label' column.
        score_col: Name of predicted probability/score column.
    """
    if len(candidates_df) == 0:
        return 0.0

    df = candidates_df
    idx_top = df.groupby(["trip_id", "point_idx"])[score_col].idxmax().to_numpy()
    top_candidates = df.loc[idx_top, ["trip_id", "point_idx", "arc_id"]].copy()

    # Ground truth determination
    if isinstance(ref_arcs, dict) and len(ref_arcs) > 0:
        top_candidates["ref_arc"] = [
            ref_arcs.get((r.trip_id, r.point_idx), -1)
            for r in top_candidates.itertuples()
        ]
        valid = top_candidates[top_candidates["ref_arc"] >= 0]
        if len(valid) == 0:
            return 0.0
        return float((valid["arc_id"] == valid["ref_arc"]).mean())
    elif "ref_arc" in df.columns:
        top_candidates["ref_arc"] = df.loc[idx_top, "ref_arc"].to_numpy()
        valid = top_candidates[top_candidates["ref_arc"] >= 0]
        if len(valid) == 0:
            return 0.0
        return float((valid["arc_id"] == valid["ref_arc"]).mean())
    elif "label" in df.columns:
        # Check if the highest-scoring candidate has label == 1
        labels_at_top = df.loc[idx_top, "label"].to_numpy()
        # Find points that had at least one candidate with label == 1 (i.e. ground truth exists in candidates)
        has_true_label = df.groupby(["trip_id", "point_idx"])["label"].max().to_numpy()
        valid_mask = has_true_label == 1
        if not np.any(valid_mask):
            return 0.0
        correct = (labels_at_top == 1) & valid_mask
        return float(correct.sum() / valid_mask.sum())
    else:
        raise ValueError("Must provide ref_arcs dict or include 'ref_arc'/'label' in candidates_df")

    # Filter out points with no valid reference (-1)
    valid = top_candidates[top_candidates["ref_arc"] >= 0]
    if len(valid) == 0:
        return 0.0
    return float((valid["arc_id"] == valid["ref_arc"]).mean())


def candidate_miss_rate(
    candidates_df: pd.DataFrame,
    ref_arcs: Union[np.ndarray, pd.Series, Dict[Tuple[str, int], int]],
) -> float:
    """Compute the candidate miss rate.
    
    Proportion of GPS observation points where the ground-truth ref_arc >= 0
    is NOT present among the generated candidate arcs for that point.
    """
    if isinstance(ref_arcs, dict):
        points = list(ref_arcs.keys())
        total_valid = 0
        missed = 0
        grouped = candidates_df.groupby(["trip_id", "point_idx"])["arc_id"].apply(set)
        for (trip_id, pt_idx), true_arc in ref_arcs.items():
            if true_arc < 0:
                continue
            total_valid += 1
            cands = grouped.get((trip_id, pt_idx), set())
            if true_arc not in cands:
                missed += 1
        return float(missed / total_valid) if total_valid > 0 else 0.0
    elif "label" in candidates_df.columns:
        # If 'label' is 1 for true arc and 0 otherwise:
        # A point has a hit if max(label) == 1, else missed.
        has_hit = candidates_df.groupby(["trip_id", "point_idx"])["label"].max()
        # Only evaluate points that actually have a reference (where label is present)
        return float((has_hit == 0).mean())
    else:
        raise ValueError("Must provide ref_arcs dict or 'label' column in candidates_df")


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
        res["top1_accuracy"] = top1_accuracy(candidates_df, ref_arcs={}, score_col="score")
    if "label" in candidates_df.columns:
        res["candidate_miss_rate"] = candidate_miss_rate(candidates_df, ref_arcs={})
        
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
