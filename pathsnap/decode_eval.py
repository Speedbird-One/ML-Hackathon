"""Shared helpers for evaluating and tuning the HMM/Viterbi decoder (used by scripts/tune_hmm.py and scripts/eval_hmm.py).

Design points
  * The decoder is run on EVERY GPS point's candidates, as it will be at inference time. Only points that have a
    reference arc are scored (points without one are decoded but not counted).
  * Trips are decoded in parallel worker processes. Workers load their own trips (bounded in-memory cache), so no Trip
    object is ever pickled and memory stays bounded.
  * Metrics: point-level top-1 over all labelled points (a point whose true arc is not a candidate, or that got no
    prediction, counts as wrong), route continuity, sequence-level route recall/precision/F1, and top-1 split into slow
    (< 1 m/s, where heading is unreliable) and moving points.
"""
import functools
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import Trip, load_trip
from .hmm import DEFAULT_PARAMS, decode
from .labels import derive_ref_arcs
from .metrics import build_adjacency, route_continuity, sequence_recovery

FEATURES = ["dist_m", "along_frac", "arc_len_m", "speed_mps", "cos_heading",
            "dist_rank", "dist_minus_min", "n_cand"]
DECODE_COLS = ["point_idx", "arc_id", "dist_m", "along_frac", "arc_len_m", "score"]
SLOW_MPS = 1.0


# --------------------------------------------------------------------------- parameters
def load_hmm_params(path: Optional[Path]) -> Dict[str, Any]:
    """Decoder parameters from a JSON file written by tune_hmm.py ({"params": {...}}), else the defaults."""
    if path is None or not Path(path).exists():
        return dict(DEFAULT_PARAMS)
    return {**DEFAULT_PARAMS, **json.loads(Path(path).read_text())["params"]}


# --------------------------------------------------------------------------- scoring
def score_candidates(model: Any, cands: pd.DataFrame) -> pd.DataFrame:
    """Add a `score` column = P(candidate is the reference arc) for every row (labelled or not)."""
    out = cands.copy()
    out["score"] = model.predict_proba(out[FEATURES].to_numpy())[:, 1]
    return out


# --------------------------------------------------------------------------- reference data
def load_trips_and_refs(partition: str, trip_ids) -> Tuple[Dict[str, Trip], Dict[str, np.ndarray], Dict[str, dict]]:
    """Trips (with route), derived per-point reference arcs (-1 = unknown) and road adjacency, per trip."""
    trips = {t: load_trip(partition, t) for t in trip_ids}
    refs = {t: derive_ref_arcs(tr).ref_arc for t, tr in trips.items()}
    adjacency = {t: build_adjacency(tr.arcs) for t, tr in trips.items()}
    return trips, refs, adjacency


def point_speeds(scored: pd.DataFrame, trips: Dict[str, Trip]) -> Dict[str, np.ndarray]:
    """Speed (m/s) per GPS point from the candidate table (NaN where a point has no candidate)."""
    out = {}
    first = scored.groupby(["trip_id", "point_idx"], sort=False)["speed_mps"].first()
    for t, trip in trips.items():
        sp = np.full(len(trip.track), np.nan)
        if t in first.index.get_level_values(0):
            s = first.loc[t]
            sp[s.index.to_numpy()] = s.to_numpy()
        out[t] = sp
    return out


# --------------------------------------------------------------------------- predictions
def argmax_predictions(scored: pd.DataFrame, trips: Dict[str, Trip]) -> Dict[str, np.ndarray]:
    """Classifier-only prediction: the highest-scoring candidate per point (-1 where a point has no candidate)."""
    top = scored.loc[scored.groupby(["trip_id", "point_idx"], sort=False)["score"].idxmax()]
    preds = {}
    for t, trip in trips.items():
        p = np.full(len(trip.track), -1, dtype=np.int64)
        g = top[top.trip_id == t]
        p[g.point_idx.to_numpy()] = g.arc_id.to_numpy()
        preds[t] = p
    return preds


@functools.lru_cache(maxsize=16)
def _worker_trip(partition: str, trip_id: str) -> Trip:
    return load_trip(partition, trip_id, with_route=False)


def _decode_task(partition: str, trip_id: str, cands: pd.DataFrame, params: Dict[str, Any]):
    trip = _worker_trip(partition, trip_id)
    t0 = time.perf_counter()
    pred = decode(trip, cands, params)
    return trip_id, pred, time.perf_counter() - t0


def decode_trips(partition: str, scored: pd.DataFrame, params: Dict[str, Any],
                 parallel: Optional[Parallel] = None) -> Tuple[Dict[str, np.ndarray], float]:
    """HMM-decode every trip in `scored`. Returns ({trip_id: predicted arc per point}, total decode seconds summed over trips)."""
    groups = {t: g[DECODE_COLS] for t, g in scored.groupby("trip_id", sort=False)}
    order = sorted(groups, key=lambda t: -len(groups[t]))            # biggest first for load balance
    par = parallel if parallel is not None else Parallel(n_jobs=1)
    results = par(delayed(_decode_task)(partition, t, groups[t], params) for t in order)
    return {t: p for t, p, _ in results}, float(sum(s for _, _, s in results))


# --------------------------------------------------------------------------- metrics
def top1_from_predictions(preds: Dict[str, np.ndarray], refs: Dict[str, np.ndarray]) -> float:
    """Point-level top-1 over all labelled points (fast path used inside the tuning loop)."""
    correct = labelled = 0
    for t, p in preds.items():
        lab = refs[t] >= 0
        correct += int((p[lab] == refs[t][lab]).sum())
        labelled += int(lab.sum())
    return correct / max(labelled, 1)


def evaluate(preds: Dict[str, np.ndarray], trips: Dict[str, Trip], refs: Dict[str, np.ndarray],
             adjacency: Dict[str, dict], speeds: Dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per trip: top-1 (overall, slow, moving), route continuity, sequence-level route recall/precision/F1."""
    rows = []
    for t, pred in preds.items():
        trip, ref, sp = trips[t], refs[t], speeds[t]
        lab = ref >= 0
        slow = lab & (sp < SLOW_MPS)
        moving = lab & (sp >= SLOW_MPS)
        seq = sequence_recovery(pred, trip.route)
        rows.append({
            "trip": t, "points": len(pred), "labelled": int(lab.sum()),
            "correct": int((pred[lab] == ref[lab]).sum()),
            "slow_labelled": int(slow.sum()), "slow_correct": int((pred[slow] == ref[slow]).sum()),
            "moving_labelled": int(moving.sum()), "moving_correct": int((pred[moving] == ref[moving]).sum()),
            "continuity": route_continuity(pred, trip.arcs, adjacency[t]),
            "route_recall": seq["route_recall"], "route_precision": seq["route_precision"], "route_f1": seq["route_f1"],
        })
    df = pd.DataFrame(rows).sort_values("trip").reset_index(drop=True)
    df["top1"] = df.correct / df.labelled.clip(lower=1)
    return df


def summarise(per_trip: pd.DataFrame) -> Dict[str, float]:
    """Aggregate per-trip rows: top-1 pooled over all labelled points; other metrics averaged over trips."""
    return {
        "top1": float(per_trip.correct.sum() / max(per_trip.labelled.sum(), 1)),
        "top1_slow": float(per_trip.slow_correct.sum() / max(per_trip.slow_labelled.sum(), 1)),
        "top1_moving": float(per_trip.moving_correct.sum() / max(per_trip.moving_labelled.sum(), 1)),
        "continuity": float(per_trip.continuity.mean()),
        "route_recall": float(per_trip.route_recall.mean()),
        "route_precision": float(per_trip.route_precision.mean()),
        "route_f1": float(per_trip.route_f1.mean()),
        "labelled_points": int(per_trip.labelled.sum()),
        "slow_points": int(per_trip.slow_labelled.sum()),
    }
