"""End-to-end inference for one trip: candidates -> classifier scores -> HMM/Viterbi route.

Used by the FastAPI backend (api/main.py) and by scripts/eval_pipeline.py, so the numbers shown
in the UI and the numbers in the report come from exactly the same code path.

Configuration lives in artifacts/inference_config.json (model file, feature list, candidate
radius, HMM parameters). To serve a different / tuned model, change "model_file" there.
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .config import ARTIFACTS_DIR
from .data import Trip, list_trips, load_trip
from .experiment import score_model
from .hmm import DEFAULT_PARAMS, decode
from .labels import UNKNOWN, derive_ref_arcs
from .metrics import build_adjacency, route_continuity, sequence_recovery

CONFIG_PATH = ARTIFACTS_DIR / "inference_config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_file": "lgbm.joblib",
    "model_name": "LightGBM",
    "features": ["dist_m", "along_frac", "arc_len_m", "speed_mps", "cos_heading",
                 "dist_rank", "dist_minus_min", "n_cand"],
    "candidate_radius_m": 40.0,
    "hmm_top_k": 8,
    "hmm_params": dict(DEFAULT_PARAMS),
}


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(path or os.environ.get("PATHSNAP_INFERENCE_CONFIG") or CONFIG_PATH)
    cfg = dict(DEFAULT_CONFIG)
    if path.exists():
        cfg.update(json.loads(path.read_text()))
    cfg["hmm_params"] = {**DEFAULT_PARAMS, **cfg.get("hmm_params", {})}
    return cfg


def _compact(arcs: np.ndarray) -> List[int]:
    """Per-point arc ids -> ordered route (drop -1 and consecutive duplicates)."""
    out: List[int] = []
    for a in np.asarray(arcs, dtype=np.int64):
        if a >= 0 and (not out or out[-1] != a):
            out.append(int(a))
    return out


def _segments(trip: Trip, arc_ids: List[int], nd: int = 6) -> List[List[List[float]]]:
    """Arc ids -> [[lat_u, lon_u], [lat_v, lon_v]] segments for Folium."""
    segs = []
    n_arcs = len(trip.arcs)
    for a in arc_ids:
        if 0 <= a < n_arcs:
            u, v = trip.arcs[a]
            segs.append([[round(float(trip.nodes[u, 1]), nd), round(float(trip.nodes[u, 0]), nd)],
                         [round(float(trip.nodes[v, 1]), nd), round(float(trip.nodes[v, 0]), nd)]])
    return segs


def _route_metrics(trip: Trip, pred: np.ndarray, ref_arc: Optional[np.ndarray],
                   adjacency: Dict[int, List[int]]) -> Dict[str, Optional[float]]:
    m: Dict[str, Optional[float]] = {
        "route_continuity": route_continuity(pred, trip.arcs, adjacency),
        "decoded_fraction": float(np.mean(pred >= 0)) if len(pred) else 0.0,
        "n_route_arcs": len(_compact(pred)),
        "top1_accuracy": None, "route_recall": None, "route_precision": None, "route_f1": None,
    }
    if ref_arc is not None:
        mask = ref_arc != UNKNOWN
        if mask.any():
            m["top1_accuracy"] = float(np.mean(pred[mask] == ref_arc[mask]))
        m.update(sequence_recovery(pred, trip.route))
    return m


class PathSnapPredictor:
    """Loads the trained classifier once and reconstructs routes for trips on demand."""

    def __init__(self, partition: str, config_path: Optional[Path] = None):
        self.partition = partition
        self.cfg = load_config(config_path)
        self.model_path = ARTIFACTS_DIR / self.cfg["model_file"]
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.features: List[str] = list(self.cfg["features"])
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ info
    def trip_ids(self) -> List[str]:
        return list_trips(self.partition)

    def model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.cfg["model_name"],
            "model_file": self.cfg["model_file"],
            "features": self.features,
            "candidate_radius_m": self.cfg["candidate_radius_m"],
            "hmm_top_k": self.cfg["hmm_top_k"],
            "hmm_params": self.cfg["hmm_params"],
        }

    def is_cached(self, trip_id: str) -> bool:
        return f"{trip_id}:server" in self._cache

    # --------------------------------------------------------------- predict
    def predict(self, trip_id: str, track: Optional[np.ndarray] = None,
                use_cache: bool = True) -> Dict[str, Any]:
        """Full pipeline for one trip. `track` (n,3 lon/lat/t) overrides the server's GPS file."""
        if trip_id not in self.trip_ids():
            raise KeyError(f"Unknown trip '{trip_id}' in partition '{self.partition}'")
        key = f"{trip_id}:{'client' if track is not None else 'server'}"
        if track is not None:
            key += f":{len(track)}:{float(np.asarray(track, dtype=float).sum()):.6f}"

        with self._lock:
            if use_cache and key in self._cache:
                out = dict(self._cache[key])
                out["cached"] = True
                return out
            out = self._run(trip_id, track)
            self._cache[key] = out
            return dict(out, cached=False)

    def _run(self, trip_id: str, track: Optional[np.ndarray]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        trip = load_trip(self.partition, trip_id, with_route=True)
        gps_source = "server"
        if track is not None:
            track = np.asarray(track, dtype=float)
            if track.ndim != 2 or track.shape[1] < 3 or len(track) < 2:
                raise ValueError("track must be a list of [lon, lat, t] rows (at least 2)")
            trip = Trip(trip_id=trip_id, track=track[:, :3], nodes=trip.nodes,
                        arcs=trip.arcs, route=trip.route)
            gps_source = "client"
        n = len(trip.track)
        t_load = time.perf_counter()

        # 1. candidate road arcs + features (same code as training)
        cands = generate_candidates(trip, radius_m=float(self.cfg["candidate_radius_m"]))
        t_cand = time.perf_counter()

        # 2. classifier scores P(candidate is the true arc)
        if len(cands):
            cands["score"] = score_model(self.model, cands[self.features].to_numpy())
        else:
            cands["score"] = pd.Series(dtype=float)
        t_score = time.perf_counter()

        # 3a. classifier-only route: highest-scoring candidate per point
        clf_arcs = np.full(n, -1, dtype=np.int64)
        if len(cands):
            top = cands.loc[cands.groupby("point_idx")["score"].idxmax()]
            clf_arcs[top["point_idx"].to_numpy()] = top["arc_id"].to_numpy()

        # 3b. HMM / Viterbi over the top-k candidates per point
        k = int(self.cfg.get("hmm_top_k") or 0)
        hmm_in = cands
        if k > 0 and len(cands):
            hmm_in = (cands.sort_values(["point_idx", "score"], ascending=[True, False])
                      .groupby("point_idx", sort=False).head(k))
        t_hmm0 = time.perf_counter()
        hmm_arcs = decode(trip, hmm_in, self.cfg["hmm_params"])
        t_hmm = time.perf_counter()

        # 4. metrics (reference available for train / student_test; maybe not for evaluator)
        has_ref = trip.route is not None and len(trip.route) > 0
        ref_arc = derive_ref_arcs(trip).ref_arc if has_ref else None
        adjacency = build_adjacency(trip.arcs)
        metrics = {
            "classifier": _route_metrics(trip, clf_arcs, ref_arc, adjacency),
            "hmm": _route_metrics(trip, hmm_arcs, ref_arc, adjacency),
        }
        changed = (clf_arcs != hmm_arcs) & (hmm_arcs >= 0)
        t_end = time.perf_counter()

        pts_with_cand = int(cands["point_idx"].nunique()) if len(cands) else 0
        cand_arc_ids = sorted(int(a) for a in cands["arc_id"].unique()) if len(cands) else []

        def ms(a, b):
            return round((b - a) * 1000.0, 1)

        return {
            "trip_id": trip_id,
            "partition": self.partition,
            "gps_source": gps_source,
            "n_points": int(n),
            "model": {"name": self.cfg["model_name"], "file": self.cfg["model_file"]},
            "has_reference": bool(has_ref),
            "timings_ms": {
                "load": ms(t0, t_load),
                "candidates": ms(t_load, t_cand),
                "scoring": ms(t_cand, t_score),
                "hmm_viterbi": ms(t_hmm0, t_hmm),
                "metrics": ms(t_hmm, t_end),
                "total": ms(t0, t_end),
            },
            "inference_ms_per_point": round((t_hmm - t_load) * 1000.0 / max(n, 1), 3),
            "candidates": {
                "n_rows": int(len(cands)),
                "mean_per_point": round(len(cands) / max(pts_with_cand, 1), 2),
                "points_without_candidates": int(n - pts_with_cand),
                "hmm_top_k": k,
            },
            "metrics": metrics,
            "n_points_changed_by_hmm": int(changed.sum()),
            "predictions": {
                "classifier_arcs": clf_arcs.tolist(),
                "hmm_arcs": hmm_arcs.tolist(),
                "classifier_route": _compact(clf_arcs),
                "hmm_route": _compact(hmm_arcs),
            },
            "geometry": {
                "gps": [[round(float(la), 6), round(float(lo), 6), float(t)]
                        for lo, la, t in trip.track[:, :3]],
                "road": _segments(trip, cand_arc_ids),
                "reference": _segments(trip, [int(a) for a in trip.route]) if has_ref else None,
                "classifier": _segments(trip, _compact(clf_arcs)),
                "hmm": _segments(trip, _compact(hmm_arcs)),
                "changed_points": np.flatnonzero(changed).tolist(),
            },
        }
