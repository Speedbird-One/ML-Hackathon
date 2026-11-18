"""Candidate road arcs for every GPS point, with base features.

One row per (GPS point, nearby arc). Arcs are straight segments between their two nodes.
Rows carry `label` (1 if arc == reference arc) and `has_ref` (False where the point has no
reference arc, i.e. label -1 from labels.py). Training and top-1 metrics must keep has_ref only.
"""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .data import Trip
from .labels import UNKNOWN


def _motion(P: np.ndarray, t: np.ndarray, half_window: int = 2):
    """Velocity (m/s) per point by central difference over +-half_window samples."""
    n = len(P)
    lo = np.clip(np.arange(n) - half_window, 0, n - 1)
    hi = np.clip(np.arange(n) + half_window, 0, n - 1)
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    return (P[hi] - P[lo]) / dt[:, None]


def generate_candidates(trip: Trip, radius_m: float = 40.0, ref_arc: np.ndarray | None = None) -> pd.DataFrame:
    P = trip.track_xy()
    t = trip.track[:, 2]
    xy = trip.nodes_xy()
    A_all, B_all = xy[trip.arcs[:, 0]], xy[trip.arcs[:, 1]]

    tree = STRtree(shapely.linestrings(np.stack([A_all, B_all], axis=1)))
    pt_idx, arc_id = tree.query(shapely.points(P), predicate="dwithin", distance=radius_m)

    A, B, p = A_all[arc_id], B_all[arc_id], P[pt_idx]
    AB = B - A
    seg_len = np.hypot(AB[:, 0], AB[:, 1])
    u = np.clip(((p - A) * AB).sum(1) / np.maximum(seg_len ** 2, 1e-9), 0.0, 1.0)
    dist = np.hypot(*(A + u[:, None] * AB - p).T)
    keep = dist <= radius_m  # dwithin is exact; guard against float edge cases
    pt_idx, arc_id, u, dist, seg_len, AB = pt_idx[keep], arc_id[keep], u[keep], dist[keep], seg_len[keep], AB[keep]

    v = _motion(P, t)[pt_idx]
    speed = np.hypot(v[:, 0], v[:, 1])
    cos_head = np.where(speed > 1.0, (v * AB).sum(1) / np.maximum(speed * seg_len, 1e-9), 0.0)

    df = pd.DataFrame({
        "trip_id": trip.trip_id,
        "point_idx": pt_idx.astype(np.int32),
        "arc_id": arc_id.astype(np.int32),
        "dist_m": dist.astype(np.float32),
        "along_frac": u.astype(np.float32),
        "arc_len_m": seg_len.astype(np.float32),
        "speed_mps": speed.astype(np.float32),
        "cos_heading": cos_head.astype(np.float32),
    })
    g = df.groupby("point_idx")["dist_m"]
    df["dist_rank"] = g.rank(method="first").astype(np.int16)
    df["dist_minus_min"] = (df["dist_m"] - g.transform("min")).astype(np.float32)
    df["n_cand"] = g.transform("size").astype(np.int16)

    if ref_arc is not None:
        r = ref_arc[df["point_idx"].to_numpy()]
        df["has_ref"] = r != UNKNOWN
        df["label"] = (df["arc_id"].to_numpy() == r).astype(np.int8)
    return df.sort_values(["point_idx", "dist_m"], kind="stable", ignore_index=True)
