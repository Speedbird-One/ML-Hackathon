"""Derive a reference arc for every GPS point from the ground-truth route.

The .route file is an ordered list of arc ids but does not say which arc each GPS
point belongs to. We recover that by aligning the track to the route in order:

  * every point is matched to a *position* k in the route (non-decreasing in time),
    minimising the sum of point-to-arc distances (dynamic programming);
  * distances are capped at `cap_m` inside the DP so a long stretch of off-route
    points (e.g. a parked vehicle with drifting GPS) cannot drag the alignment;
  * after alignment, points further than `max_dist_m` from their matched arc get
    label -1 (unknown) instead of being forced onto the nearest road.
"""
from dataclasses import dataclass

import numpy as np

from .data import Trip

UNKNOWN = -1


@dataclass
class RefLabels:
    ref_arc: np.ndarray   # (n,) arc id, or -1 when unknown
    route_pos: np.ndarray  # (n,) index into trip.route of the matched arc (even if unknown)
    dist_m: np.ndarray    # (n,) distance from point to its matched arc


def _point_segment_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Distances (n_points, n_segments) in metres, computed in chunks of points."""
    AB = B - A
    den = np.maximum((AB ** 2).sum(1), 1e-9)
    out = np.empty((len(P), len(A)), dtype=np.float32)
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        u = np.clip(((p - A) * AB).sum(2) / den, 0.0, 1.0)
        q = A + u[..., None] * AB
        out[s:s + chunk] = np.sqrt(((q - p) ** 2).sum(2))
    return out


def derive_ref_arcs(trip: Trip, max_dist_m: float = 50.0, cap_m: float = 50.0) -> RefLabels:
    if trip.route is None or len(trip.route) == 0:
        n = len(trip.track)
        return RefLabels(np.full(n, UNKNOWN), np.full(n, -1), np.full(n, np.inf, dtype=np.float32))

    xy = trip.nodes_xy()
    P = trip.track_xy()
    r = trip.route
    A, B = xy[trip.arcs[r, 0]], xy[trip.arcs[r, 1]]

    d = _point_segment_dist(P, A, B)               # (n, m)
    cost = np.minimum(d, cap_m)
    n, m = cost.shape

    # dp[i, k] = best total cost with point i matched to route position k and k non-decreasing.
    dp = np.empty((n, m), dtype=np.float32)
    dp[0] = cost[0]
    for i in range(1, n):
        dp[i] = cost[i] + np.minimum.accumulate(dp[i - 1])

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(dp[-1]))
    for i in range(n - 2, -1, -1):
        pos[i] = int(np.argmin(dp[i, : pos[i + 1] + 1]))

    dist = d[np.arange(n), pos]
    ref = np.where(dist <= max_dist_m, r[pos], UNKNOWN)
    return RefLabels(ref_arc=ref.astype(np.int64), route_pos=pos, dist_m=dist)
