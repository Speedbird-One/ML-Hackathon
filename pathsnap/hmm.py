"""Hidden Markov Model (HMM) and Viterbi decoder for route reconstruction.

This module turns per-candidate road segments and classifier scores into
a coherent, topologically connected route across an entire GPS trip.

Key Concepts:
- Hidden State: The true road arc (arc_id) traversed at each GPS point.
- Emission: P(GPS observation | candidate arc) based on classifier score
  or Gaussian error on distance (Newson-Krumm).
- Transition: P(arc_j | arc_i) based on road-network shortest path distance
  versus GPS straight-line distance (Newson-Krumm exponential model).
- Viterbi: Log-probability dynamic programming with gap handling and
  sequence restart for disconnected segments.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import Trip

DEFAULT_PARAMS: Dict[str, Any] = {
    # Emission parameter: GPS error standard deviation in metres
    "sigma": 4.07,
    # Transition parameter: exponential scale in metres
    "beta": 5.0,
    # Shortest path network search cutoff in metres
    "max_net_dist_m": 300.0,
    # Temporal gap threshold: jump > gap_s breaks the sequence (seconds)
    "gap_s": 30.0,
    # Spatial gap threshold: jump > gap_m breaks the sequence (metres)
    "gap_m": 200.0,
    # Log-probability penalty for unreachable transitions
    "unreachable_penalty": -1000.0,
    # Whether to use candidate['score'] if present in the DataFrame
    "use_score": True,
    # Tolerance for small negative displacement on the same arc due to GPS jitter (metres)
    "jitter_tol_m": 10.0,
}


@dataclass
class RoadGraph:
    """Precomputed road network graph and arc geometry in projected metres."""
    graph: nx.DiGraph
    nodes_xy: np.ndarray       # (n_nodes, 2)
    arcs: np.ndarray           # (n_arcs, 2) [from_node, to_node]
    arc_vecs: np.ndarray       # (n_arcs, 2)
    arc_lengths: np.ndarray    # (n_arcs,)


def build_road_graph(trip: Trip) -> RoadGraph:
    """Build directed graph and vector geometries from Trip."""
    nodes_xy = trip.nodes_xy()
    arcs = trip.arcs.astype(np.int64)

    from_nodes = arcs[:, 0]
    to_nodes = arcs[:, 1]

    # Direction vectors and segment lengths in metres
    arc_vecs = nodes_xy[to_nodes] - nodes_xy[from_nodes]
    arc_lengths = np.maximum(np.hypot(arc_vecs[:, 0], arc_vecs[:, 1]), 0.1)

    # Build directed graph with edge weights in metres
    g = nx.DiGraph()
    for arc_id in range(len(arcs)):
        u = int(from_nodes[arc_id])
        v = int(to_nodes[arc_id])
        w = float(arc_lengths[arc_id])
        # Keep shortest edge weight if duplicate (u, v) pairs exist
        if g.has_edge(u, v):
            if w < g[u][v]["weight"]:
                g[u][v]["weight"] = w
                g[u][v]["arc_id"] = arc_id
        else:
            g.add_edge(u, v, weight=w, arc_id=arc_id)

    return RoadGraph(
        graph=g,
        nodes_xy=nodes_xy,
        arcs=arcs,
        arc_vecs=arc_vecs,
        arc_lengths=arc_lengths,
    )


class ShortestPathCache:
    """Cached Dijkstra shortest-path query handler with search cutoff."""

    def __init__(self, graph: nx.DiGraph, cutoff: float):
        self.graph = graph
        self.cutoff = cutoff
        self._cache: Dict[int, Dict[int, float]] = {}

    def get_distance(self, u: int, v: int) -> Optional[float]:
        """Return shortest network distance from u to v in metres, or None if unreachable."""
        if u == v:
            return 0.0

        if u not in self._cache:
            try:
                self._cache[u] = nx.single_source_dijkstra_path_length(
                    self.graph, source=u, cutoff=self.cutoff, weight="weight"
                )
            except nx.NetworkXError:
                self._cache[u] = {}

        return self._cache[u].get(v, None)


def _compute_projection(
    point_xy: np.ndarray,
    arc_id: int,
    road: RoadGraph,
) -> Tuple[float, float, float, float]:
    """Compute point-to-arc projection, offset from start, dist to end, and dist to arc."""
    u = road.arcs[arc_id, 0]
    v = road.arcs[arc_id, 1]
    a_pos = road.nodes_xy[u]
    vec = road.arc_vecs[arc_id]
    l_arc = road.arc_lengths[arc_id]

    w = point_xy - a_pos
    l2 = l_arc * l_arc
    t_proj = float(np.clip(np.dot(w, vec) / l2, 0.0, 1.0))
    proj_xy = a_pos + t_proj * vec

    dist_m = float(np.hypot(point_xy[0] - proj_xy[0], point_xy[1] - proj_xy[1]))
    d_start = t_proj * l_arc
    d_end = (1.0 - t_proj) * l_arc

    return t_proj, d_start, d_end, dist_m


def generate_standin_candidates(
    trip: Trip,
    k: int = 5,
    max_dist_m: float = 100.0,
) -> pd.DataFrame:
    """Generate stand-in candidate road arcs for each GPS observation.

    Used for self-contained testing before task (2) candidate generation arrives.
    """
    track_xy = trip.track_xy()
    nodes_xy = trip.nodes_xy()
    arcs = trip.arcs

    from_nodes = arcs[:, 0]
    to_nodes = arcs[:, 1]
    midpoints = (nodes_xy[from_nodes] + nodes_xy[to_nodes]) / 2.0

    tree = cKDTree(midpoints)
    n_points = len(track_xy)
    search_k = min(k * 4, len(arcs))

    rows = []
    # Query spatial tree
    dists_approx, indices = tree.query(track_xy, k=search_k)
    if search_k == 1:
        dists_approx = dists_approx[:, None]
        indices = indices[:, None]

    road = build_road_graph(trip)

    for i in range(n_points):
        p_xy = track_xy[i]
        cand_arcs = indices[i]

        pt_candidates = []
        for arc_id in cand_arcs:
            _, _, _, dist_m = _compute_projection(p_xy, arc_id, road)
            if dist_m <= max_dist_m:
                pt_candidates.append((arc_id, dist_m))

        # Sort by distance and keep top k
        pt_candidates.sort(key=lambda x: x[1])
        pt_candidates = pt_candidates[:k]

        # If none within max_dist_m, take the single closest
        if not pt_candidates and len(cand_arcs) > 0:
            best_arc = cand_arcs[0]
            _, _, _, best_d = _compute_projection(p_xy, best_arc, road)
            pt_candidates.append((best_arc, best_d))

        for arc_id, dist_m in pt_candidates:
            rows.append({
                "trip_id": trip.trip_id,
                "point_idx": i,
                "arc_id": int(arc_id),
                "dist_m": float(dist_m),
                # Stand-in score using Newson-Krumm Gaussian (reads sigma from DEFAULT_PARAMS)
                "score": float(np.exp(-0.5 * (dist_m / DEFAULT_PARAMS["sigma"]) ** 2)),
            })

    return pd.DataFrame(rows)


def decode(
    trip: Trip,
    candidates: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Decode the most likely sequence of road arcs for a GPS trajectory.

    Parameters
    ----------
    trip : Trip
        The trip object with track, nodes, and arcs.
    candidates : pd.DataFrame
        DataFrame with columns ['point_idx', 'arc_id'] and optionally
        ['dist_m', 'score'].
    params : dict, optional
        Tuning parameters overriding DEFAULT_PARAMS.

    Returns
    -------
    np.ndarray
        Array of shape (n_points,) of predicted arc_ids (-1 where no prediction).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    sigma = float(p["sigma"])
    beta = float(p["beta"])
    max_net_dist_m = float(p["max_net_dist_m"])
    gap_s = float(p["gap_s"])
    gap_m = float(p["gap_m"])
    unreach_pen = float(p["unreachable_penalty"])
    use_score = bool(p["use_score"])
    jitter_tol_m = float(p["jitter_tol_m"])

    track_xy = trip.track_xy()
    timestamps = trip.track[:, 2]
    n_points = len(track_xy)

    predictions = np.full(n_points, -1, dtype=np.int64)
    if n_points == 0 or len(candidates) == 0:
        return predictions

    # 1. Build road graph and shortest-path cache
    road = build_road_graph(trip)
    sp_cache = ShortestPathCache(road.graph, cutoff=max_net_dist_m)

    # 2. Group candidates by point_idx
    has_score_col = use_score and ("score" in candidates.columns)
    has_dist_col = "dist_m" in candidates.columns
    # Tanmay's generate_candidates() produces along_frac and arc_len_m; use them directly
    # to skip redundant _compute_projection calls per row.
    has_along_frac = "along_frac" in candidates.columns
    has_arc_len = "arc_len_m" in candidates.columns

    # Pre-parse candidate records into per-point candidate lists
    by_point: Dict[int, List[Dict[str, Any]]] = {}
    for row in candidates.itertuples(index=False):
        pt_idx = int(getattr(row, "point_idx"))
        arc_id = int(getattr(row, "arc_id"))

        if pt_idx < 0 or pt_idx >= n_points:
            continue

        p_xy = track_xy[pt_idx]

        # Use pre-computed projection columns from generate_candidates() when available
        # (along_frac = u parameter 0-1, arc_len_m = segment length in metres).
        # Fall back to _compute_projection only if those columns are absent.
        if has_along_frac and has_arc_len:
            t_proj  = float(getattr(row, "along_frac"))
            l_arc   = float(getattr(row, "arc_len_m"))
            d_start = t_proj * l_arc
            d_end   = (1.0 - t_proj) * l_arc
            d_m     = float(getattr(row, "dist_m")) if has_dist_col else _compute_projection(p_xy, arc_id, road)[3]
        else:
            t_proj, d_start, d_end, d_m = _compute_projection(p_xy, arc_id, road)

        dist_m = float(getattr(row, "dist_m")) if has_dist_col else d_m

        # Compute log emission
        if has_score_col:
            sc = float(getattr(row, "score"))
            if np.isnan(sc) or sc <= 0.0:
                # Fallback to Gaussian
                log_emission = -0.5 * (dist_m / sigma) ** 2 - np.log(np.sqrt(2.0 * np.pi) * sigma)
            else:
                log_emission = np.log(np.clip(sc, 1e-7, 1.0))
        else:
            log_emission = -0.5 * (dist_m / sigma) ** 2 - np.log(np.sqrt(2.0 * np.pi) * sigma)

        cand_info = {
            "arc_id": arc_id,
            "dist_m": dist_m,
            "t_proj": t_proj,
            "d_start": d_start,
            "d_end": d_end,
            "log_emission": float(log_emission),
        }

        if pt_idx not in by_point:
            by_point[pt_idx] = []
        by_point[pt_idx].append(cand_info)

    # 3. Identify continuous segments (split on missing candidates, time gaps, or space gaps)
    segments: List[List[int]] = []
    current_segment: List[int] = []

    for i in range(n_points):
        if i not in by_point or len(by_point[i]) == 0:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue

        if current_segment:
            prev_i = current_segment[-1]
            dt = timestamps[i] - timestamps[prev_i]
            d_straight = np.hypot(
                track_xy[i, 0] - track_xy[prev_i, 0],
                track_xy[i, 1] - track_xy[prev_i, 1],
            )

            if dt > gap_s or d_straight > gap_m:
                # Break sequence at gap
                segments.append(current_segment)
                current_segment = [i]
            else:
                current_segment.append(i)
        else:
            current_segment.append(i)

    if current_segment:
        segments.append(current_segment)

    # 4. Run Viterbi on each segment
    log_beta = np.log(beta)

    for seg in segments:
        seg_len = len(seg)
        if seg_len == 0:
            continue

        if seg_len == 1:
            pt = seg[0]
            best_cand = max(by_point[pt], key=lambda c: c["log_emission"])
            predictions[pt] = best_cand["arc_id"]
            continue

        # Viterbi table: list of dicts mapping candidate_idx -> log_prob
        # Backpointers: list of dicts mapping candidate_idx -> prev_candidate_idx
        v_probs: List[np.ndarray] = []
        v_backptr: List[np.ndarray] = []

        # Step 0: Initial emissions
        init_cands = by_point[seg[0]]
        v0 = np.array([c["log_emission"] for c in init_cands], dtype=np.float64)
        v_probs.append(v0)
        v_backptr.append(np.full(len(init_cands), -1, dtype=np.int32))

        # Forward DP
        for t in range(1, seg_len):
            prev_pt = seg[t - 1]
            curr_pt = seg[t]

            prev_cands = by_point[prev_pt]
            curr_cands = by_point[curr_pt]
            n_prev = len(prev_cands)
            n_curr = len(curr_cands)

            d_great = float(np.hypot(
                track_xy[curr_pt, 0] - track_xy[prev_pt, 0],
                track_xy[curr_pt, 1] - track_xy[prev_pt, 1],
            ))

            curr_v = np.full(n_curr, -np.inf, dtype=np.float64)
            curr_bp = np.full(n_curr, -1, dtype=np.int32)

            for j in range(n_curr):
                cj = curr_cands[j]
                arc_j = cj["arc_id"]
                log_em_j = cj["log_emission"]

                best_p = -np.inf
                best_i = -1

                for i in range(n_prev):
                    ci = prev_cands[i]
                    arc_i = ci["arc_id"]
                    v_prev_i = v_probs[t - 1][i]

                    # Transition distance computation
                    if arc_i == arc_j:
                        # Same arc
                        if cj["t_proj"] >= ci["t_proj"]:
                            d_route = cj["d_start"] - ci["d_start"]
                        else:
                            backwards = ci["d_start"] - cj["d_start"]
                            if backwards <= jitter_tol_m:
                                d_route = 0.0
                            else:
                                # Vehicle went backwards beyond tolerance; check network loop
                                u_node = road.arcs[arc_i, 1]
                                v_node = road.arcs[arc_j, 0]
                                net_d = sp_cache.get_distance(u_node, v_node)
                                if net_d is not None:
                                    d_route = ci["d_end"] + net_d + cj["d_start"]
                                else:
                                    d_route = None
                    else:
                        u_node = road.arcs[arc_i, 1]
                        v_node = road.arcs[arc_j, 0]
                        # Short-circuit: if the end of arc_i IS the start of arc_j, net_d = 0
                        if u_node == v_node:
                            d_route = ci["d_end"] + cj["d_start"]
                        else:
                            net_d = sp_cache.get_distance(u_node, v_node)
                            if net_d is not None:
                                d_route = ci["d_end"] + net_d + cj["d_start"]
                            else:
                                d_route = None

                    if d_route is None:
                        log_trans = unreach_pen
                    else:
                        diff = abs(d_route - d_great)
                        log_trans = -log_beta - (diff / beta)

                    total_p = v_prev_i + log_trans + log_em_j
                    if total_p > best_p:
                        best_p = total_p
                        best_i = i

                # Unreachable transitions already carry the finite penalty `unreach_pen`, so Viterbi
                # handles them without special casing. (An earlier version reset the chain whenever
                # the ACCUMULATED log-probability fell below unreach_pen + 1; that is true after a few
                # hundred points on any trip, so it silently restarted decoding at almost every step.)
                curr_v[j] = best_p
                curr_bp[j] = best_i

            # Check if all states in current step reset: ensure finite probabilities
            if np.all(np.isneginf(curr_v)):
                curr_v = np.array([c["log_emission"] for c in curr_cands], dtype=np.float64)
                curr_bp = np.full(n_curr, -1, dtype=np.int32)

            v_probs.append(curr_v)
            v_backptr.append(curr_bp)

        # Backtrack optimal path for this segment
        best_end_idx = int(np.argmax(v_probs[-1]))
        chosen_indices = [best_end_idx]

        for t in range(seg_len - 1, 0, -1):
            prev_idx = v_backptr[t][chosen_indices[-1]]
            if prev_idx == -1:
                # Sequence had restarted; choose best available state at t-1
                prev_idx = int(np.argmax(v_probs[t - 1]))
            chosen_indices.append(prev_idx)

        chosen_indices.reverse()

        for t in range(seg_len):
            pt = seg[t]
            chosen_cand = by_point[pt][chosen_indices[t]]
            predictions[pt] = chosen_cand["arc_id"]

    return predictions
