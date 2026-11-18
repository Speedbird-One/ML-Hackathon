"""Evaluation and verification script for HMM / Viterbi decoder.

Runs decode() on a real trip, checks execution speed, route continuity,
and sequence recovery against the reference route.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from pathsnap.data import list_trips, load_trip
from pathsnap.hmm import decode, generate_standin_candidates, DEFAULT_PARAMS
from pathsnap.metrics import route_continuity, build_adjacency


def evaluate_trip(trip_id: str, partition: str = "train"):
    print(f"\n==========================================")
    print(f" Evaluating HMM Decoder on Trip: {trip_id} ({partition})")
    print(f"==========================================")

    # 1. Load trip
    t_start = time.perf_counter()
    trip = load_trip(partition, trip_id)
    t_load = time.perf_counter() - t_start
    n_points = len(trip.track)
    print(f"[1/4] Loaded trip: {n_points} GPS points, {len(trip.arcs)} arcs ({t_load*1000:.1f} ms)")

    # 2. Generate candidate set
    t_start = time.perf_counter()
    candidates = generate_standin_candidates(trip, k=5, max_dist_m=100.0)
    t_cand = time.perf_counter() - t_start
    print(f"[2/4] Generated candidates: {len(candidates)} rows for {n_points} points ({t_cand*1000:.1f} ms)")

    # 3. Run HMM Viterbi decode
    t_start = time.perf_counter()
    predicted_arcs = decode(trip, candidates, DEFAULT_PARAMS)
    t_decode = time.perf_counter() - t_start
    print(f"[3/4] HMM Viterbi decoded in: {t_decode*1000:.1f} ms ({n_points / max(t_decode, 1e-6):.0f} pts/sec)")

    # 4. Analyze Route Quality
    valid_mask = predicted_arcs != -1
    n_valid = int(np.sum(valid_mask))
    print(f"\n--- Output Statistics ---")
    print(f"Total GPS points:         {n_points}")
    print(f"Decoded points:           {n_valid} / {n_points} ({100.0 * n_valid / n_points:.1f}%)")
    print(f"Missing (-1) points:      {n_points - n_valid}")

    # Route continuity — uses the same logic as pathsnap.metrics.route_continuity()
    # so our numbers match exactly what Teammate 3's experiment harness reports.
    adjacency = build_adjacency(trip.arcs)
    continuity_rate = route_continuity(predicted_arcs, trip.arcs, adjacency) * 100.0
    n_valid_transitions = sum(
        1 for i in range(len(predicted_arcs) - 1)
        if predicted_arcs[i] != -1 and predicted_arcs[i + 1] != -1
    )
    print(f"Route continuity (metrics.py): {continuity_rate:.2f}%  ({n_valid_transitions} valid transitions)")

    # Sequence collapse (consecutive duplicate arcs collapsed into a route sequence)
    compact_route = []
    for a in predicted_arcs:
        if a != -1 and (not compact_route or compact_route[-1] != a):
            compact_route.append(a)

    print(f"Predicted route length:   {len(compact_route)} unique road segments")

    # Ground truth comparison if route is available
    if trip.route is not None:
        gt_route = trip.route
        print(f"\n--- Ground Truth Comparison (.route) ---")
        print(f"Ground-truth route length: {len(gt_route)} road segments")

        # Set-level overlap of predicted vs ground-truth arcs
        pred_set = set(compact_route)
        gt_set = set(gt_route)
        intersection = pred_set.intersection(gt_set)
        jaccard = len(intersection) / len(pred_set.union(gt_set)) * 100.0
        gt_coverage = len(intersection) / len(gt_set) * 100.0

        print(f"Route recall (GT arcs hit): {len(intersection)} / {len(gt_set)} ({gt_coverage:.1f}%)")
        print(f"Route Jaccard similarity:   {jaccard:.1f}%")

    print(f"==========================================\n")
    return predicted_arcs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HMM Decoder on a trip")
    parser.add_argument("--trip_id", type=str, default="00000085", help="Trip ID to evaluate")
    parser.add_argument("--partition", type=str, default="train", help="train or student_test")
    args = parser.parse_args()

    evaluate_trip(args.trip_id, args.partition)
