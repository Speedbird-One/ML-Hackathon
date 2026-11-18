"""Regression tests for pathsnap.metrics. Run: python tests/test_metrics.py (or pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.metrics import (  # noqa: E402
    build_adjacency,
    candidate_miss_rate,
    evaluate_all,
    route_continuity,
    top1_accuracy,
)


def _table():
    """4 labelled points + 1 unlabelled point (has_ref False), one trip.

    p0: top pick correct          p1: top pick wrong, true arc present
    p2: true arc NOT a candidate  p3: top pick correct
    p4: no reference (all rows label 0, has_ref False)
    """
    rows = [
        # trip, point, arc, score, label, has_ref
        ("t", 0, 10, 0.9, 1, True), ("t", 0, 11, 0.1, 0, True),
        ("t", 1, 20, 0.8, 0, True), ("t", 1, 21, 0.2, 1, True),
        ("t", 2, 30, 0.7, 0, True), ("t", 2, 31, 0.3, 0, True),
        ("t", 3, 40, 0.6, 1, True), ("t", 3, 41, 0.4, 0, True),
        ("t", 4, 50, 0.9, 0, False), ("t", 4, 51, 0.1, 0, False),
    ]
    return pd.DataFrame(rows, columns=["trip_id", "point_idx", "arc_id", "score", "label", "has_ref"])


def test_miss_rate_is_not_silently_zero():
    df = _table()
    # labelled points: 4; point 2 is missed -> 1/4. The old code returned 0.0 here.
    assert abs(candidate_miss_rate(df) - 0.25) < 1e-9
    assert abs(evaluate_all(np.array([1, 0]), np.array([1, 0]), np.array([0.9, 0.1]), df)["candidate_miss_rate"] - 0.25) < 1e-9


def test_miss_rate_ignores_unlabelled_points():
    df = _table()
    # point 4 has no reference; it must not be counted as a miss (it would give 2/5 = 0.4)
    assert candidate_miss_rate(df) != 0.4


def test_miss_rate_counts_points_with_no_candidates():
    df = _table()
    # 6 labelled points in truth, 2 of which never produced a candidate row -> 3 misses of 6
    assert abs(candidate_miss_rate(df, n_labelled_points=6) - 0.5) < 1e-9


def test_top1_counts_misses_as_wrong_and_skips_unlabelled():
    df = _table()
    # correct: p0, p3 -> 2 of 4 labelled points (p2 is a miss => wrong, p4 excluded)
    assert abs(top1_accuracy(df) - 0.5) < 1e-9


def test_top1_dict_reference():
    df = _table()
    ref = {("t", 0): 10, ("t", 1): 21, ("t", 2): 99, ("t", 3): 40, ("t", 4): -1}
    assert abs(top1_accuracy(df, ref_arcs=ref) - 0.5) < 1e-9


def test_route_continuity():
    arcs = np.array([[0, 1], [1, 2], [2, 3], [5, 6]])  # arc0->arc1->arc2 chain; arc3 isolated
    adj = build_adjacency(arcs)
    assert route_continuity([0, 1, 2], arcs, adj) == 1.0
    assert route_continuity([0, 0, 1], arcs, adj) == 1.0           # staying on an arc is allowed
    assert route_continuity([0, 3, 1], arcs, adj) == 0.0           # jumps are not connected
    assert route_continuity([0, -1, 1], arcs, adj) == 0.0          # unknown breaks the chain
    assert route_continuity([0], arcs, adj) == 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
