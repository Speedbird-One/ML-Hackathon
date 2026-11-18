"""Regression tests for pathsnap.hmm. Run: python tests/test_hmm.py (or pytest)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.data import Trip  # noqa: E402
from pathsnap.hmm import decode  # noqa: E402

STEP_DEG = 9e-5  # about 10 m of longitude at the equator


def _line_trip(n_points: int):
    """A straight chain of n_points arcs (node k -> node k+1, ~10 m each) plus an isolated 'decoy' arc."""
    n_nodes = n_points + 1
    nodes = np.column_stack([np.arange(n_nodes) * STEP_DEG, np.zeros(n_nodes)])
    chain = np.column_stack([np.arange(n_points), np.arange(1, n_points + 1)])
    decoy_nodes = np.array([[0.0, 0.0002], [STEP_DEG, 0.0002]])  # not connected to the chain
    all_nodes = np.vstack([nodes, decoy_nodes])
    decoy_arc = np.array([[n_nodes, n_nodes + 1]])
    arcs = np.vstack([chain, decoy_arc])
    track = np.column_stack([np.arange(n_points) * STEP_DEG, np.zeros(n_points), np.arange(1, n_points + 1)])
    return Trip("synthetic", track, all_nodes, arcs, None), n_nodes


def test_long_trip_is_not_reset_by_accumulated_score():
    """After ~600 points the accumulated log-probability is below -1000. The decoder must still
    follow the connected chain, not jump to a higher-scoring but unreachable decoy arc."""
    n = 1000
    trip, _ = _line_trip(n)
    decoy_arc = n  # index of the isolated arc
    rows = []
    for k in range(n):
        rows.append((k, k, 0.0, 0.0, 10.0, 0.90))                  # true chain arc
        if k >= 700:                                                 # decoy outscores the truth late in the trip
            rows.append((k, decoy_arc, 3.0, 0.5, 10.0, 0.95))
    cands = pd.DataFrame(rows, columns=["point_idx", "arc_id", "dist_m", "along_frac", "arc_len_m", "score"])
    pred = decode(trip, cands)
    assert (pred == np.arange(n)).all(), f"{int((pred != np.arange(n)).sum())} of {n} points left the chain"


def test_short_trip_follows_chain():
    n = 50
    trip, _ = _line_trip(n)
    rows = [(k, k, 0.0, 0.0, 10.0, 0.9) for k in range(n)]
    cands = pd.DataFrame(rows, columns=["point_idx", "arc_id", "dist_m", "along_frac", "arc_len_m", "score"])
    assert (decode(trip, cands) == np.arange(n)).all()


def test_points_without_candidates_return_minus_one():
    n = 20
    trip, _ = _line_trip(n)
    rows = [(k, k, 0.0, 0.0, 10.0, 0.9) for k in range(n) if k != 10]
    cands = pd.DataFrame(rows, columns=["point_idx", "arc_id", "dist_m", "along_frac", "arc_len_m", "score"])
    pred = decode(trip, cands)
    assert pred[10] == -1 and (np.delete(pred, 10) == np.delete(np.arange(n), 10)).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
