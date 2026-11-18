"""Build candidate tables (with reference labels) for a partition and cache them as parquet.

Usage: python scripts/build_candidates.py train student_test [--radius 40]
Output: cache/candidates_<partition>.parquet  (git-ignored)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.candidates import generate_candidates
from pathsnap.config import CACHE_DIR
from pathsnap.data import list_trips, load_trip
from pathsnap.labels import derive_ref_arcs

ap = argparse.ArgumentParser()
ap.add_argument("partitions", nargs="+")
ap.add_argument("--radius", type=float, default=40.0)
args = ap.parse_args()

CACHE_DIR.mkdir(exist_ok=True)
for part in args.partitions:
    frames = []
    for tid in tqdm(list_trips(part), desc=part):
        trip = load_trip(part, tid)
        frames.append(generate_candidates(trip, args.radius, derive_ref_arcs(trip).ref_arc))
    df = pd.concat(frames, ignore_index=True)
    out = CACHE_DIR / f"candidates_{part}.parquet"
    df.to_parquet(out, index=False)
    ref = df[df.has_ref]
    print(f"{part}: {len(df):,} rows, {df.trip_id.nunique()} trips, radius {args.radius} m, "
          f"positive rate (labelled points) {ref.label.mean():.3f} -> {out}")
