"""Evaluate the HMM/Viterbi decoder against classifier-only predictions on the student-test trips.

This is the report's section-5 comparison. It scores every candidate with the final classifier, then reports, on all
10 student-test trips:
  * classifier only        : highest-scoring candidate per GPS point
  * classifier + HMM (default parameters)
  * classifier + HMM (tuned parameters), if artifacts/hmm_params.json exists (written by scripts/tune_hmm.py)
Metrics: point-level top-1 (all labelled points; a point with no prediction counts as wrong), top-1 split into slow
(< 1 m/s, heading unreliable) and moving points, route continuity, sequence-level route recall/precision/F1, and
decoder runtime. It also lists every trip where the HMM did not help, and the trip-by-trip change.
The decoder runs on EVERY point's candidates, as at inference; only points with a reference arc are scored.
Only student-test data is used (the evaluator trips are never touched).

Usage (repo root):
    python scripts/eval_hmm.py                                  # final model: artifacts/lgbm_tuned.joblib
    python scripts/eval_hmm.py --model mlp                      # any saved model in artifacts/
    python scripts/eval_hmm.py --params artifacts/hmm_params.json --jobs 4
Outputs in results/hmm/: hmm_eval_student_test.csv/.md, hmm_eval_per_trip.csv, hmm_eval_where_not_helped.md
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from joblib import Parallel

from pathsnap.config import ARTIFACTS_DIR, CACHE_DIR, RESULTS_DIR
from pathsnap.decode_eval import (argmax_predictions, decode_trips, evaluate, load_hmm_params, load_trips_and_refs,
                                  point_speeds, score_candidates, summarise)
from pathsnap.hmm import DEFAULT_PARAMS

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--model", default="lgbm_tuned", help="artifact name in artifacts/ (without .joblib)")
ap.add_argument("--params", type=Path, default=ARTIFACTS_DIR / "hmm_params.json", help="tuned decoder parameters (optional)")
ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "hmm")
args = ap.parse_args()
warnings.filterwarnings("ignore")
args.out_dir.mkdir(parents=True, exist_ok=True)

model_path = ARTIFACTS_DIR / f"{args.model}.joblib"
if not model_path.exists():
    sys.exit(f"{model_path} not found")

cands = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")
trip_ids = sorted(cands.trip_id.unique())
trips, refs, adjacency = load_trips_and_refs("student_test", trip_ids)
scored = score_candidates(joblib.load(model_path), cands)
speeds = point_speeds(scored, trips)
n_lab = int(sum((r >= 0).sum() for r in refs.values()))
print(f"student-test: {len(trips)} trips, {n_lab} labelled points, {len(scored):,} candidate rows; model: {args.model}; workers: {args.jobs}")

decoders = [("HMM, default parameters", dict(DEFAULT_PARAMS))]
have_tuned = args.params.exists()
if have_tuned:
    decoders.append(("HMM, tuned parameters", load_hmm_params(args.params)))
else:
    print(f"[note] {args.params} not found: only default parameters evaluated (run scripts/tune_hmm.py first for tuned ones)")

results = {"Classifier only": (argmax_predictions(scored, trips), None, None)}
with Parallel(n_jobs=args.jobs) as pool:
    for name, params in decoders:
        t0 = time.perf_counter()
        preds, cpu_s = decode_trips("student_test", scored, params, pool)
        results[name] = (preds, cpu_s, time.perf_counter() - t0)

n_points = int(sum(len(t.track) for t in trips.values()))
rows, per_trip = [], {}
for name, (preds, cpu_s, wall_s) in results.items():
    df = evaluate(preds, trips, refs, adjacency, speeds)
    per_trip[name] = df
    m = summarise(df)
    rows.append({"Method": name, "Top-1": m["top1"], "Top-1 moving": m["top1_moving"], "Top-1 slow (<1 m/s)": m["top1_slow"],
                 "Route continuity": m["continuity"], "Route recall": m["route_recall"], "Route precision": m["route_precision"],
                 "Route F1": m["route_f1"], "Decode ms/point (CPU)": (1000 * cpu_s / n_points) if cpu_s is not None else float("nan"),
                 "Decode wall s": wall_s if wall_s is not None else float("nan")})
summary = pd.DataFrame(rows)
summary.to_csv(args.out_dir / "hmm_eval_student_test.csv", index=False)
(args.out_dir / "hmm_eval_student_test.md").write_text(summary.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

# trip-by-trip table and the cases where the HMM did not help
base = per_trip["Classifier only"].set_index("trip")
long = []
for name, df in per_trip.items():
    d = df.set_index("trip")
    long.append(d[["labelled", "top1", "continuity", "route_f1"]].assign(method=name).reset_index())
pd.concat(long).to_csv(args.out_dir / "hmm_eval_per_trip.csv", index=False)

lines = ["# Where the HMM did not help (student-test)\n"]
for name, _ in decoders:
    d = per_trip[name].set_index("trip")
    delta = (d.top1 - base.top1) * 100
    worse = delta[delta < 0].sort_values()
    lines.append(f"## {name}")
    lines.append(f"- top-1 change per trip (points): min {delta.min():+.2f}, median {delta.median():+.2f}, max {delta.max():+.2f}")
    lines.append(f"- trips where the HMM was worse than classifier-only: {len(worse)} of {len(delta)}")
    for t, v in worse.items():
        lines.append(f"  - trip {t}: {v:+.2f} points ({int(base.loc[t, 'labelled'])} labelled points)")
    sl_b = per_trip['Classifier only'].slow_correct.sum() / max(per_trip['Classifier only'].slow_labelled.sum(), 1)
    sl_h = per_trip[name].slow_correct.sum() / max(per_trip[name].slow_labelled.sum(), 1)
    lines.append(f"- slow points (<1 m/s): classifier-only {sl_b:.3f} -> HMM {sl_h:.3f} "
                 f"({int(per_trip[name].slow_labelled.sum())} points)")
    lines.append(f"- continuity: classifier-only {per_trip['Classifier only'].continuity.mean():.3f} -> HMM {d.continuity.mean():.3f}\n")
(args.out_dir / "hmm_eval_where_not_helped.md").write_text("\n".join(lines), encoding="utf-8")

print("\n=== CLASSIFIER ONLY vs CLASSIFIER + HMM (student-test) ===")
print(summary.to_markdown(index=False, floatfmt=".4f"))
print("\n" + "\n".join(lines))
print(f"\nSaved to {args.out_dir}")
