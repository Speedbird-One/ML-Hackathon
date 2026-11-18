"""Classifier-only vs classifier + HMM/Viterbi on every trip of a partition (report Section 5 & 6).

Uses the exact inference code the API serves (pathsnap/inference.py + artifacts/inference_config.json).

    python scripts/eval_pipeline.py                       # student_test, all 10 trips
    python scripts/eval_pipeline.py --partition student_test --trips 00000000 00000003

Outputs:
    results/logs/pipeline_<partition>.csv          per-trip + mean metrics and timings
    results/figures/hmm_vs_classifier_<partition>.png
    results/maps/route_<partition>_<trip>.html     interactive Folium map per trip
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import RESULTS_DIR
from pathsnap.inference import PathSnapPredictor

METRICS = ["top1_accuracy", "route_continuity", "route_recall", "route_precision", "route_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--trips", nargs="*")
    ap.add_argument("--no-maps", action="store_true")
    args = ap.parse_args()

    pred = PathSnapPredictor(args.partition)
    ids = args.trips or pred.trip_ids()
    rows = []
    for tid in ids:
        r = pred.predict(tid, use_cache=False)
        row = {"trip_id": tid, "n_points": r["n_points"],
               "points_changed_by_hmm": r["n_points_changed_by_hmm"],
               "candidate_ms": r["timings_ms"]["candidates"], "scoring_ms": r["timings_ms"]["scoring"],
               "hmm_ms": r["timings_ms"]["hmm_viterbi"], "ms_per_point": r["inference_ms_per_point"]}
        for stage in ("classifier", "hmm"):
            for k in METRICS:
                row[f"{stage}_{k}"] = r["metrics"][stage][k]
        rows.append(row)
        print(f"{tid}: top-1 {row['classifier_top1_accuracy']} -> {row['hmm_top1_accuracy']} | "
              f"continuity {row['classifier_route_continuity']:.3f} -> {row['hmm_route_continuity']:.3f} | "
              f"HMM {row['hmm_ms']:.0f} ms")
        if not args.no_maps:
            from ui.route_map import build_route_map
            out = RESULTS_DIR / "maps" / f"route_{args.partition}_{tid}.html"
            out.parent.mkdir(parents=True, exist_ok=True)
            build_route_map(r).save(str(out))

    df = pd.DataFrame(rows)
    mean = df.drop(columns=["trip_id"]).mean(numeric_only=True)
    df = pd.concat([df, pd.DataFrame([{"trip_id": "MEAN", **mean.to_dict()}])], ignore_index=True)
    (RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    csv = RESULTS_DIR / "logs" / f"pipeline_{args.partition}.csv"
    df.to_csv(csv, index=False)
    print(f"\nSaved {csv}")
    print(df[["trip_id"] + [f"{s}_{k}" for k in METRICS[:2] for s in ("classifier", "hmm")]].to_string(index=False))

    # bar chart of mean metrics: classifier-only vs + HMM (real results only)
    labels = [k.replace("_", " ") for k in METRICS]
    c = [mean.get(f"classifier_{k}", np.nan) for k in METRICS]
    h = [mean.get(f"hmm_{k}", np.nan) for k in METRICS]
    x = np.arange(len(METRICS))
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.bar(x - 0.2, c, 0.4, label=f"{pred.cfg['model_name']} only", color="#ff7f0e")
    ax.bar(x + 0.2, h, 0.4, label=f"{pred.cfg['model_name']} + HMM/Viterbi", color="#d62728")
    for xi, v in zip(x - 0.2, c):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + 0.2, h):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Classifier-only vs HMM decoding ({args.partition}, mean over {len(ids)} trips)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png = RESULTS_DIR / "figures" / f"hmm_vs_classifier_{args.partition}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png)
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
