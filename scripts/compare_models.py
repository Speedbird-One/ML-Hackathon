"""Compare the five trained models on the 10 student-test trips, from their SAVED model files.

Every model is loaded from artifacts/<name>.joblib and scored on the same candidate table, so the numbers
are directly comparable and do not depend on who trained which model or on which machine.

Reported per model:
  * candidate level (is this candidate the reference arc?): precision, recall, macro F1, log loss,
    Brier score, expected calibration error (probability quality matters because the HMM uses the scores)
  * point level: top-1 accuracy (highest-scoring candidate vs reference arc, on ALL labelled points; a
    point whose true arc is not a candidate counts as wrong), route continuity of those picks
  * the same two numbers after HMM/Viterbi decoding of that model's scores
  * scoring time per GPS point and model file size

Only student-test data is used. Usage (repo root, candidate tables built, models in artifacts/):
    python scripts/compare_models.py            # full comparison incl. HMM decoding (about 1-2 minutes)
    python scripts/compare_models.py --no-hmm   # skip the HMM columns
Outputs: results/model_comparison/ (student_test_comparison.csv/.md, per_trip_top1.csv, comparison.png)
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from pathsnap.config import ARTIFACTS_DIR, CACHE_DIR, RESULTS_DIR
from pathsnap.data import list_trips, load_trip
from pathsnap.hmm import decode
from pathsnap.labels import derive_ref_arcs
from pathsnap.metrics import build_adjacency, candidate_metrics, route_continuity

FEATURES = ["dist_m", "along_frac", "arc_len_m", "speed_mps", "cos_heading",
            "dist_rank", "dist_minus_min", "n_cand"]
MODELS = [  # (display name, artifact file stem)
    ("Logistic Regression", "logreg"),
    ("Gaussian Naive Bayes", "nb"),
    ("Random Forest", "rf"),
    ("LightGBM", "lgbm"),
    ("MLP", "mlp"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--no-hmm", action="store_true", help="skip HMM decoding")
ap.add_argument("--models", nargs="*", help="subset of artifact names, e.g. logreg lgbm")
args = ap.parse_args()
models = [(n, s) for n, s in MODELS if not args.models or s in args.models]

out_dir = RESULTS_DIR / "model_comparison"
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data (student-test only)
cands = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")
cands = cands[cands.has_ref].reset_index(drop=True)
trips = {t: load_trip("student_test", t) for t in list_trips("student_test")}
refs = {t: derive_ref_arcs(tr).ref_arc for t, tr in trips.items()}
adjacency = {t: build_adjacency(tr.arcs) for t, tr in trips.items()}
n_labelled = int(sum((r >= 0).sum() for r in refs.values()))
X = cands[FEATURES].to_numpy()
y = cands["label"].to_numpy()
points = cands.groupby(["trip_id", "point_idx"]).ngroups
print(f"student-test: {len(trips)} trips, {n_labelled} labelled points, {len(cands)} candidate rows, "
      f"{points} points with candidates; scikit-learn {sklearn.__version__}")


def ece(p, labels, bins=10):
    """Expected calibration error with equal-width probability bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(labels[m].mean() - p[m].mean())
    return float(err)


def point_predictions(df, trip_id, n_points):
    """Highest-scoring candidate per point as an array (-1 where the point has no candidate)."""
    pred = np.full(n_points, -1, dtype=np.int64)
    top = df.loc[df.groupby("point_idx")["score"].idxmax()]
    pred[top.point_idx.to_numpy()] = top.arc_id.to_numpy()
    return pred


rows, per_trip = [], []
for display, stem in models:
    path = ARTIFACTS_DIR / f"{stem}.joblib"
    if not path.exists():
        print(f"[skip] {display}: {path} not found")
        continue
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(path)
    if any("Version" in type(c.message).__name__ for c in caught):
        print(f"[note] {display}: saved with a different scikit-learn version than {sklearn.__version__} (loads fine)")

    # scoring time: median of 3 repeats, per GPS point
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        scores = model.predict_proba(X)[:, 1]
        times.append(time.perf_counter() - t0)
    ms_per_point = float(np.median(times)) / points * 1000.0

    cm = candidate_metrics(y, (scores >= 0.5).astype(int), scores)
    d = cands[["trip_id", "point_idx", "arc_id", "dist_m", "along_frac", "arc_len_m"]].copy()
    d["score"] = scores

    correct_clf = correct_hmm = 0
    cont_clf, cont_hmm = [], []
    t_hmm = 0.0
    for tid, trip in trips.items():
        n_pts = len(trip.track)
        ref = refs[tid]
        lab = ref >= 0
        dt = d[d.trip_id == tid]
        clf = point_predictions(dt, tid, n_pts)
        c1 = int((clf[lab] == ref[lab]).sum())
        correct_clf += c1
        cont_clf.append(route_continuity(clf, trip.arcs, adjacency[tid]))
        rec = {"model": display, "trip": tid, "labelled_points": int(lab.sum()), "top1_clf": c1 / lab.sum()}
        if not args.no_hmm:
            t0 = time.perf_counter()
            hmm = decode(trip, dt[["point_idx", "arc_id", "dist_m", "along_frac", "arc_len_m", "score"]])
            t_hmm += time.perf_counter() - t0
            c2 = int((hmm[lab] == ref[lab]).sum())
            correct_hmm += c2
            cont_hmm.append(route_continuity(hmm, trip.arcs, adjacency[tid]))
            rec["top1_hmm"] = c2 / lab.sum()
        per_trip.append(rec)

    row = {
        "Model": display,
        "Top-1 (classifier only)": correct_clf / n_labelled,
        "Top-1 (+HMM)": correct_hmm / n_labelled if not args.no_hmm else np.nan,
        "Macro F1": cm["macro_f1"],
        "Precision": cm["precision"],
        "Recall": cm["recall"],
        "Log loss": cm["log_loss"],
        "Brier": float(np.mean((scores - y) ** 2)),
        "ECE": ece(scores, y),
        "Continuity (clf)": float(np.mean(cont_clf)),
        "Continuity (+HMM)": float(np.mean(cont_hmm)) if not args.no_hmm else np.nan,
        "Score ms/pt": ms_per_point,
        "Size MB": path.stat().st_size / 1e6,
    }
    rows.append(row)
    print(f"[done] {display:22s} top-1 {row['Top-1 (classifier only)']:.4f}"
          + (f" -> {row['Top-1 (+HMM)']:.4f} with HMM ({t_hmm:.0f}s decode)" if not args.no_hmm else ""), flush=True)

if not rows:
    sys.exit("No models found in artifacts/.")

table = pd.DataFrame(rows)
table.to_csv(out_dir / "student_test_comparison.csv", index=False)
pd.DataFrame(per_trip).to_csv(out_dir / "per_trip_top1.csv", index=False)
md = table.copy()
for c in md.columns[1:]:
    md[c] = md[c].map(lambda v: "" if pd.isna(v) else (f"{v:.5f}" if c in ("Log loss", "Brier", "ECE", "Score ms/pt") else f"{v:.4f}" if c != "Size MB" else f"{v:.3f}"))
(out_dir / "student_test_comparison.md").write_text(md.to_markdown(index=False), encoding="utf-8")

# ---------------------------------------------------------------- figure (real numbers only)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
x = np.arange(len(table))
w = 0.38
axes[0].bar(x - w / 2, table["Top-1 (classifier only)"], w, label="classifier only", color="#4c72b0")
if not args.no_hmm:
    axes[0].bar(x + w / 2, table["Top-1 (+HMM)"], w, label="+ HMM", color="#55a868")
axes[0].set_title("Student-test top-1 accuracy")
axes[0].set_ylim(0.7, 1.0)
axes[0].legend()
axes[1].bar(x - w / 2, table["Log loss"], w, label="log loss", color="#c44e52")
axes[1].bar(x + w / 2, table["Brier"], w, label="Brier", color="#8172b2")
axes[1].set_title("Probability quality (lower is better)")
axes[1].legend()
axes[2].bar(x - w / 2, table["Macro F1"], w, label="macro F1", color="#4c72b0")
axes[2].bar(x + w / 2, table["Continuity (clf)"], w, label="route continuity", color="#dd8452")
axes[2].set_title("Macro F1 and route continuity")
axes[2].set_ylim(0.7, 1.0)
axes[2].legend()
for a in axes:
    a.set_xticks(x)
    a.set_xticklabels(table["Model"], rotation=20, ha="right")
fig.tight_layout()
fig.savefig(out_dir / "comparison.png", dpi=200)

# ---------------------------------------------------------------- console summary (paste-friendly)
print("\n=== STUDENT-TEST COMPARISON (saved models, same candidate table) ===")
print(md.to_markdown(index=False))
print(f"\nSaved to {out_dir}")
