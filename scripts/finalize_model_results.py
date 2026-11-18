"""Build the consolidated model-comparison results (report sections 3.1-3.3 and 4) from the saved models and the logs.

Why this script exists: the five models were trained by different people at different times with different versions of
the harness, so their logs have different columns and inconsistent timing. This script puts everything on one footing:

  * Student-test and TRAIN metrics are re-computed here from each saved model (artifacts/<name>.joblib), with the same
    functions (pathsnap.experiment.evaluate_split) on the same candidate tables. The train-vs-CV-vs-test numbers are the
    evidence for the report's overfitting/underfitting question.
  * Cross-validation metrics (grouped 5-fold over the 80 train trips) cannot be recomputed cheaply, so they are taken from
    each model's log; the log file and row used are recorded in the output.
  * Inference time is the scoring-only, per-GPS-point measure from scripts/compare_models.py (results/model_comparison/).
  * LightGBM here is our rerun of the untuned configuration on the shared candidate tables (artifacts/lgbm_untuned.joblib,
    row 'LightGBM (untuned)' of results/hpo/experiments_lgbm_hpo.csv); Teammate 3's own saved copy of that configuration is
    not used.

Usage (repo root; candidate tables built; scripts/compare_models.py and scripts/optimize_lgbm.py already run):
    python scripts/finalize_model_results.py
Outputs:
    results/logs/final_models.csv        one authoritative row per model (CV from logs; train/test recomputed; provenance)
    results/logs/experiments.csv         every logged run from every per-person log, concatenated
    results/tables/model_comparison_table.md/.csv   report table 3.1
    results/tables/overfitting_evidence.md/.csv     train vs CV vs student-test
    results/tables/student_test_metrics.md/.csv     accuracy, precision, recall, macro F1, top-1, log loss per model
    results/figures/cv_macro_f1_comparison.png, student_test_comparison.png, log_loss_comparison.png
    results/hpo/tuned_vs_untuned.png     section 4 comparison figure
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import ARTIFACTS_DIR, CACHE_DIR, RESULTS_DIR
from pathsnap.data import list_trips, load_trip
from pathsnap.experiment import evaluate_split
from pathsnap.plots import plot_cv_macro_f1, plot_log_loss_comparison, plot_test_comparison

warnings.filterwarnings("ignore")
LOGS = RESULTS_DIR / "logs"
TABLES = RESULTS_DIR / "tables"
FIGS = RESULTS_DIR / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FEATURES = ["dist_m", "along_frac", "arc_len_m", "speed_mps", "cos_heading",
            "dist_rank", "dist_minus_min", "n_cand"]

HPO_LOG = RESULTS_DIR / "hpo" / "experiments_lgbm_hpo.csv"
# display name, family, artifact, log file, how to choose the row, key configuration shown in the table
SPEC = [
    ("Logistic Regression", "Linear", "logreg", LOGS / "experiments_logreg.csv", lambda d: d[d.key_config.str.contains("class_weight=None")].iloc[-1],
     None),
    ("Gaussian Naive Bayes", "Probabilistic (generative)", "nb", LOGS / "experiments_nb.csv", lambda d: d.iloc[-1], None),
    ("Random Forest", "Bagging ensemble", "rf", LOGS / "experiments_rf.csv", lambda d: d.iloc[-1], None),
    ("LightGBM", "Gradient boosting", "lgbm_untuned", HPO_LOG, lambda d: d[d.model_name == "LightGBM (untuned)"].iloc[-1],
     "n_estimators=300, learning_rate=0.05, num_leaves=63, other parameters default (untuned)"),
    ("MLP", "Neural network", "mlp", LOGS / "experiments_mlp.csv", lambda d: d.iloc[-1], None),
]

# ------------------------------------------------------------------------------- data (read once)
train = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
train = train[train.has_ref].reset_index(drop=True)
test = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")
test = test[test.has_ref].reset_index(drop=True)
test_trips = {t: load_trip("student_test", t, with_route=False) for t in list_trips("student_test")}
cmp_ = pd.read_csv(RESULTS_DIR / "model_comparison" / "student_test_comparison.csv").set_index("Model")
print(f"train: {len(train):,} labelled rows | student-test: {len(test):,} labelled rows", flush=True)

rows = []
for display, family, artifact, log_path, pick, cfg in SPEC:
    log_row = pick(pd.read_csv(log_path))
    model = joblib.load(ARTIFACTS_DIR / f"{artifact}.joblib")
    tr = evaluate_split(model, train[FEATURES].to_numpy(), train.label.to_numpy(), train, None)
    te = evaluate_split(model, test[FEATURES].to_numpy(), test.label.to_numpy(), test, test_trips)
    row = {
        "model_name": display, "family": family, "key_config": cfg or log_row.key_config, "seed": int(log_row.seed),
        # cross-validation (from the model's log)
        "cv_macro_f1": log_row.cv_macro_f1, "cv_top1_acc": log_row.cv_top1_acc, "cv_log_loss": log_row.cv_log_loss,
        "cv_route_continuity": log_row.cv_route_continuity,
        # student-test (recomputed from the saved model)
        "test_macro_f1": te["macro_f1"], "test_top1_acc": te["top1_acc"], "test_log_loss": te["log_loss"],
        "test_accuracy": te["accuracy"], "test_precision": te["precision"], "test_recall": te["recall"],
        "test_binary_f1": te["binary_f1"], "test_route_continuity": te["route_continuity"],
        # train (recomputed from the saved model)
        "train_macro_f1": tr["macro_f1"], "train_top1_acc": tr["top1_acc"], "train_log_loss": tr["log_loss"],
        "train_accuracy": tr["accuracy"],
        # cost
        "train_time_sec": log_row.train_time_sec, "inference_time_ms_per_point": cmp_.loc[display, "Score ms/pt"],
        "model_size_mb": (ARTIFACTS_DIR / f"{artifact}.joblib").stat().st_size / 1e6,
        "artifact": f"artifacts/{artifact}.joblib", "cv_source": f"{log_path.relative_to(RESULTS_DIR.parent).as_posix()} (row logged {log_row.timestamp})",
        "timestamp": log_row.timestamp,
    }
    # cross-check the recomputed student-test numbers against what the model's own log recorded
    drift = abs(row["test_top1_acc"] - log_row.test_top1_acc)
    row["test_top1_vs_log_diff"] = float(drift)
    rows.append(row)
    print(f"  {display:22s} train F1 {tr['macro_f1']:.4f} | CV F1 {row['cv_macro_f1']:.4f} | test F1 {te['macro_f1']:.4f} | "
          f"test top-1 {te['top1_acc']:.4f} (log said {log_row.test_top1_acc:.4f})", flush=True)

final = pd.DataFrame(rows)
final.to_csv(LOGS / "final_models.csv", index=False)

# ------------------------------------------------------------------------------- combined log of every run
parts = []
for p in sorted(LOGS.glob("experiments_*.csv")) + [HPO_LOG]:
    d = pd.read_csv(p)
    d.insert(0, "source_file", p.relative_to(RESULTS_DIR.parent).as_posix())
    parts.append(d)
pd.concat(parts, ignore_index=True).to_csv(LOGS / "experiments.csv", index=False)

# ------------------------------------------------------------------------------- required plots (3.3), real data only
plot_cv_macro_f1(final)
plot_test_comparison(final)
plot_log_loss_comparison(final)

# ------------------------------------------------------------------------------- table 3.1 (the column order the PDF specifies)
t31 = pd.DataFrame({
    "Model": final.model_name, "Algorithm Family": final.family, "Key Configuration": final.key_config,
    "CV Macro F1-score": final.cv_macro_f1, "Top-1 Accuracy (student-test)": final.test_top1_acc,
    "Route Continuity (student-test)": final.test_route_continuity, "Student-Test Macro F1-score": final.test_macro_f1,
    "Log Loss (student-test)": final.test_log_loss, "Inference Time (ms per GPS point)": final.inference_time_ms_per_point,
})
t31.to_csv(TABLES / "model_comparison_table.csv", index=False)
(TABLES / "model_comparison_table.md").write_text(
    t31.to_markdown(index=False, floatfmt=".4f")
    + "\n\nCV Macro F1: grouped 5-fold over the 80 train trips (from each model's log). Top-1, route continuity, macro F1 and log loss: "
    "the 10 student-test trips, recomputed from the saved models. Inference time: scoring only, per GPS point. No metric is N/A for any model.\n",
    encoding="utf-8")

# ------------------------------------------------------------------------------- overfitting / underfitting evidence (numbers only)
over = pd.DataFrame({
    "Model": final.model_name,
    "Macro F1 train": final.train_macro_f1, "Macro F1 CV": final.cv_macro_f1, "Macro F1 test": final.test_macro_f1,
    "Top-1 train": final.train_top1_acc, "Top-1 CV": final.cv_top1_acc, "Top-1 test": final.test_top1_acc,
    "Log loss train": final.train_log_loss, "Log loss CV": final.cv_log_loss, "Log loss test": final.test_log_loss,
})
over["Macro F1 gap (train - CV)"] = over["Macro F1 train"] - over["Macro F1 CV"]
over["Top-1 gap (train - CV)"] = over["Top-1 train"] - over["Top-1 CV"]
over["Log loss gap (CV - train)"] = over["Log loss CV"] - over["Log loss train"]
over.to_csv(TABLES / "overfitting_evidence.csv", index=False)
(TABLES / "overfitting_evidence.md").write_text(
    over.to_markdown(index=False, floatfmt=".4f")
    + "\n\nTrain = all labelled rows of the 80 train trips scored by the final saved model (which was fitted on them); CV = grouped 5-fold "
    "estimate from the log; test = 10 student-test trips. A large positive train-minus-CV gap suggests overfitting; low scores on all three "
    "suggest underfitting. (Interpretation is for the report.)\n", encoding="utf-8")

# ------------------------------------------------------------------------------- per-model student-test metrics (3.2 questions 7-8)
st = pd.DataFrame({
    "Model": final.model_name, "Accuracy": final.test_accuracy, "Precision": final.test_precision, "Recall": final.test_recall,
    "Macro F1": final.test_macro_f1, "Binary F1": final.test_binary_f1, "Top-1 (observation level)": final.test_top1_acc,
    "Log loss": final.test_log_loss, "Route continuity": final.test_route_continuity,
})
st.to_csv(TABLES / "student_test_metrics.csv", index=False)
(TABLES / "student_test_metrics.md").write_text(st.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

# ------------------------------------------------------------------------------- section 4: tuned vs untuned figure
h = pd.read_csv(HPO_LOG)
u = h[h.model_name == "LightGBM (untuned)"].iloc[-1]
t = h[h.model_name == "LightGBM (tuned)"].iloc[-1]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
x, w = np.arange(3), 0.38
for ax, labels, cols, title, ylim in (
    (axes[0], ["CV macro F1", "Test top-1", "Test macro F1"], ["cv_macro_f1", "test_top1_acc", "test_macro_f1"], "Accuracy-type metrics (higher is better)", (0.8, 1.0)),
    (axes[1], ["CV log loss", "Test log loss"], ["cv_log_loss", "test_log_loss"], "Log loss (lower is better)", None),
):
    xx = np.arange(len(cols))
    ax.bar(xx - w / 2, [u[c] for c in cols], w, label="untuned", color="#4c72b0")
    ax.bar(xx + w / 2, [t[c] for c in cols], w, label="tuned", color="#55a868")
    for i, c in enumerate(cols):
        ax.text(i - w / 2, u[c], f"{u[c]:.4f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, t[c], f"{t[c]:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xx)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
ax = axes[2]
ax.bar([0 - w / 2, 1 - w / 2], [u.inference_time_ms_per_point, u.train_time_sec], w, color="#4c72b0", label="untuned")
ax.bar([0 + w / 2, 1 + w / 2], [t.inference_time_ms_per_point, t.train_time_sec], w, color="#55a868", label="tuned")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Inference (ms/point)", "Final fit (s)"])
ax.set_yscale("log")
ax.set_title("Cost (log scale, lower is better)")
ax.legend()
fig.suptitle("LightGBM: tuned vs untuned, same cross-validation and student-test partitions", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "hpo" / "tuned_vs_untuned.png", dpi=200)
plt.close(fig)

print("\n=== TABLE 3.1 ===")
print(t31.to_markdown(index=False, floatfmt=".4f"))
print("\n=== OVERFITTING EVIDENCE ===")
print(over.to_markdown(index=False, floatfmt=".4f"))
bad = final[final.test_top1_vs_log_diff > 0.003]
print("\nlogged vs recomputed student-test top-1 differ by more than 0.003 for:", list(bad.model_name) or "none")
print("done")
