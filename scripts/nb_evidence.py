# %% [markdown]
# # Gaussian Naive Bayes: report evidence (run AFTER `scripts/train_nb.py`)
# Produces every number, table and figure the PathSnap report asks for about this model
# (Section 3.1 comparison row, 3.2 questions A-C, 3.3 visualisations). Everything is computed from
# the saved model `artifacts/nb.joblib` and the log `results/logs/experiments_nb.csv`.
# Outputs: `results/tables/nb_*.csv|md|json` and `results/figures/nb_*.png`, `cm_nb.png`.

# %%
import json
import sys
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running inside a notebook
    ROOT = Path.cwd() if (Path.cwd() / "pathsnap").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import ARTIFACTS_DIR, CACHE_DIR, RESULTS_DIR
from pathsnap.metrics import candidate_confusion_matrix, candidate_metrics, top1_accuracy
from pathsnap.plots import plot_confusion_matrix_heatmap

FEATURES = ["dist_m", "along_frac", "arc_len_m", "speed_mps", "cos_heading",
            "dist_rank", "dist_minus_min", "n_cand"]
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


def finish(fig, name):
    """Save a figure under results/figures; show it only inside a notebook."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=300)
    print("saved", FIG_DIR / name)
    plt.show() if "ipykernel" in sys.modules else plt.close(fig)


def save_table(df, name):
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)
    try:
        (TAB_DIR / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    except ImportError:  # tabulate not installed
        (TAB_DIR / f"{name}.md").write_text(df.to_string(index=False), encoding="utf-8")
    print("saved", TAB_DIR / f"{name}.csv")


# %% [markdown]
# ## 1. Section 3.1: row for the consolidated comparison table
# Taken from the harness log (the latest Naive Bayes run). Fields the harness cannot measure are N/A with a reason.

# %%
log = pd.read_csv(RESULTS_DIR / "logs" / "experiments_nb.csv")
run = log.iloc[-1]          # latest run; earlier runs stay in the CSV
print(f"{len(log)} run(s) logged; using the one from {run['timestamp']}")

comparison_row = pd.DataFrame([{
    "Model": run["model_name"],
    "Algorithm Family": run["family"],
    "Key Configuration": run["key_config"],
    "CV Macro F1": round(run["cv_macro_f1"], 4),
    "Top-1 Accuracy (student-test)": round(run["test_top1_acc"], 4),
    "Route Continuity (student-test)": round(run["test_route_continuity"], 4),
    "Student-Test Macro F1": round(run["test_macro_f1"], 4),
    "Log Loss (student-test)": round(run["test_log_loss"], 4),
    "Inference Time (ms/point)": round(run["inference_time_ms_per_point"], 5),
}])
save_table(comparison_row, "nb_comparison_row")
comparison_row.T

# %% [markdown]
# ## 2. Score train and student-test with the saved model
# Rows without a reference arc (`has_ref` False) are excluded everywhere.

# %%
model = joblib.load(ARTIFACTS_DIR / "nb.joblib")
train_df = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
test_df = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")
train_df = train_df[train_df.has_ref].reset_index(drop=True)
test_df = test_df[test_df.has_ref].reset_index(drop=True)

train_df["score"] = model.predict_proba(train_df[FEATURES].to_numpy())[:, 1]
test_df["score"] = model.predict_proba(test_df[FEATURES].to_numpy())[:, 1]

m_train = candidate_metrics(train_df.label, (train_df.score >= 0.5).astype(int), train_df.score)
m_test = candidate_metrics(test_df.label, (test_df.score >= 0.5).astype(int), test_df.score)
top1_train = top1_accuracy(train_df)
top1_test = top1_accuracy(test_df)
print("TRAIN (all 80 trips):", {k: round(v, 4) for k, v in m_train.items()}, "top1", round(top1_train, 4))
print("TEST  (10 trips)    :", {k: round(v, 4) for k, v in m_test.items()}, "top1", round(top1_test, 4))

# %% [markdown]
# ## 3. Questions 7-8: student-test performance, precision/recall, confusion matrix

# %%
perf = pd.DataFrame([{
    "Top-1": top1_test, "Macro F1": m_test["macro_f1"], "Log loss": m_test["log_loss"],
    "Precision": m_test["precision"], "Recall": m_test["recall"],
    "Binary F1": m_test["binary_f1"], "Accuracy": m_test["accuracy"],
}]).round(4)
save_table(perf, "nb_student_test_metrics")

cm = candidate_confusion_matrix(test_df.label, (test_df.score >= 0.5).astype(int))
print("Confusion matrix (rows = true 0/1, cols = predicted 0/1):\n", cm)
plot_confusion_matrix_heatmap(cm, "Gaussian Naive Bayes", save_path=FIG_DIR / "cm_nb.png")
perf.T

# %% [markdown]
# ## 4. Question 9: overfitting or underfitting?
# Same metrics on the training set, in cross-validation (grouped by trip) and on student-test.

# %%
fit_check = pd.DataFrame({
    "Split": ["Train (80 trips)", "CV mean (5 folds)", "Student-test (10 trips)"],
    "Macro F1": [m_train["macro_f1"], run["cv_macro_f1"], m_test["macro_f1"]],
    "Top-1": [top1_train, run["cv_top1_acc"], top1_test],
    "Log loss": [m_train["log_loss"], run["cv_log_loss"], m_test["log_loss"]],
}).round(4)
save_table(fit_check, "nb_train_cv_test")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, col in zip(axes, ["Macro F1", "Top-1", "Log loss"]):
    ax.bar(["Train", "CV", "Test"], fit_check[col], color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_title(col)
    for i, v in enumerate(fit_check[col]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Gaussian Naive Bayes: train vs CV vs student-test")
finish(fig, "nb_train_cv_test.png")
fit_check

# %% [markdown]
# ## 5. Question 2 (limitations): why the probabilities are overconfident
# (a) Feature correlation, which violates the independence assumption. (b) Reliability: predicted probability vs actual positive rate.

# %%
corr = train_df[FEATURES].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
ax.set_yticks(range(len(FEATURES)), FEATURES)
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("Feature correlation (train, labelled rows)")
finish(fig, "nb_feature_correlation.png")
save_table(corr.round(3).reset_index().rename(columns={"index": "feature"}), "nb_feature_correlation")

# %%
edges = [0, .01, .1, .3, .5, .7, .9, .99, 1.0]
bins = pd.cut(test_df.score, edges, include_lowest=True)
calib = (test_df.assign(bin=bins).groupby("bin", observed=True)
         .agg(n=("label", "size"), mean_predicted=("score", "mean"), actual_rate=("label", "mean"))
         .round(4).reset_index())
calib["bin"] = calib["bin"].astype(str)
save_table(calib, "nb_calibration")

fig, ax = plt.subplots(figsize=(5, 4.5))
ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
ax.plot(calib.mean_predicted, calib.actual_rate, "o-", color="#c44e52", label="Naive Bayes")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("actual share of true arcs")
ax.set_title("Reliability (student-test)")
ax.legend()
finish(fig, "nb_calibration.png")
calib

# %% [markdown]
# ## 6. Where does it fail? Top-1 for moving vs slow points (student-test)
# Slow points (speed < 1 m/s) have `cos_heading = 0` by definition, so heading cannot separate the two arcs of a two-way road.

# %%
top_idx = test_df.groupby(["trip_id", "point_idx"])["score"].idxmax()
picked = test_df.loc[top_idx, ["speed_mps", "label"]]
picked["group"] = np.where(picked.speed_mps < 1.0, "slow (<1 m/s)", "moving (>=1 m/s)")
speed_tbl = (picked.groupby("group").agg(points=("label", "size"), top1=("label", "mean")).round(4).reset_index())
save_table(speed_tbl, "nb_top1_by_speed")
speed_tbl

# %% [markdown]
# ## 7. Save one summary file (so every number in the report can be traced)
# Note: Gaussian Naive Bayes is fitted in closed form (no epochs), so there is **no** training/validation loss curve for it.
# None is drawn, on purpose (the guidelines forbid artificial curves).

# %%
summary = {
    "model": run["model_name"], "key_config": run["key_config"], "seed": int(run["seed"]),
    "log_timestamp": run["timestamp"],
    "cv": {"macro_f1": run["cv_macro_f1"], "top1": run["cv_top1_acc"], "log_loss": run["cv_log_loss"]},
    "student_test": {**m_test, "top1": top1_test, "route_continuity": run["test_route_continuity"]},
    "train": {**m_train, "top1": top1_train},
    "confusion_matrix_student_test": cm.tolist(),
    "train_time_sec": run["train_time_sec"],
    "inference_time_ms_per_point": run["inference_time_ms_per_point"],
}
(TAB_DIR / "nb_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
print("saved", TAB_DIR / "nb_summary.json")
