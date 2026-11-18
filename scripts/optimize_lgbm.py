"""Bayesian hyperparameter optimisation of LightGBM (Optuna TPE) with group-aware cross-validation.

What it does
  1. Searches LightGBM hyperparameters with Optuna's TPE sampler (a Bayesian optimiser: it models which regions of
     the search space gave good scores and proposes new trials there). A median pruner stops clearly bad trials
     early, after two folds.
  2. Every trial is scored by grouped 5-fold cross-validation over the 80 TRAIN trips (GroupKFold on trip_id, so a
     trip is never split across folds), using exactly the harness metrics (pathsnap.experiment.evaluate_split).
  3. Refits the untuned (Teammate 3's) and the tuned configuration through the shared harness, so both are evaluated
     identically: CV metrics, then the 10 student-test trips, with train/inference time and the saved model.
  4. Writes the full optimisation record under results/hpo/.

Objective (default): minimise mean CV LOG LOSS. Why: it is a smooth, proper scoring rule, so it can rank hyperparameter
settings whose top-1 accuracy differs by only a few hundredths of a point; and the HMM decoder consumes these
probabilities, so probability quality matters directly. Alternatives: --objective top1 | macro_f1. Every trial logs all
CV metrics whichever objective is used.

Search space and why
  n_estimators 100-600 (int)         number of boosting rounds; trades fit against overfit
  learning_rate 0.02-0.2 (log)       step size; interacts with n_estimators
  num_leaves 15-255 (int, log)       tree complexity: the main capacity control
  min_child_samples 20-500 (log)     minimum rows per leaf; larger = smoother, less overfit to single trips
  subsample 0.5-1.0                  row sampling per tree (variance reduction), used with subsample_freq=1
  colsample_bytree 0.5-1.0           feature sampling per tree; we have only 8 features so the floor is 0.5
  reg_alpha 1e-3-10 (log)            L1 penalty on leaf weights
  reg_lambda 1e-3-10 (log)           L2 penalty on leaf weights
Only the 80 train trips are used to choose hyperparameters; student-test is evaluated once, afterwards, for the
tuned-vs-untuned comparison. Evaluator data is never touched.

Usage (repo root, candidate tables built):
    python scripts/optimize_lgbm.py                    # 30 trials, then refit + compare (about 30-60 minutes)
    python scripts/optimize_lgbm.py --trials 5         # quick check
    python scripts/optimize_lgbm.py --resume           # continue an interrupted study (same --out-dir)
Outputs in results/hpo/: lgbm_trials.csv, best_params.json, optimisation_history.png, param_importance.png,
    hpo_comparison.md, experiments_lgbm_hpo.csv, lgbm_study.db (resume file, not for git)
    plus artifacts/lgbm_untuned.joblib and artifacts/lgbm_tuned.joblib (and score files / confusion matrices).
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.config import CACHE_DIR, RESULTS_DIR, SEED
from pathsnap.data import list_trips, load_trip
from pathsnap.experiment import ExperimentLogger, evaluate_split, run_experiment
from pathsnap.metrics import _labelled_only

FEATURES = ["dist_m", "along_frac", "arc_len_m", "speed_mps", "cos_heading",
            "dist_rank", "dist_minus_min", "n_cand"]
# Teammate 3's untuned configuration (scripts/train_lgbm.py); every other parameter is the LightGBM default.
BASELINE = dict(n_estimators=300, learning_rate=0.05, num_leaves=63)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--trials", type=int, default=30, help="total number of trials in the study")
ap.add_argument("--objective", choices=["logloss", "top1", "macro_f1"], default="logloss")
ap.add_argument("--splits", type=int, default=5, help="grouped CV folds")
ap.add_argument("--timeout-min", type=float, default=None, help="stop the search after this many minutes")
ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "hpo")
ap.add_argument("--resume", action="store_true", help="continue an existing study in --out-dir")
ap.add_argument("--skip-final", action="store_true", help="only run the search; skip refit and comparison")
ap.add_argument("--no-continuity", action="store_true", help="skip route continuity in the final comparison (faster)")
ap.add_argument("--no-save", action="store_true", help="do not save models/scores in the final stage (testing only)")
ap.add_argument("--max-trips", type=int, default=None, help="TESTING ONLY: use only the first N train trips")
args = ap.parse_args()

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
args.out_dir.mkdir(parents=True, exist_ok=True)
t_start = time.time()

# --------------------------------------------------------------------------- data (train only for the search)
train_df = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
test_df = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")
if args.max_trips:
    keep = sorted(train_df.trip_id.unique())[: args.max_trips]
    train_df = train_df[train_df.trip_id.isin(keep)]
    print(f"[TEST MODE] using only {len(keep)} train trips; results are not meaningful")
train_df = _labelled_only(train_df).reset_index(drop=True)

X = train_df[FEATURES].to_numpy()
y = train_df["label"].to_numpy()
groups = train_df["trip_id"].to_numpy()
folds = list(GroupKFold(n_splits=args.splits).split(X, y, groups=groups))
val_dfs = [train_df.iloc[va] for _, va in folds]
print(f"{train_df.trip_id.nunique()} train trips, {len(train_df):,} labelled rows, {args.splits} grouped folds; objective: {args.objective}")

METRIC_KEY = {"logloss": "log_loss", "top1": "top1_acc", "macro_f1": "macro_f1"}[args.objective]
DIRECTION = "minimize" if args.objective == "logloss" else "maximize"


def make_model(p):
    return LGBMClassifier(**p, subsample_freq=1 if "subsample" in p else 0, random_state=SEED, verbose=-1, n_jobs=-1)


def objective(trial: optuna.Trial) -> float:
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 500, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )
    fold_metrics, fit_time = [], 0.0
    for k, ((tr, va), vdf) in enumerate(zip(folds, val_dfs)):
        model = make_model(p)
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        fit_time += time.perf_counter() - t0
        fold_metrics.append(evaluate_split(model, X[va], y[va], vdf, None))
        running = float(np.mean([m[METRIC_KEY] for m in fold_metrics]))
        trial.report(running, step=k)
        if k >= 1 and k < len(folds) - 1 and trial.should_prune():   # prune only after at least two folds
            trial.set_user_attr("folds_run", k + 1)
            raise optuna.TrialPruned()
    for key in ("log_loss", "top1_acc", "macro_f1", "precision", "recall"):
        vals = [m[key] for m in fold_metrics]
        trial.set_user_attr(f"cv_{key}", float(np.mean(vals)))
        trial.set_user_attr(f"cv_{key}_std", float(np.std(vals)))
    trial.set_user_attr("fit_time_s", fit_time)
    trial.set_user_attr("folds_run", len(folds))
    return float(np.mean([m[METRIC_KEY] for m in fold_metrics]))


# --------------------------------------------------------------------------- the Bayesian search
db = args.out_dir / "lgbm_study.db"
if db.exists() and not args.resume:
    sys.exit(f"{db} exists. Use --resume to continue that study, or a different --out-dir / delete the file for a fresh run.")
study = optuna.create_study(
    study_name="lgbm_hpo", storage=f"sqlite:///{db.as_posix()}", load_if_exists=args.resume, direction=DIRECTION,
    sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),   # 8 random trials, then Bayesian proposals
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
)
if len(study.trials):
    # The sampler's random state is not stored with the study; reseed on resume so the new random start-up
    # trials do not replay the earlier ones.
    study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials), n_startup_trials=8)
remaining = max(0, args.trials - len(study.trials))
print(f"study already has {len(study.trials)} trial(s); running {remaining} more")


def progress(study_, trial):
    best = study_.best_value if any(t.state == optuna.trial.TrialState.COMPLETE for t in study_.trials) else float("nan")
    print(f"trial {trial.number:3d} {trial.state.name:9s} value={trial.value if trial.value is not None else float('nan'):.5f} "
          f"best={best:.5f} elapsed={(time.time() - t_start) / 60:.1f} min", flush=True)


search_t0 = time.time()
if remaining:
    study.optimize(objective, n_trials=remaining, timeout=args.timeout_min * 60 if args.timeout_min else None, callbacks=[progress])
search_seconds = time.time() - search_t0

done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if not done:
    sys.exit("No completed trials.")
best = study.best_trial
best_params = dict(best.params)

# --------------------------------------------------------------------------- record (real results only)
trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
trials.to_csv(args.out_dir / "lgbm_trials.csv", index=False)
summary = {
    "objective": f"mean CV {METRIC_KEY} ({DIRECTION})", "cv": f"GroupKFold({args.splits}) by trip_id on {train_df.trip_id.nunique()} train trips",
    "sampler": "TPE (Bayesian), 8 random start-up trials", "pruner": "MedianPruner after 2 folds", "seed": SEED,
    "trials_total": len(study.trials), "trials_completed": len(done),
    "trials_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
    "search_wall_time_s": round(search_seconds, 1), "best_trial": best.number, "best_value": best.value,
    "best_params": best_params, "baseline_untuned": BASELINE,
    "best_cv_metrics": {k[3:]: v for k, v in best.user_attrs.items() if k.startswith("cv_")},
    "search_space": {"n_estimators": [100, 600], "learning_rate": [0.02, 0.2, "log"], "num_leaves": [15, 255, "log"],
                     "min_child_samples": [20, 500, "log"], "subsample": [0.5, 1.0], "colsample_bytree": [0.5, 1.0],
                     "reg_alpha": [1e-3, 10, "log"], "reg_lambda": [1e-3, 10, "log"]},
    "test_mode_max_trips": args.max_trips,
}
(args.out_dir / "best_params.json").write_text(json.dumps(summary, indent=2))

c = trials[trials.state == "COMPLETE"].sort_values("number")
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.scatter(c.number, c.value, s=22, color="#4c72b0", label="completed trial")
run_best = c.value.cummin() if DIRECTION == "minimize" else c.value.cummax()
ax.plot(c.number, run_best, color="#c44e52", lw=2, label="best so far")
ax.set_xlabel("trial")
ax.set_ylabel(f"mean CV {METRIC_KEY}")
ax.set_title(f"Optimisation history ({summary['trials_pruned']} further trials pruned early)")
ax.legend()
fig.tight_layout()
fig.savefig(args.out_dir / "optimisation_history.png", dpi=200)
plt.close(fig)
try:
    imp = optuna.importance.get_param_importances(study)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#55a868")
    ax.set_xlabel("importance (fANOVA, share of variance in the objective)")
    ax.set_title("Which hyperparameters mattered")
    fig.tight_layout()
    fig.savefig(args.out_dir / "param_importance.png", dpi=200)
    plt.close(fig)
except Exception as e:  # needs enough completed trials
    print(f"[note] parameter-importance plot skipped: {e}")

print("\n=== SEARCH FINISHED ===")
print(f"{summary['trials_completed']} completed, {summary['trials_pruned']} pruned, {search_seconds / 60:.1f} min")
print(f"best trial #{best.number}: mean CV {METRIC_KEY} = {best.value:.5f}")
print(json.dumps(best_params, indent=2))

if args.skip_final:
    sys.exit(0)

# --------------------------------------------------------------------------- untuned vs tuned, same harness
print("\n=== REFIT: untuned vs tuned through the shared harness (CV + student-test) ===")
train_trips = test_trips = None
if not args.no_continuity:
    train_trips = {t: load_trip("train", t, with_route=False) for t in list_trips("train")}
    test_trips = {t: load_trip("student_test", t, with_route=False) for t in list_trips("student_test")}
logger = ExperimentLogger(args.out_dir / "experiments_lgbm_hpo.csv")
results = {}
for label, params, save in (("LightGBM (untuned)", BASELINE, "lgbm_untuned"), ("LightGBM (tuned)", best_params, "lgbm_tuned")):
    results[label] = run_experiment(
        model_name=label, family="Boosting", key_config=json.dumps(params), model_fn=lambda p=params: make_model(p),
        train_df=train_df, test_df=test_df, feature_cols=FEATURES, seed=SEED, n_splits=args.splits,
        train_trips=train_trips, test_trips=test_trips, logger=logger, save_name=None if args.no_save else save)

rows = []
for label, r in results.items():
    rows.append({"Model": label, "CV macro F1": r.cv_macro_f1, "CV top-1": r.cv_top1_acc, "CV log loss": r.cv_log_loss,
                 "Student-test top-1": r.test_top1_acc, "Student-test macro F1": r.test_macro_f1,
                 "Student-test log loss": r.test_log_loss, "Inference ms/pt": r.inference_time_ms_per_point,
                 "Final fit s": r.train_time_sec})
cmp_ = pd.DataFrame(rows)
(args.out_dir / "hpo_comparison.md").write_text(
    cmp_.to_markdown(index=False, floatfmt=".5f")
    + f"\n\nSearch cost: {len(study.trials)} trials ({summary['trials_pruned']} pruned) in {search_seconds / 60:.1f} min "
    f"(objective: mean CV {METRIC_KEY}, {DIRECTION}); best trial #{best.number}.\n", encoding="utf-8")
print("\n=== TUNED vs UNTUNED ===")
print(cmp_.to_markdown(index=False, floatfmt=".5f"))
print(f"\nAll outputs in {args.out_dir}   (total {(time.time() - t_start) / 60:.1f} min)")
