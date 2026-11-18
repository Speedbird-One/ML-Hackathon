"""Tune the HMM/Viterbi decoder parameters (Bayesian search, Optuna TPE) on the FINAL classifier's scores.

Why this design
  * The decoder's emission probability IS the classifier's score, so its best parameters depend on the classifier. This
    script tunes on the final LightGBM (hyperparameters from results/hpo/best_params.json).
  * Tuning on the 80 train trips needs scores a model has NOT overfit. So train-trip scores are OUT-OF-FOLD: grouped
    5-fold cross-validation where every trip is scored by a model that never saw it (cached in cache/).
  * Parameters are SELECTED on those train out-of-fold scores only. Student-test is used afterwards, once, to check the
    selected parameters against the defaults (the rules allow train and student-test for decoder tuning; keeping
    student-test out of the selection keeps it an honest check). Evaluator data is never touched.
  * Objective: maximise point-level top-1 over all labelled train points. Every trial also logs route continuity.
  * The decoder is run on every GPS point's candidates, exactly as at inference.

Searched parameters (default in brackets) and why
  beta               0.5-50, log   [5]    transition tolerance (m): how much |route distance - straight-line distance| is accepted
  max_net_dist_m     100-1000      [300]  shortest-path search cutoff (m): larger finds more connections but is slower
  gap_s              10-120        [30]   time gap (s) at which the chain is broken and restarted
  gap_m              100-600       [200]  spatial jump (m) at which the chain is broken and restarted
  unreachable_penalty 20-5000, log [1000] log-probability cost of an unreachable transition (hard vs soft constraint)
  jitter_tol_m       0-30          [10]   tolerated backwards movement on one arc (GPS jitter)
Not tuned: sigma (only used when no classifier score is present). The default parameters are trial 0, so the study
measures the baseline with the same code path.

Usage (repo root; candidate tables built; results/hpo and artifacts/lgbm_tuned.joblib exist):
    python scripts/tune_hmm.py                      # 30 trials, then final comparison
    python scripts/tune_hmm.py --trials 3 --max-trips 8 --out-dir results/hmm_test   # quick check (not for results)
    python scripts/tune_hmm.py --resume             # continue an interrupted study
Outputs in results/hmm/: hmm_trials.csv, hmm_best_params.json, hmm_optimisation_history.png, hmm_tuning_comparison.md,
    per_trip_student_test_default.csv / _tuned.csv, hmm_study.db (resume file, not for git)
    and artifacts/hmm_params.json (the tuned decoder configuration used by the API).
"""
import argparse
import json
import os
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
import optuna
import pandas as pd
from joblib import Parallel
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.config import ARTIFACTS_DIR, CACHE_DIR, RESULTS_DIR, SEED
from pathsnap.decode_eval import (FEATURES, argmax_predictions, decode_trips, evaluate, load_trips_and_refs,
                                  point_speeds, score_candidates, summarise, top1_from_predictions)
from pathsnap.hmm import DEFAULT_PARAMS
from pathsnap.metrics import route_continuity

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--trials", type=int, default=30, help="total number of trials in the study (trial 0 = default parameters)")
ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="parallel worker processes for decoding")
ap.add_argument("--hpo-json", type=Path, default=RESULTS_DIR / "hpo" / "best_params.json")
ap.add_argument("--final-model", type=Path, default=ARTIFACTS_DIR / "lgbm_tuned.joblib", help="model used to score student-test")
ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "hmm")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--refresh-oof", action="store_true", help="recompute the cached out-of-fold train scores")
ap.add_argument("--max-trips", type=int, default=None, help="TESTING ONLY: use only the first N train trips")
ap.add_argument("--no-save-artifact", action="store_true", help="do not write artifacts/hmm_params.json (testing)")
args = ap.parse_args()

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
args.out_dir.mkdir(parents=True, exist_ok=True)
t_start = time.time()
TUNED_KEYS = ["beta", "max_net_dist_m", "gap_s", "gap_m", "unreachable_penalty", "jitter_tol_m"]


def lgbm_from(hpo):
    p = dict(hpo["best_params"])
    return LGBMClassifier(**p, subsample_freq=1 if "subsample" in p else 0, random_state=SEED, verbose=-1, n_jobs=-1)


# --------------------------------------------------------------------------- train out-of-fold scores
train = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
if args.max_trips:
    train = train[train.trip_id.isin(sorted(train.trip_id.unique())[: args.max_trips])]
    print(f"[TEST MODE] first {args.max_trips} train trips only; results are not meaningful")
hpo = json.loads(args.hpo_json.read_text())
oof_path = CACHE_DIR / f"oof_scores_lgbm_tuned{'_test' if args.max_trips else ''}.parquet"
if oof_path.exists() and not args.refresh_oof:
    oof = pd.read_parquet(oof_path)
    print(f"loaded cached out-of-fold scores: {oof_path.name}")
else:
    print("computing out-of-fold train scores (5 grouped folds, tuned LightGBM)...", flush=True)
    lab = train[train.has_ref]
    parts, seen = [], set()
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(lab, lab.label, groups=lab.trip_id)):
        model = lgbm_from(hpo).fit(lab.iloc[tr][FEATURES].to_numpy(), lab.iloc[tr].label.to_numpy())
        val_trips = set(lab.iloc[va].trip_id)
        parts.append(score_candidates(model, train[train.trip_id.isin(val_trips)]))   # ALL rows of held-out trips
        seen |= val_trips
    rest = train[~train.trip_id.isin(seen)]                    # trips with no labelled rows: never trained on, any fold model is out-of-fold
    if len(rest):
        parts.append(score_candidates(model, rest))
    oof = pd.concat(parts, ignore_index=True)
    oof.to_parquet(oof_path, index=False)
    print(f"  done ({len(oof):,} rows), cached to {oof_path.name}")

train_ids = sorted(oof.trip_id.unique())
print(f"loading {len(train_ids)} train trips and reference arcs...", flush=True)
trips, refs, adjacency = load_trips_and_refs("train", train_ids)
speeds = point_speeds(oof, trips)
n_lab = int(sum((r >= 0).sum() for r in refs.values()))
print(f"{len(train_ids)} trips, {n_lab:,} labelled points; decoding with {args.jobs} worker process(es)")

# --------------------------------------------------------------------------- the Bayesian search
db = args.out_dir / "hmm_study.db"
if db.exists() and not args.resume:
    sys.exit(f"{db} exists. Use --resume to continue that study, or delete it for a fresh run.")
study = optuna.create_study(study_name="hmm_tuning", storage=f"sqlite:///{db.as_posix()}", load_if_exists=args.resume,
                            direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
if len(study.trials):
    study.sampler = optuna.samplers.TPESampler(seed=SEED + len(study.trials), n_startup_trials=8)
else:
    study.enqueue_trial({"beta": DEFAULT_PARAMS["beta"], "max_net_dist_m": DEFAULT_PARAMS["max_net_dist_m"],
                         "gap_s": DEFAULT_PARAMS["gap_s"], "gap_m": DEFAULT_PARAMS["gap_m"],
                         "unreachable_penalty_magnitude": -DEFAULT_PARAMS["unreachable_penalty"],
                         "jitter_tol_m": DEFAULT_PARAMS["jitter_tol_m"]})


def params_from_trial(trial):
    p = dict(DEFAULT_PARAMS)
    p["beta"] = trial.suggest_float("beta", 0.5, 50.0, log=True)
    p["max_net_dist_m"] = trial.suggest_float("max_net_dist_m", 100.0, 1000.0)
    p["gap_s"] = trial.suggest_float("gap_s", 10.0, 120.0)
    p["gap_m"] = trial.suggest_float("gap_m", 100.0, 600.0)
    p["unreachable_penalty"] = -trial.suggest_float("unreachable_penalty_magnitude", 20.0, 5000.0, log=True)
    p["jitter_tol_m"] = trial.suggest_float("jitter_tol_m", 0.0, 30.0)
    return p


def objective(trial, pool):
    params = params_from_trial(trial)
    t0 = time.perf_counter()
    preds, cpu_s = decode_trips("train", oof, params, pool)
    trial.set_user_attr("decode_cpu_s", cpu_s)
    trial.set_user_attr("wall_s", time.perf_counter() - t0)
    trial.set_user_attr("continuity", float(np.mean([route_continuity(p, trips[t].arcs, adjacency[t]) for t, p in preds.items()])))
    return top1_from_predictions(preds, refs)


def progress(study_, trial):
    print(f"trial {trial.number:3d} top-1={trial.value:.5f} best={study_.best_value:.5f} "
          f"({trial.user_attrs.get('wall_s', 0):.0f}s, elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)


remaining = max(0, args.trials - len(study.trials))
print(f"study has {len(study.trials)} trial(s); running {remaining} more (trial 0 is the default parameters)")
search_t0 = time.time()
with Parallel(n_jobs=args.jobs) as pool:
    if remaining:
        study.optimize(lambda tr: objective(tr, pool), n_trials=remaining, callbacks=[progress])
    search_seconds = time.time() - search_t0

    best = study.best_trial
    default_trial = study.trials[0]
    tuned = dict(DEFAULT_PARAMS)
    tuned.update({k: best.params[k] for k in ("beta", "max_net_dist_m", "gap_s", "gap_m", "jitter_tol_m")})
    tuned["unreachable_penalty"] = -best.params["unreachable_penalty_magnitude"]

    # ----------------------------------------------------------------------- record
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    trials.to_csv(args.out_dir / "hmm_trials.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(trials.number, trials.value, s=22, color="#4c72b0", label="trial")
    ax.plot(trials.number, trials.value.cummax(), color="#c44e52", lw=2, label="best so far")
    ax.axhline(default_trial.value, color="#888888", ls="--", lw=1, label="default parameters (trial 0)")
    ax.set_xlabel("trial")
    ax.set_ylabel("train out-of-fold top-1 (after HMM)")
    ax.set_title("HMM decoder tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "hmm_optimisation_history.png", dpi=200)
    plt.close(fig)

    # ----------------------------------------------------------------------- final comparison
    print("\n=== FINAL COMPARISON ===", flush=True)
    clf_train = argmax_predictions(oof, trips)
    sections = {}
    for name, preds in (("classifier only", clf_train),
                        ("HMM default", decode_trips("train", oof, DEFAULT_PARAMS, pool)[0]),
                        ("HMM tuned", decode_trips("train", oof, tuned, pool)[0])):
        sections[("train out-of-fold (used for tuning)", name)] = summarise(evaluate(preds, trips, refs, adjacency, speeds))

    test = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")
    st_trips, st_refs, st_adj = load_trips_and_refs("student_test", sorted(test.trip_id.unique()))
    st_scored = score_candidates(joblib.load(args.final_model), test)
    st_speeds = point_speeds(st_scored, st_trips)
    per_trip = {}
    for name, preds in (("classifier only", argmax_predictions(st_scored, st_trips)),
                        ("HMM default", decode_trips("student_test", st_scored, DEFAULT_PARAMS, pool)[0]),
                        ("HMM tuned", decode_trips("student_test", st_scored, tuned, pool)[0])):
        df = evaluate(preds, st_trips, st_refs, st_adj, st_speeds)
        per_trip[name] = df
        sections[("student-test (check only)", name)] = summarise(df)
    per_trip["HMM default"].to_csv(args.out_dir / "per_trip_student_test_default.csv", index=False)
    per_trip["HMM tuned"].to_csv(args.out_dir / "per_trip_student_test_tuned.csv", index=False)

rows = []
for (part, name), m in sections.items():
    rows.append({"Partition": part, "Decoder": name, "Top-1": m["top1"], "Top-1 moving": m["top1_moving"], "Top-1 slow (<1 m/s)": m["top1_slow"],
                 "Route continuity": m["continuity"], "Route recall": m["route_recall"], "Route precision": m["route_precision"], "Route F1": m["route_f1"]})
cmp_ = pd.DataFrame(rows)
(args.out_dir / "hmm_tuning_comparison.md").write_text(
    cmp_.to_markdown(index=False, floatfmt=".4f")
    + f"\n\nSearch: {len(study.trials)} trials in {search_seconds / 60:.1f} min on {args.jobs} worker(s); objective = train out-of-fold top-1 after HMM; "
    f"default parameters were trial 0 (top-1 {default_trial.value:.4f}); best trial #{best.number} (top-1 {best.value:.4f}).\n", encoding="utf-8")
out = {
    "scores_from": "tuned LightGBM (artifacts/lgbm_tuned.joblib); train trips scored out-of-fold (grouped 5-fold)",
    "objective": "maximise train out-of-fold point-level top-1 after HMM decoding (labelled points only)",
    "search_space": {"beta": [0.5, 50, "log"], "max_net_dist_m": [100, 1000], "gap_s": [10, 120], "gap_m": [100, 600],
                     "unreachable_penalty_magnitude": [20, 5000, "log"], "jitter_tol_m": [0, 30]},
    "sampler": "TPE, 8 random start-up trials; trial 0 = default parameters", "seed": SEED,
    "trials": len(study.trials), "best_trial": best.number, "search_wall_time_s": round(search_seconds, 1),
    "default_train_top1": default_trial.value, "best_train_top1": best.value,
    "params": {k: tuned[k] for k in tuned}, "default_params": dict(DEFAULT_PARAMS), "test_mode_max_trips": args.max_trips,
}
(args.out_dir / "hmm_best_params.json").write_text(json.dumps(out, indent=2))
if not args.no_save_artifact:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "hmm_params.json").write_text(json.dumps(out, indent=2))

print(cmp_.to_markdown(index=False, floatfmt=".4f"))
print("\ntuned parameters:")
print(json.dumps({k: tuned[k] for k in TUNED_KEYS}, indent=2))
print(f"\nAll outputs in {args.out_dir}" + ("" if args.no_save_artifact else " and artifacts/hmm_params.json") + f"   (total {(time.time() - t_start) / 60:.1f} min)")
