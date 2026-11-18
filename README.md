# PathSnap: ML map-matching with HMM/Viterbi

PathSnap reconstructs a vehicle's route from noisy 1 Hz GPS points by snapping them to a road network.

1. **Candidates.** For every GPS point, list the road arcs within 40 m and compute features (distance, heading agreement, speed, ...).
2. **Classifier.** A model scores each candidate with the probability that it is the arc the vehicle was on. Five model families were trained and compared; the final classifier is a **tuned LightGBM**.
3. **HMM/Viterbi decoder.** Uses those probabilities and the road graph's connectivity to pick one coherent route per trip.

Data note: the 90 provided trips (80 train, 10 student-test) are in many cities worldwide and every trip is its own local map, so each trip is projected into its own metric frame (`Trip.track_xy()`). The evaluator trips are never used in development.

## Installation
Python 3.13 (3.11/3.12 should also work; avoid 3.14). Full steps, including how to place the data, are in [SETUP.md](SETUP.md); in short:
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# copy the organiser-provided train/ and student_test/ folders into data/  (or set PATHSNAP_DATA)
python -c "from pathsnap.data import *; print(len(list_trips('train')), len(list_trips('student_test')))"   # expect: 80 10
```

## Repository layout
| Path | Contents |
|---|---|
| `pathsnap/` | Library: `data.py` (loading, per-trip projection), `labels.py` (per-point reference arcs from the route), `candidates.py` (candidate arcs + features), `metrics.py`, `experiment.py` (grouped-CV harness with timing and logging), `plots.py`, `hmm.py` (HMM/Viterbi decoder), `decode_eval.py` (decoder evaluation helpers) |
| `scripts/` | Runnable entry points (below) |
| `notebooks/` | `01_eda.ipynb` (EDA), `02_teammate1_label_speed_checks.ipynb` (label and speed checks) |
| `artifacts/` | Saved models (`*.joblib`) and the decoder configuration (`hmm_params.json`, written by `tune_hmm.py`) |
| `results/` | Everything the report cites: logs, tables, figures, maps, HPO and HMM outputs (see "Where the results are") |
| `tests/` | `python tests/test_metrics.py`, `python tests/test_hmm.py` |
| `data/` | Raw data (git-ignored; see `data/README.md`) |
| `report/` | `EDA_REPORT.md` (EDA notes); the final technical report is added separately |

## Running the pipeline (in order)
All commands run from the repository root with the environment active. Every script uses seed 42.
```bash
# 1. Build the candidate tables (about 30 s). Needed by everything below.
python scripts/build_candidates.py train student_test

# 2. Exploratory analysis
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb     # tables + figures in results/eda/
python scripts/eda_extras.py             # missing-value analysis, candidate coverage, trip locations, trajectory samples

# 3. Train the five models (each: grouped 5-fold CV on the 80 train trips, then final fit + student-test evaluation)
python scripts/train_logreg.py           # Logistic Regression
python scripts/train_nb.py               # Gaussian Naive Bayes
python scripts/train_rf.py               # Random Forest
python scripts/train_lgbm.py             # LightGBM (the untuned configuration)
python scripts/train_mlp.py              # MLP (about 25 min)
python scripts/learning_curve_mlp.py     # real per-epoch training/validation loss for the MLP

# 4. Compare the saved models on student-test (with and without HMM decoding)
python scripts/compare_models.py

# 5. Bayesian hyperparameter optimisation of LightGBM (Optuna TPE, grouped CV; about 20-40 min)
python scripts/optimize_lgbm.py

# 6. Consolidated tables and plots for the model comparison (section 3) and tuning (section 4)
python scripts/finalize_model_results.py

# 7. Tune and evaluate the HMM decoder on the final classifier's scores
python scripts/tune_hmm.py               # writes artifacts/hmm_params.json
python scripts/eval_hmm.py               # classifier-only vs classifier + HMM on student-test

# 8. Folium maps (any model, any trip)
python scripts/make_map.py --model lgbm --scores-path results/scores/lgbm_tuned_student_test.parquet --trip-id 00000000
python scripts/make_map.py --model hmm --trip-id 00000000     # the final pipeline: classifier + HMM-decoded route (after eval_hmm.py)

# Tests
python tests/test_metrics.py && python tests/test_hmm.py
```
Per-model scripts, `scripts/optimize_lgbm.py` and `scripts/tune_hmm.py` document their options in their module docstrings (`--help`). `scripts/README.md` explains `make_map.py`.

## Where the results are
| What | Where |
|---|---|
| EDA tables and figures (partition stated in each title) | `results/eda/` (`dataset_summary.csv`, `trip_summary.csv`, `missing_value_*.csv/.png`, `candidate_coverage.csv`, `network_connectivity.csv/.png`, `trip_locations.png`, `trajectory_samples.png`, `representative_trip.png`, `trip_distributions.png`) |
| Per-model experiment logs (every run, with seeds and configurations) | `results/logs/experiments_<model>.csv`; all runs combined in `results/logs/experiments.csv`; the authoritative row per model in `results/logs/final_models.csv` |
| Model comparison table, overfitting evidence, student-test metrics | `results/tables/` (`model_comparison_table`, `overfitting_evidence`, `student_test_metrics`) |
| Comparison plots (CV macro F1, student-test top-1/macro F1, log loss) and confusion matrices | `results/figures/` |
| MLP training/validation loss | `results/figures/mlp_learning_curve_training_history.png`, `results/logs/mlp_learning_curve_history.json` |
| Five-model comparison on student-test with HMM | `results/model_comparison/` |
| Hyperparameter optimisation (trials, best parameters, plots, tuned vs untuned) | `results/hpo/` |
| HMM tuning and evaluation | `results/hmm/` and `artifacts/hmm_params.json` |
| Folium maps | `results/maps/` (static versions in `results/figures/`) |
| Saved models and their student-test scores | `artifacts/`, `results/scores/` |

## Metrics, partitions and rules
- **Partitions.** Train (80 trips) for fitting, cross-validation (grouped by trip) and tuning; student-test (10 trips) for development comparison only; the evaluator partition is never read.
- **Labels.** Reference arcs per GPS point are derived from the supplied route (`pathsnap/labels.py`); points with no reference arc are excluded from training and from all metrics.
- **Top-1** is observation-level: for each labelled GPS point, is the highest-scoring candidate the reference arc. **Route continuity** is the share of consecutive predicted arcs that are the same or directly connected on the road graph.

## API, UI and deployment
Not implemented yet. This section will describe the FastAPI backend (laptop 1), the evaluator UI (laptop 2) and the two-laptop deployment once they exist.
