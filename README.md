# PathSnap: ML Map-Matching with HMM/Viterbi

## Team Members
| SRN | Name |
|---|---|
| PES1UG24CS492 | TANMAY T A |
| PES1UG24CS493 | TANMAYI NAGABHAIRAVA |
| PES1UG24CS925 | PRATHIKSHA B P |
| PES1UG24CS928 | RITU RAVISH |

---

## Overview
PathSnap constructs a vehicle’s trajectory from noisy 1 Hz GPS points by snapping them to a directed road network. The pipeline runs in three main steps:

1. **Candidates:** For each GPS point, determine road arcs within 40 m and calculate spatial and kinematic features (distance, heading alignment, velocity, etc.).
2. **Classifier:** A model assigns a probability for each candidate arc to be the correct one. Five model families were trained and compared; the final champion classifier is a tuned LightGBM.
3. **HMM/Viterbi Decoding:** Uses these emission probabilities combined with road graph connectivity to find the most likely continuous trajectory for each trip.

> **Data note:** The 90 trips in the dataset (80 training, 10 student-test) come from different cities around the world. Because each trip represents its own local map, every trip is projected into its own metric coordinate system (`Trip.track_xy()`). Evaluator trips were kept strictly blind during development.

## Setup
Python 3.13 (should work with 3.11 or 3.12; definitely won’t work with 3.14). Complete instructions on how to install everything and where to put your data are in [SETUP.md](SETUP.md); brief summary:
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# copy the organiser-provided train/ and student_test/ folders into data/  (or set PATHSNAP_DATA)
python -c "from pathsnap.data import *; print(len(list_trips('train')), len(list_trips('student_test')))"   # expect: 80 10
```

## Repository structure
| Path | Contents |
|---|---|
| `pathsnap/` | Library: `data.py` (data loading, trip-wise projections), `labels.py` (point-wise reference arcs of the route), `candidates.py` (candidate arcs + their features), `metrics.py`, `experiment.py` (CV harness with timings & logging), `plots.py`, `hmm.py` (HMM/Viterbi decoding), `decode_eval.py` (helpers for decoder evaluations) |
| `scripts/` | Entry point scripts (listed below) |
| `notebooks/` | `01_eda.ipynb` (EDA), `02_teammate1_label_speed_checks.ipynb` (label/speed checks) |
| `artifacts/` | Model checkpoints (`*.joblib`) and HMM configuration (`hmm_params.json`, saved by `tune_hmm.py`) |
| `results/` | All referenced by the report: logs, tables, figures, maps, HPO and HMM outputs (refer to "Where the results can be found" for more) |
| `tests/` | `python tests/test_metrics.py`, `python tests/test_hmm.py` |
| `data/` | Raw data (git-ignored; refer to `data/README.md`) |
| `report/` | `EDA_REPORT.md` (EDA notes); the final report is saved as `492_493_925_928_ml_report.pdf` |

```text
ML-Hackathon/
├── 492_493_925_928_ml_report.pdf       # Final technical report (PDF)
├── DEPLOY.md                           # Two-laptop Wi-Fi deployment guide
├── README.md                           # Main repository guide
├── SETUP.md                            # Environment & candidate build steps
├── requirements.txt                    # Project dependencies
├── api/                                # FastAPI backend service
│   ├── __init__.py
│   └── main.py                         # REST API (POST /predict, GET /trips)
├── ui/                                 # Streamlit frontend dashboard
│   ├── __init__.py
│   ├── app.py                          # User interface (interactive Folium maps)
│   ├── api_client.py                   # API communication layer
│   ├── route_map.py                    # Folium route rendering
│   └── requirements.txt                # Client-specific requirements
├── pathsnap/                           # Core map-matching & sequence decoding library
│   ├── data.py                         # Projection & data ingestion
│   ├── candidates.py                   # Spatial arc search & 8 feature computation
│   ├── labels.py                       # Ground-truth route alignment
│   ├── hmm.py                          # HMM emission & Viterbi trellis decoding
│   ├── decode_eval.py                  # Route continuity & evaluation helpers
│   ├── experiment.py                   # 5-fold GroupKFold cross-validation
│   ├── metrics.py                      # Top-1 accuracy, Macro F1, Log Loss, ECE
│   └── plots.py                        # Plotting utilities
├── scripts/                            # Standalone execution scripts
│   ├── build_candidates.py             # Precomputes candidate tables
│   ├── eda_extras.py                   # Missing values & connectivity checks
│   ├── train_lgbm.py                   # Trains LightGBM model
│   ├── train_rf.py                     # Trains Random Forest model
│   ├── train_mlp.py                    # Trains Multi-Layer Perceptron
│   ├── train_logreg.py                 # Trains Logistic Regression
│   ├── train_nb.py                     # Trains Gaussian Naive Bayes
│   ├── learning_curve_mlp.py           # Epoch-wise loss history
│   ├── compare_models.py               # Benchmark comparison on student-test
│   ├── optimize_lgbm.py                # Bayesian HPO via Optuna TPE
│   ├── finalize_model_results.py       # Consolidates comparison tables & figures
│   ├── tune_hmm.py                     # Tunes HMM transition & emission parameters
│   ├── eval_hmm.py                     # Evaluates HMM vs classifier-only performance
│   ├── make_map.py                     # Generates Folium HTML maps & static PNGs
│   └── README.md                       # Guide for running scripts
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis notebook
│   └── 02_teammate1_label_speed_checks.ipynb # Speed & spatial label checks
├── artifacts/                          # Trained models & config files
│   ├── lgbm.joblib                     # LightGBM untuned baseline
│   ├── lgbm_tuned.joblib               # LightGBM champion model
│   ├── rf.joblib                       # Random Forest model
│   ├── mlp.joblib                      # Multi-Layer Perceptron model
│   ├── logreg.joblib                   # Logistic Regression model
│   ├── nb.joblib                       # Gaussian Naive Bayes model
│   ├── hmm_params.json                 # Tuned HMM parameters
│   └── inference_config.json           # Model serving configuration
├── results/                            # All experimental outputs cited in report
│   ├── eda/                            # Missing value, connectivity & trajectory plots
│   ├── figures/                        # Comparison bar plots, confusion matrices, map PNGs
│   ├── tables/                         # CSV and Markdown tables matching rubric
│   ├── logs/                           # CSV experiment runs per model
│   ├── scores/                         # Scored prediction parquet files
│   ├── hpo/                            # Optuna trial logs & comparison curves
│   ├── hmm/                            # HMM evaluation & where HMM did not help
│   └── maps/                           # Interactive Folium HTML maps
├── report/
│   ├── EDA_REPORT.md                   # Early data exploratory notes
│   └── 492_493_925_928_ml_report.docx  # Editable Word report
└── data/                               # Raw data placeholder (git-ignored)
    └── README.md
```

## Pipeline execution
All commands to be executed from the repository root with the environment activated. All scripts employ seed 42.
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
Scripts per model (`scripts/optimize_lgbm.py` and `scripts/tune_hmm.py`) have their options described in the module's docstrings (`--help`). Script to run `make_map.py` is described in `scripts/README.md`.

## Where the results can be found
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

## Metrics, partitions, and rules
- **Partitions.** Training (80 trips) for training, validation (by trips), and parameter tuning; Student test set (10 trips) for development comparisons; the evaluator partition is never seen.
- **Labels.** The reference arcs per GPS point are determined from the provided route (`pathsnap/labels.py`). If there is no reference arc at a point, then it will not be used in training and all metrics.
- **Top-1** is point-level: for each labeled GPS point, is the best scoring candidate the reference arc. **Route continuity** is the proportion of sequential arcs which are the same or adjacent on the graph.

## API, UI and deployment
PathSnap runs across two laptops connected over a local Wi-Fi or mobile hotspot:
- **Laptop 1 (Model Server):** Runs the FastAPI backend (`uvicorn api.main:app --host 0.0.0.0 --port 8000`) with our trained LightGBM model and HMM decoder. It processes candidate generation, scoring, and Viterbi sequence decoding in ~122 ms per trip.
- **Laptop 2 (Evaluator Dashboard):** Runs the Streamlit interface (`streamlit run ui/app.py`), letting users select any evaluation trip from a dropdown and view the snapped route on an interactive Folium map with layer controls.

For network setup, firewall rules, and testing commands, see [DEPLOY.md](DEPLOY.md).
