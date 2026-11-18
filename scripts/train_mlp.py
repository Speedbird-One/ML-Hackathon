"""Train an MLP (neural network) through the shared harness.

Grouped 5-fold cross-validation on the 80 train trips (a trip is never split across folds), then a final fit
on all 80 and evaluation on the 10 student-test trips. Timing, metrics and logging come from
pathsnap.experiment.run_experiment; this script only supplies the data and the model.

Usage (from the repo root, with the candidate tables built):  python scripts/train_mlp.py
Outputs: results/logs/experiments_mlp.csv, artifacts/mlp.joblib, results/scores/mlp_student_test.parquet
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.config import CACHE_DIR, RESULTS_DIR, SEED
from pathsnap.data import list_trips, load_trip
from pathsnap.experiment import ExperimentLogger, run_experiment

# Identifiers (trip_id, point_idx, arc_id) are deliberately not features.
FEATURES = ["dist_m", "along_frac", "arc_len_m", "speed_mps", "cos_heading",
            "dist_rank", "dist_minus_min", "n_cand"]

train_df = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
test_df = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")

# Road networks are only needed for route continuity (about 40 s and 0.6 GB for the 80 train trips).
train_trips = {t: load_trip("train", t, with_route=False) for t in list_trips("train")}
test_trips = {t: load_trip("student_test", t, with_route=False) for t in list_trips("student_test")}

logger = ExperimentLogger(RESULTS_DIR / "logs" / "experiments_mlp.csv")

def make_model_1():
    # Fresh, unfitted model on every call (once per CV fold and once for the final fit).
    # Early stopping holds out a random 10% of rows, not whole trips; it only decides when to stop.
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=50, random_state=SEED))


run_experiment(
    model_name="MLP",
    family="Neural network",
    key_config="hidden=(64,32), relu, adam, early_stopping (10% val), max_iter=50, StandardScaler",
    model_fn=make_model_1,
    train_df=train_df,
    test_df=test_df,
    feature_cols=FEATURES,
    seed=SEED,
    n_splits=5,
    train_trips=train_trips,
    test_trips=test_trips,
    logger=logger,
    save_name="mlp",
)

