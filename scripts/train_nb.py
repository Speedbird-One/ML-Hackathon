"""Train Gaussian Naive Bayes through the shared harness.  Usage: python scripts/train_nb.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.naive_bayes import GaussianNB

from pathsnap.config import CACHE_DIR, RESULTS_DIR, SEED
from pathsnap.data import list_trips, load_trip
from pathsnap.experiment import ExperimentLogger, run_experiment

FEATURES = ["dist_m", "along_frac", "arc_len_m", "speed_mps", "cos_heading",
            "dist_rank", "dist_minus_min", "n_cand"]

train_df = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
test_df = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")

# Road networks, used only to compute route continuity (~40 s, ~0.6 GB for all 80 trips).
train_trips = {t: load_trip("train", t, with_route=False) for t in list_trips("train")}
test_trips = {t: load_trip("student_test", t, with_route=False) for t in list_trips("student_test")}


def make_model():
    # Fresh, unfitted model on every call (once per CV fold, once for the final fit).
    # No scaler: Gaussian NB fits a per-feature mean/variance, so scaling does not change its predictions.
    return GaussianNB()


run_experiment(
    model_name="GaussianNaiveBayes",
    family="Probabilistic",
    key_config="var_smoothing=1e-9 (default), priors=None (learned from data), no scaler",
    model_fn=make_model,
    train_df=train_df,
    test_df=test_df,
    feature_cols=FEATURES,
    seed=SEED,
    n_splits=5,
    train_trips=train_trips,
    test_trips=test_trips,
    logger=ExperimentLogger(RESULTS_DIR / "logs" / "experiments_nb.csv"),
    save_name="nb",
)
