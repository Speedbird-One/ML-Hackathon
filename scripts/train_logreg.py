"""Train Logistic Regression with the shared PathSnap experiment harness."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.config import CACHE_DIR, RESULTS_DIR, SEED
from pathsnap.data import list_trips, load_trip
from pathsnap.experiment import ExperimentLogger, run_experiment


FEATURES = [
    "dist_m",
    "along_frac",
    "arc_len_m",
    "speed_mps",
    "cos_heading",
    "dist_rank",
    "dist_minus_min",
    "n_cand",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--skip-route-continuity", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    class_weight = None if args.class_weight == "none" else "balanced"
    train_df = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
    test_df = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")

    if args.skip_route_continuity:
        train_trips = None
        test_trips = None
    else:
        train_trips = {trip_id: load_trip("train", trip_id, with_route=False) for trip_id in list_trips("train")}
        test_trips = {
            trip_id: load_trip("student_test", trip_id, with_route=False)
            for trip_id in list_trips("student_test")
        }

    def make_model():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=300,
                class_weight=class_weight,
                random_state=SEED,
            ),
        )

    run_experiment(
        model_name="LogisticRegression",
        family="Linear",
        key_config=f"C=1.0, max_iter=300, StandardScaler, class_weight={class_weight}",
        model_fn=make_model,
        train_df=train_df,
        test_df=test_df,
        feature_cols=FEATURES,
        seed=SEED,
        n_splits=5,
        train_trips=train_trips,
        test_trips=test_trips,
        logger=ExperimentLogger(RESULTS_DIR / "logs" / "experiments_logreg.csv"),
    )


if __name__ == "__main__":
    main()
