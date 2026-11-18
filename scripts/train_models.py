"""Master training and experimentation script for PathSnap (Teammate 3).

Trains and compares at least 5 distinct machine learning algorithm families:
1. Linear & Probabilistic: Logistic Regression & Gaussian Naive Bayes
2. Instance-Based: K-Nearest Neighbors (KNN)
3. Tree-based / Ensemble: Random Forest
4. Boosting: LightGBM / HistGradientBoosting
5. Neural Network: Multi-Layer Perceptron (MLP)

Outputs:
- Full evaluation logs in results/logs/experiments.csv
- Figures in results/figures/ (CV Macro F1, Student Test Top-1 & F1, Log Loss, Confusion Matrices, MLP Loss History)
- Formatted Markdown table in results/figures/model_comparison_table.md
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path so pathsnap package is importable
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pathsnap  # Handles Windows DLL search paths for Cython extensions

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pathsnap.config import ARTIFACTS_DIR, CACHE_DIR, RESULTS_DIR, SEED
from pathsnap.data import list_trips, load_trip
from pathsnap.experiment import ExperimentLogger, run_experiment
from pathsnap.metrics import candidate_confusion_matrix
from pathsnap.plots import (
    generate_all_plots_and_tables,
    plot_confusion_matrix_heatmap,
    plot_mlp_loss_curve,
)


def generate_standin_candidate_table(
    partition: str = "train",
    max_trips: int = 15,
    candidates_per_point: int = 4
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build a fast stand-in candidate table directly from raw trip files.
    
    Used when Tanmay & Claude are still assembling the production candidate table.
    Generates realistic geometric features (dist_m, bearing_diff, etc.).
    """
    trip_ids = list_trips(partition)[:max_trips]
    print(f"[Stand-in Data] Generating candidate dataset from {len(trip_ids)} '{partition}' trips...")
    
    records = []
    trips_dict = {}
    
    for tid in trip_ids:
        trip = load_trip(partition, tid, with_route=True)
        trips_dict[tid] = trip
        track_xy = trip.track_xy()
        nodes_xy = trip.nodes_xy()
        arcs = trip.arcs
        n_points = min(len(track_xy), 150) # Sample points for fast stand-in run

        # Simple road segment midpoints for candidate generation
        from_xy = nodes_xy[arcs[:, 0]]
        to_xy = nodes_xy[arcs[:, 1]]
        arc_midpoints = 0.5 * (from_xy + to_xy)
        
        # Ground truth route set for labelling stand-in data
        route_set = set(trip.route.tolist()) if trip.route is not None else set()

        for pt_idx in range(n_points):
            pt = track_xy[pt_idx]
            # Euclidean distances to all arc midpoints
            dists = np.hypot(arc_midpoints[:, 0] - pt[0], arc_midpoints[:, 1] - pt[1])
            # Nearest K candidates
            nearest_arc_ids = np.argpartition(dists, candidates_per_point)[:candidates_per_point]
            nearest_arc_ids = nearest_arc_ids[np.argsort(dists[nearest_arc_ids])]

            # Assign label (1 if in route and nearest, else 0)
            has_pos = False
            for rank, arc_id in enumerate(nearest_arc_ids):
                d_m = float(dists[arc_id])
                # Arc vector
                vec = to_xy[arc_id] - from_xy[arc_id]
                arc_len = float(np.hypot(vec[0], vec[1]))
                bearing = float(np.degrees(np.arctan2(vec[1], vec[0])) % 360)

                # Stand-in label
                if not has_pos and (arc_id in route_set or rank == 0):
                    lbl = 1
                    has_pos = True
                else:
                    lbl = 0

                records.append({
                    "trip_id": tid,
                    "point_idx": pt_idx,
                    "arc_id": int(arc_id),
                    "dist_m": d_m,
                    "arc_length_m": arc_len,
                    "arc_bearing": bearing,
                    "candidate_rank": rank,
                    "label": lbl,
                })

    df = pd.DataFrame(records)
    print(f"[Stand-in Data] Generated {len(df)} candidate rows across {len(trip_ids)} trips.")
    return df, trips_dict


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Extract list of numeric feature columns for modeling."""
    excluded = {"trip_id", "point_idx", "arc_id", "label", "score", "ref_arc"}
    return [col for col in df.columns if col not in excluded]


def run_all_model_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_trips: Dict[str, object],
    test_trips: Dict[str, object],
):
    """Train and evaluate 5+ model families using standardized experiment harness."""
    feature_cols = get_feature_cols(train_df)
    print(f"\nTraining features: {feature_cols}")

    logger = ExperimentLogger()

    # -------------------------------------------------------------
    # 1. Linear & Probabilistic: Logistic Regression
    # -------------------------------------------------------------
    lr_pipeline = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, penalty="l2", max_iter=300, random_state=SEED))
    ])
    res_lr = run_experiment(
        model_name="Logistic Regression",
        family="Linear and Probabilistic",
        key_config="StandardScaler + L2 LogisticRegression (C=1.0, max_iter=300)",
        model_fn=lr_pipeline,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        train_trips=train_trips,
        test_trips=test_trips,
        logger=logger,
    )

    # -------------------------------------------------------------
    # 2. Linear & Probabilistic: Gaussian Naive Bayes
    # -------------------------------------------------------------
    gnb_pipeline = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ])
    res_gnb = run_experiment(
        model_name="Gaussian Naive Bayes",
        family="Linear and Probabilistic",
        key_config="StandardScaler + GaussianNB (var_smoothing=1e-9)",
        model_fn=gnb_pipeline,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        train_trips=train_trips,
        test_trips=test_trips,
        logger=logger,
    )

    # -------------------------------------------------------------
    # 3. Instance-based: K-Nearest Neighbors (KNN)
    # -------------------------------------------------------------
    knn_pipeline = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance", algorithm="kd_tree"))
    ])
    res_knn = run_experiment(
        model_name="K-Nearest Neighbors",
        family="Instance-based",
        key_config="StandardScaler + KNN (k=7, weights='distance', kd_tree)",
        model_fn=knn_pipeline,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        train_trips=train_trips,
        test_trips=test_trips,
        logger=logger,
    )

    # -------------------------------------------------------------
    # 4. Ensemble: Random Forest Classifier
    # -------------------------------------------------------------
    rf_model = lambda: RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=SEED,
        n_jobs=-1
    )
    res_rf = run_experiment(
        model_name="Random Forest",
        family="Ensemble",
        key_config="RandomForestClassifier (n_trees=100, max_depth=12, min_samples=5)",
        model_fn=rf_model,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        train_trips=train_trips,
        test_trips=test_trips,
        logger=logger,
    )

    # -------------------------------------------------------------
    # 5. Boosting: HistGradientBoosting / LightGBM
    # -------------------------------------------------------------
    hgb_model = lambda: HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.08,
        max_depth=8,
        random_state=SEED
    )
    res_hgb = run_experiment(
        model_name="Gradient Boosting (HistGBM)",
        family="Boosting",
        key_config="HistGradientBoosting (n_iter=100, lr=0.08, max_depth=8)",
        model_fn=hgb_model,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        train_trips=train_trips,
        test_trips=test_trips,
        logger=logger,
    )

    # -------------------------------------------------------------
    # 6. Neural Network: Multi-Layer Perceptron (MLP)
    # -------------------------------------------------------------
    print("\nTraining MLPClassifier to capture epoch loss curve...")
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=0.005,
        max_iter=120,
        random_state=SEED,
        early_stopping=False
    )
    mlp_pipeline = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", mlp_clf)
    ])
    res_mlp = run_experiment(
        model_name="Multi-Layer Perceptron (MLP)",
        family="Neural Networks",
        key_config="MLPClassifier (layers=[64, 32], relu, lr=0.005, max_iter=120)",
        model_fn=mlp_pipeline,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        train_trips=train_trips,
        test_trips=test_trips,
        logger=logger,
    )

    # Plot real training loss history of MLP
    if hasattr(mlp_clf, "loss_curve_"):
        plot_mlp_loss_curve(mlp_clf.loss_curve_)

    # -------------------------------------------------------------
    # Generate all summary plots and markdown tables
    # -------------------------------------------------------------
    print("\nGenerating report figures and tables...")
    generate_all_plots_and_tables()
    print("\nAll model experiments completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="PathSnap Model Experimentation Harness")
    parser.add_argument("--train-data", type=str, default=None, help="Path to training candidate table (.parquet or .csv)")
    parser.add_argument("--test-data", type=str, default=None, help="Path to student-test candidate table (.parquet or .csv)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic stand-in candidate generation")
    args = parser.parse_args()

    # Load production data if supplied or present in cache
    train_candidates_path = args.train_data or (CACHE_DIR / "candidates_train.parquet")
    test_candidates_path = args.test_data or (CACHE_DIR / "candidates_student_test.parquet")

    if not args.synthetic and Path(train_candidates_path).exists() and Path(test_candidates_path).exists():
        print(f"Loading candidate tables from cache: {train_candidates_path}")
        train_df = pd.read_parquet(train_candidates_path) if str(train_candidates_path).endswith(".parquet") else pd.read_csv(train_candidates_path)
        test_df = pd.read_parquet(test_candidates_path) if str(test_candidates_path).endswith(".parquet") else pd.read_csv(test_candidates_path)
        train_trips, test_trips = {}, {}
    else:
        print("Using stand-in candidate data generation from raw trip files...")
        train_df, train_trips = generate_standin_candidate_table(partition="train", max_trips=20)
        test_df, test_trips = generate_standin_candidate_table(partition="student_test", max_trips=10)

    run_all_model_experiments(train_df, test_df, train_trips, test_trips)


if __name__ == "__main__":
    main()
