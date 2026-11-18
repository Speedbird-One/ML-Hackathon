"""Generate Teammate 1 Logistic Regression visualisations from real model runs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.config import CACHE_DIR, RESULTS_DIR, SEED
from pathsnap.data import load_trip
from pathsnap.plots import plot_cv_macro_f1, plot_log_loss_comparison, plot_test_comparison


FEATURES = [
    "dist_m", "along_frac", "arc_len_m", "speed_mps", "cos_heading",
    "dist_rank", "dist_minus_min", "n_cand",
]
OUTPUT_DIR = RESULTS_DIR / "logreg_analysis"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-trip", default="00000054")
    parser.add_argument("--log-path", type=Path, default=RESULTS_DIR / "logs" / "experiments_logreg.csv")
    return parser.parse_args()


def weight_name(key_config):
    return "balanced" if "class_weight=balanced" in key_config else "none"


def make_model(class_weight):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=300, class_weight=class_weight, random_state=SEED),
    )


def save_bar_chart(df, columns, labels, title, filename):
    positions = np.arange(len(df))
    width = 0.7 / len(columns)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for index, (column, label) in enumerate(zip(columns, labels)):
        offset = (index - (len(columns) - 1) / 2) * width
        bars = axis.bar(positions + offset, df[column], width, label=label)
        for bar in bars:
            axis.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
            )
    axis.set_xticks(positions, df["class_weight"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def add_arc_lines(group, trip, arc_ids, color, weight, label):
    for arc_id in np.unique(np.asarray(arc_ids, dtype=int)):
        if arc_id < 0 or arc_id >= len(trip.arcs):
            continue
        start_node, end_node = trip.arcs[arc_id]
        start_lon, start_lat = trip.nodes[start_node]
        end_lon, end_lat = trip.nodes[end_node]
        folium.PolyLine(
            [(start_lat, start_lon), (end_lat, end_lon)],
            color=color,
            weight=weight,
            opacity=0.85,
            tooltip=f"{label}: {arc_id}",
        ).add_to(group)


def create_prediction_map(trip_id, predicted_arcs, output_path):
    trip = load_trip("train", trip_id, with_route=True)
    track_xy = trip.track_xy()
    nodes_xy = trip.nodes_xy()
    lower = track_xy.min(axis=0) - 150
    upper = track_xy.max(axis=0) + 150
    near_nodes = ((nodes_xy >= lower) & (nodes_xy <= upper)).all(axis=1)
    nearby_arcs = np.flatnonzero(near_nodes[trip.arcs[:, 0]] | near_nodes[trip.arcs[:, 1]])[:10_000]
    track = trip.track
    map_view = folium.Map(
        location=(float(track[:, 1].mean()), float(track[:, 0].mean())),
        tiles="CartoDB positron",
        zoom_start=14,
    )
    gps_layer = folium.FeatureGroup(name="GPS points", show=True)
    for index, (lon, lat, timestamp) in enumerate(track):
        folium.CircleMarker((lat, lon), radius=2, color="#2563eb", fill=True, fill_opacity=0.7, tooltip=f"Point {index}, t={timestamp:.1f}s").add_to(gps_layer)
    gps_layer.add_to(map_view)
    roads_layer = folium.FeatureGroup(name="Road arcs near track", show=True)
    add_arc_lines(roads_layer, trip, nearby_arcs, "#94a3b8", 1, "Road arc")
    roads_layer.add_to(map_view)
    reference_layer = folium.FeatureGroup(name="Reference route", show=True)
    add_arc_lines(reference_layer, trip, trip.route, "#f97316", 4, "Reference arc")
    reference_layer.add_to(map_view)
    prediction_layer = folium.FeatureGroup(name="Classifier-only predicted route", show=True)
    add_arc_lines(prediction_layer, trip, predicted_arcs, "#16a34a", 3, "Predicted arc")
    prediction_layer.add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    map_view.fit_bounds([(float(track[:, 1].min()), float(track[:, 0].min())), (float(track[:, 1].max()), float(track[:, 0].max()))])
    map_view.save(output_path)


def generate_global_comparisons():
    log_paths = sorted((RESULTS_DIR / "logs").glob("experiments_*.csv"))
    if not log_paths:
        return
    all_runs = pd.concat([pd.read_csv(path) for path in log_paths], ignore_index=True)
    best_runs = all_runs.loc[all_runs.groupby("model_name")["cv_macro_f1"].idxmax()].copy()
    if best_runs["model_name"].nunique() < 5:
        print("Global comparison charts need logs for all five required models. Skipping them for now.")
        return
    output_dir = RESULTS_DIR / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_cv_macro_f1(best_runs, output_dir / "cross_validation_macro_f1_all_models.png")
    plot_test_comparison(best_runs, output_dir / "student_test_top1_and_macro_f1_all_models.png")
    plot_log_loss_comparison(best_runs, output_dir / "student_test_log_loss_all_models.png")


def main():
    args = parse_args()
    figures_dir = OUTPUT_DIR / "figures"
    maps_dir = OUTPUT_DIR / "maps"
    figures_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    if not args.log_path.exists():
        raise FileNotFoundError("Run train_logreg.py for both class-weight settings before generating visualisations.")

    runs = pd.read_csv(args.log_path)
    runs["class_weight"] = runs["key_config"].map(weight_name)
    runs = runs.sort_values("timestamp").drop_duplicates("class_weight", keep="last").sort_values("class_weight")
    if set(runs["class_weight"]) != {"none", "balanced"}:
        raise ValueError("The log must contain both class_weight=None and class_weight=balanced runs.")

    save_bar_chart(runs, ["cv_macro_f1"], ["CV Macro F1"], "Cross-validation Macro F1 by class weight", figures_dir / "01_cv_macro_f1_by_class_weight.png")
    save_bar_chart(runs, ["test_top1_acc", "test_macro_f1"], ["Top-1 accuracy", "Macro F1"], "Student-test performance by class weight", figures_dir / "02_student_test_top1_and_macro_f1_by_class_weight.png")
    save_bar_chart(runs, ["test_log_loss"], ["Log loss"], "Student-test log loss by class weight", figures_dir / "03_student_test_log_loss_by_class_weight.png")
    save_bar_chart(runs, ["test_route_continuity"], ["Route continuity"], "Student-test route continuity by class weight", figures_dir / "04_student_test_route_continuity_by_class_weight.png")

    train_df = pd.read_parquet(CACHE_DIR / "candidates_train.parquet")
    test_df = pd.read_parquet(CACHE_DIR / "candidates_student_test.parquet")
    train_labelled = train_df[train_df["has_ref"]].copy()
    test_labelled = test_df[test_df["has_ref"]].copy()
    X_train = train_labelled[FEATURES].to_numpy()
    y_train = train_labelled["label"].to_numpy()
    X_test = test_labelled[FEATURES].to_numpy()
    y_test = test_labelled["label"].to_numpy()
    map_scores = None

    for class_weight in [None, "balanced"]:
        name = "none" if class_weight is None else "balanced"
        model = make_model(class_weight)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        fig, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, predictions), display_labels=["Candidate", "Reference arc"]).plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(f"Logistic Regression confusion matrix ({name})")
        fig.tight_layout()
        fig.savefig(figures_dir / f"05_confusion_matrix_class_weight_{name}.png", dpi=180)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(6, 4.5))
        CalibrationDisplay.from_predictions(y_test, probabilities, n_bins=10, strategy="quantile", ax=axis, name=f"class_weight={name}")
        axis.set_title(f"Probability calibration ({name})")
        fig.tight_layout()
        fig.savefig(figures_dir / f"06_probability_calibration_class_weight_{name}.png", dpi=180)
        plt.close(fig)

        probability_df = pd.DataFrame({"Probability": probabilities, "Label": np.where(y_test == 1, "Reference arc", "Other candidate")})
        fig, axis = plt.subplots(figsize=(7, 4.5))
        sns.histplot(probability_df, x="Probability", hue="Label", bins=30, stat="density", common_norm=False, element="step", ax=axis)
        axis.set_title(f"Predicted-probability distribution ({name})")
        fig.tight_layout()
        fig.savefig(figures_dir / f"07_probability_distribution_class_weight_{name}.png", dpi=180)
        plt.close(fig)

        if class_weight is None:
            map_scores = train_df[["trip_id", "point_idx", "arc_id"]].copy()
            map_scores["score"] = model.predict_proba(train_df[FEATURES].to_numpy())[:, 1]
            print(f"Unweighted student-test log loss: {log_loss(y_test, probabilities):.4f}")

    selected = map_scores[map_scores["trip_id"].astype(str) == args.map_trip]
    if selected.empty:
        raise ValueError(f"Trip {args.map_trip!r} is not present in the training candidate table.")
    top_rows = selected.loc[selected.groupby("point_idx")["score"].idxmax()].sort_values("point_idx")
    create_prediction_map(args.map_trip, top_rows["arc_id"].to_numpy(), maps_dir / f"08_folium_classifier_prediction_map_{args.map_trip}.html")
    print("Logistic Regression has no epoch-wise loss history, so no training-loss curve is generated.")
    generate_global_comparisons()
    print(f"Saved Logistic Regression visualisations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
