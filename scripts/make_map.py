"""Create a layered Folium map for one training or student-test trip."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import numpy as np
import pandas as pd

from pathsnap.config import RESULTS_DIR
from pathsnap.data import list_trips, load_trip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=["train", "student_test"])
    parser.add_argument("trip_id", choices=None)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--max-road-arcs", type=int, default=10_000)
    return parser.parse_args()


def add_arc_lines(group, trip, arc_ids, color, weight, tooltip_prefix):
    unique_ids = np.unique(np.asarray(arc_ids, dtype=int))
    valid_ids = unique_ids[(unique_ids >= 0) & (unique_ids < len(trip.arcs))]
    for arc_id in valid_ids:
        start_node, end_node = trip.arcs[arc_id]
        start_lon, start_lat = trip.nodes[start_node]
        end_lon, end_lat = trip.nodes[end_node]
        folium.PolyLine(
            locations=[(start_lat, start_lon), (end_lat, end_lon)],
            color=color,
            weight=weight,
            opacity=0.85,
            tooltip=f"{tooltip_prefix}: {arc_id}",
        ).add_to(group)


def nearby_arc_ids(trip, padding_m=150.0):
    track_xy = trip.track_xy()
    nodes_xy = trip.nodes_xy()
    lower = track_xy.min(axis=0) - padding_m
    upper = track_xy.max(axis=0) + padding_m
    near_nodes = ((nodes_xy >= lower) & (nodes_xy <= upper)).all(axis=1)
    return np.flatnonzero(near_nodes[trip.arcs[:, 0]] | near_nodes[trip.arcs[:, 1]])


def predicted_arc_ids(scores_path, trip_id):
    scores = pd.read_parquet(scores_path)
    required = {"trip_id", "point_idx", "arc_id", "score"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Scores file is missing columns: {sorted(missing)}")
    trip_scores = scores[scores["trip_id"].astype(str) == trip_id]
    if trip_scores.empty:
        return np.array([], dtype=int)
    top_rows = trip_scores.loc[trip_scores.groupby("point_idx")["score"].idxmax()]
    return top_rows.sort_values("point_idx")["arc_id"].to_numpy(dtype=int)


def main():
    args = parse_args()
    if args.trip_id not in list_trips(args.partition):
        raise ValueError(f"Trip {args.trip_id!r} was not found in {args.partition}.")

    trip = load_trip(args.partition, args.trip_id, with_route=True)
    track = trip.track
    map_view = folium.Map(
        location=(float(track[:, 1].mean()), float(track[:, 0].mean())),
        tiles="CartoDB positron",
        zoom_start=14,
    )

    gps_layer = folium.FeatureGroup(name="GPS points", show=True)
    for point_idx, (lon, lat, timestamp) in enumerate(track):
        folium.CircleMarker(
            location=(lat, lon),
            radius=2,
            color="#2563eb",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"Point {point_idx}, t={timestamp:.1f}s",
        ).add_to(gps_layer)
    gps_layer.add_to(map_view)

    road_ids = nearby_arc_ids(trip)
    if len(road_ids) > args.max_road_arcs:
        road_ids = road_ids[: args.max_road_arcs]
        print(f"Road layer limited to {args.max_road_arcs:,} nearby arcs.")
    roads_layer = folium.FeatureGroup(name="Road arcs near track", show=True)
    add_arc_lines(roads_layer, trip, road_ids, "#94a3b8", 1, "Road arc")
    roads_layer.add_to(map_view)

    if trip.route is not None and len(trip.route):
        reference_layer = folium.FeatureGroup(name="Reference route", show=True)
        add_arc_lines(reference_layer, trip, trip.route, "#f97316", 4, "Reference arc")
        reference_layer.add_to(map_view)

    if args.scores:
        prediction_layer = folium.FeatureGroup(name="Predicted route", show=True)
        add_arc_lines(prediction_layer, trip, predicted_arc_ids(args.scores, args.trip_id), "#16a34a", 3, "Predicted arc")
        prediction_layer.add_to(map_view)

    folium.LayerControl(collapsed=False).add_to(map_view)
    legend = """
    <div style='position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; border: 1px solid #94a3b8; padding: 10px;'>
      <b>PathSnap layers</b><br>
      <span style='color:#2563eb;'>●</span> GPS points<br>
      <span style='color:#94a3b8;'>━</span> Road arcs<br>
      <span style='color:#f97316;'>━</span> Reference route<br>
      <span style='color:#16a34a;'>━</span> Predicted route
    </div>
    """
    map_view.get_root().html.add_child(folium.Element(legend))
    map_view.fit_bounds([(float(track[:, 1].min()), float(track[:, 0].min())), (float(track[:, 1].max()), float(track[:, 0].max()))])

    output = RESULTS_DIR / "maps" / f"{args.partition}_{args.trip_id}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(output)
    print(f"Saved map to {output}")


if __name__ == "__main__":
    main()
