"""Interactive Folium and static publication map generation for PathSnap.

Renders 4 mandatory layers as required by Section 6 of Hackathon Guidelines:
1. Original GPS observations (blue circle markers with timestamp tooltips + trajectory line)
2. Relevant road-network geometry (grey candidate arcs near trajectory)
3. Reference route (ground-truth, solid emerald green)
4. Predicted route (model top-1 predictions, dashed crimson/coral)

Teammate Usage:
    # 1. Any specific model:
    python scripts/make_map.py --model lgbm --trip-id 00000000
    python scripts/make_map.py --model rf   --trip-id 00000000
    python scripts/make_map.py --model logreg --trip-id 00000000
    python scripts/make_map.py --model nb   --trip-id 00000000
    python scripts/make_map.py --model mlp  --trip-id 00000000

    # 2. Automatically generate maps for all available scored models in results/scores/:
    python scripts/make_map.py --model all --trip-id 00000000

    # 3. Pure EDA / Reference map (no model predictions, just GPS + Roads + Reference route):
    python scripts/make_map.py --model none --trip-id 00000000

    # 4. Explicit custom scores file:
    python scripts/make_map.py --scores-path path/to/custom_scores.parquet --model-name "HMM Decoded"
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import RESULTS_DIR
from pathsnap.data import Trip, list_trips, load_trip

MODEL_NAMES = {
    "lgbm": "LightGBM",
    "rf": "Random Forest",
    "logreg": "Logistic Regression",
    "nb": "Gaussian Naive Bayes",
    "mlp": "MLP (Neural Net)",
    "hmm": "HMM / Viterbi Decoded",
}


def load_parquet_scores(path: Path) -> pd.DataFrame:
    """Load scores dataframe using pyarrow or duckdb fallback."""
    try:
        return pd.read_parquet(path)
    except Exception:
        import duckdb
        return duckdb.read_parquet(str(path)).df()


def extract_arc_segments(trip: Trip, arc_ids: List[int]) -> List[List[List[float]]]:
    """Convert a sequence of arc IDs into [[lat_u, lon_u], [lat_v, lon_v]] segments for Folium."""
    segments = []
    for aid in arc_ids:
        if aid < 0 or aid >= len(trip.arcs):
            continue
        u, v = trip.arcs[aid]
        lon_u, lat_u = trip.nodes[u]
        lon_v, lat_v = trip.nodes[v]
        segments.append([[lat_u, lon_u], [lat_v, lon_v]])
    return segments


def build_folium_map(
    trip: Trip,
    pred_arcs: Optional[List[int]],
    cand_arcs: Optional[List[int]],
    model_label: str,
    partition: str,
) -> folium.Map:
    """Construct an interactive multi-layer Folium map."""
    track = trip.track
    lats = track[:, 1]
    lons = track[:, 0]
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # -------------------------------------------------------------
    # Layer 1: Road Network (Candidate Arcs near trajectory)
    # -------------------------------------------------------------
    if cand_arcs:
        fg_road = folium.FeatureGroup(name="1. Road Network (Candidate Arcs)", show=False)
        cand_segments = extract_arc_segments(trip, cand_arcs)
        if cand_segments:
            folium.PolyLine(
                locations=cand_segments,
                color="#888888",
                weight=2.0,
                opacity=0.45,
                tooltip="Candidate Road Arc",
            ).add_to(fg_road)
        fg_road.add_to(m)

    # -------------------------------------------------------------
    # Layer 2: Original GPS Observations
    # -------------------------------------------------------------
    fg_gps = folium.FeatureGroup(name="2. GPS Observations (Raw)", show=True)
    # Connecting trajectory line
    folium.PolyLine(
        locations=[[p[1], p[0]] for p in track],
        color="#1f77b4",
        weight=1.5,
        opacity=0.4,
        dash_array="3, 5",
        tooltip="GPS Trajectory Line",
    ).add_to(fg_gps)

    # GPS Point Markers (stride to keep HTML lightweight if > 1500 points)
    step = 1 if len(track) <= 1500 else 2
    for idx in range(0, len(track), step):
        p = track[idx]
        folium.CircleMarker(
            location=[p[1], p[0]],
            radius=2.5,
            color="#1f77b4",
            fill=True,
            fill_color="#3182bd",
            fill_opacity=0.6,
            weight=1,
            tooltip=f"GPS #{idx} (t={p[2]:.0f}s, [{p[1]:.5f}, {p[0]:.5f}])",
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # -------------------------------------------------------------
    # Layer 3: Reference Route (Ground Truth)
    # -------------------------------------------------------------
    gt_count = len(trip.route) if trip.route is not None else 0
    if trip.route is not None and gt_count > 0:
        fg_ref = folium.FeatureGroup(name=f"3. Reference Route (Ground Truth, {gt_count} arcs)", show=True)
        ref_segments = extract_arc_segments(trip, trip.route)
        folium.PolyLine(
            locations=ref_segments,
            color="#2ca02c",
            weight=6.0,
            opacity=0.75,
            tooltip="Reference Route (Ground Truth)",
        ).add_to(fg_ref)
        fg_ref.add_to(m)

    # -------------------------------------------------------------
    # Layer 4: Model Predicted Route (if model evaluated)
    # -------------------------------------------------------------
    clean_pred_arcs = []
    if pred_arcs:
        for a in pred_arcs:
            if not clean_pred_arcs or clean_pred_arcs[-1] != a:
                clean_pred_arcs.append(a)

        fg_pred = folium.FeatureGroup(
            name=f"4. Predicted Route ({model_label}, {len(clean_pred_arcs)} arcs)",
            show=True,
        )
        pred_segments = extract_arc_segments(trip, clean_pred_arcs)
        folium.PolyLine(
            locations=pred_segments,
            color="#d62728",
            weight=3.5,
            opacity=0.95,
            dash_array="6, 6",
            tooltip=f"Predicted Route: {model_label}",
        ).add_to(fg_pred)
        fg_pred.add_to(m)

    # -------------------------------------------------------------
    # Layer 5: Start & End Markers
    # -------------------------------------------------------------
    fg_markers = folium.FeatureGroup(name="5. Trip Endpoints (Start/End)", show=True)
    folium.Marker(
        location=[track[0, 1], track[0, 0]],
        popup=f"<b>Trip Start</b><br>ID: {trip.trip_id}<br>t=0s",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(fg_markers)

    folium.Marker(
        location=[track[-1, 1], track[-1, 0]],
        popup=f"<b>Trip End</b><br>ID: {trip.trip_id}<br>t={track[-1, 2]:.0f}s",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(fg_markers)
    fg_markers.add_to(m)

    # -------------------------------------------------------------
    # Layer Control
    # -------------------------------------------------------------
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # -------------------------------------------------------------
    # Title & Metadata Dashboard Overlay
    # -------------------------------------------------------------
    overlap_text = ""
    if trip.route is not None and gt_count > 0 and clean_pred_arcs:
        gt_set = set(trip.route)
        pred_set = set(clean_pred_arcs)
        overlap = len(gt_set.intersection(pred_set))
        pct = (overlap / len(gt_set)) * 100.0
        overlap_text = f"<b>Ground Truth Arc Match:</b> {overlap}/{gt_count} ({pct:.1f}%)<br>"

    header_html = f"""
    <div style="
        position: fixed;
        top: 15px;
        left: 60px;
        z-index: 9999;
        background-color: rgba(255, 255, 255, 0.95);
        border: 2px solid #2b5c8f;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        font-family: Arial, sans-serif;
        font-size: 13px;
        line-height: 1.4;
        max-width: 380px;
    ">
        <h4 style="margin: 0 0 6px 0; color: #2b5c8f; font-size: 15px; font-weight: bold;">
            PathSnap Map-Matching | {model_label}
        </h4>
        <b>Partition:</b> {partition} | <b>Trip ID:</b> <code>{trip.trip_id}</code><br>
        <b>GPS Observations:</b> {len(track):,} points<br>
        {overlap_text}
        <span style="font-size: 11px; color: #666;">Toggle layers via top-right controls to inspect alignment.</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(header_html))

    # -------------------------------------------------------------
    # Legend Overlay
    # -------------------------------------------------------------
    pred_legend_line = f'<span style="color: #d62728; font-weight: bold;">- - -</span> Predicted Route ({model_label})<br>' if pred_arcs else ""
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        right: 15px;
        z-index: 9999;
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px 14px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 1.6;
    ">
        <b style="font-size: 13px; color: #333;">Map Legend</b><br>
        <span style="color: #1f77b4; font-size: 14px;">&#9679;</span> GPS Points (Noisy observations)<br>
        <span style="color: #888888; font-weight: bold;">&#8213;</span> Candidate Road Arcs (Network)<br>
        <span style="color: #2ca02c; font-weight: bold;">&#9644;</span> Reference Route (Ground Truth)<br>
        {pred_legend_line}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Auto-fit viewport bounds
    m.fit_bounds([[float(np.min(lats)), float(np.min(lons))], [float(np.max(lats)), float(np.max(lons))]])
    return m


def export_static_figure(
    trip: Trip,
    pred_arcs: Optional[List[int]],
    cand_arcs: Optional[List[int]],
    model_label: str,
    partition: str,
    save_path: Path,
):
    """Generate high-resolution 300 DPI publication figure for Technical Report Section 6."""
    track = trip.track
    lons = track[:, 0]
    lats = track[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 1. Road network candidate arcs
    if cand_arcs:
        cand_segs = extract_arc_segments(trip, cand_arcs)
        for seg in cand_segs:
            ax.plot([seg[0][1], seg[1][1]], [seg[0][0], seg[1][0]], color="#cccccc", lw=1.0, zorder=1)
        if cand_segs:
            ax.plot([], [], color="#aaaaaa", lw=1.2, label=f"Candidate Road Arcs ({len(cand_arcs)})")

    # 2. Raw GPS observations
    ax.scatter(lons, lats, c="#1f77b4", s=9, alpha=0.45, label=f"GPS Points ({len(track):,})", zorder=2)
    ax.plot(lons, lats, c="#1f77b4", lw=0.6, ls="--", alpha=0.3, zorder=2)

    # 3. Reference route
    if trip.route is not None and len(trip.route) > 0:
        ref_segs = extract_arc_segments(trip, trip.route)
        for seg in ref_segs:
            ax.plot([seg[0][1], seg[1][1]], [seg[0][0], seg[1][0]], color="#2ca02c", lw=4.5, alpha=0.75, zorder=3)
        ax.plot([], [], color="#2ca02c", lw=4.5, label=f"Reference Route ({len(trip.route)} arcs)")

    # 4. Predicted route
    if pred_arcs:
        clean_pred_arcs = []
        for a in pred_arcs:
            if not clean_pred_arcs or clean_pred_arcs[-1] != a:
                clean_pred_arcs.append(a)
        pred_segs = extract_arc_segments(trip, clean_pred_arcs)
        for seg in pred_segs:
            ax.plot([seg[0][1], seg[1][1]], [seg[0][0], seg[1][0]], color="#d62728", lw=2.2, ls="--", alpha=0.95, zorder=4)
        ax.plot([], [], color="#d62728", lw=2.2, ls="--", label=f"Predicted Route: {model_label}")

    # 5. Start and End markers
    ax.scatter(lons[0], lats[0], c="green", s=90, marker="^", edgecolors="black", label="Trip Start", zorder=5)
    ax.scatter(lons[-1], lats[-1], c="red", s=90, marker="s", edgecolors="black", label="Trip End", zorder=5)

    ax.set_xlabel("Longitude (deg)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (deg)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"PathSnap Map-Matching Geographical Visualisation\nModel: {model_label} | Trip: {trip.trip_id} ({partition})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(frameon=True, loc="best", facecolor="white", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Bounding margin
    dx = np.ptp(lons) * 0.05
    dy = np.ptp(lats) * 0.05
    ax.set_xlim(np.min(lons) - dx, np.max(lons) + dx)
    ax.set_ylim(np.min(lats) - dy, np.max(lats) + dy)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Map] Saved high-resolution static figure to {save_path}")


def generate_map_for_model(
    model_key: str,
    partition: str = "student_test",
    trip_id: str = "00000000",
    scores_file: Optional[Path] = None,
    custom_model_label: Optional[str] = None,
    export_png: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """Generate both interactive Folium HTML map and static PNG figure."""
    model_key_clean = model_key.lower().strip()
    model_label = custom_model_label or MODEL_NAMES.get(model_key_clean, model_key.upper())

    print(f"\n[Map] Loading trip {trip_id} ({partition})...")
    trip = load_trip(partition, trip_id)

    pred_arcs = None
    cand_arcs = None

    if model_key_clean not in ["none", "eda"]:
        # Find scores parquet
        scores_path = scores_file or (RESULTS_DIR / "scores" / f"{model_key_clean}_{partition}.parquet")
        if not scores_path.exists():
            raise FileNotFoundError(
                f"Scores parquet not found at '{scores_path}'.\n"
                f"To generate scores first, run your training script (e.g., 'python scripts/train_{model_key_clean}.py'), "
                f"or pass an explicit scores file via '--scores-path <path>'."
            )

        print(f"[Map] Loading scores from {scores_path}...")
        scores_df = load_parquet_scores(scores_path)
        sub = scores_df[scores_df["trip_id"] == trip_id].copy()

        if sub.empty:
            raise ValueError(f"No scores found for trip {trip_id} in {scores_path}")

        # Top-1 candidate arc per point
        best_cands = sub.sort_values(["point_idx", "score"], ascending=[True, False]).groupby("point_idx", as_index=False).first()
        pred_arcs = best_cands["arc_id"].tolist()
        cand_arcs = sub["arc_id"].unique().tolist()
    else:
        model_label = "EDA / Ground Truth"

    print(f"[Map] Building interactive Folium map for {model_label}...")
    folium_map = build_folium_map(trip, pred_arcs, cand_arcs, model_label, partition)

    html_out = RESULTS_DIR / "maps" / f"map_{model_key_clean}_{partition}_{trip_id}.html"
    html_out.parent.mkdir(parents=True, exist_ok=True)
    folium_map.save(str(html_out))
    print(f"[Map] Saved interactive Folium map to {html_out} ({html_out.stat().st_size:,} bytes)")

    png_out = None
    if export_png:
        png_out = RESULTS_DIR / "figures" / f"map_{model_key_clean}_{partition}_{trip_id}.png"
        export_static_figure(trip, pred_arcs, cand_arcs, model_label, partition, png_out)

    return html_out, png_out


def main():
    parser = argparse.ArgumentParser(description="Generate interactive Folium maps and report figures for any PathSnap model.")
    parser.add_argument("--model", type=str, default="lgbm", help="Model key ('lgbm', 'rf', 'logreg', 'nb', 'mlp', 'none', or 'all')")
    parser.add_argument("--partition", type=str, default="student_test", help="Dataset partition ('train' or 'student_test')")
    parser.add_argument("--trip-id", type=str, default="00000000", help="Trip ID to map (e.g. '00000000')")
    parser.add_argument("--scores-path", type=str, default=None, help="Explicit path to scores parquet/CSV")
    parser.add_argument("--model-name", type=str, default=None, help="Custom model label to display on map")
    parser.add_argument("--no-png", action="store_true", help="Skip static PNG export")
    args = parser.parse_args()

    scores_p = Path(args.scores_path) if args.scores_path else None

    if args.model.lower() == "all":
        # Find all score parquets in results/scores
        scores_dir = RESULTS_DIR / "scores"
        available_scores = list(scores_dir.glob(f"*_{args.partition}.parquet"))
        if not available_scores:
            print(f"[Map] No score files found in {scores_dir} matching *_{args.partition}.parquet")
            return

        for sc in available_scores:
            # model_key is prefix before _{partition}.parquet
            m_key = sc.name.replace(f"_{args.partition}.parquet", "")
            try:
                generate_map_for_model(
                    model_key=m_key,
                    partition=args.partition,
                    trip_id=args.trip_id,
                    scores_file=sc,
                    export_png=not args.no_png,
                )
            except Exception as e:
                print(f"[Map Warning] Failed generating map for {m_key}: {e}")
    else:
        generate_map_for_model(
            model_key=args.model,
            partition=args.partition,
            trip_id=args.trip_id,
            scores_file=scores_p,
            custom_model_label=args.model_name,
            export_png=not args.no_png,
        )


if __name__ == "__main__":
    main()
