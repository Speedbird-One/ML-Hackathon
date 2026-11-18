"""EDA additions required by the report (section 2.2): missing-value / validity analysis, candidate coverage, and
extra trajectory and location plots. Each analysis states the partition it uses (train and student_test; the evaluator
trips are never read).

Usage (repo root; candidate tables built):
    python scripts/eda_extras.py
Outputs in results/eda/:
    missing_value_by_trip.csv         per-trip validity and missing-data counts (both partitions)
    missing_value_summary.csv/.md     per-partition totals and percentages
    candidate_coverage.csv            labelled points, points without candidates, candidate miss rate (40 m radius)
    network_connectivity.csv/.png     per-trip road-network connectivity: components, isolated nodes, route inside one component
    missing_value_analysis.png        figure: reference/candidate coverage and raw-data issue counts
    trip_locations.png                where the trips are (the dataset is worldwide)
    trajectory_samples.png            GPS points and reference route for six train trips
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.config import CACHE_DIR, RESULTS_DIR
from pathsnap.data import list_trips, load_trip
from pathsnap.labels import derive_ref_arcs

warnings.filterwarnings("ignore")
OUT = RESULTS_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)
PARTS = ["train", "student_test"]
COLORS = {"train": "#4c72b0", "student_test": "#dd8452"}

rows, cov_rows, centroids, trips_by_part, net_rows = [], [], [], {}, []
for part in PARTS:
    cands = pd.read_parquet(CACHE_DIR / f"candidates_{part}.parquet")
    pos = cands[cands.label == 1][["trip_id", "point_idx"]].drop_duplicates()
    pos_count = pos.groupby("trip_id").size()
    with_cand = cands.groupby("trip_id").point_idx.nunique()
    n_rows = cands.groupby("trip_id").size()
    trips_by_part[part] = {}
    for tid in list_trips(part):
        trip = load_trip(part, tid)
        trips_by_part[part][tid] = trip
        tr, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
        dt = np.diff(tr[:, 2])
        ref = derive_ref_arcs(trip).ref_arc
        labelled = int((ref >= 0).sum())
        n_pts = len(tr)
        hits = int(pos_count.get(tid, 0))
        rows.append({
            "partition": part, "trip": tid, "points": n_pts, "nodes": len(nodes), "arcs": len(arcs), "route_arcs": len(route),
            # raw-file validity / missing values
            "track_nan_inf": int((~np.isfinite(tr)).sum()), "node_nan_inf": int((~np.isfinite(nodes)).sum()),
            "coords_out_of_range": int(((np.abs(tr[:, 0]) > 180) | (np.abs(tr[:, 1]) > 90)).sum()),
            "nonpositive_time_steps": int((dt <= 0).sum()), "time_gaps_over_1_5s": int((dt > 1.5).sum()),
            "max_time_gap_s": float(dt.max()) if len(dt) else 0.0,
            "arc_node_ids_out_of_range": int(((arcs < 0) | (arcs >= len(nodes))).any(axis=1).sum()),
            "self_loop_arcs": int((arcs[:, 0] == arcs[:, 1]).sum()),
            "route_ids_out_of_range": int(((route < 0) | (route >= len(arcs))).sum()),
            "route_breaks": int((arcs[route[:-1], 1] != arcs[route[1:], 0]).sum()) if len(route) > 1 else 0,
            # reference / candidate coverage (derived labels, 40 m candidate radius)
            "points_without_reference_arc": n_pts - labelled, "labelled_points": labelled,
            "points_without_any_candidate": n_pts - int(with_cand.get(tid, 0)),
            "labelled_points_true_arc_not_candidate": labelled - hits,
            "candidate_rows": int(n_rows.get(tid, 0)),
        })
        centroids.append({"partition": part, "trip": tid, "lon": tr[:, 0].mean(), "lat": tr[:, 1].mean()})
        # road-network connectivity (weak components ignore arc direction; strong components respect it)
        g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(len(nodes), len(nodes))).tocsr()
        n_weak, lab_weak = connected_components(g, directed=True, connection="weak")
        n_strong, _ = connected_components(g, directed=True, connection="strong")
        sizes = np.bincount(lab_weak)
        deg = np.bincount(arcs.ravel(), minlength=len(nodes))
        route_nodes = np.unique(arcs[route].ravel())
        net_rows.append({"partition": part, "trip": tid, "nodes": len(nodes), "arcs": len(arcs), "weak_components": int(n_weak),
                         "strong_components": int(n_strong), "isolated_nodes": int((deg == 0).sum()),
                         "largest_weak_component_share": float(sizes.max() / len(nodes)),
                         "route_inside_one_weak_component": bool(len(np.unique(lab_weak[route_nodes])) == 1)})
    print(f"{part}: {len(trips_by_part[part])} trips analysed", flush=True)

by_trip = pd.DataFrame(rows)
by_trip.to_csv(OUT / "missing_value_by_trip.csv", index=False)

# ------------------------------------------------------------------------------- network connectivity
net = pd.DataFrame(net_rows)
net.to_csv(OUT / "network_connectivity.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for part in PARTS:
    g_ = net[net.partition == part]
    axes[0].hist(g_.weak_components, bins=20, alpha=0.6, label=part, color=COLORS[part])
    axes[1].hist(100 * g_.largest_weak_component_share, bins=20, alpha=0.6, label=part, color=COLORS[part])
    axes[2].hist(g_.isolated_nodes, bins=20, alpha=0.6, label=part, color=COLORS[part])
axes[0].set_xlabel("weakly connected components per trip network")
axes[1].set_xlabel("% of nodes in the largest component")
axes[2].set_xlabel("isolated nodes per trip network")
for a in axes:
    a.set_ylabel("trips")
    a.legend()
fig.suptitle("Road-network connectivity (partitions: train, student_test)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "network_connectivity.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------------------- per-partition summary
summ = []
for part, g in by_trip.groupby("partition"):
    n = int(g.points.sum())
    s = {"partition": part, "trips": len(g), "gps_points": n}
    for c in ["track_nan_inf", "node_nan_inf", "coords_out_of_range", "nonpositive_time_steps", "arc_node_ids_out_of_range",
              "self_loop_arcs", "route_ids_out_of_range", "route_breaks", "time_gaps_over_1_5s"]:
        s[c] = int(g[c].sum())
    s["trips_with_time_gaps"] = int((g.time_gaps_over_1_5s > 0).sum())
    s["longest_time_gap_s"] = float(g.max_time_gap_s.max())
    s["points_without_reference_arc"] = int(g.points_without_reference_arc.sum())
    s["pct_points_without_reference_arc"] = 100 * s["points_without_reference_arc"] / n
    s["points_without_any_candidate"] = int(g.points_without_any_candidate.sum())
    s["pct_points_without_any_candidate"] = 100 * s["points_without_any_candidate"] / n
    summ.append(s)
summary = pd.DataFrame(summ)
summary.to_csv(OUT / "missing_value_summary.csv", index=False)
(OUT / "missing_value_summary.md").write_text(summary.T.to_markdown(floatfmt=".3f"), encoding="utf-8")

# ------------------------------------------------------------------------------- candidate coverage (model-independent)
for part, g in by_trip.groupby("partition"):
    lab, miss, n_rows = int(g.labelled_points.sum()), int(g.labelled_points_true_arc_not_candidate.sum()), int(g.candidate_rows.sum())
    with_c = int(g.points.sum() - g.points_without_any_candidate.sum())
    cov_rows.append({"partition": part, "candidate_radius_m": 40, "labelled_points": lab, "labelled_points_true_arc_not_candidate": miss,
                     "candidate_miss_rate_pct": 100 * miss / lab, "points_with_candidates": with_c,
                     "mean_candidates_per_point_with_candidates": n_rows / max(with_c, 1)})
pd.DataFrame(cov_rows).to_csv(OUT / "candidate_coverage.csv", index=False)

# ------------------------------------------------------------------------------- figure: missing-value analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
ax = axes[0]
x = np.arange(len(summary))
w = 0.38
ax.bar(x - w / 2, summary.pct_points_without_reference_arc, w, label="no reference arc (derived label = unknown)", color="#c44e52")
ax.bar(x + w / 2, summary.pct_points_without_any_candidate, w, label="no candidate arc within 40 m", color="#8172b2")
ax.set_xticks(x)
ax.set_xticklabels(summary.partition)
ax.set_ylabel("% of GPS points")
ax.set_title("Points missing a reference arc / any candidate")
ax.set_ylim(0, 1.45 * max(summary.pct_points_without_reference_arc.max(), 1))
ax.legend(fontsize=8, loc="upper left")
ax = axes[1]
for part in PARTS:
    g = by_trip[by_trip.partition == part]
    ax.hist(100 * g.points_without_reference_arc / g.points, bins=20, alpha=0.6, label=f"{part} ({len(g)} trips)", color=COLORS[part])
ax.set_xlabel("% of a trip's points without a reference arc")
ax.set_ylabel("trips")
ax.set_title("Per-trip share of unlabelled points")
ax.legend()
ax = axes[2]
issues = ["track_nan_inf", "node_nan_inf", "coords_out_of_range", "nonpositive_time_steps", "arc_node_ids_out_of_range",
          "route_ids_out_of_range", "route_breaks", "time_gaps_over_1_5s"]
y = np.arange(len(issues))
for i, part in enumerate(PARTS):
    vals = [int(summary[summary.partition == part][c].iloc[0]) for c in issues]
    ax.barh(y + (i - 0.5) * 0.38, vals, 0.38, label=part, color=COLORS[part])
ax.set_yticks(y)
ax.set_yticklabels([c.replace("_", " ") for c in issues], fontsize=8)
ax.set_xscale("symlog")
ax.set_xlabel("count (symlog scale; 0 = none found)")
ax.set_title("Raw-data validity checks")
ax.legend()
fig.suptitle("Missing-value and validity analysis (partitions: train, student_test)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "missing_value_analysis.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------------------- figure: where the trips are
cen = pd.DataFrame(centroids)
fig, ax = plt.subplots(figsize=(11, 5.2))
for part in PARTS:
    g = cen[cen.partition == part]
    ax.scatter(g.lon, g.lat, s=34, alpha=0.85, label=f"{part} ({len(g)} trips)", color=COLORS[part], edgecolor="white", linewidth=0.4)
ax.set_xlim(-180, 180)
ax.set_ylim(-60, 80)
ax.grid(alpha=0.3)
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_title("Trip locations (track centroids): every trip is a separate local map, in cities worldwide")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "trip_locations.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------------------- figure: sample trajectories
tr_rows = by_trip[by_trip.partition == "train"].sort_values("points").reset_index(drop=True)
picks = [tr_rows.trip[int(q * (len(tr_rows) - 1))] for q in (0.08, 0.25, 0.45, 0.65, 0.85, 0.97)]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, tid in zip(axes.ravel(), picks):
    trip = trips_by_part["train"][tid]
    xy, nodes_xy = trip.track_xy(), trip.nodes_xy()
    seg = np.stack([nodes_xy[trip.arcs[trip.route, 0]], nodes_xy[trip.arcs[trip.route, 1]]], axis=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=4, color="#2b6cb0", label="GPS points", zorder=2)
    ax.add_collection(LineCollection(seg, colors="#f5a623", linewidths=1.1, label="reference route", zorder=3))
    ax.scatter(*xy[0], s=60, color="#38a169", label="start", zorder=4)
    ax.scatter(*xy[-1], s=60, color="#c53030", label="end", zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"train trip {tid}: {len(xy):,} points", fontsize=10)
    ax.set_xlabel("metres east of track centroid")
    ax.set_ylabel("metres north")
axes[0, 0].legend(fontsize=8, loc="best")
fig.suptitle("GPS trajectories and reference routes, six train trips (local metric frame per trip)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "trajectory_samples.png", dpi=200)
plt.close(fig)

print("\n=== SUMMARY (per partition) ===")
print(summary.T.to_markdown(floatfmt=".3f"))
print("\n=== NETWORK CONNECTIVITY (per partition) ===")
print(net.groupby("partition").agg(trips=("trip", "count"), median_components=("weak_components", "median"),
                                   max_components=("weak_components", "max"), median_isolated=("isolated_nodes", "median"),
                                   median_largest_share=("largest_weak_component_share", "median"),
                                   routes_in_one_component=("route_inside_one_weak_component", "sum")).to_markdown(floatfmt=".3f"))
print("\n=== CANDIDATE COVERAGE ===")
print(pd.DataFrame(cov_rows).to_markdown(index=False, floatfmt=".3f"))
print(f"\nSaved to {OUT}")
