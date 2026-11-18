"""Folium map of one PathSnap result (the JSON returned by the API's /predict endpoint).

Depends only on folium, so it runs on the UI laptop (Laptop 2) without the ML stack.
Also used by api/main.py (/map/<trip_id>) and scripts/eval_pipeline.py for report maps.

Layers (each toggleable in the layer control):
  1. Road network      - candidate road arcs within the search radius (grey)
  2. GPS observations  - raw noisy points (blue) + dashed trajectory
  3. Reference route   - ground truth, when available (green, thick)
  4. Classifier only   - top-1 arc per point, no HMM (orange, dashed)
  5. HMM / Viterbi     - final decoded route (red, solid)
  6. HMM corrections   - points where Viterbi changed the classifier's choice (purple)
  7. Start / end markers
"""
from typing import Any, Dict, Optional

import folium

COLORS = {
    "road": "#8c8c8c",
    "gps": "#1f77b4",
    "reference": "#2ca02c",
    "classifier": "#ff7f0e",
    "hmm": "#d62728",
    "changed": "#9467bd",
}


def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return "N/A"
    return f"{100 * v:.1f}%" if pct else f"{v}"


def build_route_map(result: Dict[str, Any], max_gps_markers: int = 1500) -> folium.Map:
    geo = result["geometry"]
    gps = geo["gps"]  # [[lat, lon, t], ...]
    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    clf_arcs = result.get("predictions", {}).get("classifier_arcs")
    hmm_arcs = result.get("predictions", {}).get("hmm_arcs")

    m = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)],
                   zoom_start=14, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light (CartoDB)").add_to(m)

    # 1. road network
    if geo.get("road"):
        fg = folium.FeatureGroup(name=f"Road network near track ({len(geo['road'])} arcs)", show=False)
        folium.PolyLine(geo["road"], color=COLORS["road"], weight=2, opacity=0.5,
                        tooltip="Road arc (candidate)").add_to(fg)
        fg.add_to(m)

    # 2. GPS observations
    fg = folium.FeatureGroup(name=f"GPS observations ({len(gps)} pts)", show=True)
    folium.PolyLine([[p[0], p[1]] for p in gps], color=COLORS["gps"], weight=1.5,
                    opacity=0.5, dash_array="3,5").add_to(fg)
    step = max(1, len(gps) // max_gps_markers + (1 if len(gps) % max_gps_markers else 0))
    for i in range(0, len(gps), step):
        la, lo, t = gps[i]
        tip = f"GPS #{i} | t={t:.0f}s"
        if clf_arcs is not None and hmm_arcs is not None:
            tip += f" | classifier arc {clf_arcs[i]} | HMM arc {hmm_arcs[i]}"
        folium.CircleMarker([la, lo], radius=2.5, color=COLORS["gps"], fill=True,
                            fill_opacity=0.7, weight=1, tooltip=tip).add_to(fg)
    fg.add_to(m)

    # 3. reference route
    if geo.get("reference"):
        fg = folium.FeatureGroup(name=f"Reference route ({len(geo['reference'])} arcs)", show=True)
        folium.PolyLine(geo["reference"], color=COLORS["reference"], weight=7, opacity=0.6,
                        tooltip="Reference route (ground truth)").add_to(fg)
        fg.add_to(m)

    # 4. classifier only
    if geo.get("classifier"):
        fg = folium.FeatureGroup(name=f"Classifier only ({len(geo['classifier'])} arcs)", show=False)
        folium.PolyLine(geo["classifier"], color=COLORS["classifier"], weight=3.5, opacity=0.95,
                        dash_array="6,6", tooltip="Classifier-only top-1 route").add_to(fg)
        fg.add_to(m)

    # 5. HMM route
    if geo.get("hmm"):
        fg = folium.FeatureGroup(name=f"HMM / Viterbi route ({len(geo['hmm'])} arcs)", show=True)
        folium.PolyLine(geo["hmm"], color=COLORS["hmm"], weight=4, opacity=0.9,
                        tooltip="Predicted route (classifier + HMM/Viterbi)").add_to(fg)
        fg.add_to(m)

    # 6. points corrected by HMM
    changed = geo.get("changed_points") or []
    if changed:
        fg = folium.FeatureGroup(name=f"Points changed by HMM ({len(changed)})", show=False)
        for i in changed:
            la, lo, t = gps[i]
            folium.CircleMarker([la, lo], radius=4, color=COLORS["changed"], fill=True,
                                fill_opacity=0.9, weight=1,
                                tooltip=f"GPS #{i}: classifier {clf_arcs[i]} -> HMM {hmm_arcs[i]}").add_to(fg)
        fg.add_to(m)

    # 7. endpoints
    fg = folium.FeatureGroup(name="Start / end", show=True)
    folium.Marker([gps[0][0], gps[0][1]], tooltip="Trip start",
                  icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(fg)
    folium.Marker([gps[-1][0], gps[-1][1]], tooltip=f"Trip end (t={gps[-1][2]:.0f}s)",
                  icon=folium.Icon(color="red", icon="stop", prefix="fa")).add_to(fg)
    fg.add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # info box + legend
    mc = result.get("metrics", {}).get("classifier", {})
    mh = result.get("metrics", {}).get("hmm", {})
    model = result.get("model", {}).get("name", "model")
    rows = ""
    for label, key in [("Top-1 accuracy", "top1_accuracy"), ("Route continuity", "route_continuity"),
                       ("Route recall", "route_recall"), ("Route precision", "route_precision")]:
        rows += (f"<tr><td>{label}</td><td style='text-align:right'>{_fmt(mc.get(key))}</td>"
                 f"<td style='text-align:right'><b>{_fmt(mh.get(key))}</b></td></tr>")
    info = f"""
    <div style="position:fixed;top:12px;left:55px;z-index:9999;background:rgba(255,255,255,.95);
         border:1px solid #bbb;border-radius:8px;padding:10px 12px;font:12px Arial,sans-serif;
         box-shadow:0 2px 8px rgba(0,0,0,.2);max-width:340px">
      <div style="font-size:14px;font-weight:bold;margin-bottom:4px">PathSnap | Trip {result['trip_id']}</div>
      {result.get('partition','')} &middot; {result['n_points']} GPS points &middot; {model} + HMM<br>
      <table style="margin-top:6px;border-collapse:collapse;width:100%">
        <tr style="color:#555"><td></td><td style='text-align:right'>Classifier</td>
        <td style='text-align:right'>+ HMM</td></tr>{rows}
      </table>
    </div>"""
    legend = f"""
    <div style="position:fixed;bottom:25px;left:12px;z-index:9999;background:rgba(255,255,255,.95);
         border:1px solid #bbb;border-radius:8px;padding:8px 12px;font:12px Arial,sans-serif;line-height:1.7">
      <b>Legend</b><br>
      <span style="color:{COLORS['gps']}">&#9679;</span> GPS observations (raw)<br>
      <span style="color:{COLORS['road']}">&#9472;&#9472;</span> Road network near track<br>
      <span style="color:{COLORS['reference']};font-weight:bold">&#9473;&#9473;</span> Reference route<br>
      <span style="color:{COLORS['classifier']};font-weight:bold">- - -</span> Classifier only<br>
      <span style="color:{COLORS['hmm']};font-weight:bold">&#9473;&#9473;</span> HMM / Viterbi route<br>
      <span style="color:{COLORS['changed']}">&#9679;</span> Point changed by HMM
    </div>"""
    m.get_root().html.add_child(folium.Element(info))
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))
    return m


def map_html(result: Dict[str, Any]) -> str:
    """Full standalone HTML document for the map."""
    return build_route_map(result).get_root().render()
