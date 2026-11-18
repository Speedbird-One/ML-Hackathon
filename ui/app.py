"""PathSnap evaluator UI (Laptop 2).

Run from the repo root:
    streamlit run ui/app.py
Optionally preset the API address:
    PATHSNAP_API_URL=http://192.168.43.10:8000 streamlit run ui/app.py   (bash)
    $env:PATHSNAP_API_URL="http://192.168.43.10:8000"; streamlit run ui/app.py   (PowerShell)

Needs only: streamlit, requests, folium (see ui/requirements.txt). No model or ML stack here:
everything is computed on Laptop 1 and returned through the API.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_map import map_html  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))

st.set_page_config(page_title="PathSnap Evaluator", page_icon="🗺️", layout="wide")


def pct(v):
    return "N/A" if v is None else f"{100 * v:.1f}%"


def local_track(partition: str, trip_id: str):
    """GPS track from this laptop's copy of the data, if present: [[lon, lat, t], ...]."""
    p = LOCAL_DATA / partition / trip_id / f"{trip_id}.track"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return rows


@st.cache_data(ttl=10, show_spinner=False)
def get_json(url: str):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ sidebar
st.sidebar.title("PathSnap")
api = st.sidebar.text_input("API address (Laptop 1)",
                            os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")).rstrip("/")
try:
    health = get_json(f"{api}/health")
    st.sidebar.success(f"Connected · {health['partition']} · {health['n_trips']} trips")
    with st.sidebar.expander("Model served by Laptop 1"):
        st.json(health["model"])
except Exception as e:
    st.sidebar.error(f"Cannot reach {api}\n\n{e}")
    st.title("🗺️ PathSnap — Route Reconstruction")
    st.info("Start the API on Laptop 1, then enter its address (e.g. http://192.168.x.x:8000) "
            "in the sidebar.")
    st.stop()

if st.sidebar.button("Refresh"):
    get_json.clear()
    st.rerun()

# ------------------------------------------------------------------ trip choice
st.title("🗺️ PathSnap — Route Reconstruction")
st.caption("Pick an evaluation trip. Laptop 2 sends it to the API on Laptop 1, which scores candidate "
           "road arcs with the trained classifier and decodes the route with HMM/Viterbi.")

trips = get_json(f"{api}/trips")
partition = trips["partition"]
options = {t["trip_id"]: t for t in trips["trips"]}
labels = {tid: f"Trip {tid}  ·  {t['n_points']:,} GPS points" + ("  ·  ✓ ready" if t["cached"] else "")
          for tid, t in options.items()}

c1, c2 = st.columns([3, 2])
with c1:
    trip_id = st.radio(f"Evaluation trips ({partition})", list(options), format_func=labels.get,
                       horizontal=False)
with c2:
    has_local = local_track(partition, trip_id) is not None
    send_track = st.checkbox("Send GPS track from this laptop", value=has_local, disabled=not has_local,
                             help=f"Uses {LOCAL_DATA / partition}. If off, Laptop 1 uses its own copy "
                                  "of the trip's GPS file.")
    run = st.button("▶ Reconstruct route", type="primary")

if run:
    payload = {"trip_id": trip_id, "include_geometry": True}
    if send_track:
        payload["track"] = local_track(partition, trip_id)
    with st.spinner(f"Laptop 1 is reconstructing trip {trip_id}…"):
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{api}/predict", json=payload, timeout=600)
            resp.raise_for_status()
        except Exception as e:
            st.error(f"API call failed: {e}")
            st.stop()
        rtt = (time.perf_counter() - t0) * 1000
    res = resp.json()
    res["_round_trip_ms"] = rtt
    st.session_state["result"] = res
    m = res["metrics"]
    print(f"[UI] POST {api}/predict trip={trip_id} -> HTTP {resp.status_code} in {rtt:.0f} ms | "
          f"points={res['n_points']} | top-1 clf={pct(m['classifier']['top1_accuracy'])} "
          f"hmm={pct(m['hmm']['top1_accuracy'])} | continuity clf={pct(m['classifier']['route_continuity'])} "
          f"hmm={pct(m['hmm']['route_continuity'])}", flush=True)

res = st.session_state.get("result")
if not res:
    st.stop()

# ------------------------------------------------------------------ results
st.divider()
st.subheader(f"Trip {res['trip_id']} — {res['model']['name']} + HMM/Viterbi")
mc, mh = res["metrics"]["classifier"], res["metrics"]["hmm"]

k = st.columns(5)
k[0].metric("GPS points", f"{res['n_points']:,}")
k[1].metric("Top-1 accuracy (HMM)", pct(mh["top1_accuracy"]),
            None if mh["top1_accuracy"] is None else
            f"{100 * (mh['top1_accuracy'] - mc['top1_accuracy']):+.1f} pts vs classifier")
k[2].metric("Route continuity (HMM)", pct(mh["route_continuity"]),
            f"{100 * (mh['route_continuity'] - mc['route_continuity']):+.1f} pts vs classifier")
k[3].metric("Server time", f"{res['timings_ms']['total'] / 1000:.2f} s",
            "cached" if res.get("cached") else None, delta_color="off")
k[4].metric("Round trip", f"{res['_round_trip_ms'] / 1000:.2f} s")

if not res["has_reference"]:
    st.info("No reference route for this trip on Laptop 1, so accuracy metrics are N/A; "
            "the reconstructed route and continuity are still shown.")

html_doc = map_html(res)
components.html(html_doc, height=640)

left, right = st.columns(2)
with left:
    st.markdown("**Classifier-only vs classifier + HMM/Viterbi**")
    rows = [("Top-1 accuracy (per GPS point)", "top1_accuracy", True),
            ("Route continuity", "route_continuity", True),
            ("Route recall (ref arcs found)", "route_recall", True),
            ("Route precision", "route_precision", True),
            ("Route F1", "route_f1", True),
            ("Points decoded", "decoded_fraction", True),
            ("Arcs in route", "n_route_arcs", False)]
    st.table({"Metric": [r[0] for r in rows],
              "Classifier only": [pct(mc[r[1]]) if r[2] else str(mc[r[1]]) for r in rows],
              "+ HMM/Viterbi": [pct(mh[r[1]]) if r[2] else str(mh[r[1]]) for r in rows]})
    st.caption(f"HMM changed the classifier's arc at {res['n_points_changed_by_hmm']:,} points.")
with right:
    st.markdown("**Server-side timings (ms)**")
    st.table({"Stage": list(res["timings_ms"]), "ms": [f"{v:,.1f}" for v in res["timings_ms"].values()]})
    st.caption(f"Inference: {res['inference_ms_per_point']} ms per GPS point · "
               f"{res['candidates']['n_rows']:,} candidates "
               f"({res['candidates']['mean_per_point']} per point) · GPS source: {res['gps_source']}")

slim = {k: v for k, v in res.items() if k not in ("geometry", "predictions")}
with st.expander("Raw API response (without geometry)"):
    st.json(slim)

d1, d2 = st.columns(2)
d1.download_button("Download map (HTML)", html_doc, f"pathsnap_map_{res['trip_id']}.html", "text/html")
d2.download_button("Download result (JSON)", json.dumps(res), f"pathsnap_result_{res['trip_id']}.json",
                   "application/json")
