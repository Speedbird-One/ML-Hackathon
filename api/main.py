"""PathSnap FastAPI backend (Laptop 1: model + API).

Run from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables:
    PATHSNAP_PARTITION   which trips to serve (default "evaluator"; falls back to "student_test"
                         if data/evaluator has no trips)
    PATHSNAP_DATA        folder that contains train/, student_test/, evaluator/ (default ./data)
    PATHSNAP_PRECOMPUTE  "1" = decode every trip in the background at startup so the demo is instant

Endpoints:
    GET  /health              model + partition info
    GET  /trips               the trips the UI can choose from
    POST /predict             {"trip_id": "...", "track": [[lon,lat,t],...] (optional)}
    GET  /predict/{trip_id}   same, convenient for a browser / curl
    GET  /map/{trip_id}       interactive Folium map (HTML) for the trip
"""
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pathsnap.config import PARTITIONS
from pathsnap.data import list_trips
from pathsnap.inference import PathSnapPredictor


def _pick_partition() -> str:
    wanted = os.environ.get("PATHSNAP_PARTITION", "evaluator")
    if list_trips(wanted):
        return wanted
    for fallback in ("student_test", "train"):
        if list_trips(fallback):
            print(f"[PathSnap] WARNING: no trips in '{PARTITIONS[wanted]}', serving '{fallback}' instead.")
            return fallback
    raise RuntimeError(f"No trip data found. Put the trip folders under {PARTITIONS[wanted].parent} "
                       f"or set PATHSNAP_DATA.")


predictor: Optional[PathSnapPredictor] = None


def _precompute():
    for tid in predictor.trip_ids():
        try:
            r = predictor.predict(tid)
            print(f"[PathSnap] precomputed {tid} in {r['timings_ms']['total']:.0f} ms")
        except Exception as e:  # keep going; the error will show again on request
            print(f"[PathSnap] precompute failed for {tid}: {e}")
    print("[PathSnap] precompute finished: all trips cached.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = PathSnapPredictor(_pick_partition())
    info = predictor.model_info()
    print("=" * 64)
    print(f"[PathSnap] model     : {info['model_name']} ({info['model_file']})")
    print(f"[PathSnap] partition : {predictor.partition} ({len(predictor.trip_ids())} trips)")
    print(f"[PathSnap] HMM       : top_k={info['hmm_top_k']} params={info['hmm_params']}")
    print("=" * 64)
    if os.environ.get("PATHSNAP_PRECOMPUTE") == "1":
        threading.Thread(target=_precompute, daemon=True).start()
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str = Field(..., examples=["00000000"])
    track: Optional[List[List[float]]] = Field(
        None, description="Optional GPS track [[lon, lat, t], ...] sent by the UI laptop. "
                          "If omitted the server's copy of the trip's .track file is used.")
    include_geometry: bool = True


def _log_request(client: str, r: dict, wall_ms: float) -> None:
    mc, mh = r["metrics"]["classifier"], r["metrics"]["hmm"]

    def f(v):
        return "  N/A " if v is None else f"{100 * v:5.1f}%"

    print("-" * 64)
    print(f"[PathSnap] request from {client}  trip={r['trip_id']}  gps_source={r['gps_source']}"
          f"  cached={r.get('cached')}")
    print(f"           points={r['n_points']}  candidates={r['candidates']['n_rows']}"
          f"  server_time={r['timings_ms']['total']:.0f} ms  wall={wall_ms:.0f} ms")
    print(f"           timings: {r['timings_ms']}")
    print(f"           {'':18s}{'classifier':>12s}{'+ HMM':>10s}")
    for label, k in [("top-1 accuracy", "top1_accuracy"), ("route continuity", "route_continuity"),
                     ("route recall", "route_recall"), ("route precision", "route_precision")]:
        print(f"           {label:18s}{f(mc[k]):>12s}{f(mh[k]):>10s}")
    print(f"           points changed by HMM: {r['n_points_changed_by_hmm']}")
    print("-" * 64, flush=True)


def _check_trip(trip_id: str) -> None:
    if trip_id not in predictor.trip_ids():
        raise HTTPException(404, f"Unknown trip_id '{trip_id}'. GET /trips for the valid ids.")


@app.get("/")
def root():
    return {"service": "PathSnap API", "docs": "/docs",
            "endpoints": ["/health", "/trips", "/predict", "/predict/{trip_id}", "/map/{trip_id}"]}


@app.get("/health")
def health():
    return {"status": "ok", "partition": predictor.partition,
            "n_trips": len(predictor.trip_ids()), "model": predictor.model_info()}


@app.get("/trips")
def trips():
    out = []
    for tid in predictor.trip_ids():
        d = PARTITIONS[predictor.partition] / tid
        track = d / f"{tid}.track"
        with open(track) as fh:
            n_pts = sum(1 for line in fh if line.strip())
        out.append({"trip_id": tid, "n_points": n_pts,
                    "has_reference": (d / f"{tid}.route").exists(),
                    "cached": predictor.is_cached(tid)})
    return {"partition": predictor.partition, "trips": out}


def _predict(req: PredictRequest, request: Request) -> dict:
    _check_trip(req.trip_id)
    t0 = time.perf_counter()
    try:
        track = np.asarray(req.track, dtype=float) if req.track else None
        r = predictor.predict(req.trip_id, track=track)
    except ValueError as e:
        raise HTTPException(422, str(e))
    wall = (time.perf_counter() - t0) * 1000
    _log_request(request.client.host if request.client else "?", r, wall)
    if not req.include_geometry:
        r = {k: v for k, v in r.items() if k not in ("geometry", "predictions")}
    return r


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    return _predict(req, request)


@app.get("/predict/{trip_id}")
def predict_get(trip_id: str, request: Request, include_geometry: bool = False):
    return _predict(PredictRequest(trip_id=trip_id, include_geometry=include_geometry), request)


@app.get("/map/{trip_id}", response_class=HTMLResponse)
def map_view(trip_id: str):
    from ui.route_map import map_html
    _check_trip(trip_id)
    return HTMLResponse(map_html(predictor.predict(trip_id)))
