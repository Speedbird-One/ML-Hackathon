"""Command-line client for the PathSnap API. Run it on Laptop 2 to show API responses in the terminal.

    python ui/api_client.py --api http://192.168.43.10:8000              # health + trip list
    python ui/api_client.py --api http://192.168.43.10:8000 --trip 00000003
    python ui/api_client.py --api http://192.168.43.10:8000 --all        # every trip, summary table
"""
import argparse
import json
import time

import requests


def pct(v):
    return "  N/A " if v is None else f"{100 * v:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--trip", help="trip id to reconstruct")
    ap.add_argument("--all", action="store_true", help="reconstruct every trip")
    args = ap.parse_args()
    api = args.api.rstrip("/")

    h = requests.get(f"{api}/health", timeout=10).json()
    print(f"GET {api}/health ->")
    print(json.dumps(h, indent=2))
    trips = requests.get(f"{api}/trips", timeout=10).json()
    print(f"\nGET {api}/trips -> {len(trips['trips'])} trips ({trips['partition']})")
    for t in trips["trips"]:
        print(f"  {t['trip_id']}  {t['n_points']:6d} pts  reference={t['has_reference']}  cached={t['cached']}")

    ids = [t["trip_id"] for t in trips["trips"]] if args.all else ([args.trip] if args.trip else [])
    for tid in ids:
        t0 = time.perf_counter()
        r = requests.post(f"{api}/predict", json={"trip_id": tid, "include_geometry": False}, timeout=600)
        rtt = (time.perf_counter() - t0) * 1000
        print(f"\nPOST {api}/predict {{trip_id: {tid}}} -> HTTP {r.status_code} in {rtt:.0f} ms")
        r.raise_for_status()
        d = r.json()
        if not args.all:
            print(json.dumps(d, indent=2))
        mc, mh = d["metrics"]["classifier"], d["metrics"]["hmm"]
        print(f"  {'':18s}{'classifier':>12s}{'+ HMM':>10s}")
        for label, k in [("top-1 accuracy", "top1_accuracy"), ("route continuity", "route_continuity"),
                         ("route recall", "route_recall"), ("route precision", "route_precision")]:
            print(f"  {label:18s}{pct(mc[k]):>12s}{pct(mh[k]):>10s}")


if __name__ == "__main__":
    main()
