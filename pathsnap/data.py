"""Raw-data loading. One Trip = one folder of .track/.nodes/.arcs/.route files."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import METRIC_CRS, PARTITIONS

try:
    from pyproj import Transformer
    _to_metric = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    def _project_xy(lon, lat):
        return _to_metric.transform(lon, lat)
except Exception:
    # Pure-numpy fallback for UTM 32N (EPSG:32632) when pyproj native DLL is unavailable
    def _project_xy(lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        a, f = 6378137.0, 1 / 298.257223563
        b = a * (1 - f)
        e2 = (a**2 - b**2) / (a**2)
        e02 = (a**2 - b**2) / (b**2)
        k0 = 0.9996
        lon0 = 9.0 * np.pi / 180.0
        phi = np.radians(lat)
        lam = np.radians(lon)
        N = a / np.sqrt(1 - e2 * np.sin(phi)**2)
        T = np.tan(phi)**2
        C = e02 * np.cos(phi)**2
        A = (lam - lon0) * np.cos(phi)
        M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * phi
                 - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2*phi)
                 + (15*e2**2/256 + 45*e2**3/1024) * np.sin(4*phi)
                 - (35*e2**3/3072) * np.sin(6*phi))
        x = k0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T**2 + 72*C - 58*e02) * A**5 / 120) + 500000.0
        y = k0 * (M + N * np.tan(phi) * (A**2 / 2 + (5 - T + 9*C + 4*C**2) * A**4 / 24 + (61 - 58*T + T**2 + 600*C - 330*e02) * A**6 / 720))
        return x, y


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (n, 3) lon, lat, t
    nodes: np.ndarray            # (n_nodes, 2) lon, lat; row index = node id
    arcs: np.ndarray             # (n_arcs, 2) from_node, to_node; row index = arc id
    route: Optional[np.ndarray]  # (m,) ground-truth arc ids, or None

    def track_xy(self) -> np.ndarray:
        """GPS points in metres (projected CRS)."""
        x, y = _project_xy(self.track[:, 0], self.track[:, 1])
        return np.column_stack([x, y])

    def nodes_xy(self) -> np.ndarray:
        x, y = _project_xy(self.nodes[:, 0], self.nodes[:, 1])
        return np.column_stack([x, y])


def _read(path: Path, dtype=float, ncols=None) -> np.ndarray:
    a = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype).to_numpy()
    return a


def list_trips(partition: str) -> list[str]:
    root = PARTITIONS[partition]
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def load_trip(partition: str, trip_id: str, with_route: bool = True) -> Trip:
    d = PARTITIONS[partition] / trip_id
    route_path = d / f"{trip_id}.route"
    route = None
    if with_route and route_path.exists():
        route = _read(route_path, dtype=np.int64).ravel()
    return Trip(
        trip_id=trip_id,
        track=_read(d / f"{trip_id}.track", float),
        nodes=_read(d / f"{trip_id}.nodes", float),
        arcs=_read(d / f"{trip_id}.arcs", np.int64),
        route=route,
    )
