"""Raw-data loading. One Trip = one folder of .track/.nodes/.arcs/.route files."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import METRIC_CRS, PARTITIONS

_to_metric = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (n, 3) lon, lat, t
    nodes: np.ndarray            # (n_nodes, 2) lon, lat; row index = node id
    arcs: np.ndarray             # (n_arcs, 2) from_node, to_node; row index = arc id
    route: Optional[np.ndarray]  # (m,) ground-truth arc ids, or None

    def track_xy(self) -> np.ndarray:
        """GPS points in metres (projected CRS)."""
        x, y = _to_metric.transform(self.track[:, 0], self.track[:, 1])
        return np.column_stack([x, y])

    def nodes_xy(self) -> np.ndarray:
        x, y = _to_metric.transform(self.nodes[:, 0], self.nodes[:, 1])
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
