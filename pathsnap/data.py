"""Raw-data loading. One Trip = one folder of .track/.nodes/.arcs/.route files."""
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import PARTITIONS


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (n, 3) lon, lat, t
    nodes: np.ndarray            # (n_nodes, 2) lon, lat; row index = node id
    arcs: np.ndarray             # (n_arcs, 2) from_node, to_node; row index = arc id
    route: Optional[np.ndarray]  # (m,) ground-truth arc ids, or None

    @cached_property
    def _to_metric(self) -> Transformer:
        """Local azimuthal-equidistant projection centred on this trip's track.

        The dataset spans many cities worldwide (each trip is its own local map), so no single
        fixed UTM zone works. Distances from the centre are exact; distortion is negligible
        at trip scale (<= ~80 km extent).
        """
        lon0, lat0 = self.track[:, :2].mean(axis=0)
        crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
        return Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def track_xy(self) -> np.ndarray:
        """GPS points in metres (trip-local projection; origin = track centroid)."""
        x, y = self._to_metric.transform(self.track[:, 0], self.track[:, 1])
        return np.column_stack([x, y])

    def nodes_xy(self) -> np.ndarray:
        """Road-network nodes in the same metric frame as track_xy()."""
        x, y = self._to_metric.transform(self.nodes[:, 0], self.nodes[:, 1])
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
