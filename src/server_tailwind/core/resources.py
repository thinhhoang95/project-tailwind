from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import threading

import geopandas as gpd
import numpy as np

# Ensure imports from project src are available
import sys
project_root = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.impact_eval.distance_computation import haversine_vectorized


@dataclass(frozen=True)
class ResourcePaths:
    occupancy_file_path: Path = Path("output/so6_occupancy_matrix_with_times.json")
    tvtw_indexer_path: Path = Path("output/tvtw_indexer.json")
    traffic_volumes_path: Path = Path("/Volumes/CrucialX/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson")
    fallback_traffic_volumes_path: Path = Path("D:/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson")
    fallback_traffic_volumes_path_2: Path = Path("/mnt/d/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson")


class AppResources:
    """
    Process-wide resource container with lazy, thread-safe loaders for heavy artifacts.
    """

    def __init__(self, paths: Optional[ResourcePaths] = None):
        self.paths = paths or ResourcePaths()
        self._lock = threading.RLock()
        self._flight_list: Optional[FlightList] = None
        self._indexer: Optional[TVTWIndexer] = None
        self._traffic_volumes_gdf: Optional[Any] = None
        self._hourly_capacity_by_tv: Optional[Dict[str, Dict[int, float]]] = None
        self._capacity_per_bin_matrix: Optional[np.ndarray] = None
        self._travel_minutes: Optional[Dict[str, Dict[str, float]]] = None

    def preload_all(self) -> "AppResources":
        _ = self.flight_list
        _ = self.indexer
        _ = self.traffic_volumes_gdf
        _ = self.hourly_capacity_by_tv
        _ = self.capacity_per_bin_matrix
        return self

    @property
    def flight_list(self) -> FlightList:
        with self._lock:
            if self._flight_list is None:
                self._flight_list = FlightList(
                    occupancy_file_path=str(self.paths.occupancy_file_path),
                    tvtw_indexer_path=str(self.paths.tvtw_indexer_path),
                )
            return self._flight_list

    @property
    def indexer(self) -> TVTWIndexer:
        with self._lock:
            if self._indexer is None:
                self._indexer = TVTWIndexer.load(str(self.paths.tvtw_indexer_path))
            return self._indexer

    @property
    def traffic_volumes_gdf(self):
        with self._lock:
            if self._traffic_volumes_gdf is None:
                p = self.paths.traffic_volumes_path
                if not p.exists():
                    p = self.paths.fallback_traffic_volumes_path
                if not p.exists():
                    p = self.paths.fallback_traffic_volumes_path_2
                self._traffic_volumes_gdf = gpd.read_file(str(p))
            return self._traffic_volumes_gdf

    @property
    def hourly_capacity_by_tv(self) -> Dict[str, Dict[int, float]]:
        with self._lock:
            if self._hourly_capacity_by_tv is None:
                gdf = self.traffic_volumes_gdf
                mapping: Dict[str, Dict[int, float]] = {}
                for _, row in gdf.iterrows():
                    tv_id = row.get("traffic_volume_id")
                    if tv_id is None:
                        continue
                    tv_id = str(tv_id)
                    raw = row.get("capacity")
                    if raw is None:
                        mapping[tv_id] = {}
                        continue
                    if isinstance(raw, str):
                        import json as _json
                        try:
                            raw = _json.loads(str(raw).replace("'", '"'))
                        except Exception:
                            raw = None
                    d: Dict[int, float] = {}
                    if isinstance(raw, dict):
                        for k, v in raw.items():
                            try:
                                if isinstance(k, str) and ":" in k:
                                    hour = int(k.split(":")[0])
                                else:
                                    hour = int(k)
                                d[int(hour)] = float(v)
                            except Exception:
                                continue
                    mapping[tv_id] = d
                self._hourly_capacity_by_tv = mapping
            return self._hourly_capacity_by_tv

    @property
    def capacity_per_bin_matrix(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._capacity_per_bin_matrix is None:
                per_hour = self.hourly_capacity_by_tv
                fl = self.flight_list
                num_tvs = len(fl.tv_id_to_idx)
                bins_per_tv = int(fl.num_time_bins_per_tv)
                bins_per_hour = max(1, 60 // int(fl.time_bin_minutes))
                mat = np.full((num_tvs, bins_per_tv), -1.0, dtype=np.float32)
                for tv_id, row_idx in fl.tv_id_to_idx.items():
                    caps = per_hour.get(tv_id, {})
                    if not caps:
                        continue
                    for h, cap in caps.items():
                        try:
                            hh = int(h)
                            start = hh * bins_per_hour
                            end = min(start + bins_per_hour, bins_per_tv)
                            if 0 <= start < bins_per_tv:
                                mat[int(row_idx), start:end] = float(cap)
                        except Exception:
                            continue
                self._capacity_per_bin_matrix = mat
            return self._capacity_per_bin_matrix

    def travel_minutes(self, speed_kts: float = 475.0) -> Dict[str, Dict[str, float]]:
        with self._lock:
            if self._travel_minutes is not None:
                return self._travel_minutes
            gdf = self.traffic_volumes_gdf
            fl = self.flight_list
            try:
                g = gdf.to_crs(epsg=4326) if gdf.crs and "4326" not in str(gdf.crs) else gdf
            except Exception:
                g = gdf
            lat: Dict[str, float] = {}
            lon: Dict[str, float] = {}
            tv_ids = set(fl.tv_id_to_idx.keys())
            for _, row in g.iterrows():
                tv_id = row.get("traffic_volume_id")
                if tv_id is None:
                    continue
                tv_id = str(tv_id)
                if tv_id not in tv_ids:
                    continue
                geom = row.get("geometry")
                if geom is None:
                    continue
                try:
                    c = geom.centroid
                    lat[tv_id] = float(c.y)
                    lon[tv_id] = float(c.x)
                except Exception:
                    continue
            ids = sorted(tv_ids & lat.keys() & lon.keys())
            if not ids:
                self._travel_minutes = {}
                return self._travel_minutes
            lat_arr = np.asarray([lat[i] for i in ids], dtype=np.float64)
            lon_arr = np.asarray([lon[i] for i in ids], dtype=np.float64)
            dist_nm = haversine_vectorized(lat_arr[:, None], lon_arr[:, None], lat_arr[None, :], lon_arr[None, :])
            minutes = (dist_nm / float(speed_kts)) * 60.0
            nested: Dict[str, Dict[str, float]] = {}
            for i, src in enumerate(ids):
                nested[src] = {ids[j]: float(minutes[i, j]) for j in range(len(ids))}
            self._travel_minutes = nested
            return self._travel_minutes


_GLOBAL_RESOURCES: Optional[AppResources] = None
_GLOBAL_LOCK = threading.Lock()


def get_resources() -> AppResources:
    global _GLOBAL_RESOURCES
    with _GLOBAL_LOCK:
        if _GLOBAL_RESOURCES is None:
            _GLOBAL_RESOURCES = AppResources()
        return _GLOBAL_RESOURCES


