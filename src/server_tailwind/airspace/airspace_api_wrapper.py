"""
Airspace API Wrapper for handling network requests and data processing.

This module provides the API layer that interfaces between the FastAPI endpoints
and the data science logic in NetworkEvaluator.
"""

import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import geopandas as gpd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np

# Import relative to the project structure
import sys
project_root = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.features.flight_features import FlightFeatures
from .network_evaluator_for_api import NetworkEvaluator
from project_tailwind.impact_eval.distance_computation import haversine_vectorized


class AirspaceAPIWrapper:
    """
    API wrapper that handles network requests and manages the data science components.
    
    This class acts as the bridge between HTTP requests and the underlying
    NetworkEvaluator data science logic.
    """
    
    def __init__(self):
        """Initialize the API wrapper with data loading."""
        self._evaluator: Optional[NetworkEvaluator] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._flight_features: Optional[FlightFeatures] = None
        self._features_lock = threading.Lock()
        # Caches
        self._slack_vector: Optional[np.ndarray] = None
        self._total_occupancy_vector: Optional[np.ndarray] = None
        self._slack_lock = threading.Lock()
        self._tv_travel_minutes: Optional[Dict[str, Dict[str, float]]] = None
        self._travel_lock = threading.Lock()
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize the NetworkEvaluator with required data."""
        try:
            # Load traffic volumes GeoDataFrame
            # Update this path based on your actual data location
            # traffic_volumes_path = "D:/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"
            traffic_volumes_path = "/Volumes/CrucialX/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"
            
            if not Path(traffic_volumes_path).exists():
                # Fallback to a relative path if absolute doesn't exist
                traffic_volumes_path = "data/traffic_volumes_with_capacity.geojson"
            
            self._traffic_volumes_gdf = gpd.read_file(traffic_volumes_path)
            
            # Load flight list
            self._flight_list = FlightList(
                occupancy_file_path="output/so6_occupancy_matrix_with_times.json",
                tvtw_indexer_path="output/tvtw_indexer.json",
            )
            
            # Initialize evaluator
            self._evaluator = NetworkEvaluator(self._traffic_volumes_gdf, self._flight_list)
            
        except Exception as e:
            print(f"Warning: Failed to initialize data: {e}")
            self._evaluator = None
    
    def _ensure_evaluator_ready(self):
        """Ensure the evaluator is initialized, raise error if not."""
        if self._evaluator is None:
            raise RuntimeError("NetworkEvaluator is not initialized. Check data file paths.")

    # ----- Internal helpers for slack distribution -----
    def _get_or_build_slack_vector(self) -> np.ndarray:
        """
        Build and cache the slack vector (capacity_per_bin - occupancy_per_bin) for every TVTW.
        Also caches the total occupancy vector for reuse.
        """
        if self._slack_vector is not None and self._total_occupancy_vector is not None:
            return self._slack_vector

        with self._slack_lock:
            if self._slack_vector is not None and self._total_occupancy_vector is not None:
                return self._slack_vector

            total_capacity_per_bin = self._evaluator._get_total_capacity_vector()
            total_occupancy = self._flight_list.get_total_occupancy_by_tvtw()
            slack = total_capacity_per_bin.astype(np.float32, copy=False) - total_occupancy.astype(np.float32, copy=False)

            self._total_occupancy_vector = total_occupancy.astype(np.float32, copy=False)
            self._slack_vector = slack
            return self._slack_vector

    def _format_time_window(self, bin_offset: int) -> str:
        """Format a time-window label like "HH:MM-HH:MM" for a bin offset within a TV."""
        start_total_min = int(bin_offset * self._evaluator.time_bin_minutes)
        start_hour = start_total_min // 60
        start_min = start_total_min % 60

        end_total_min = start_total_min + self._evaluator.time_bin_minutes
        if end_total_min == 24 * 60:
            end_str = "24:00"
        else:
            end_hour = (end_total_min // 60) % 24
            end_min = end_total_min % 60
            end_str = f"{end_hour:02d}:{end_min:02d}"

        start_str = f"{start_hour:02d}:{start_min:02d}"
        return f"{start_str}-{end_str}"

    def _compute_tv_centroid_latlon_map(self) -> Dict[str, Dict[str, float]]:
        """Return mapping of tv_id -> {"lat": float, "lon": float} using centroid in EPSG:4326."""
        gdf = self._traffic_volumes_gdf
        if gdf.crs is None or str(gdf.crs).lower() not in ("epsg:4326", "epsg: 4326", "wgs84", "wgs 84"):
            try:
                gdf = gdf.to_crs(epsg=4326)
            except Exception:
                # Best effort: proceed without reprojection
                pass

        tv_ids = set(self._evaluator.tv_id_to_idx.keys())
        result: Dict[str, Dict[str, float]] = {}
        for _, row in gdf.iterrows():
            tv_id = row.get("traffic_volume_id")
            if tv_id not in tv_ids:
                continue
            geom = row.get("geometry")
            if geom is None:
                continue
            try:
                c = geom.centroid
                lat = float(c.y)
                lon = float(c.x)
                result[str(tv_id)] = {"lat": lat, "lon": lon}
            except Exception:
                continue
        return result

    def _ensure_travel_minutes(self, speed_kts: float = 475.0) -> Dict[str, Dict[str, float]]:
        """
        Ensure pairwise travel minutes between TVs are available (persisted and cached).
        minutes = (distance_nm / speed_kts) * 60
        """
        # If cached and matches speed, return
        if self._tv_travel_minutes is not None:
            return self._tv_travel_minutes

        with self._travel_lock:
            if self._tv_travel_minutes is not None:
                return self._tv_travel_minutes

            filename = f"output/tv_travel_minutes_{int(speed_kts)}.json"
            fpath = Path(filename)
            if fpath.exists():
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    meta = data.get("metadata", {})
                    if float(meta.get("speed_kts", -1)) == float(speed_kts):
                        self._tv_travel_minutes = data.get("travel_minutes", {})
                        return self._tv_travel_minutes
                except Exception:
                    pass

            # Compute from centroids
            tv_id_list = list(self._evaluator.tv_id_to_idx.keys())
            centroid_map = self._compute_tv_centroid_latlon_map()
            lat_list: List[float] = []
            lon_list: List[float] = []
            valid_ids: List[str] = []
            for tv_id in tv_id_list:
                pt = centroid_map.get(tv_id)
                if pt is None:
                    continue
                valid_ids.append(tv_id)
                lat_list.append(pt["lat"])
                lon_list.append(pt["lon"])

            if not valid_ids:
                self._tv_travel_minutes = {}
                return self._tv_travel_minutes

            lat_arr = np.asarray(lat_list, dtype=np.float64)
            lon_arr = np.asarray(lon_list, dtype=np.float64)

            lat1 = lat_arr[:, None]
            lon1 = lon_arr[:, None]
            lat2 = lat_arr[None, :]
            lon2 = lon_arr[None, :]

            dist_nm_matrix = haversine_vectorized(lat1, lon1, lat2, lon2)
            minutes_matrix = (dist_nm_matrix / float(speed_kts)) * 60.0

            # Build nested dict
            nested: Dict[str, Dict[str, float]] = {}
            for i, src in enumerate(valid_ids):
                inner: Dict[str, float] = {}
                row_minutes = minutes_matrix[i]
                for j, dst in enumerate(valid_ids):
                    inner[dst] = float(row_minutes[j])
                nested[src] = inner

            # Persist
            try:
                fpath.parent.mkdir(parents=True, exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump({
                        "metadata": {"speed_kts": float(speed_kts)},
                        "travel_minutes": nested,
                    }, f, indent=2)
            except Exception:
                pass

            self._tv_travel_minutes = nested
            return self._tv_travel_minutes

    async def get_slack_distribution(self, traffic_volume_id: str, ref_time_str: str, sign: str, delta_min: float = 0.0) -> Dict[str, Any]:
        """
        For the given source traffic volume and reference time, compute slack at the
        corresponding query bin for every TV by shifting by nominal travel time, then
        applying an additional shift of delta_min minutes.
        """
        self._ensure_evaluator_ready()

        if sign not in ("plus", "minus"):
            raise ValueError("sign must be one of 'plus' or 'minus'")

        tv_map = self._evaluator.tv_id_to_idx
        if traffic_volume_id not in tv_map:
            raise ValueError(f"Traffic volume ID '{traffic_volume_id}' not found")

        # Parse ref time to seconds since midnight
        def _parse_ref_time_to_seconds(ts: str) -> int:
            s = str(ts).strip()
            if ":" in s:
                parts = s.split(":")
                if len(parts) not in (2, 3):
                    raise ValueError("ref_time_str must be HH:MM or HH:MM:SS")
                hour = int(parts[0]); minute = int(parts[1]); second = int(parts[2]) if len(parts) == 3 else 0
            else:
                if not s.isdigit() or len(s) not in (4, 6):
                    raise ValueError("ref_time_str must be numeric 'HHMM' or 'HHMMSS'")
                hour = int(s[0:2]); minute = int(s[2:4]); second = int(s[4:6]) if len(s) == 6 else 0
            if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
                raise ValueError("ref_time_str contains invalid time components")
            return hour * 3600 + minute * 60 + second

        ref_seconds = _parse_ref_time_to_seconds(ref_time_str)

        # Heavy compute wrapped for thread pool
        loop = asyncio.get_event_loop()

        def _compute() -> Dict[str, Any]:
            slack_vec = self._get_or_build_slack_vector()
            # Use cached occupancy if present; else compute
            if self._total_occupancy_vector is not None:
                total_occ = self._total_occupancy_vector
            else:
                total_occ = self._flight_list.get_total_occupancy_by_tvtw().astype(np.float32, copy=False)

            travel_minutes_map = self._ensure_travel_minutes(speed_kts=475.0)

            num_tvtws = self._flight_list.num_tvtws
            num_tvs = len(tv_map)
            if num_tvs == 0 or num_tvtws == 0:
                return {
                    "traffic_volume_id": traffic_volume_id,
                    "ref_time_str": ref_time_str,
                    "sign": sign,
                    "delta_min": float(delta_min),
                    "time_bin_minutes": self._evaluator.time_bin_minutes,
                    "nominal_speed_kts": 475.0,
                    "count": 0,
                    "results": [],
                }
            bins_per_tv = num_tvtws // num_tvs
            bins_per_hour = 60 // self._evaluator.time_bin_minutes
            bin_seconds = self._evaluator.time_bin_minutes * 60
            ref_bin = int(ref_seconds // bin_seconds)
            # Clamp ref_bin to range within a TV
            ref_bin = max(0, min(ref_bin, bins_per_tv - 1))

            src = traffic_volume_id
            results: List[Dict[str, Any]] = []
            for dst, dst_row in tv_map.items():
                # Travel minutes from src to dst
                m = float(travel_minutes_map.get(src, {}).get(dst, 0.0))
                sign_mult = 1 if sign == "plus" else -1
                # Combine travel-time minutes and delta_min first, then discretize once
                total_offset_minutes = sign_mult * m + float(delta_min)
                total_offset_bins = int(round(total_offset_minutes / float(self._evaluator.time_bin_minutes)))
                query_bin = ref_bin + total_offset_bins
                bin_offset = total_offset_bins

                clamped = False
                if query_bin < 0:
                    clamped = True
                    query_bin = 0
                elif query_bin >= bins_per_tv:
                    clamped = True
                    query_bin = bins_per_tv - 1

                tvtw_idx = dst_row * bins_per_tv + int(query_bin)
                if tvtw_idx < 0 or tvtw_idx >= num_tvtws:
                    continue

                slack = float(slack_vec[tvtw_idx])
                occupancy = float(total_occ[tvtw_idx])
                hour = int(query_bin // bins_per_hour)
                hourly_cap = self._evaluator.hourly_capacity_by_tv.get(dst, {}).get(hour, -1)
                cap_bin = (float(hourly_cap) / float(bins_per_hour)) if hourly_cap is not None and hourly_cap > -1 else 0.0
                distance_nm = (m * 475.0) / 60.0

                results.append({
                    "traffic_volume_id": dst,
                    "time_window": self._format_time_window(int(query_bin)),
                    "slack": float(slack),
                    "occupancy": float(occupancy),
                    "capacity_per_bin": float(cap_bin),
                    "distance_nm": float(distance_nm),
                    "travel_minutes": float(m),
                    "bin_offset": int(bin_offset),
                    "clamped": bool(clamped),
                })

            # Sort by slack descending
            results.sort(key=lambda x: x["slack"], reverse=True)

            return {
                "traffic_volume_id": traffic_volume_id,
                "ref_time_str": ref_time_str,
                "sign": sign,
                "delta_min": float(delta_min),
                "time_bin_minutes": self._evaluator.time_bin_minutes,
                "nominal_speed_kts": 475.0,
                "count": len(results),
                "results": results,
            }

        return await loop.run_in_executor(self._executor, _compute)
    
    async def get_traffic_volume_occupancy(self, traffic_volume_id: str) -> Dict[str, Any]:
        """
        Get occupancy counts for all time windows of a specific traffic volume.
        
        Args:
            traffic_volume_id: The traffic volume ID to analyze
            
        Returns:
            JSON-serializable dictionary with time windows and occupancy counts
        """
        self._ensure_evaluator_ready()
        
        # Run the data science computation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self._executor,
                self._evaluator.get_traffic_volume_occupancy_counts,
                traffic_volume_id
            )
            
            # Ensure the result is JSON-serializable
            json_result = {
                "traffic_volume_id": traffic_volume_id,
                "occupancy_counts": result,
                "metadata": {
                    "time_bin_minutes": self._evaluator.time_bin_minutes,
                    "total_time_windows": len(result),
                    "total_flights_in_tv": sum(result.values())
                }
            }
            
            return json_result
            
        except ValueError as e:
            # Re-raise ValueError for 404 handling
            raise e
        except Exception as e:
            # Log the error and re-raise as a generic exception
            print(f"Error computing occupancy for {traffic_volume_id}: {e}")
            raise e

    async def get_traffic_volume_occupancy_with_capacity(self, traffic_volume_id: str) -> Dict[str, Any]:
        """
        Get occupancy counts for all time windows of a specific traffic volume,
        along with the hourly capacity from the GeoJSON.
        
        Returns a JSON-serializable dictionary including:
        - occupancy_counts: {"HH:MM-HH:MM": count}
        - hourly_capacity: {"HH:00-HH+1:00": capacity}
        """
        self._ensure_evaluator_ready()

        loop = asyncio.get_event_loop()
        try:
            # Compute occupancy counts as in the base method
            occupancy_counts = await loop.run_in_executor(
                self._executor,
                self._evaluator.get_traffic_volume_occupancy_counts,
                traffic_volume_id,
            )

            # Retrieve hourly capacity map for this traffic volume
            hourly_caps_raw = self._evaluator.hourly_capacity_by_tv.get(traffic_volume_id)
            if hourly_caps_raw is None:
                raise ValueError(f"Traffic volume ID '{traffic_volume_id}' not found or has no capacity data")

            def _format_hour_label(h: int) -> str:
                start = f"{h:02d}:00"
                end = f"{(h + 1) % 24:02d}:00"
                return f"{start}-{end}"

            hourly_capacity = { _format_hour_label(int(h)): float(c) for h, c in hourly_caps_raw.items() }

            return {
                "traffic_volume_id": traffic_volume_id,
                "occupancy_counts": occupancy_counts,
                "hourly_capacity": hourly_capacity,
                "metadata": {
                    "time_bin_minutes": self._evaluator.time_bin_minutes,
                    "total_time_windows": len(occupancy_counts),
                    "total_flights_in_tv": sum(occupancy_counts.values()),
                },
            }
        except ValueError:
            raise
        except Exception as e:
            print(f"Error computing occupancy with capacity for {traffic_volume_id}: {e}")
            raise e

    async def get_traffic_volume_flight_ids(self, traffic_volume_id: str) -> Dict[str, Any]:
        """
        Get flight identifiers for each time window of a specific traffic volume.

        Returns:
            Dictionary mapping time window labels to lists of flight IDs.
        """
        self._ensure_evaluator_ready()

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                self._evaluator.get_traffic_volume_flight_ids_by_time_window,
                traffic_volume_id,
            )
            return result
        except ValueError as e:
            raise e
        except Exception as e:
            print(f"Error retrieving flight IDs for {traffic_volume_id}: {e}")
            raise e

    async def get_traffic_volume_flights_ordered_by_ref_time(
        self, traffic_volume_id: str, ref_time_str: str
    ) -> Dict[str, Any]:
        """
        Get all flights for a traffic volume ordered by proximity to a reference time.

        Args:
            traffic_volume_id: Traffic volume identifier
            ref_time_str: Reference time string like "08:00:10"
        """
        self._ensure_evaluator_ready()

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                self._evaluator.get_traffic_volume_flights_ordered_by_ref_time,
                traffic_volume_id,
                ref_time_str,
            )
            return result
        except ValueError as e:
            raise e
        except Exception as e:
            print(
                f"Error retrieving ordered flights for {traffic_volume_id} at {ref_time_str}: {e}"
            )
            raise e
    
    async def get_available_traffic_volumes(self) -> Dict[str, Any]:
        """
        Get list of available traffic volume IDs.
        
        Returns:
            Dictionary with available traffic volume IDs and metadata
        """
        self._ensure_evaluator_ready()
        
        available_tvs = list(self._evaluator.tv_id_to_idx.keys())
        
        return {
            "available_traffic_volumes": available_tvs,
            "count": len(available_tvs),
            "metadata": {
                "time_bin_minutes": self._evaluator.time_bin_minutes,
                "total_tvtws": self._evaluator.flight_list.num_tvtws,
                "total_flights": self._evaluator.flight_list.num_flights
            }
        }

    async def get_hotspots(self, threshold: float = 0.0) -> Dict[str, Any]:
        """
        Get list of hotspots (traffic volumes where capacity exceeds demand).
        
        Args:
            threshold: Minimum excess traffic to consider as overloaded
            
        Returns:
            Dictionary with hotspot information including traffic_volume_id, 
            time_bin, z_max, z_sum, and other statistics
        """
        self._ensure_evaluator_ready()
        
        loop = asyncio.get_event_loop()
        try:
            hotspots = await loop.run_in_executor(
                self._executor,
                self._evaluator.get_hotspots,
                threshold
            )
            
            return {
                "hotspots": hotspots,
                "count": len(hotspots),
                "metadata": {
                    "threshold": threshold,
                    "time_bin_minutes": self._evaluator.time_bin_minutes,
                    "analysis_type": "hourly_excess_capacity"
                }
            }
            
        except Exception as e:
            print(f"Error computing hotspots: {e}")
            raise e

    async def get_regulation_ranking_tv_flights_ordered(
        self,
        traffic_volume_id: str,
        ref_time_str: str,
        seed_flight_ids: str,
        top_k: Optional[int] = None,
        *,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Return candidate flights in a traffic volume ordered by proximity to ref time,
        augmented with regulation ranking scores based on FlightFeatures.

        Args:
            traffic_volume_id: Traffic volume identifier
            ref_time_str: Reference time string in HHMMSS (or HHMM) format
            seed_flight_ids: Comma-separated seed flight IDs
            top_k: Optionally limit number of ranked flights returned
            normalize: Whether to normalize feature components to [0, 1]

        Returns:
            Dictionary containing ranked flights with arrival info and score components.
        """
        self._ensure_evaluator_ready()

        # Get ordered flights and arrival details using existing functionality
        ordered = await self.get_traffic_volume_flights_ordered_by_ref_time(
            traffic_volume_id, ref_time_str
        )

        candidate_flight_ids = list(ordered.get("ordered_flights", []))
        details_list = ordered.get("details", [])
        detail_by_fid = {d.get("flight_id"): d for d in details_list}

        # Parse seeds (comma-separated)
        seeds = [s.strip() for s in str(seed_flight_ids).split(",") if s.strip()]

        loop = asyncio.get_event_loop()

        def _compute_rank() -> List[Dict[str, Any]]:
            # Reuse a single cached FlightFeatures instance built on the master flight list
            # Lazily initialize to avoid heavy startup cost
            time_start = time.time()
            if self._flight_features is None:
                with self._features_lock:
                    if self._flight_features is None:
                        self._flight_features = FlightFeatures(
                            self._flight_list,
                            self._evaluator,
                        )
                        time_end = time.time()
                        print(f"FlightFeatures computation took {time_end - time_start} seconds")
            feats = self._flight_features
            time_start = time.time()
            seed_footprint = feats.compute_seed_footprint(seeds)
            time_end = time.time()
            print(f"Seed footprint computation took {time_end - time_start} seconds")
            time_start = time.time()
            return feats.rank_candidates(
                seed_footprint,
                candidate_flight_ids=candidate_flight_ids,
                normalize=normalize,
                top_k=top_k,
            )

        try:
            ranked = await loop.run_in_executor(self._executor, _compute_rank)
        except ValueError:
            raise
        except Exception as e:
            print(
                f"Error computing regulation ranking for {traffic_volume_id} at {ref_time_str}: {e}"
            )
            raise e

        # Merge ranking with arrival details
        ranked_with_details: List[Dict[str, Any]] = []
        for item in ranked:
            fid = item.get("flight_id")
            det = detail_by_fid.get(fid, {})
            ranked_with_details.append(
                {
                    "flight_id": fid,
                    "arrival_time": det.get("arrival_time"),
                    "time_window": det.get("time_window"),
                    "delta_seconds": det.get("delta_seconds"),
                    "score": item.get("score"),
                    "components": item.get("components", {}),
                }
            )

        return {
            "traffic_volume_id": traffic_volume_id,
            "ref_time_str": ref_time_str,
            "seed_flight_ids": seeds,
            "ranked_flights": ranked_with_details,
            "metadata": {
                "num_candidates": len(candidate_flight_ids),
                "num_ranked": len(ranked_with_details),
                "time_bin_minutes": self._evaluator.time_bin_minutes,
            },
        }
    
    def __del__(self):
        """Clean up the thread pool executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)