"""
Airspace API Wrapper for handling network requests and data processing.

This module provides the API layer that interfaces between the FastAPI endpoints
and the data science logic in NetworkEvaluator.
"""

import json
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import geopandas as gpd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from datetime import timedelta

# Import relative to the project structure
import sys
project_root = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.features.flight_features import FlightFeatures
from .network_evaluator_for_api import NetworkEvaluator
from project_tailwind.impact_eval.distance_computation import haversine_vectorized
from project_tailwind.optimize.eval.plan_evaluator import PlanEvaluator
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.regulation import Regulation
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.subflows.flow_extractor import assign_communities_for_hotspot
from server_tailwind.core.resources import get_resources


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
            res = get_resources()
            self._traffic_volumes_gdf = res.traffic_volumes_gdf
            self._flight_list = res.flight_list
            # Initialize evaluator with shared resources
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
        Ensure pairwise travel minutes between TVs are available (cached globally).
        minutes = (distance_nm / speed_kts) * 60
        """
        if self._tv_travel_minutes is not None:
            return self._tv_travel_minutes
        with self._travel_lock:
            if self._tv_travel_minutes is not None:
                return self._tv_travel_minutes
            self._tv_travel_minutes = get_resources().travel_minutes(speed_kts)
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
        Get sliding rolling-hour hotspots as contiguous overloaded segments per TV.
        
        Args:
            threshold: Minimum excess traffic to consider as overloaded
            
        Returns:
            Dictionary with segment-based hotspot information. Each segment groups
            consecutive overloaded bins where rolling-hour count exceeds per-bin capacity.
        """
        self._ensure_evaluator_ready()
        
        loop = asyncio.get_event_loop()
        try:
            segments = await loop.run_in_executor(
                self._executor,
                self._evaluator.get_hotspot_segments,
                threshold
            )
            # Map segments into the legacy-like "hotspots" list while preserving new semantics
            hotspots = []
            for seg in segments:
                try:
                    time_bin = f"{seg.get('start_label')}-{seg.get('end_label')}"
                    cap_stats = seg.get("capacity_stats", {}) or {}
                    # Maintain legacy keys: map rolling metrics into analogous fields
                    peak_rolling = float(seg.get("peak_rolling_count", 0.0))
                    # Use conservative capacity across the segment
                    cap_min = float(cap_stats.get("min", -1.0)) if cap_stats else -1.0
                    hotspots.append(
                        {
                            "traffic_volume_id": seg.get("traffic_volume_id"),
                            "time_bin": time_bin,
                            "z_max": float(seg.get("max_excess", 0.0)),
                            "z_sum": float(seg.get("sum_excess", 0.0)),
                            # Legacy naming: provide peak rolling occupancy under hourly_occupancy
                            "hourly_occupancy": peak_rolling,
                            # Provide a single capacity value; choose min across segment for safety
                            "hourly_capacity": cap_min,
                            # Always overloaded for returned segments
                            "is_overloaded": True,
                        }
                    )
                except Exception:
                    continue

            return {
                "hotspots": hotspots,
                "count": len(hotspots),
                "metadata": {
                    "threshold": threshold,
                    "time_bin_minutes": self._evaluator.time_bin_minutes,
                    "analysis_type": "rolling_hour_sliding"
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
        duration_min: Optional[int],
        top_k: Optional[int] = None,
        *,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Return candidate flights in a traffic volume ordered by proximity to ref time.
        Scores and component breakdown are omitted, and FlightFeatures are not computed.

        Args:
            traffic_volume_id: Traffic volume identifier
            ref_time_str: Reference time string in HHMMSS (or HHMM) format
            seed_flight_ids: Comma-separated seed flight IDs (accepted but not used)
            top_k: Optionally limit number of returned flights
            normalize: Unused; present for backward compatibility

        Returns:
            Dictionary containing flights with arrival info ordered by proximity to ref time.
        """
        self._ensure_evaluator_ready()

        # Get ordered flights and arrival details using existing functionality
        ordered = await self.get_traffic_volume_flights_ordered_by_ref_time(
            traffic_volume_id, ref_time_str
        )

        candidate_flight_ids = list(ordered.get("ordered_flights", []))
        details_list = ordered.get("details", [])
        detail_by_fid = {d.get("flight_id"): d for d in details_list}

        # Parse seeds (comma-separated) but do not use them
        seeds = [s.strip() for s in str(seed_flight_ids).split(",") if s.strip()]

        # Optional: filter by duration window after ranking
        def _parse_ref_time_to_seconds(ts: str) -> int:
            s = str(ts).strip()
            if s.isdigit() and len(s) in (4, 6):
                hour = int(s[0:2])
                minute = int(s[2:4])
                second = int(s[4:6]) if len(s) == 6 else 0
            else:
                raise ValueError("ref_time_str must be numeric in 'HHMMSS' (or 'HHMM') format")
            if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
                raise ValueError("ref_time_str contains invalid time components")
            return hour * 3600 + minute * 60 + second

        selected_fids: List[str] = candidate_flight_ids
        if duration_min is not None:
            try:
                dur_min_int = int(duration_min)
            except Exception:
                dur_min_int = None
            if dur_min_int is not None and dur_min_int > 0:
                ref_seconds = _parse_ref_time_to_seconds(ref_time_str)
                end_seconds = min(ref_seconds + int(dur_min_int) * 60, 24 * 3600 - 1)
                filtered_fids: List[str] = []
                for fid in candidate_flight_ids:
                    det = detail_by_fid.get(fid)
                    if not det:
                        continue
                    arr_sec = det.get("arrival_seconds")
                    try:
                        arr_sec_int = int(arr_sec)
                    except Exception:
                        continue
                    if ref_seconds <= arr_sec_int <= end_seconds:
                        filtered_fids.append(fid)
                selected_fids = filtered_fids

        # Apply top_k last (after optional duration filter)
        if top_k is not None:
            try:
                tk = int(top_k)
            except Exception:
                tk = None
            if tk is not None and tk > 0:
                selected_fids = selected_fids[:tk]

        # Merge ranking with arrival details
        ranked_with_details: List[Dict[str, Any]] = []
        for fid in selected_fids:
            det = detail_by_fid.get(fid, {})
            ranked_with_details.append(
                {
                    "flight_id": fid,
                    "arrival_time": det.get("arrival_time"),
                    "time_window": det.get("time_window"),
                    "delta_seconds": det.get("delta_seconds"),
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
                "duration_min": int(duration_min) if duration_min is not None else None,
            },
        }

    async def get_flow_extraction(
        self,
        traffic_volume_id: str,
        ref_time_str: str,
        *,
        threshold: float = 0.8,
        resolution: float = 1.0,
        flight_ids: Optional[str] = None,
        seed: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Assign community labels to flights that pass through a given traffic volume,
        using Jaccard similarity over TV footprints and Leiden clustering.

        This composes the existing ordered-flights query with the subflows flow_extractor.

        Args:
            traffic_volume_id: TV identifier
            ref_time_str: Reference time string (HHMMSS or HHMM)
            threshold: Similarity threshold for building the graph (default 0.8)
            resolution: Leiden resolution parameter (default 1.0)
            flight_ids: Optional comma-separated list of flight IDs to cluster directly. If provided,
                community detection runs only on these flights and no retrieval of ordered flights is performed.
            seed: Optional random seed for Leiden
            limit: Optional cap on number of candidate flights (nearest to ref time)

        Returns:
            Dictionary with per-flight community labels and basic metadata.
        """
        self._ensure_evaluator_ready()

        # Normalize ref_time_str to numeric HHMMSS for ordered flights API
        s_ref = str(ref_time_str).strip()
        if s_ref.isdigit() and len(s_ref) in (4, 6):
            _hour = int(s_ref[0:2])
            _minute = int(s_ref[2:4])
            _second = int(s_ref[4:6]) if len(s_ref) == 6 else 0
        else:
            parts_ref = s_ref.split(":")
            if len(parts_ref) not in (2, 3):
                raise ValueError("ref_time_str must be numeric 'HHMM' or 'HHMMSS' or formatted 'HH:MM'/'HH:MM:SS'")
            _hour = int(parts_ref[0]); _minute = int(parts_ref[1]); _second = int(parts_ref[2]) if len(parts_ref) == 3 else 0
        if not (0 <= _hour < 24 and 0 <= _minute < 60 and 0 <= _second < 60):
            raise ValueError("ref_time_str contains invalid time components")
        normalized_ref_str = f"{_hour:02d}{_minute:02d}{_second:02d}"

        # Determine candidate flights: prefer explicitly provided flight_ids if present,
        # otherwise fall back to retrieving flights ordered by proximity to ref time.
        candidate_flight_ids: List[str]
        used_source: str
        if flight_ids is not None and str(flight_ids).strip() != "":
            parsed = [s.strip() for s in str(flight_ids).split(",") if s.strip()]
            candidate_flight_ids = list(parsed)
            used_source = "provided_flight_ids"
        else:
            ordered = await self.get_traffic_volume_flights_ordered_by_ref_time(
                traffic_volume_id, normalized_ref_str
            )
            candidate_flight_ids = list(ordered.get("ordered_flights", []))
            if limit is not None:
                try:
                    lmt = int(limit)
                    if lmt > 0:
                        candidate_flight_ids = candidate_flight_ids[:lmt]
                except Exception:
                    pass
            used_source = "ordered_by_ref_time"

        # Construct hotspot-like item for the flow extractor API
        # The flow extractor accepts either traffic_volume_id or tvtw_index; hour is optional
        try:
            ref_seconds = _hour * 3600 + _minute * 60 + _second
            ref_hour = int(ref_seconds // 3600)
        except Exception:
            ref_hour = 0

        loop = asyncio.get_event_loop()

        def _compute_assignments() -> Dict[str, int]:
            return assign_communities_for_hotspot(
                self._flight_list,
                candidate_flight_ids,
                traffic_volume_id,
                threshold=float(threshold),
                resolution=float(resolution),
                seed=seed,
            )

        try:
            assignments = await loop.run_in_executor(self._executor, _compute_assignments)
        except ValueError:
            raise
        except Exception as e:
            print(
                f"Error computing flow extraction for {traffic_volume_id} at {ref_time_str}: {e}"
            )
            raise e

        # Build groups for convenience: label -> [flight_ids]
        groups: Dict[int, List[str]] = {}
        for fid, label in assignments.items():
            lab = int(label)
            if lab not in groups:
                groups[lab] = []
            groups[lab].append(fid)

        # Sort groups' flight lists by their proximity order when available
        order_index: Dict[str, int] = {fid: i for i, fid in enumerate(candidate_flight_ids)}
        for lab in list(groups.keys()):
            groups[lab].sort(key=lambda x: order_index.get(x, 1_000_000))

        return {
            "traffic_volume_id": traffic_volume_id,
            "ref_time_str": ref_time_str,
            "flight_ids": candidate_flight_ids,
            "communities": assignments,
            "groups": {int(k): v for k, v in groups.items()},
            "metadata": {
                "num_flights": len(candidate_flight_ids),
                "time_bin_minutes": self._evaluator.time_bin_minutes,
                "threshold": float(threshold),
                "resolution": float(resolution),
                "source": used_source,
            },
        }
    
    async def run_regulation_plan_simulation(
        self,
        regulations: List[Any],
        *,
        weights: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
        include_excess_vector: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a regulation plan on the baseline flight list and return post-regulation
        delays, metrics, and rolling-hour occupancy arrays for all TVs that changed.

        - regulations: list of raw strings (Regulation DSL) or dicts with keys
            {location, rate, time_windows[], filter_type?, filter_value?, target_flight_ids?}
        - weights: optional objective weights
        - include_excess_vector: if True, include the full post-regulation excess vector

        Backward compatibility (one release):
        - top_k is accepted but ignored; a deprecation note is included in metadata when present.
        """
        self._ensure_evaluator_ready()

        loop = asyncio.get_event_loop()

        def _compute() -> Dict[str, Any]:
            # Build indexer and parser based on the same files backing the server's FlightList
            tvtw_indexer = TVTWIndexer.load(self._flight_list.tvtw_indexer_path)
            parser = RegulationParser(
                flights_file=self._flight_list.occupancy_file_path,
                tvtw_indexer=tvtw_indexer,
            )

            # Normalize regulations into Regulation objects or raw strings
            normalized_regs: List[Any] = []
            for item in regulations:
                if isinstance(item, str):
                    normalized_regs.append(item)
                elif isinstance(item, dict):
                    loc = str(item.get("location"))
                    rate = int(item.get("rate"))
                    time_windows = [int(w) for w in item.get("time_windows", [])]
                    filter_type = str(item.get("filter_type", "IC"))
                    filter_value = str(item.get("filter_value", "__"))
                    target_flight_ids = item.get("target_flight_ids")
                    reg = Regulation.from_components(
                        location=loc,
                        rate=rate,
                        time_windows=time_windows,
                        filter_type=filter_type,
                        filter_value=filter_value,
                        target_flight_ids=target_flight_ids,
                    )
                    normalized_regs.append(reg)
                else:
                    raise ValueError("Each regulation must be a string or an object with required fields")

            # Debug: print out the regulations
            print("DEBUG: Normalized regulations:")
            for i, reg in enumerate(normalized_regs):
                if isinstance(reg, str):
                    print(f"  {i}: (string) {reg}")
                else:
                    print(f"  {i}: (object) loc={reg.location}, rate={reg.rate}, time_windows={reg.time_windows}, filter_type={reg.filter_type}, filter_value={reg.filter_value}, target_flight_ids={reg.target_flight_ids}")


            network_plan = NetworkPlan(normalized_regs)

            # Evaluate plan -> delays, overlay view, metrics
            evaluator = PlanEvaluator(
                traffic_volumes_gdf=self._traffic_volumes_gdf,
                parser=parser,
                tvtw_indexer=tvtw_indexer,
            )
            time_start = time.time()
            plan_result = evaluator.evaluate_plan(network_plan, self._flight_list, weights=weights)
            time_end = time.time()
            print(f"Plan evaluation took {time_end - time_start} seconds")

            # Rolling-hour occupancy arrays for all bins per TV (length = bins_per_tv)
            # 1) Build flat occupancy vectors pre/post
            time_start = time.time()
            pre_total = self._flight_list.get_total_occupancy_by_tvtw().astype(np.float32, copy=False)
            time_end = time.time()
            print(f"Pre-regulation occupancy computation took {time_end - time_start} seconds")
            time_start = time.time()
            post_total = plan_result["delta_view"].get_total_occupancy_by_tvtw().astype(np.float32, copy=False)
            time_end = time.time()
            print(f"Post-regulation occupancy computation took {time_end - time_start} seconds")

            # 2) Dimensions and helpers
            num_tvtws = int(self._flight_list.num_tvtws)
            num_tvs = int(len(self._flight_list.tv_id_to_idx))
            bins_per_hour = 60 // int(self._flight_list.time_bin_minutes)
            bins_per_tv = num_tvtws // max(num_tvs, 1)
            if bins_per_tv <= 0:
                raise RuntimeError("Invalid bins_per_tv computed from occupancy vector")

            # 3) Stable ordering of TVs by row index
            tv_items = sorted(self._flight_list.tv_id_to_idx.items(), key=lambda kv: kv[1])
            tv_ids_in_row_order = [tv for tv, _ in tv_items]

            # 4) Reshape pre/post into [num_tvs, bins_per_tv]
            pre_by_tv = np.zeros((num_tvs, bins_per_tv), dtype=np.float32)
            post_by_tv = np.zeros((num_tvs, bins_per_tv), dtype=np.float32)
            for tv_id, row in tv_items:
                start = row * bins_per_tv
                end = start + bins_per_tv
                pre_by_tv[row, :] = pre_total[start:end]
                post_by_tv[row, :] = post_total[start:end]

            # 5) Rolling-hour (forward) sums per bin: sum over bins [i, i+W-1] (clamped at end)
            time_start = time.time()
            def rolling_forward_sum_full(mat: np.ndarray, window: int) -> np.ndarray:
                # pad zeros at end to keep length the same (forward-looking window)
                pad = [(0, 0), (0, window - 1)]
                padded = np.pad(mat, pad, mode="constant", constant_values=0.0)
                cs = np.cumsum(padded, axis=1, dtype=np.float64)
                # out[i] = cs[i+W-1] - cs[i-1]; implement vectorized with shifted cs
                left = np.concatenate([np.zeros((mat.shape[0], 1), dtype=np.float64), cs[:, :-window]], axis=1)
                right = cs[:, window - 1 : window - 1 + mat.shape[1]]
                return (right - left).astype(np.float32, copy=False)

            pre_roll = rolling_forward_sum_full(pre_by_tv, bins_per_hour)
            post_roll = rolling_forward_sum_full(post_by_tv, bins_per_hour)
            time_end = time.time()
            print(f"Rolling-hour sums computation took {time_end - time_start} seconds")
            
            # 6) Capacity per bin for each TV (repeat hourly capacity across bins in that hour)
            # Reuse the cached evaluator for consistent capacity parsing
            cap_by_tv = {}
            for tv_id, row in tv_items:
                hourly_caps = self._evaluator.hourly_capacity_by_tv.get(tv_id, {})
                if not hourly_caps:
                    # mark as missing capacity with -1.0 per bin
                    cap_by_tv[tv_id] = np.full(bins_per_tv, -1.0, dtype=np.float32)
                    continue
                arr = np.full(bins_per_tv, -1.0, dtype=np.float32)
                for h, c in hourly_caps.items():
                    if 0 <= int(h) < (bins_per_tv // bins_per_hour):
                        start = int(h) * bins_per_hour
                        end = start + bins_per_hour
                        arr[start:end] = float(c)
                cap_by_tv[tv_id] = arr

            # 7) Build per-TV active window mask from the plan and also the UNION mask across all TVs
            tv_to_active_mask = {tv: np.zeros(bins_per_tv, dtype=bool) for tv, _ in tv_items}
            union_active_mask = np.zeros(bins_per_tv, dtype=bool)
            for reg in network_plan.regulations:
                loc = getattr(reg, "location", None)
                wins = getattr(reg, "time_windows", []) or []
                if not loc or loc not in tv_to_active_mask:
                    continue
                mask = tv_to_active_mask[loc]
                for w in wins:
                    wi = int(w)
                    if 0 <= wi < bins_per_tv:
                        mask[wi] = True
                        union_active_mask[wi] = True

            # 8) Determine changed TVs by comparing raw pre/post occupancy per TV (integer semantics)
            # If arrays are float, use tolerance to be safe.
            if np.issubdtype(pre_by_tv.dtype, np.floating) or np.issubdtype(post_by_tv.dtype, np.floating):
                diff_mask = np.abs(pre_by_tv - post_by_tv) > 0.5
            else:
                diff_mask = (pre_by_tv != post_by_tv)
            changed_rows = np.where(np.any(diff_mask, axis=1))[0].tolist()

            # 9) Package rolling arrays for changed TVs in stable row order
            rolling_changed_tvs: List[Dict[str, Any]] = []
            for tv_id, row in tv_items:
                if row not in changed_rows:
                    continue
                active_wins = np.where(tv_to_active_mask[tv_id])[0].tolist()
                rolling_changed_tvs.append(
                    {
                        "traffic_volume_id": tv_id,
                        "pre_rolling_counts": pre_roll[row, :].astype(float).tolist(),
                        "post_rolling_counts": post_roll[row, :].astype(float).tolist(),
                        "capacity_per_bin": cap_by_tv[tv_id].astype(float).tolist(),
                        "active_time_windows": [int(w) for w in active_wins],
                    }
                )

            # Optionally include the full post-regulation excess vector
            post_excess = plan_result.get("excess_vector")
            excess_payload: Dict[str, Any]
            if include_excess_vector and post_excess is not None:
                try:
                    excess_payload = {"excess_vector": [float(x) for x in post_excess.tolist()]}  # type: ignore[attr-defined]
                except Exception:
                    # Fallback if not numpy
                    excess_payload = {"excess_vector": list(map(float, post_excess))}
            else:
                # Compact stats
                import numpy as _np
                ev = _np.asarray(post_excess, dtype=float)
                excess_payload = {
                    "excess_vector_stats": {
                        "sum": float(_np.sum(ev)) if ev.size > 0 else 0.0,
                        "max": float(_np.max(ev)) if ev.size > 0 else 0.0,
                        "mean": float(_np.mean(ev)) if ev.size > 0 else 0.0,
                        "count": int(ev.size),
                    }
                }

            # Build pre-flight context: takeoff time and baseline arrival time to any regulated TV
            pre_flight_context: Dict[str, Dict[str, Optional[str]]] = {}
            delays_by_flight = plan_result.get("delays_by_flight", {}) or {}
            if delays_by_flight:
                # Collect regulated TV ranges [start, end) in the flat TVTW index space
                regulated_tv_ids: List[str] = []
                for reg in network_plan.regulations:
                    loc = getattr(reg, "location", None)
                    if loc and loc in self._flight_list.tv_id_to_idx:
                        regulated_tv_ids.append(loc)

                # Deduplicate while preserving order
                seen: set = set()
                reg_tv_ids_dedup: List[str] = []
                for tv in regulated_tv_ids:
                    if tv not in seen:
                        seen.add(tv)
                        reg_tv_ids_dedup.append(tv)
                regulated_tv_ids = reg_tv_ids_dedup

                tv_ranges: List[Tuple[str, int, int]] = []
                for tv_id in regulated_tv_ids:
                    row = self._flight_list.tv_id_to_idx[tv_id]
                    start = row * bins_per_tv
                    end = start + bins_per_tv
                    tv_ranges.append((tv_id, start, end))

                for fid in delays_by_flight.keys():
                    meta = self._flight_list.flight_metadata.get(fid)
                    if not meta:
                        continue
                    takeoff_dt = meta.get("takeoff_time")
                    if not takeoff_dt:
                        continue

                    # Earliest entry to any regulated TV
                    earliest_entry_s: Optional[int] = None
                    for interval in meta.get("occupancy_intervals", []):
                        col = interval.get("tvtw_index")
                        if col is None:
                            continue
                        in_reg_tv = False
                        for _, s, e in tv_ranges:
                            if s <= int(col) < e:
                                in_reg_tv = True
                                break
                        if not in_reg_tv:
                            continue
                        entry_s = int(interval.get("entry_time_s", 0))
                        if earliest_entry_s is None or entry_s < earliest_entry_s:
                            earliest_entry_s = entry_s

                    # Format times
                    takeoff_time_str = f"{takeoff_dt.hour:02d}:{takeoff_dt.minute:02d}:{takeoff_dt.second:02d}"
                    if earliest_entry_s is not None:
                        arrival_dt = takeoff_dt + timedelta(seconds=int(earliest_entry_s))
                        tv_arrival_time_str: Optional[str] = f"{arrival_dt.hour:02d}:{arrival_dt.minute:02d}:{arrival_dt.second:02d}"
                    else:
                        tv_arrival_time_str = None

                    pre_flight_context[fid] = {
                        "takeoff_time": takeoff_time_str,
                        "tv_arrival_time": tv_arrival_time_str,
                    }

            result_payload = {
                "delays_by_flight": plan_result.get("delays_by_flight", {}),
                "delay_stats": plan_result.get("delay_stats", {}),
                "objective": plan_result.get("objective", 0.0),
                "objective_components": plan_result.get("objective_components", {}),
                "pre_flight_context": pre_flight_context,
                # New key with changed TVs only
                "rolling_changed_tvs": rolling_changed_tvs,
                # Backward-compatibility alias for one release
                "rolling_top_tvs": rolling_changed_tvs,
                "metadata": {
                    "time_bin_minutes": int(self._flight_list.time_bin_minutes),
                    "bins_per_tv": int(bins_per_tv),
                    "bins_per_hour": int(bins_per_hour),
                    "num_traffic_volumes": int(num_tvs),
                    "num_changed_tvs": int(len(rolling_changed_tvs)),
                },
                **excess_payload,
            }

            # Add deprecation note if top_k was provided in the request
            if top_k is not None:
                try:
                    meta = result_payload.get("metadata", {})
                    meta["deprecated"] = {
                        "top_k": "accepted but ignored; will be removed in next release",
                        "rolling_top_tvs": "alias of rolling_changed_tvs for one release",
                    }
                    result_payload["metadata"] = meta
                except Exception:
                    pass

            return result_payload

        try:
            return await loop.run_in_executor(self._executor, _compute)
        except Exception as e:
            print(f"Error in run_regulation_plan_simulation: {e}")
            raise
    
    def __del__(self):
        """Clean up the thread pool executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
