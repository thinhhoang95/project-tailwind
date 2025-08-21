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

# Import relative to the project structure
import sys
project_root = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.features.flight_features import FlightFeatures
from .network_evaluator_for_api import NetworkEvaluator


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
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize the NetworkEvaluator with required data."""
        try:
            # Load traffic volumes GeoDataFrame
            # Update this path based on your actual data location
            traffic_volumes_path = "D:/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"
            
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