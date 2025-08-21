"""
Airspace API Wrapper for handling network requests and data processing.

This module provides the API layer that interfaces between the FastAPI endpoints
and the data science logic in NetworkEvaluator.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
import geopandas as gpd
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import relative to the project structure
import sys
project_root = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

from project_tailwind.optimize.eval.flight_list import FlightList
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
    
    def __del__(self):
        """Clean up the thread pool executor."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)