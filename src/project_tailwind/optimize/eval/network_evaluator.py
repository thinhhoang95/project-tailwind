"""
NetworkEvaluator class for computing excess traffic vectors from flight occupancy data.

Some implementation notes:
_get_total_capacity_vector() also “distributes capacity per bin,” but it is only for reporting; it is not used in excess calculation. Excess uses hourly capacity vs hourly occupancy, then spreads excess across the bins.
"""

import json
import numpy as np
import geopandas as gpd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path

from .flight_list import FlightList


class NetworkEvaluator:
    """
    A class to evaluate network overload by comparing flight occupancy with traffic volume capacities.

    This class handles the conversion between different time window formats and computes
    excess traffic vectors to identify overloaded Traffic Volume Time Windows (TVTWs).
    """

    def __init__(self, traffic_volumes_gdf: gpd.GeoDataFrame, flight_list: FlightList):
        """
        Initialize NetworkEvaluator with traffic volumes and flight data.

        Args:
            traffic_volumes_gdf: GeoDataFrame containing traffic volume data with capacities
            flight_list: FlightList object with loaded occupancy data
        """
        self.traffic_volumes_gdf = traffic_volumes_gdf
        self.flight_list = flight_list
        # If we are given a DeltaFlightList, snapshot the original baseline from its base
        if hasattr(flight_list, "_base"):
            # Use the immutable base as the baseline to avoid a heavy copy
            self.original_flight_list = flight_list._base  # type: ignore[attr-defined]
        else:
            self.original_flight_list = flight_list.copy()

        # Extract time window information
        self.time_bin_minutes = flight_list.time_bin_minutes
        self.tv_id_to_idx = flight_list.tv_id_to_idx

        # For caching hourly occupancy results
        self.last_hourly_occupancy_matrix: Optional[np.ndarray] = None
        self.tv_id_to_row_idx = self.tv_id_to_idx

        # Process capacity data and handle time window conversion
        self.hourly_capacity_by_tv: Dict[str, Dict[int, float]] = {}
        self._process_capacity_data()

    def update_flight_list(self, new_flight_list: FlightList) -> None:
        """
        Update the flight_list reference for evaluation with a new state.
        
        Args:
            new_flight_list: Updated FlightList object representing the new state
        """
        self.flight_list = new_flight_list
        # Clear cached results that depend on flight_list
        self.last_hourly_occupancy_matrix = None

    def _process_capacity_data(self):
        """
        Process and store hourly capacity data for each traffic volume.

        The traffic volumes have hourly capacity data (e.g., "6:00-7:00": 23),
        which represents the throughput for that hour.
        """
        for _, tv_row in self.traffic_volumes_gdf.iterrows():
            tv_id = tv_row["traffic_volume_id"]
            capacity_data = tv_row["capacity"]

            if tv_id not in self.tv_id_to_idx:
                continue  # Skip traffic volumes not in indexer

            # Handle case where capacity might be a string (from JSON parsing)
            if isinstance(capacity_data, str):
                try:
                    capacity_data = json.loads(capacity_data.replace("'", '"'))
                except (json.JSONDecodeError, AttributeError):
                    continue  # Skip if can't parse capacity data
            elif not isinstance(capacity_data, dict):
                continue  # Skip if capacity data is not in expected format

            self.hourly_capacity_by_tv[tv_id] = {}
            # Process each hourly capacity entry
            for time_range, hourly_capacity in capacity_data.items():
                start_hour, _ = self._parse_time_range(time_range)

                if start_hour is None:
                    continue

                self.hourly_capacity_by_tv[tv_id][start_hour] = float(hourly_capacity)

    def _get_total_capacity_vector(self) -> np.ndarray:
        """
        Create a capacity vector for all TVTWs for reporting purposes.
        It distributes the hourly capacity across the time bins within each hour.
        """
        num_tvtws = self.flight_list.num_tvtws
        bins_per_hour = 60 // self.time_bin_minutes
        total_capacity = np.zeros(num_tvtws)

        num_time_bins_per_tv = num_tvtws // len(self.tv_id_to_idx)

        for tv_id, hourly_capacities in self.hourly_capacity_by_tv.items():
            tv_row = self.tv_id_to_idx.get(tv_id)
            if tv_row is None:
                continue

            tv_start = tv_row * num_time_bins_per_tv
            for hour, hourly_capacity in hourly_capacities.items():
                start_bin = hour * bins_per_hour
                end_bin = start_bin + bins_per_hour

                # For reporting, distribute capacity over bins
                # This is NOT for excess calculation
                capacity_per_bin = hourly_capacity / bins_per_hour

                for bin_offset in range(start_bin, end_bin):
                    if bin_offset < num_time_bins_per_tv:
                        tvtw_idx = tv_start + bin_offset
                        if tvtw_idx < num_tvtws:
                            total_capacity[tvtw_idx] = capacity_per_bin
        return total_capacity

    def _parse_time_range(self, time_range: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse time range string like "6:00-7:00" into hour integers.

        Args:
            time_range: Time range string in format "HH:MM-HH:MM"

        Returns:
            Tuple of (start_hour, end_hour) or (None, None) if parsing fails
        """
        try:
            start_time, end_time = time_range.split("-")
            start_hour = int(start_time.split(":")[0])
            end_hour = int(end_time.split(":")[0])
            return start_hour, end_hour
        except (ValueError, IndexError):
            return None, None

    def compute_excess_traffic_vector(self) -> np.ndarray:
        """
        Compute excess traffic vector showing overload for each TVTW.

        This implementation aggregates traffic into hourly bins, compares with
        hourly throughput capacity, and distributes any excess traffic equally
        back to the TVTWs within that hour.

        Output Format:
        Excess Count per TVTW: [(TV1, TW1), (TV1, TW2), ... (TV2, TW1), ...]
        i.e., the first 96 values are the excess counts for TV1, the next 96 are for TV2, etc.
        This format is consistent with tvtw_indexer.py's output format.

        Returns:
            1D numpy array where:
            - 0 indicates no overload
            - positive values indicate excess traffic for overloaded TVTWs
        """
        total_occupancy = self.flight_list.get_total_occupancy_by_tvtw()
        num_tvtws = len(total_occupancy)
        num_tvs = len(self.tv_id_to_idx)
        bins_per_tv = 24 * (60 // self.time_bin_minutes)
        # Defensive check: ensure consistent shape
        if num_tvtws != num_tvs * bins_per_tv:
            raise ValueError(f"num_tvtws ({num_tvtws}) != num_tvs * bins_per_tv ({num_tvs} * {bins_per_tv} expected)")
        excess_vector = np.zeros(num_tvtws)

        bins_per_hour = 60 // self.time_bin_minutes

        # This will be an occupancy matrix indexed by tv_id and hour
        # For simplicity, we create a dense matrix, assuming number of hours is not huge
        max_hour = 24
        hourly_occupancy_matrix = np.zeros((len(self.tv_id_to_idx), max_hour))

        # Aggregate occupancy into hourly buckets for each traffic volume
        num_time_bins_per_tv = num_tvtws // len(self.tv_id_to_idx)
        expected_bins_per_tv = 24 * (60 // self.time_bin_minutes)
        if num_time_bins_per_tv != expected_bins_per_tv:
            raise ValueError(
                f"num_time_bins_per_tv mismatch: expected {expected_bins_per_tv}, got {num_time_bins_per_tv}"
            )

        for tv_id, tv_row in self.tv_id_to_idx.items():
            row_idx = self.tv_id_to_row_idx[tv_id]
            tv_start = tv_row * num_time_bins_per_tv
            for hour in range(max_hour):
                start_bin = hour * bins_per_hour
                end_bin = start_bin + bins_per_hour

                hourly_occupancy_for_tv = 0
                for bin_offset in range(start_bin, end_bin):
                    if bin_offset < num_time_bins_per_tv:
                        tvtw_idx = tv_start + bin_offset
                        if tvtw_idx < num_tvtws:
                            hourly_occupancy_for_tv += total_occupancy[tvtw_idx]

                hourly_occupancy_matrix[row_idx, hour] = hourly_occupancy_for_tv

        self.last_hourly_occupancy_matrix = hourly_occupancy_matrix

        # Compute excess traffic for each traffic volume and hour
        for tv_id, hourly_capacities in self.hourly_capacity_by_tv.items():
            row_idx = self.tv_id_to_row_idx.get(tv_id)
            if row_idx is None:
                continue

            for hour, hourly_capacity in hourly_capacities.items():
                if hour >= max_hour:
                    continue

                hourly_occupancy = hourly_occupancy_matrix[row_idx, hour]
                hourly_excess = max(0, hourly_occupancy - hourly_capacity)

                if hourly_excess > 0:
                    # Distribute excess back to TVTWs
                    start_bin_of_hour = hour * bins_per_hour
                    end_bin_of_hour = start_bin_of_hour + bins_per_hour

                    tvtw_indices_for_hour = []
                    tv_row = self.tv_id_to_idx[tv_id]
                    tv_start = tv_row * num_time_bins_per_tv
                    for bin_offset in range(start_bin_of_hour, end_bin_of_hour):
                        if bin_offset < num_time_bins_per_tv:
                            tvtw_idx = tv_start + bin_offset
                            if tvtw_idx < num_tvtws:
                                tvtw_indices_for_hour.append(tvtw_idx)

                    if tvtw_indices_for_hour:
                        # Instead of equal split, allocate hourly excess proportionally to the per-bin occupancy within the hour.
                        # This preserves z_sum and makes z_95 more indicative of peak bins:
                        hour_bins = np.arange(start_bin_of_hour, end_bin_of_hour)
                        mask = hour_bins < num_time_bins_per_tv
                        tvtw_indices_for_hour = (tv_start + hour_bins[mask]).tolist()
                        bin_demands = total_occupancy[tvtw_indices_for_hour]
                        den = float(np.sum(bin_demands))
                        if den > 0:
                            weights = bin_demands / den
                        else:
                            weights = np.full_like(bin_demands, 1.0 / len(bin_demands), dtype=float)
                        excess_vector[tvtw_indices_for_hour] += hourly_excess * weights

        return excess_vector

    def get_overloaded_tvtws(self, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get list of overloaded TVTWs with details using a vectorized approach.

        Args:
            threshold: Minimum excess traffic to consider as overloaded

        Returns:
            List of dictionaries with overload information
        """
        excess_vector = self.compute_excess_traffic_vector()
        total_occupancy = self.flight_list.get_total_occupancy_by_tvtw()
        num_tvtws = len(total_occupancy)

        # Find overloaded TVTWs
        overloaded_indices = np.where(excess_vector > threshold)[0]

        # If no overloads, return early
        if len(overloaded_indices) == 0:
            return []

        # Build reverse mapping from TVTW index to traffic volume ID
        tvtw_to_tv_id = np.full(num_tvtws, None, dtype=object)

        num_time_bins_per_tv = num_tvtws // len(self.tv_id_to_idx)
        for tv_id, tv_row in self.tv_id_to_idx.items():
            start_idx = tv_row * num_time_bins_per_tv
            end_idx = start_idx + num_time_bins_per_tv
            tvtw_to_tv_id[start_idx:end_idx] = tv_id

        # Collect details for overloaded TVTWs
        overloaded_tvtws = []
        bins_per_hour = 60 // self.time_bin_minutes

        for idx in overloaded_indices:
            occupancy = total_occupancy[idx]

            # Find which hour this TVTW belongs to
            tv_id = tvtw_to_tv_id[idx]
            tv_row = self.tv_id_to_idx[tv_id]
            tv_start = tv_row * num_time_bins_per_tv
            bin_offset = idx - tv_start
            hour = bin_offset // bins_per_hour

            # Get hourly capacity for this tv and hour
            hourly_capacity = self.hourly_capacity_by_tv.get(tv_id, {}).get(hour, -1)
            capacity_per_bin = (
                hourly_capacity / bins_per_hour if hourly_capacity > -1 else -1
            )

            row_idx = self.tv_id_to_row_idx[tv_id]
            hourly_occupancy = -1
            if self.last_hourly_occupancy_matrix is not None:
                hourly_occupancy = self.last_hourly_occupancy_matrix[row_idx, hour]

            overloaded_tvtws.append(
                {
                    "tvtw_index": int(idx),
                    "traffic_volume_id": tv_id,
                    "occupancy": float(occupancy),
                    "capacity_per_bin": float(capacity_per_bin),
                    "hourly_capacity": float(hourly_capacity),
                    "hourly_occupancy": float(hourly_occupancy),
                    "excess": float(excess_vector[idx]),
                    "utilization_ratio": (
                        float(occupancy / capacity_per_bin)
                        if capacity_per_bin > 0
                        else float("inf")
                    ),
                }
            )

        # Sort by excess traffic (highest first)
        overloaded_tvtws.sort(key=lambda x: x["excess"], reverse=True)

        return overloaded_tvtws

    def get_hotspot_flights(
        self,
        threshold: float = 0.0,
        mode: str = "bin",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve flight identifiers for each hotspot.

        Args:
            threshold: Minimum excess traffic to consider as overloaded.
            mode: "bin" to return per-TVTW flights; "hour" to return per (traffic_volume_id, hour).

        Returns:
            If mode == "bin": list of {"tvtw_index": int, "flight_ids": List[str]}.
            If mode == "hour": list of {"traffic_volume_id": str, "hour": int, "flight_ids": List[str], 
                                        "hourly_occupancy": float, "unique_flights": int}.
        """
        # Compute excess once and determine overloaded indices
        excess_vector = self.compute_excess_traffic_vector()
        overloaded_indices = np.where(excess_vector > threshold)[0]
        if overloaded_indices.size == 0:
            return []

        # Get total occupancy for reporting
        total_occupancy = self.flight_list.get_total_occupancy_by_tvtw()

        # Ensure the CSR is up-to-date, then convert to CSC for fast column accesses
        try:
            # Finalize any pending updates (no-op if none)
            self.flight_list.finalize_occupancy_updates()
        except Exception:
            pass

        occ_csc = self.flight_list.occupancy_matrix.tocsc(copy=False)
        num_tvtws = self.flight_list.num_tvtws

        # Precompute TVTW -> (tv_id, hour) mapping helpers
        num_time_bins_per_tv = num_tvtws // len(self.tv_id_to_idx)
        bins_per_hour = 60 // self.time_bin_minutes

        # Build reverse mapping from contiguous TVTW ranges to tv_id for vectorized math
        tvtw_to_tv_id = np.full(num_tvtws, None, dtype=object)
        for tv_id, tv_row in self.tv_id_to_idx.items():
            start_idx = tv_row * num_time_bins_per_tv
            end_idx = start_idx + num_time_bins_per_tv
            tvtw_to_tv_id[start_idx:end_idx] = tv_id

        if mode == "bin":
            # ... (bin mode code remains the same) ...
            # Slice once to a compact submatrix with only hotspot columns
            sub = occ_csc[:, overloaded_indices]
            rows, sub_cols = sub.nonzero()
            # Group rows by sub-column index (within the submatrix)
            flights_by_subcol: Dict[int, List[str]] = {}
            for r, c in zip(rows.tolist(), sub_cols.tolist()):
                flights_by_subcol.setdefault(c, []).append(self.flight_list.flight_ids[r])

            # Map back to original TVTW indices
            results: List[Dict[str, Any]] = []
            for local_col_idx, tvtw_idx in enumerate(overloaded_indices.tolist()):
                # Derive tv/hour to compute capacity
                tv_id = tvtw_to_tv_id[tvtw_idx]
                tv_row = self.tv_id_to_idx[tv_id]
                tv_start = tv_row * num_time_bins_per_tv
                bin_offset = int(tvtw_idx - tv_start)
                hour = int(bin_offset // bins_per_hour)
                hourly_capacity = self.hourly_capacity_by_tv.get(tv_id, {}).get(hour, -1)
                capacity_per_bin = (
                    float(hourly_capacity) / float(bins_per_hour) if hourly_capacity > -1 else -1
                )
                results.append(
                    {
                        "tvtw_index": int(tvtw_idx),
                        "flight_ids": flights_by_subcol.get(local_col_idx, []),
                        "hourly_capacity": float(hourly_capacity),
                        "capacity_per_bin": float(capacity_per_bin),
                    }
                )
            return results

        if mode == "hour":
            # Group overloaded indices by (tv_id, hour)
            groups: Dict[Tuple[str, int], None] = {}
            for idx in overloaded_indices.tolist():
                tv_id = tvtw_to_tv_id[idx]
                tv_row = self.tv_id_to_idx[tv_id]
                tv_start = tv_row * num_time_bins_per_tv
                bin_offset = idx - tv_start
                hour = int(bin_offset // bins_per_hour)
                groups[(tv_id, hour)] = None

            # For each (tv, hour), take union of flights over all bins in the hour
            results: List[Dict[str, Any]] = []
            for (tv_id, hour) in groups.keys():
                tv_row = self.tv_id_to_idx[tv_id]
                tv_start = tv_row * num_time_bins_per_tv
                start_bin = tv_start + hour * bins_per_hour
                end_bin = min(start_bin + bins_per_hour, tv_start + num_time_bins_per_tv)

                # Calculate the ACTUAL hourly occupancy (sum of bin occupancies)
                # This is what's used in excess calculation
                hourly_occupancy_sum = 0
                for bin_idx in range(start_bin, end_bin):
                    if bin_idx < num_tvtws:
                        hourly_occupancy_sum += total_occupancy[bin_idx]

                # Get unique flights for this hour
                sub = occ_csc[:, start_bin:end_bin]
                hour_rows = np.unique(sub.nonzero()[0])
                flight_ids = [self.flight_list.flight_ids[r] for r in hour_rows.tolist()]
                
                hourly_capacity = self.hourly_capacity_by_tv.get(tv_id, {}).get(int(hour), -1)
                capacity_per_bin = (
                    float(hourly_capacity) / float(bins_per_hour) if hourly_capacity > -1 else -1
                )
                
                results.append(
                    {
                        "traffic_volume_id": tv_id,
                        "hour": int(hour),
                        "flight_ids": flight_ids,
                        "unique_flights": len(flight_ids),  # Add unique flight count
                        "hourly_occupancy": float(hourly_occupancy_sum),  # This is the sum used in excess calc
                        "hourly_capacity": float(hourly_capacity),
                        "capacity_per_bin": float(capacity_per_bin),
                        "is_overloaded": hourly_occupancy_sum > hourly_capacity,  # Add explicit flag
                    }
                )
            # Optional: sort by tv_id then hour for stability
            results.sort(key=lambda x: (x["traffic_volume_id"], x["hour"]))
            return results

        raise ValueError("mode must be either 'bin' or 'hour'")



    def compute_delay_stats(self) -> Dict[str, float]:
        """
        Compute delay statistics by comparing current assigned takeoff times
        against the original takeoff times preserved at initialization.

        Returns:
            Dictionary with:
            - total_delay_seconds: Sum of delays over all flights (current - original)
            - mean_delay_seconds: Mean delay
            - max_delay_seconds: Maximum delay
            - min_delay_seconds: Minimum delay
            - delayed_flights_count: Number of flights with positive delay
        Notes:
            Positive values mean delays; negative values mean advances.
            Implementation is vectorized for efficiency.
        """
        # Fast path: both flight lists were loaded from the same sources in the same order.
        # We rely on identical ordering of flight_ids (copy() preserves order).
        current_fids = self.flight_list.flight_ids
        original_fids = self.original_flight_list.flight_ids

        # Optional light sanity check (O(1) + O(n) worst if mismatch); can be disabled for max speed.
        if len(current_fids) != len(original_fids):
            raise ValueError(
                "Flight counts differ between current and original flight lists."
            )

        # Vectorize extraction of takeoff times as seconds since epoch
        # Using a list comprehension once is O(n) and then numpy ops are vectorized.
        # datetime.timestamp() returns float seconds; cast to float64 for safe aggregation.
        curr_seconds = np.asarray(
            [
                self.flight_list.flight_metadata[fid]["takeoff_time"].timestamp()
                for fid in current_fids
            ],
            dtype=np.float64,
        )
        orig_seconds = np.asarray(
            [
                self.original_flight_list.flight_metadata[fid][
                    "takeoff_time"
                ].timestamp()
                for fid in original_fids
            ],
            dtype=np.float64,
        )

        # If the ordering is guaranteed identical, the above aligns by index.
        # If you want to be extra safe with minimal overhead, do a quick spot check on a few positions.
        # Skipped here for maximal performance.

        # Compute per-flight delay in seconds (positive = delayed, negative = advanced)
        delays = curr_seconds - orig_seconds

        # Aggregate stats (vectorized)
        total_delay_seconds = float(np.sum(delays))
        mean_delay_seconds = float(np.mean(delays)) if delays.size > 0 else 0.0
        max_delay_seconds = float(np.max(delays)) if delays.size > 0 else 0.0
        min_delay_seconds = float(np.min(delays)) if delays.size > 0 else 0.0
        delayed_flights_count = int(np.count_nonzero(delays > 0.0))

        # # For debugging: print top 10 most delayed flights (by positive delay)
        # # Build a list of (flight_id, delay_seconds) and sort descending by delay
        # try:
        #     delays_list = list(zip(current_fids, delays.tolist()))
        #     delayed_only = [(fid, d) for fid, d in delays_list if d > 0]
        #     delayed_only.sort(key=lambda x: x[1], reverse=True)
        #     top_n = delayed_only[:10]

        #     if top_n:
        #         print("\n=== NetworkEvaluator Delay Debug ===")
        #         print(f"Total flights: {len(current_fids)} | Delayed flights: {delayed_flights_count}")
        #         print("Top delayed flights (seconds):")
        #         for i, (fid, dsec) in enumerate(top_n, 1):
        #             # also show minutes for readability
        #             dmin = dsec / 60.0
        #             print(f"  {i:2d}. Flight {fid}: {dsec:.1f}s ({dmin:.1f} min)")
        #         print("=== End Delay Debug ===\n")
        # except Exception as _e:
        #     # Debug printing should never break the evaluator; swallow any unexpected issues.
        #     pass

        return {
            "total_delay_seconds": total_delay_seconds,
            "mean_delay_seconds": mean_delay_seconds,
            "max_delay_seconds": max_delay_seconds,
            "min_delay_seconds": min_delay_seconds,
            "delayed_flights_count": delayed_flights_count,
            "num_flights": int(delays.size),
        }

    def compute_horizon_metrics(
        self, horizon_time_windows: int, percentile_for_z_max: int = 95
    ) -> Dict[str, float]:
        """
        Compute z_max and z_sum metrics within a specified horizon.

        Args:
            horizon_time_windows: Number of time windows to consider from start

        Returns:
            Dictionary with z_max (maximum excess) and z_sum (total excess)
        """

        excess_vector = self.compute_excess_traffic_vector()
        # reshape by (num_tvs, num_time_bins_per_tv), keep first K time bins for all TVs
        num_tvtws = self.flight_list.num_tvtws
        num_tvs = len(self.tv_id_to_idx)
        bins_per_tv = 24 * (60 // self.time_bin_minutes)
        # Defensive check: ensure consistent shape
        if num_tvtws != num_tvs * bins_per_tv:
            raise ValueError(f"num_tvtws ({num_tvtws}) != num_tvs * bins_per_tv ({num_tvs} * {bins_per_tv} expected)")
            # excess_vector = np.resize(excess_vector, (num_tvs * bins_per_tv,))  # or raise
        # Apply horizon across time axis
        if horizon_time_windows > 0:
            k = min(horizon_time_windows, bins_per_tv)
            ev2d = excess_vector.reshape(num_tvs, bins_per_tv)
            excess_vector = ev2d[:, :k].ravel()

        # delay_vector = self.compute_delay_stats()

        # Limit to horizon
        # if horizon_time_windows > 0:
        #     excess_vector = excess_vector[:horizon_time_windows]

        z_95 = (
            float(np.percentile(excess_vector, percentile_for_z_max))
            if len(excess_vector) > 0
            else 0.0
        )
        z_max = float(np.max(excess_vector)) if len(excess_vector) > 0 else 0.0
        z_sum = float(np.sum(excess_vector))

        return {
            "z_95": z_95,
            "z_max": z_max,
            "z_sum": z_sum,
            "horizon_windows": len(excess_vector),
            # "total_delay_seconds": delay_vector["total_delay_seconds"],
            # "max_delay_seconds": delay_vector["max_delay_seconds"],
        }

    def get_capacity_utilization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about capacity utilization across all traffic volumes.

        Returns:
            Dictionary with utilization statistics
        """
        total_occupancy = self.flight_list.get_total_occupancy_by_tvtw()
        excess_vector = self.compute_excess_traffic_vector()

        # Get capacity vector for reporting
        total_capacity_per_bin = self._get_total_capacity_vector()

        # Find indices where capacity is defined
        active_indices = np.where(total_capacity_per_bin > 0)[0]

        # If no active TVTWs with capacity, return zero stats
        if len(active_indices) == 0:
            return {
                "mean_utilization": 0.0,
                "max_utilization": 0.0,
                "std_utilization": 0.0,
                "total_capacity": 0.0,
                "total_demand": 0.0,
                "system_utilization": 0.0,
                "overloaded_tvtws": 0,
                "total_tvtws_with_capacity": 0,
                "overload_percentage": 0.0,
            }

        # Filter data for active TVTWs
        active_capacity_bins = total_capacity_per_bin[active_indices]
        active_demand = total_occupancy[active_indices]

        # Calculate utilization
        utilizations = active_demand / active_capacity_bins

        # Calculate statistics
        # Note: total_system_capacity is hourly, so we need to sum hourly capacities
        total_system_capacity = sum(
            sum(h.values()) for h in self.hourly_capacity_by_tv.values()
        )
        total_system_demand = float(np.sum(total_occupancy[active_indices]))
        overloaded_count = int(np.sum(excess_vector > 0))

        return {
            "mean_utilization": float(np.mean(utilizations)),
            "max_utilization": float(np.max(utilizations)),
            "std_utilization": float(np.std(utilizations)),
            "total_capacity": total_system_capacity,
            "total_demand": total_system_demand,
            "system_utilization": (
                (total_system_demand / total_system_capacity)
                if total_system_capacity > 0
                else 0.0
            ),
            "overloaded_tvtws": overloaded_count,
            "total_tvtws_with_capacity": len(utilizations),
            "overload_percentage": (
                (overloaded_count / len(utilizations) * 100)
                if len(utilizations) > 0
                else 0.0
            ),
        }

    def export_results(self, filepath: str, horizon_time_windows: Optional[int] = None):
        """
        Export overload analysis results to JSON file.

        Args:
            filepath: Output file path
            horizon_time_windows: Optional horizon limit for metrics
        """
        results = {
            "metadata": {
                "time_bin_minutes": self.time_bin_minutes,
                "num_flights": self.flight_list.num_flights,
                "num_tvtws": self.flight_list.num_tvtws,
                "num_traffic_volumes": len(self.hourly_capacity_by_tv),
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "excess_traffic_vector": self.compute_excess_traffic_vector().tolist(),
            "overloaded_tvtws": self.get_overloaded_tvtws(),
            "utilization_stats": self.get_capacity_utilization_stats(),
        }

        if horizon_time_windows:
            results["horizon_metrics"] = self.compute_horizon_metrics(
                horizon_time_windows
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def get_traffic_volume_summary(self) -> pd.DataFrame:
        """
        Get summary of traffic volumes with capacity and utilization info.

        Returns:
            DataFrame with traffic volume statistics
        """
        total_occupancy = self.flight_list.get_total_occupancy_by_tvtw()
        excess_vector = self.compute_excess_traffic_vector()

        summary_data = []

        num_tvtws = self.flight_list.num_tvtws
        num_time_bins_per_tv = num_tvtws // len(self.tv_id_to_idx)
        bins_per_hour = 60 // self.time_bin_minutes

        for tv_id, tv_row in self.tv_id_to_idx.items():

            tv_hourly_capacity = self.hourly_capacity_by_tv.get(tv_id, {})
            total_tv_capacity = sum(tv_hourly_capacity.values())

            start_idx = tv_row * num_time_bins_per_tv
            end_idx = start_idx + num_time_bins_per_tv
            tv_indices = np.arange(start_idx, end_idx)

            tv_demand = np.sum(total_occupancy[tv_indices])
            tv_excess = np.sum(excess_vector[tv_indices])

            active_bins_count = 0
            for hour in tv_hourly_capacity:
                active_bins_count += bins_per_hour

            overloaded_bins = np.count_nonzero(excess_vector[tv_indices] > 0)

            summary_data.append(
                {
                    "traffic_volume_id": tv_id,
                    "total_capacity": total_tv_capacity,
                    "total_demand": tv_demand,
                    "total_excess": tv_excess,
                    "utilization_ratio": (
                        tv_demand / total_tv_capacity if total_tv_capacity > 0 else 0
                    ),
                    "overloaded_bins": overloaded_bins,
                    "active_time_bins": active_bins_count,
                }
            )

        return pd.DataFrame(summary_data).sort_values("total_excess", ascending=False)

    def get_raw_tvtw_counts_df(self) -> pd.DataFrame:
        """
        Build a DataFrame with the raw flight count for every Traffic-Volume-Time-Window (TVTW).

        Returns:
            DataFrame with columns: traffic_volume_id, time_window, raw_count
        """
        total_occupancy = self.flight_list.get_total_occupancy_by_tvtw()
        num_tvtws = self.flight_list.num_tvtws

        # Dimensions
        num_traffic_volumes = len(self.tv_id_to_idx)
        if num_traffic_volumes == 0:
            return pd.DataFrame(columns=["traffic_volume_id", "time_window", "raw_count"])

        num_time_bins_per_tv = num_tvtws // num_traffic_volumes
        bins_per_hour = 60 // self.time_bin_minutes

        def _format_time_window(bin_offset: int) -> str:
            # Start time (minutes from 00:00)
            start_total_min = int(bin_offset * self.time_bin_minutes)
            start_hour = start_total_min // 60
            start_min = start_total_min % 60

            # End time
            end_total_min = start_total_min + self.time_bin_minutes
            if end_total_min == 24 * 60:
                end_str = "24:00"
            else:
                end_hour = (end_total_min // 60) % 24
                end_min = end_total_min % 60
                end_str = f"{end_hour:02d}:{end_min:02d}"

            start_str = f"{start_hour:02d}:{start_min:02d}"
            return f"{start_str}-{end_str}"

        rows: List[Dict[str, Any]] = []
        for tv_id, tv_row in self.tv_id_to_idx.items():
            tv_start = tv_row * num_time_bins_per_tv
            tv_end = tv_start + num_time_bins_per_tv

            for bin_offset in range(num_time_bins_per_tv):
                tvtw_idx = tv_start + bin_offset
                if tvtw_idx >= num_tvtws:
                    break
                rows.append(
                    {
                        "traffic_volume_id": tv_id,
                        "time_window": _format_time_window(bin_offset),
                        "raw_count": int(total_occupancy[tvtw_idx]),
                    }
                )

        df = pd.DataFrame(rows, columns=["traffic_volume_id", "time_window", "raw_count"])
        # Stable sort by TV then by time window label
        if not df.empty:
            df.sort_values(["traffic_volume_id", "time_window"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def export_raw_tvtw_counts_csv(self, filepath: str) -> None:
        """
        Export a CSV with the raw flight count for every TVTW.

        Args:
            filepath: Destination CSV path.
        """
        df = self.get_raw_tvtw_counts_df()
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
