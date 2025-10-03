"""
Flow Cache Extraction Script

Implements a script that:
- Exports all hotspot bins to a CSV (tvtw_index, traffic_volume_id, time_bin).
- For each hotspot, extracts flows (groups) and writes a second CSV with one row per flow:
  reference TV, hotspot TV, avg similarity, score, flight IDs, plus cached metrics
  CountOver(t), SumOver(t), MinSlack(t) for t=0..15.
"""

import argparse
import csv
import json
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import logging
import multiprocessing as mp
from functools import partial
from rich.console import Console
from rich.table import Table

# Import the required classes
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.flow_x.flow_extractor import FlowXExtractor


def _flowx_default_params() -> Dict[str, Any]:
    """Defaults matching FlowXExtractor.find_groups_from_hotspot_hour."""
    return {
        'auto_collapse_group_output': True,
        'min_flights_per_ref': 3,
        'max_references': 500,
        'tau': None,
        'sparsification_alpha': 0.0,
        'group_size_lam': 0.0,
        'normalize_by_degree': False,
        'average_objective': True,
        'k_max_trajectories_per_group': None,
        'max_groups': 3,
        'min_group_size': 2,
        'path_length_gamma': 0.0,
        'debug_verbose_path': None,
    }


def _build_effective_flowx_params(overrides: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _flowx_default_params()
    effective = defaults.copy()
    for k, v in overrides.items():
        if k in effective:
            effective[k] = v
    return effective


def _print_flowx_params_table(effective: Dict[str, Any], title: str = "FlowX Parameters") -> None:
    console = Console()
    table = Table(title=title)
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="white")
    for key in sorted(effective.keys()):
        table.add_row(key, str(effective[key]))
    console.print(table)


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract flow cache metrics for hotspot analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required input files
    parser.add_argument(
        "--occupancy-matrix",
        type=str,
        default="output/so6_occupancy_matrix_with_times.json",
        help="Path to SO6 occupancy matrix JSON file"
    )
    parser.add_argument(
        "--tvtw-indexer", 
        type=str,
        default="output/tvtw_indexer.json",
        help="Path to TVTW indexer JSON file"
    )
    parser.add_argument(
        "--tv-geojson",
        type=str,
        required=True,
        help="Path to traffic volume GeoJSON file with capacities"
    )
    
    # CLI options
    parser.add_argument(
        "--mode",
        choices=["bin", "hour"],
        default="hour",
        help="Mode for flow extraction (hotspots CSV always uses bin mode)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for hotspot detection"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/flow_dump",
        help="Output directory for CSV files"
    )
    
    # FlowX parameters
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Maximum number of groups per hotspot"
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=None,
        help="Maximum k value for FlowX"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha parameter for FlowX"
    )
    parser.add_argument(
        "--avg-objective",
        action="store_true",
        help="Use average objective for FlowX"
    )
    parser.add_argument(
        "--group-size-lam",
        type=float,
        default=None,
        help="Group size lambda parameter"
    )
    parser.add_argument(
        "--path-length-gamma",
        type=float,
        default=None,
        help="Path length gamma parameter"
    )
    
    # Scoping options
    parser.add_argument(
        "--only-tv",
        type=str,
        default=None,
        help="Process only this traffic volume ID"
    )
    parser.add_argument(
        "--only-hour",
        type=int,
        default=None,
        help="Process only this hour"
    )
    parser.add_argument(
        "--only-tvtw",
        type=int,
        default=None,
        help="Process only this TVTW index"
    )
    parser.add_argument(
        "--limit-hotspots",
        type=int,
        default=None,
        help="Limit to first N hotspots"
    )
    
    # Debug and performance options
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use slow but simple fallback implementation for cache computation"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def load_input_data(args) -> Tuple[FlightList, gpd.GeoDataFrame]:
    """Load required input data files."""
    logging.info("Loading input data...")
    
    # Check if input files exist
    occupancy_path = Path(args.occupancy_matrix)
    indexer_path = Path(args.tvtw_indexer)
    tv_path = Path(args.tv_geojson)
    
    if not occupancy_path.exists():
        raise FileNotFoundError(f"Occupancy matrix file not found: {occupancy_path}")
    if not indexer_path.exists():
        raise FileNotFoundError(f"TVTW indexer file not found: {indexer_path}")
    if not tv_path.exists():
        raise FileNotFoundError(f"Traffic volume GeoJSON file not found: {tv_path}")
    
    # Load FlightList
    logging.info(f"Loading occupancy matrix from {occupancy_path}")
    logging.info(f"Loading TVTW indexer from {indexer_path}")
    
    try:
        flight_list = FlightList(
            str(occupancy_path), 
            str(indexer_path)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load FlightList: {e}")
    
    if len(flight_list.flight_ids) == 0:
        raise ValueError("No flights loaded from occupancy matrix")
    
    # Load traffic volume GeoJSON
    logging.info(f"Loading traffic volumes from {tv_path}")
    try:
        tv_gdf = gpd.read_file(tv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load traffic volume GeoJSON: {e}")
    
    if len(tv_gdf) == 0:
        raise ValueError("No traffic volumes loaded from GeoJSON")
    
    # Validate required columns
    required_cols = ['traffic_volume_id', 'capacity']
    missing_cols = [col for col in required_cols if col not in tv_gdf.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in traffic volume GeoJSON: {missing_cols}")
    
    logging.info(f"Loaded {len(flight_list.flight_ids)} flights and {len(tv_gdf)} traffic volumes")
    
    return flight_list, tv_gdf


def create_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_path.absolute()}")
    return output_path


def export_hotspots_csv(
    evaluator: NetworkEvaluator,
    flow_extractor: FlowXExtractor,
    flowx_kwargs: Dict[str, Any],
    output_path: Path,
    threshold: float,
    only_tv: Optional[str] = None,
    only_hour: Optional[int] = None,
    only_tvtw: Optional[int] = None,
    limit_hotspots: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Export hotspots CSV (bin mode) and return hotspot list."""
    logging.info("Discovering hotspots in bin mode...")

    # Get hotspots in bin mode
    hotspots = evaluator.get_hotspot_flights(threshold=threshold, mode="bin")

    # Apply scoping filters
    if only_tv:
        # Need to map tvtw_index back to traffic_volume_id
        num_time_bins_per_tv = evaluator.flight_list.num_tvtws // len(
            evaluator.tv_id_to_idx
        )
        filtered_hotspots = []
        for hotspot in hotspots:
            tvtw_idx = hotspot["tvtw_index"]
            # Find which TV this tvtw belongs to
            tv_row = tvtw_idx // num_time_bins_per_tv
            tv_id = None
            for tid, row in evaluator.tv_id_to_idx.items():
                if row == tv_row:
                    tv_id = tid
                    break
            if tv_id == only_tv:
                filtered_hotspots.append(hotspot)
        hotspots = filtered_hotspots

    if only_tvtw is not None:
        hotspots = [h for h in hotspots if h["tvtw_index"] == only_tvtw]

    if only_hour is not None:
        # Convert tvtw to hour and filter
        num_time_bins_per_tv = evaluator.flight_list.num_tvtws // len(
            evaluator.tv_id_to_idx
        )
        bins_per_hour = 60 // evaluator.time_bin_minutes
        filtered_hotspots = []
        for hotspot in hotspots:
            tvtw_idx = hotspot["tvtw_index"]
            tv_row = tvtw_idx // num_time_bins_per_tv
            tv_start = tv_row * num_time_bins_per_tv
            bin_offset = tvtw_idx - tv_start
            hour = bin_offset // bins_per_hour
            if hour == only_hour:
                filtered_hotspots.append(hotspot)
        hotspots = filtered_hotspots

    if limit_hotspots:
        hotspots = hotspots[:limit_hotspots]

    logging.info(f"Found {len(hotspots)} hotspots after filtering. Analyzing flows for each...")
    
    for hotspot in hotspots:
        groups = flow_extractor.find_groups_from_evaluator_item(hotspot, **flowx_kwargs)
        candidate_flights = set(hotspot.get("flight_ids", []))
        all_group_flights = []
        if groups:
            for group in groups:
                group_flight_ids = group.get("group_flights", [])
                if group_flight_ids:
                    all_group_flights.extend(group_flight_ids)

        flights_in_flows_set = set(all_group_flights)
        non_flow_flights = candidate_flights - flights_in_flows_set

        hotspot["n_flow_flights"] = len(flights_in_flows_set)
        hotspot["n_non_flow_flights"] = len(non_flow_flights)


    if not hotspots:
        logging.warning("No hotspots found with current filters")
        return hotspots

    # Export CSV 1: hotspots.csv
    hotspots_csv_path = output_path / "hotspots.csv"

    # Compute additional derived fields
    num_time_bins_per_tv = evaluator.flight_list.num_tvtws // len(
        evaluator.tv_id_to_idx
    )
    bins_per_hour = 60 // evaluator.time_bin_minutes

    # Get total occupancy for reporting
    total_occupancy = evaluator.flight_list.get_total_occupancy_by_tvtw()

    with open(hotspots_csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "tvtw_index",
            "traffic_volume_id",
            "time_bin",
            "hour",
            "hourly_capacity",
            "hourly_occupancy",
            "capacity_per_bin",
            "n_flow_flights",
            "n_non_flow_flights",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for hotspot in hotspots:
            tvtw_idx = hotspot["tvtw_index"]

            # Derive traffic_volume_id and time_bin from tvtw_index
            tv_row = tvtw_idx // num_time_bins_per_tv
            tv_id = None
            for tid, row in evaluator.tv_id_to_idx.items():
                if row == tv_row:
                    tv_id = tid
                    break

            tv_start = tv_row * num_time_bins_per_tv
            bin_offset = tvtw_idx - tv_start
            hour = bin_offset // bins_per_hour
            time_bin = bin_offset % bins_per_hour

            # Get hourly occupancy from total occupancy
            hour_start_idx = tv_start + hour * bins_per_hour
            hour_end_idx = hour_start_idx + bins_per_hour
            hourly_occupancy = float(
                np.sum(total_occupancy[hour_start_idx:hour_end_idx])
            )

            row_data = {
                "tvtw_index": tvtw_idx,
                "traffic_volume_id": tv_id,
                "time_bin": time_bin,
                "hour": hour,
                "hourly_capacity": hotspot["hourly_capacity"],
                "hourly_occupancy": hourly_occupancy,
                "capacity_per_bin": hotspot["capacity_per_bin"],
                "n_flow_flights": hotspot["n_flow_flights"],
                "n_non_flow_flights": hotspot["n_non_flow_flights"],
            }
            writer.writerow(row_data)

    logging.info(f"Exported hotspots to {hotspots_csv_path}")
    return hotspots


def precompute_cache_context(evaluator: NetworkEvaluator) -> Dict[str, Any]:
    """Precompute global structures needed for cache computation."""
    flight_list = evaluator.flight_list
    
    # Base occupancy matrix
    occ_base = flight_list.get_total_occupancy_by_tvtw()
    
    # Derived constants
    num_tvtws = flight_list.num_tvtws
    num_tvs = len(evaluator.tv_id_to_idx)
    num_time_bins_per_tv = num_tvtws // num_tvs
    bins_per_hour = 60 // evaluator.time_bin_minutes
    
    tv_row_to_id = {v: k for k, v in evaluator.tv_id_to_idx.items()}
    
    # Build mapping arrays for vectorized computation
    tv_row_of_tvtw = np.zeros(num_tvtws, dtype=int)
    hour_of_tvtw = np.zeros(num_tvtws, dtype=int)
    
    for tv_id, tv_row in evaluator.tv_id_to_idx.items():
        start_idx = tv_row * num_time_bins_per_tv
        end_idx = start_idx + num_time_bins_per_tv
        tv_row_of_tvtw[start_idx:end_idx] = tv_row
        
        for i in range(start_idx, end_idx):
            bin_offset = i - start_idx
            hour_of_tvtw[i] = bin_offset // bins_per_hour
    
    # Capacity per bin (distributed uniformly within hour)
    cap_per_bin = np.zeros(num_tvtws)
    for tv_id, tv_row in evaluator.tv_id_to_idx.items():
        start_idx = tv_row * num_time_bins_per_tv
        for hour in range(24):
            hourly_cap = evaluator.hourly_capacity_by_tv.get(tv_id, {}).get(hour, 999)
            if hourly_cap > 0:
                hour_start = start_idx + hour * bins_per_hour
                hour_end = min(hour_start + bins_per_hour, start_idx + num_time_bins_per_tv)
                cap_per_bin[hour_start:hour_end] = hourly_cap / bins_per_hour
    
    # Hourly capacity matrix [num_tvs, 24]
    hourly_capacity_matrix = np.zeros((num_tvs, 24))
    for tv_id, tv_row in evaluator.tv_id_to_idx.items():
        for hour in range(24):
            hourly_capacity_matrix[tv_row, hour] = evaluator.hourly_capacity_by_tv.get(tv_id, {}).get(hour, 999)
    
    # Precompute hourly occupancy base
    hourly_occ_base = np.zeros((num_tvs, 24))
    for tv_row in range(num_tvs):
        start_idx = tv_row * num_time_bins_per_tv
        for hour in range(24):
            hour_start = start_idx + hour * bins_per_hour
            hour_end = min(hour_start + bins_per_hour, start_idx + num_time_bins_per_tv)
            hourly_occ_base[tv_row, hour] = np.sum(occ_base[hour_start:hour_end])
    
    return {
        'occ_base': occ_base,
        'num_tvtws': num_tvtws,
        'num_tvs': num_tvs,
        'num_time_bins_per_tv': num_time_bins_per_tv,
        'bins_per_hour': bins_per_hour,
        'tv_row_of_tvtw': tv_row_of_tvtw,
        'hour_of_tvtw': hour_of_tvtw,
        'cap_per_bin': cap_per_bin,
        'hourly_capacity_matrix': hourly_capacity_matrix,
        'hourly_occ_base': hourly_occ_base,
        'tv_row_to_id': tv_row_to_id,
    }


def compute_group_occupancy_vector(flight_list: FlightList, flight_ids: List[str]) -> np.ndarray:
    """Compute total occupancy vector for a group of flights."""
    if not flight_ids:
        return np.zeros(flight_list.num_tvtws)
    
    g0 = np.zeros(flight_list.num_tvtws)
    found_flights = 0
    
    for flight_id in flight_ids:
        if flight_id in flight_list.flight_id_to_row:
            try:
                flight_occ = flight_list.get_occupancy_vector(flight_id)
                g0 += flight_occ
                found_flights += 1
            except Exception as e:
                logging.warning(f"Failed to get occupancy for flight {flight_id}: {e}")
                continue
    
    if found_flights == 0:
        logging.warning(f"No valid flights found in group: {flight_ids}")
    
    return g0


def compute_cache_metrics_vectorized(
    g0: np.ndarray, 
    cache_context: Dict[str, Any]
) -> Dict[str, List[float]]:
    """Compute cache metrics for t=0..15 using vectorized operations."""
    
    # Handle edge case: empty group
    if g0 is None or len(g0) == 0 or np.sum(g0) == 0:
        logging.debug("Empty or zero group occupancy, returning zero metrics")
        return {
            'count_over': [0.0] * 16,
            'sum_over': [0.0] * 16,
            'min_slack': [np.nan] * 16
        }
    
    # Initialize result arrays
    count_over = []
    sum_over = []
    min_slack = []
    
    # Extract context
    occ_base = cache_context['occ_base']
    hourly_occ_base = cache_context['hourly_occ_base']
    hourly_capacity_matrix = cache_context['hourly_capacity_matrix']
    cap_per_bin = cache_context['cap_per_bin']
    tv_row_of_tvtw = cache_context['tv_row_of_tvtw']
    hour_of_tvtw = cache_context['hour_of_tvtw']
    bins_per_hour = cache_context['bins_per_hour']
    num_tvs = cache_context['num_tvs']
    tv_row_to_id = cache_context['tv_row_to_id']

    # occ_base is an array representing the total traffic count (from all flights) in every single time-volume window.
    # g0 is an array of the exact same size, but it only contains the counts for the specific group of flights you are currently analyzing.
    # g_t is a new array that represents the g0 occupancy values shifted forward in time by t intervals. 
    
    # Validate dimensions
    if len(g0) != len(occ_base):
        raise ValueError(f"Group occupancy length {len(g0)} doesn't match base length {len(occ_base)}")
    if len(g0) != len(tv_row_of_tvtw):
        raise ValueError(f"Group occupancy length {len(g0)} doesn't match mapping length {len(tv_row_of_tvtw)}")
    
    # Precompute group hourly counts for base (t=0)
    group_hourly_counts_base = np.zeros((num_tvs, 24))
    for tvtw_idx in range(len(g0)):
        if g0[tvtw_idx] > 0:
            tv_row = tv_row_of_tvtw[tvtw_idx]
            hour = hour_of_tvtw[tvtw_idx]
            group_hourly_counts_base[tv_row, hour] += g0[tvtw_idx]
    
    # Compute metrics for each time shift t
    for t in range(0, 16):  # t = 0..15
        # Compute shifted group occupancy g_t
        if t == 0:
            g_t = g0.copy()
        else:
            # Shift right by t bins, zero-fill on the left (within each TV)
            g_t = np.zeros_like(g0)
            n_bins_per_tv = cache_context['num_time_bins_per_tv']
            for tv_row in range(num_tvs):
                start = tv_row * n_bins_per_tv
                end = start + n_bins_per_tv
                
                block = g0[start:end]
                
                if t < n_bins_per_tv:
                    g_t[start + t:end] = block[:-t]
        
        # Compute shifted occupancy
        occ_t = occ_base + g_t - g0

        if np.max(g0) < 1e-6:
            raise Exception("g0 is zero")
        
        # Compute group hourly counts for shifted occupancy
        group_hourly_counts_t = np.zeros((num_tvs, 24))
        for tvtw_idx in range(len(g_t)):
            if g_t[tvtw_idx] > 0:
                tv_row = tv_row_of_tvtw[tvtw_idx]
                hour = hour_of_tvtw[tvtw_idx]
                group_hourly_counts_t[tv_row, hour] += g_t[tvtw_idx]
        
        # Compute hourly occupancy and excess
        hourly_occ_t = hourly_occ_base + group_hourly_counts_t - group_hourly_counts_base
        hourly_excess_t = np.maximum(hourly_occ_t - hourly_capacity_matrix, 0)
        
        # CountOver(t): sum of present bins in overloaded hours
        count_over_t = 0
        for tv_row in range(num_tvs):
            for hour in range(24):
                if hourly_excess_t[tv_row, hour] > 0:
                    # Count bins where group is present in this hour
                    start_idx = tv_row * cache_context['num_time_bins_per_tv']
                    hour_start = start_idx + hour * bins_per_hour
                    hour_end = min(hour_start + bins_per_hour, start_idx + cache_context['num_time_bins_per_tv'])
                    
                    present_bins = 0
                    for bin_idx in range(hour_start, hour_end):
                        if bin_idx < len(g_t) and g_t[bin_idx] > 0:
                            present_bins += 1
                            # logging.debug(f"    Group present in hour {hour} time_bin {bin_idx - hour_start} (contributes to count_over)")
                    
                    if present_bins > 0:
                        tv_id = tv_row_to_id.get(tv_row, "UNKNOWN_TV")
                        # logging.debug(
                        #     f"  Adding {present_bins} to count_over_t for TV {tv_id} at hour {hour}"
                        # )

                    count_over_t += present_bins
        
        # SumOver(t): weighted sum of excess per bin
        sum_over_t = 0
        for tv_row in range(num_tvs):
            for hour in range(24):
                if hourly_excess_t[tv_row, hour] > 0:
                    excess_per_bin = hourly_excess_t[tv_row, hour] / bins_per_hour
                    
                    # Count present bins and multiply by excess per bin
                    start_idx = tv_row * cache_context['num_time_bins_per_tv']
                    hour_start = start_idx + hour * bins_per_hour
                    hour_end = min(hour_start + bins_per_hour, start_idx + cache_context['num_time_bins_per_tv'])
                    
                    present_bins = 0
                    for bin_idx in range(hour_start, hour_end):
                        if bin_idx < len(g_t) and g_t[bin_idx] > 0:
                            present_bins += 1
                    
                    sum_over_t += present_bins * excess_per_bin
        
        # MinSlack(t): minimum slack among group-occupied bins
        # Correctly compute slack based on hourly throughput, not per-bin presence.
        # slack_per_bin = np.maximum(cap_per_bin - occ_t, 0)
        # occ_t = occ_base - g0 + g_t
        hourly_slack_t = hourly_capacity_matrix - hourly_occ_t
        slack_per_bin = hourly_slack_t[tv_row_of_tvtw, hour_of_tvtw] / bins_per_hour
        slack_per_bin = np.maximum(slack_per_bin, 0)

        mask_t = g_t > 0
        
        # Define indices_to_show for logging/debugging, fixing prior scope issue.
        indices_to_show = np.array([], dtype=int)
        if np.any(mask_t):
            min_slack_t = np.min(slack_per_bin[mask_t])
            present_indices = np.where(mask_t)[0]
            if len(present_indices) > 10:
                sorted_indices = present_indices[np.argsort(slack_per_bin[mask_t])]
                indices_to_show = sorted_indices[:10]
            else:
                indices_to_show = present_indices
        else:
            min_slack_t = np.nan

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            if np.any(mask_t):
                present_indices = np.where(mask_t)[0]
                logging.debug(f"--- Slack details for t={t} (group in {len(present_indices)} bins) ---")
                if len(present_indices) > 10:
                    logging.debug("  (showing 10 bins with lowest slack)")

                for i in indices_to_show:
                    logging.debug(
                        f"  bin {i:4d}: cap={cap_per_bin[i]:5.2f}, occ_t={occ_t[i]:5.2f} "
                        f"(occ_base={occ_base[i]:5.2f} - g0={g0[i]:5.2f} + g_t={g_t[i]:5.2f}), "
                        f"slack={slack_per_bin[i]:5.2f}"
                    )
        
        # if t == 15 and min_slack_t < 20:
        #     print(f"t={t}, min_slack_t={min_slack_t}")
        #     for i in indices_to_show:
        #         logging.debug(
        #             f"  bin {i:4d}: cap={cap_per_bin[i]:5.2f}, occ_t={occ_t[i]:5.2f} "
        #             f"(occ_base={occ_base[i]:5.2f} - g0={g0[i]:5.2f} + g_t={g_t[i]:5.2f}), "
        #             f"slack={slack_per_bin[i]:5.2f}"
        #         )
        #     raise Exception("Stop here")
        
        # Store results
        logging.debug(f"Final count_over for t={t}: {count_over_t}")
        count_over.append(float(count_over_t))
        sum_over.append(float(sum_over_t))
        min_slack.append(float(min_slack_t))
        
    return {
        'count_over': count_over,
        'sum_over': sum_over,
        'min_slack': min_slack,
    }


def compute_cache_metrics_slow(
    flight_list: FlightList,
    evaluator: NetworkEvaluator, 
    flight_ids: List[str]
) -> Dict[str, List[float]]:
    """Slow but simple fallback implementation using DeltaFlightList."""
    # This would use DeltaFlightList and evaluator calls per t
    # For now, just return dummy values
    logging.warning("Slow implementation not yet implemented, using dummy values")
    return {
        'count_over': [0.0] * 16,
        'sum_over': [0.0] * 16,
        'min_slack': [0.0] * 16
    }


def process_hotspot_worker(args):
    """Worker function for processing a single hotspot."""
    hotspot, flow_extractor, cache_context, flowx_kwargs, use_slow = args
    
    try:
        candidate_flights = set(hotspot.get("flight_ids", []))
        groups = flow_extractor.find_groups_from_evaluator_item(
            hotspot, **flowx_kwargs
        )
        
        rows = []
        all_group_flights = []
        
        if not groups:
            logging.debug(f"No groups found for hotspot {hotspot}")
        else:
            logging.debug(f"Found {len(groups)} groups for hotspot")

            for group in groups:
                group_flight_ids = group.get("group_flights", [])
                if not group_flight_ids:
                    continue

                all_group_flights.extend(group_flight_ids)

                # Compute cache metrics
                if use_slow:
                    cache_metrics = compute_cache_metrics_slow(
                        flow_extractor.flight_list,
                        None,  # Would need evaluator reference
                        group_flight_ids,
                    )
                else:
                    # Compute group occupancy vector
                    g0 = compute_group_occupancy_vector(
                        flow_extractor.flight_list, group_flight_ids
                    )
                    cache_metrics = compute_cache_metrics_vectorized(
                        g0, cache_context
                    )

                # Prepare row data
                row_data = {
                    "hotspot_traffic_volume_id": hotspot.get(
                        "traffic_volume_id", ""
                    ),
                    "hotspot_hour": hotspot.get("hour", ""),
                    "hotspot_tvtw_index": hotspot.get("tvtw_index", ""),
                    "reference_sector": group.get("reference_sector", ""),
                    "group_size": group.get("group_size", 0),
                    "avg_pairwise_similarity": group.get(
                        "avg_pairwise_similarity", 0.0
                    ),
                    "score": group.get("score", 0.0),
                    "mean_path_length": group.get("mean_path_length", 0.0),
                    "flight_ids": " ".join(group_flight_ids),
                }

                # Add cache metrics
                for t in range(16):
                    row_data[f"count_over_{t}"] = cache_metrics["count_over"][t]
                    row_data[f"sum_over_{t}"] = cache_metrics["sum_over"][t]
                    row_data[f"min_slack_{t}"] = cache_metrics["min_slack"][t]

                rows.append(row_data)

        flights_in_flows_set = set(all_group_flights)
        non_flow_flights = candidate_flights - flights_in_flows_set

        non_flow_data = None
        if non_flow_flights:
            non_flow_data = {
                "traffic_volume_id": hotspot.get("traffic_volume_id", ""),
                "hour": hotspot.get("hour", ""),
                "non_flow_flights": " ".join(
                    sorted(list(non_flow_flights))
                ),
            }

        return rows, non_flow_data, len(all_group_flights), len(flights_in_flows_set), len(non_flow_flights)
        
    except Exception as e:
        # Convert numpy types to Python types for safe logging
        safe_hotspot = {}
        for k, v in hotspot.items():
            if hasattr(v, "item"):  # numpy scalar
                safe_hotspot[k] = v.item()
            else:
                safe_hotspot[k] = v
        logging.error(f"Error processing hotspot {safe_hotspot}: {e}")
        return [], None, 0, 0, 0


def extract_flows_with_cache(
    flow_extractor: FlowXExtractor,
    hotspots: List[Dict[str, Any]],
    cache_context: Dict[str, Any],
    output_path: Path,
    mode: str,
    flowx_kwargs: Dict[str, Any],
    use_slow: bool = False,
    use_multiprocessing: bool = True,
    n_processes: int = None,
) -> List[Dict[str, Any]]:
    """Extract flows and compute cache metrics, writing to CSV.
    
    Args:
        use_multiprocessing: Whether to use multiprocessing (default: True)
        n_processes: Number of processes to use (default: cpu_count - 3, min 1)
    """

    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 3)
    
    logging.info(f"Extracting flows for {len(hotspots)} hotspots using {'multiprocessing' if use_multiprocessing else 'sequential'} with {n_processes if use_multiprocessing else 1} processes...")

    fieldnames = [
        "hotspot_traffic_volume_id",
        "hotspot_hour",
        "hotspot_tvtw_index",
        "reference_sector",
        "group_size",
        "avg_pairwise_similarity",
        "score",
        "mean_path_length",
        "flight_ids",
    ] + [f"count_over_{t}" for t in range(16)] + [f"sum_over_{t}" for t in range(16)] + [f"min_slack_{t}" for t in range(16)]

    total_flows = 0
    non_flow_flights_data = []
    
    # Process hotspots
    if use_multiprocessing and len(hotspots) > 1:
        # Prepare arguments for worker processes
        worker_args = [(hotspot, flow_extractor, cache_context, flowx_kwargs, use_slow) 
                      for hotspot in hotspots]
        
        # Use multiprocessing
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(process_hotspot_worker, worker_args)
    else:
        # Sequential processing
        results = [process_hotspot_worker((hotspot, flow_extractor, cache_context, flowx_kwargs, use_slow)) 
                  for hotspot in hotspots]

    # Write results to CSV
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, (rows, non_flow_data, n_group_flights, n_unique_flights, n_non_flow_flights) in enumerate(results):
            hotspot = hotspots[i]
            
            # Write all rows for this hotspot
            for row_data in rows:
                writer.writerow(row_data)
                total_flows += 1
            
            # Log statistics
            logging.info(
                f"Hotspot {hotspot.get('traffic_volume_id')} at hour {hotspot.get('hour')}:"
            )
            logging.info(
                f"  Flights in flows: {n_group_flights} (duplicates included), {n_unique_flights} unique flights."
            )
            logging.info(f"  Flights not in any flow: {n_non_flow_flights}")
            
            # Collect non-flow data
            if non_flow_data:
                non_flow_flights_data.append(non_flow_data)

    logging.info(f"Extracted {total_flows} flows and wrote to {output_path}")
    return non_flow_flights_data


def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.debug)
    
    try:
        # Load input data
        flight_list, tv_gdf = load_input_data(args)
        
        # Create output directory
        output_path = create_output_directory(args.output_dir)
        
        # Initialize NetworkEvaluator
        logging.info("Initializing NetworkEvaluator...")
        try:
            evaluator = NetworkEvaluator(tv_gdf, flight_list)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NetworkEvaluator: {e}")
        
        # Validate evaluator state
        if len(evaluator.tv_id_to_idx) == 0:
            raise ValueError("No traffic volumes found in evaluator mapping")
        
        # Initialize FlowXExtractor
        logging.info("Initializing FlowXExtractor...")
        try:
            flow_extractor = FlowXExtractor(flight_list)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FlowXExtractor: {e}")

        # Prepare FlowX kwargs (only overrides passed to extractor)
        flowx_kwargs: Dict[str, Any] = {}
        if args.max_groups is not None:
            flowx_kwargs["max_groups"] = args.max_groups
        if args.k_max is not None:
            flowx_kwargs["k_max_trajectories_per_group"] = args.k_max
        if args.alpha is not None:
            flowx_kwargs["sparsification_alpha"] = args.alpha
        
        flowx_kwargs["average_objective"] = bool(args.avg_objective)
        
        if args.group_size_lam is not None:
            flowx_kwargs["group_size_lam"] = args.group_size_lam
        if args.path_length_gamma is not None:
            flowx_kwargs["path_length_gamma"] = args.path_length_gamma

        # Export hotspots CSV (bin mode)
        logging.info("Discovering and exporting hotspots...")
        hotspots = export_hotspots_csv(
            evaluator,
            flow_extractor,
            flowx_kwargs,
            output_path,
            args.threshold,
            only_tv=args.only_tv,
            only_hour=args.only_hour,
            only_tvtw=args.only_tvtw,
            limit_hotspots=args.limit_hotspots,
        )

        if not hotspots:
            logging.info("No hotspots found matching criteria, exiting")
            return

        # Compute and print effective parameters table (defaults overlaid with overrides)
        effective_params = _build_effective_flowx_params(flowx_kwargs)
        _print_flowx_params_table(
            effective_params, title="FlowX Parameters (Effective)"
        )

        # Precompute global structures for cache computation
        logging.info("Precomputing global structures for cache computation...")
        try:
            cache_context = precompute_cache_context(evaluator)
        except Exception as e:
            raise RuntimeError(f"Failed to precompute cache context: {e}")
        
        # Get hotspots for flow extraction (in the requested mode)
        if args.mode == "hour":
            logging.info("Getting hotspots in hour mode for flow extraction...")
            try:
                flow_hotspots = evaluator.get_hotspot_flights(threshold=args.threshold, mode="hour")
            except Exception as e:
                raise RuntimeError(f"Failed to get hotspots in hour mode: {e}")
            
            # Apply same filtering
            initial_count = len(flow_hotspots)
            if args.only_tv:
                flow_hotspots = [h for h in flow_hotspots if h["traffic_volume_id"] == args.only_tv]
                logging.info(f"Filtered to TV {args.only_tv}: {len(flow_hotspots)}/{initial_count} hotspots")
            if args.only_hour is not None:
                flow_hotspots = [h for h in flow_hotspots if h["hour"] == args.only_hour]
                logging.info(f"Filtered to hour {args.only_hour}: {len(flow_hotspots)} hotspots")
            if args.limit_hotspots:
                flow_hotspots = flow_hotspots[:args.limit_hotspots]
                logging.info(f"Limited to {args.limit_hotspots}: {len(flow_hotspots)} hotspots")
        else:
            # Use bin mode hotspots but convert format for flow extractor
            flow_hotspots = []
            for hotspot in hotspots:
                if args.limit_hotspots and len(flow_hotspots) >= args.limit_hotspots:
                    break
                flow_hotspots.append(hotspot)
        
        if not flow_hotspots:
            logging.warning("No hotspots found for flow extraction, creating empty flows CSV")
            flows_csv_path = output_path / "flows_with_cache.csv"
            # Create empty CSV with proper headers
            fieldnames = [
                'hotspot_traffic_volume_id', 'hotspot_hour', 'hotspot_tvtw_index',
                'reference_sector', 'group_size', 'avg_pairwise_similarity', 
                'score', 'mean_path_length', 'flight_ids'
            ] + [f'count_over_{t}' for t in range(16)] + \
              [f'sum_over_{t}' for t in range(16)] + \
              [f'min_slack_{t}' for t in range(16)]
            
            with open(flows_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            logging.info(f"Created empty flows CSV at {flows_csv_path}")
            return
        
        logging.info(f"Processing {len(flow_hotspots)} hotspots for flow extraction in {args.mode} mode")
        
        # Extract flows and compute cache metrics
        flows_csv_path = output_path / "flows_with_cache.csv"
        try:
            extract_flows_with_cache(
                flow_extractor,
                flow_hotspots,
                cache_context,
                flows_csv_path,
                args.mode,
                flowx_kwargs,
                args.slow,
                use_multiprocessing=True,
                n_processes=None,  # Use default (cpu_count - 3)
            )
        except Exception as e:
            raise RuntimeError(f"Failed during flow extraction: {e}")

        logging.info("Flow extraction completed successfully")
        logging.info(f"Output files:")
        logging.info(f"  Hotspots: {output_path / 'hotspots.csv'}")
        logging.info(f"  Flows: {flows_csv_path}")
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        return
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Wrapper function to run flow cache extraction programmatically.
def run_flow_cache_extraction(
    tv_geojson: str,
    threshold: float = 0.0,
    mode: str = "hour",
    output_dir: str = "output/flow_dump",
    occupancy_matrix: str = "output/so6_occupancy_matrix_with_times.json",
    tvtw_indexer: str = "output/tvtw_indexer.json",
    max_groups: int = None,
    k_max: int = None,
    alpha: float = None,
    avg_objective: bool = False,
    group_size_lam: float = None,
    path_length_gamma: float = None,
    only_tv: str = None,
    only_hour: int = None,
    only_tvtw: int = None,
    limit_hotspots: int = None,
    slow: bool = False,
    debug: bool = False,
    use_multiprocessing: bool = True,
    n_processes: int = None,
) -> bool:
    """
    Wrapper function to run flow cache extraction programmatically.
    
    Args:
        tv_geojson: Path to traffic volume GeoJSON file with capacities
        threshold: Threshold for hotspot detection
        mode: Mode for flow extraction (bin or hour)
        output_dir: Output directory for CSV files
        occupancy_matrix: Path to SO6 occupancy matrix JSON file
        tvtw_indexer: Path to TVTW indexer JSON file
        max_groups: Maximum number of groups per hotspot
        k_max: Maximum k value for FlowX
        alpha: Alpha parameter for FlowX
        avg_objective: Use average objective for FlowX
        group_size_lam: Group size lambda parameter
        path_length_gamma: Path length gamma parameter
        only_tv: Process only this traffic volume ID
        only_hour: Process only this hour
        only_tvtw: Process only this TVTW index
        limit_hotspots: Limit to first N hotspots
        slow: Use slow but simple fallback implementation for cache computation
        debug: Enable debug logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Setup logging
    setup_logging(debug)
    
    try:
        # Load input data
        flight_list, tv_gdf = load_input_data(type('Args', (), {
            'occupancy_matrix': occupancy_matrix,
            'tvtw_indexer': tvtw_indexer,
            'tv_geojson': tv_geojson
        })())
        
        # Create output directory
        output_path = create_output_directory(output_dir)
        
        # Initialize NetworkEvaluator
        logging.info("Initializing NetworkEvaluator...")
        try:
            evaluator = NetworkEvaluator(tv_gdf, flight_list)
        except Exception as e:
            logging.error(f"Failed to initialize NetworkEvaluator: {e}")
            return False
        
        # Validate evaluator state
        if len(evaluator.tv_id_to_idx) == 0:
            logging.error("No traffic volumes found in evaluator mapping")
            return False

        # Initialize FlowXExtractor
        logging.info("Initializing FlowXExtractor...")
        try:
            flow_extractor = FlowXExtractor(flight_list)
        except Exception as e:
            logging.error(f"Failed to initialize FlowXExtractor: {e}")
            return False

        # Prepare FlowX kwargs (only overrides passed to extractor)
        flowx_kwargs: Dict[str, Any] = {}
        if max_groups is not None:
            flowx_kwargs["max_groups"] = max_groups
        if k_max is not None:
            flowx_kwargs["k_max_trajectories_per_group"] = k_max
        if alpha is not None:
            flowx_kwargs["sparsification_alpha"] = alpha
        
        flowx_kwargs["average_objective"] = bool(avg_objective)
        
        if group_size_lam is not None:
            flowx_kwargs["group_size_lam"] = group_size_lam
        if path_length_gamma is not None:
            flowx_kwargs["path_length_gamma"] = path_length_gamma

        # Export hotspots CSV (bin mode)
        logging.info("Discovering and exporting hotspots...")
        hotspots = export_hotspots_csv(
            evaluator,
            flow_extractor,
            flowx_kwargs,
            output_path,
            threshold,
            only_tv=only_tv,
            only_hour=only_hour,
            only_tvtw=only_tvtw,
            limit_hotspots=limit_hotspots,
        )

        if not hotspots:
            logging.info("No hotspots found matching criteria, exiting")
            return True

        # Compute and print effective parameters table (defaults overlaid with overrides)
        effective_params = _build_effective_flowx_params(flowx_kwargs)
        _print_flowx_params_table(
            effective_params, title="FlowX Parameters (Effective)"
        )

        # Precompute global structures for cache computation
        logging.info("Precomputing global structures for cache computation...")
        try:
            cache_context = precompute_cache_context(evaluator)
        except Exception as e:
            logging.error(f"Failed to precompute cache context: {e}")
            return False
        
        # Get hotspots for flow extraction (in the requested mode)
        if mode == "hour":
            logging.info("Getting hotspots in hour mode for flow extraction...")
            try:
                flow_hotspots = evaluator.get_hotspot_flights(threshold=threshold, mode="hour")
            except Exception as e:
                logging.error(f"Failed to get hotspots in hour mode: {e}")
                return False
            
            # Apply same filtering
            initial_count = len(flow_hotspots)
            if only_tv:
                flow_hotspots = [h for h in flow_hotspots if h["traffic_volume_id"] == only_tv]
                logging.info(f"Filtered to TV {only_tv}: {len(flow_hotspots)}/{initial_count} hotspots")
            if only_hour is not None:
                flow_hotspots = [h for h in flow_hotspots if h["hour"] == only_hour]
                logging.info(f"Filtered to hour {only_hour}: {len(flow_hotspots)} hotspots")
            if limit_hotspots:
                flow_hotspots = flow_hotspots[:limit_hotspots]
                logging.info(f"Limited to {limit_hotspots}: {len(flow_hotspots)} hotspots")
        else:
            # Use bin mode hotspots but convert format for flow extractor
            flow_hotspots = []
            for hotspot in hotspots:
                if limit_hotspots and len(flow_hotspots) >= limit_hotspots:
                    break
                flow_hotspots.append(hotspot)
        
        if not flow_hotspots:
            logging.warning("No hotspots found for flow extraction, creating empty flows CSV")
            flows_csv_path = output_path / "flows_with_cache.csv"
            # Create empty CSV with proper headers
            fieldnames = [
                'hotspot_traffic_volume_id', 'hotspot_hour', 'hotspot_tvtw_index',
                'reference_sector', 'group_size', 'avg_pairwise_similarity',
                'score', 'mean_path_length', 'flight_ids'
            ] + [f'count_over_{t}' for t in range(16)] + \
              [f'sum_over_{t}' for t in range(16)] + \
              [f'min_slack_{t}' for t in range(16)]
            
            with open(flows_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            logging.info(f"Created empty flows CSV at {flows_csv_path}")
            return True
        
        # for debugging
        # flow_hotspots = flow_hotspots[-1:] # for debugging
        # print(f"flow_hotspots: {flow_hotspots}")
        # logging.warning(f"Processing only {len(flow_hotspots)} hotspots for flow extraction in {mode} mode")
        logging.info(f"Processing {len(flow_hotspots)} hotspots for flow extraction in {mode} mode")
        
        # Extract flows and compute cache metrics
        flows_csv_path = output_path / "flows_with_cache.csv"
        non_flow_flights_data = None
        try:
            non_flow_flights_data = extract_flows_with_cache(
                flow_extractor,
                flow_hotspots,
                cache_context,
                flows_csv_path,
                mode,
                flowx_kwargs,
                slow,
                use_multiprocessing=use_multiprocessing,
                n_processes=n_processes,
            )

            if non_flow_flights_data:
                non_flow_flights_csv_path = output_path / "non_flow_flights.csv"
                logging.info(
                    f"Writing non-flow flights to {non_flow_flights_csv_path}..."
                )
                with open(non_flow_flights_csv_path, "w", newline="") as csvfile:
                    fieldnames = ["traffic_volume_id", "hour", "non_flow_flights"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(non_flow_flights_data)

        except Exception as e:
            logging.error(f"Failed during flow extraction: {e}")
            return False

        logging.info("Flow extraction completed successfully")
        logging.info(f"Output files:")
        logging.info(f"  Hotspots: {output_path / 'hotspots.csv'}")
        logging.info(f"  Flows: {flows_csv_path}")
        if non_flow_flights_data:
            logging.info(f"  Non-flow flights: {output_path / 'non_flow_flights.csv'}")

        return True

    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        return False
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    main()