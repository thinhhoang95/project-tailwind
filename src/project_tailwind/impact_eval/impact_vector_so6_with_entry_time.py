import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
import numpy as np
import datetime
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import argparse
import json
from typing import List, Tuple, Dict, Any, Set

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

warnings.filterwarnings('ignore')

def load_traffic_volumes(tv_path: str) -> gpd.GeoDataFrame:
    """
    Loads traffic volumes from a GeoJSON file and prepares them for processing.
    """
    tv_gdf = gpd.read_file(tv_path)
    if 'traffic_volume_id' not in tv_gdf.columns:
        raise ValueError("GeoJSON file must have a 'traffic_volume_id' property in each feature.")
    if 'min_fl' not in tv_gdf.columns or 'max_fl' not in tv_gdf.columns:
        raise ValueError("GeoJSON file must have 'min_fl' and 'max_fl' properties in each feature.")
    tv_gdf.sindex
    return tv_gdf

def find_tvs_for_point(point: Point, fl: float, tv_gdf: gpd.GeoDataFrame) -> List[str]:
    """
    Find all traffic volume IDs that contain a given point at a specific flight level.
    """
    if point.is_empty:
        return []
    
    possible_matches_idx = list(tv_gdf.sindex.query(point, predicate='intersects'))
    if not possible_matches_idx:
        return []
    
    possible_matches = tv_gdf.iloc[possible_matches_idx]
    
    precise_matches = possible_matches[possible_matches.geometry.covers(point)]
    # Note: min_fl and max_fl in traffic volume files are often in Flight Levels (hundreds of feet)
    vertically_relevant_tvs = precise_matches[
        (precise_matches['min_fl'] * 100 <= fl) & (precise_matches['max_fl'] * 100 >= fl)
    ]
    
    return vertically_relevant_tvs['traffic_volume_id'].tolist()

def parse_so6_time(date_val: Any, time_val: Any) -> datetime.datetime:
    """
    Parses SO6 date and time formats, handling potential float representations.
    """
    date_str = str(int(date_val))
    time_int = int(time_val)
    
    time_str = f"{time_int:06d}"
    # Pad date string if it's shortened (e.g., 'yymmdd' instead of 'yyyymmdd')
    if len(date_str) == 6:
        full_datetime_str = f"{date_str}{time_str}"
        return datetime.datetime.strptime(full_datetime_str, "%y%m%d%H%M%S")
    else:
        # Assuming format like 'yyyymmdd'
        full_datetime_str = f"{date_str}{time_str}"
        return datetime.datetime.strptime(full_datetime_str, "%Y%m%d%H%M%S")


def compute_occupancy_for_flight(
    flight_data: Tuple[str, pd.DataFrame],
    tv_gdf: gpd.GeoDataFrame,
    tvtw_indexer: TVTWIndexer,
    sampling_dist_nm: float = 5.0
) -> Tuple[str, List[Dict[str, Any]], float, str, str, str]:
    """
    Computes the occupancy intervals for a single flight.
    Entry and exit times are reported in seconds from takeoff.
    """
    flight_identifier, flight_df = flight_data
    total_distance_nm = 0.0
    all_points = []

    sorted_segments = flight_df.sort_values(by='time_begin_segment')
    
    if sorted_segments.empty:
        return flight_identifier, [], 0.0, None, None, None

    first_segment = sorted_segments.iloc[0]
    origin = first_segment.get('origin_aerodrome')
    destination = first_segment.get('destination_aerodrome')
    
    try:
        takeoff_time_dt = parse_so6_time(first_segment['date_begin_segment'], first_segment['time_begin_segment'])
        takeoff_time_iso = takeoff_time_dt.isoformat()
    except (ValueError, KeyError, TypeError):
        # If takeoff time cannot be parsed, we cannot calculate relative seconds.
        return flight_identifier, [], 0.0, None, origin, destination

    # 1. Collect all trajectory points from segments
    for _, segment in sorted_segments.iterrows():
        try:
            start_time = parse_so6_time(segment['date_begin_segment'], segment['time_begin_segment'])
            end_time = parse_so6_time(segment['date_end_segment'], segment['time_end_segment'])
            p1_geom = Point(segment['longitude_begin'], segment['latitude_begin'])
            p2_geom = Point(segment['longitude_end'], segment['latitude_end'])
            
            if p1_geom.is_empty or p2_geom.is_empty:
                continue

            dist_nm = geodesic((p1_geom.y, p1_geom.x), (p2_geom.y, p2_geom.x)).nm
            total_distance_nm += dist_nm

            alt1_ft = segment['flight_level_begin'] * 100
            alt2_ft = segment['flight_level_end'] * 100

            all_points.append({'geom': p1_geom, 'alt_ft': alt1_ft, 'time': start_time})

            if dist_nm > 0:
                line = LineString([p1_geom, p2_geom])
                num_samples = int(dist_nm / sampling_dist_nm) if sampling_dist_nm > 0 else 0
                if num_samples > 0:
                    for j in range(1, num_samples + 1):
                        fraction = j / (num_samples + 1)
                        sample_point = line.interpolate(fraction, normalized=True)
                        sample_alt = alt1_ft + fraction * (alt2_ft - alt1_ft)
                        time_delta = (end_time - start_time) * fraction
                        sample_time = start_time + time_delta
                        all_points.append({'geom': sample_point, 'alt_ft': sample_alt, 'time': sample_time})
            
            all_points.append({'geom': p2_geom, 'alt_ft': alt2_ft, 'time': end_time})

        except (ValueError, KeyError, TypeError):
            continue

    if not all_points:
        return flight_identifier, [], total_distance_nm, takeoff_time_iso, origin, destination

    # 2. Sort all points chronologically
    all_points.sort(key=lambda p: p['time'])

    # 3. Process points to find entry/exit events
    open_occupancies: Dict[Tuple[str, int], datetime.datetime] = {}
    observed_intervals: List[Dict[str, Any]] = []
    previous_tvtws: Set[Tuple[str, int]] = set()

    for point_data in all_points:
        current_time = point_data['time']
        tvs_at_sample = set(find_tvs_for_point(point_data['geom'], point_data['alt_ft'], tv_gdf))
        minute_of_day = current_time.hour * 60 + current_time.minute
        time_window_idx = minute_of_day // tvtw_indexer.time_bin_minutes
        
        current_tvtws = {(tv_id, time_window_idx) for tv_id in tvs_at_sample}

        exited_tvtws = previous_tvtws - current_tvtws
        for tvtw_tuple in exited_tvtws:
            if tvtw_tuple in open_occupancies:
                entry_time = open_occupancies.pop(tvtw_tuple)
                tvtw_index = tvtw_indexer.get_tvtw_index(tvtw_tuple[0], tvtw_tuple[1])
                if tvtw_index is not None:
                    observed_intervals.append({
                        "tvtw_index": tvtw_index,
                        "entry_time_s": (entry_time - takeoff_time_dt).total_seconds(),
                        "exit_time_s": (current_time - takeoff_time_dt).total_seconds()
                    })

        entered_tvtws = current_tvtws - previous_tvtws
        for tvtw_tuple in entered_tvtws:
            if tvtw_tuple not in open_occupancies:
                open_occupancies[tvtw_tuple] = current_time

        previous_tvtws = current_tvtws

    # 4. Close any remaining open occupancies at the end of the trajectory
    final_time = all_points[-1]['time']
    for tvtw_tuple, entry_time in open_occupancies.items():
        tvtw_index = tvtw_indexer.get_tvtw_index(tvtw_tuple[0], tvtw_tuple[1])
        if tvtw_index is not None:
            observed_intervals.append({
                "tvtw_index": tvtw_index,
                "entry_time_s": (entry_time - takeoff_time_dt).total_seconds(),
                "exit_time_s": (final_time - takeoff_time_dt).total_seconds()
            })
    
    # 5. Sort intervals by their entry time
    observed_intervals.sort(key=lambda x: x['entry_time_s'])

    return flight_identifier, observed_intervals, total_distance_nm, takeoff_time_iso, origin, destination


def main():
    parser = argparse.ArgumentParser(description="Compute occupancy matrix from SO6 flight data.")
    parser.add_argument("--so6_path", type=str, required=False, default="D:/project-cirrus/cases/flights_20230801.csv", help="Path to the SO6 flight data CSV file.")
    # Attention: this code currently uses full traffic volumes, not simplified!
    parser.add_argument("--tv_path", type=str, required=False, default="D:/project-cirrus/cases/traffic_volumes_with_capacity.geojson", help="Path to the traffic volumes GeoJSON file.") 
    parser.add_argument("--output_path", type=str, required=False, default="output/so6_occupancy_matrix_with_times.json", help="Path to save the output JSON file with occupancy intervals.")
    parser.add_argument("--tvtw_indexer_path", type=str, required=False, default="output/tvtw_indexer.json", help="Path to save or load the TVTW indexer. If not provided, defaults to a path next to the output.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs. Defaults to CPU count.")
    parser.add_argument("--time_bin_minutes", type=int, default=15, help="Time bin duration in minutes for TVTWs.")
    args = parser.parse_args()

    n_jobs = args.n_jobs if args.n_jobs is not None else (cpu_count())
    tvtw_indexer_path = args.tvtw_indexer_path or os.path.join(os.path.dirname(args.output_path), "tvtw_indexer.json")


    print("Loading data...")
    tv_gdf = load_traffic_volumes(args.tv_path)
    
    print(f"Loading SO6 data from {args.so6_path}...")
    so6_df = pd.read_csv(args.so6_path)
    
    required_cols = [
        'flight_identifier', 'date_begin_segment', 'time_begin_segment',
        'date_end_segment', 'time_end_segment', 'longitude_begin', 'latitude_begin',
        'longitude_end', 'latitude_end', 'flight_level_begin', 'flight_level_end',
        'origin_aerodrome', 'destination_aerodrome'
    ]
    so6_df.dropna(subset=required_cols, inplace=True)
    
    print("Grouping flights...")
    flights_grouped = so6_df.groupby('flight_identifier')
    flight_data_list = list(flights_grouped)

    print(f"Found {len(flight_data_list)} unique flights to process.")
    
    if os.path.exists(tvtw_indexer_path):
        print("Loading existing TVTW indexer...")
        tvtw_indexer = TVTWIndexer.load(tvtw_indexer_path)
    else:
        print("Building new TVTW indexer...")
        tvtw_indexer = TVTWIndexer(time_bin_minutes=args.time_bin_minutes)
        tvtw_indexer.build_from_tv_geojson(args.tv_path)
        tvtw_indexer.save(tvtw_indexer_path)
        print(f"TVTW indexer saved to {tvtw_indexer_path}")

    print(f"Processing flights with {n_jobs} workers...")
    
    process_func = partial(compute_occupancy_for_flight,
                           tv_gdf=tv_gdf,
                           tvtw_indexer=tvtw_indexer)

    results = {}
    with Pool(processes=n_jobs) as pool:
        with tqdm(total=len(flight_data_list)) as pbar:
            for flight_id, occupancy_intervals, total_distance, takeoff_time, origin, destination in pool.imap_unordered(process_func, flight_data_list):
                if occupancy_intervals:
                    results[str(flight_id)] = {
                        "occupancy_intervals": occupancy_intervals,
                        "distance": total_distance,
                        "takeoff_time": takeoff_time,
                        "origin": origin,
                        "destination": destination
                    }
                pbar.update()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"Saving {len(results)} occupancy results to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Done.")

if __name__ == '__main__':
    # Example usage:
    # python src/project_tailwind/impact_eval/impact_vector_so6_with_entry_time.py \
    #   --so6_path D:/project-cirrus/cases/flights_20230801.csv \
    #   --tv_path D:/project-cirrus/cases/traffic_volumes_simplified.geojson \
    #   --output_path output/so6_occupancy_matrix_with_times.json
    main()
