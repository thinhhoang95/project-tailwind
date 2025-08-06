## ATTENTION: Use the impact_vector_so6_with_entry_time.py instead.
## This is legacy code.
## It is not used in the project.

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
) -> Tuple[str, List[int], float, str, str, str]:
    """
    Computes the occupancy vector (a sequence of TVTW indices) for a single flight
    by processing all its segments.
    """
    flight_identifier, flight_df = flight_data
    occupancy_vector = set()
    total_distance_nm = 0.0

    sorted_segments = flight_df.sort_values(by='time_begin_segment')
    
    if sorted_segments.empty:
        return (flight_identifier, [], 0.0, None, None, None)

    first_segment = sorted_segments.iloc[0]
    origin = first_segment.get('origin_aerodrome')
    destination = first_segment.get('destination_aerodrome')
    takeoff_time = None
    try:
        takeoff_time_dt = parse_so6_time(first_segment['date_begin_segment'], first_segment['time_begin_segment'])
        takeoff_time = takeoff_time_dt.isoformat()
    except (ValueError, KeyError, TypeError):
        pass

    for _, segment in sorted_segments.iterrows():
        try:
            start_time = parse_so6_time(segment['date_begin_segment'], segment['time_begin_segment'])
            end_time = parse_so6_time(segment['date_end_segment'], segment['time_end_segment'])

            p1_geom = Point(segment['longitude_begin'], segment['latitude_begin'])
            p2_geom = Point(segment['longitude_end'], segment['latitude_end'])
            
            if p1_geom.is_empty or p2_geom.is_empty:
                continue

            dist_nm = geodesic((p1_geom.y, p1_geom.x), (p2_geom.y, p2_geom.x)).nm
            if dist_nm == 0:
                continue

            total_distance_nm += dist_nm

            alt1_ft = segment['flight_level_begin'] * 100
            alt2_ft = segment['flight_level_end'] * 100

            # Check occupancy at the start point of the segment
            tvs_at_p1 = find_tvs_for_point(p1_geom, alt1_ft, tv_gdf)
            for tv_id in tvs_at_p1:
                minute_of_day = start_time.hour * 60 + start_time.minute
                time_window_idx = minute_of_day // tvtw_indexer.time_bin_minutes
                tvtw_index = tvtw_indexer.get_tvtw_index(tv_id, time_window_idx)
                if tvtw_index is not None:
                    occupancy_vector.add(tvtw_index)

            # Sample along the segment to find intermediate occupancies
            line = LineString([p1_geom, p2_geom])
            num_samples = int(dist_nm / sampling_dist_nm) if sampling_dist_nm > 0 else 0

            if num_samples > 0:
                for j in range(1, num_samples + 1):
                    fraction = j / (num_samples + 1)
                    sample_point = line.interpolate(fraction, normalized=True)
                    sample_alt = alt1_ft + fraction * (alt2_ft - alt1_ft)
                    
                    time_delta = (end_time - start_time) * fraction
                    sample_time = start_time + time_delta
                    
                    tvs_at_sample = find_tvs_for_point(sample_point, sample_alt, tv_gdf)
                    for tv_id in tvs_at_sample:
                        minute_of_day = sample_time.hour * 60 + sample_time.minute
                        time_window_idx = minute_of_day // tvtw_indexer.time_bin_minutes
                        tvtw_index = tvtw_indexer.get_tvtw_index(tv_id, time_window_idx)
                        if tvtw_index is not None:
                            occupancy_vector.add(tvtw_index)

        except (ValueError, KeyError, TypeError) as e:
            # Silently skip segments that cause parsing errors.
            # print(f"Warning: Skipping segment for flight '{flight_identifier}' due to error: {e}")
            continue

    return (flight_identifier, sorted(list(occupancy_vector)), total_distance_nm, takeoff_time, origin, destination)


def main():
    parser = argparse.ArgumentParser(description="Compute occupancy matrix from SO6 flight data.")
    parser.add_argument("--so6_path", type=str, required=False, default="/Volumes/CrucialX/project-cirrus/cases/flights_20230801.csv", help="Path to the SO6 flight data CSV file.")
    parser.add_argument("--tv_path", type=str, required=False, default="/Volumes/CrucialX/project-cirrus/cases/traffic_volumes_simplified.geojson", help="Path to the traffic volumes GeoJSON file.")
    parser.add_argument("--output_path", type=str, required=False, default="output/so6_occupancy_matrix.json", help="Path to save the output JSON file with occupancy vectors.")
    parser.add_argument("--tvtw_indexer_path", type=str, required=False, default="output/tvtw_indexer.json", help="Path to save or load the TVTW indexer. If not provided, defaults to a path next to the output.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs. Defaults to CPU count.")
    parser.add_argument("--time_bin_minutes", type=int, default=15, help="Time bin duration in minutes for TVTWs.")
    args = parser.parse_args()

    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
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
            for flight_id, occupancy_vector, total_distance, takeoff_time, origin, destination in pool.imap_unordered(process_func, flight_data_list):
                if occupancy_vector:
                    results[str(flight_id)] = {
                        "occupancy_vector": occupancy_vector,
                        "distance": total_distance,
                        "takeoff_time": takeoff_time,
                        "origin": origin,
                        "destination": destination
                    }
                pbar.update()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    print(f"Saving {len(results)} occupancy vectors to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Done.")

if __name__ == '__main__':
    # Example usage:
    # python src/project_tailwind/impact_eval/impact_vector_so6.py \
    #   --so6_path D:/project-cirrus/cases/flights_20230801.csv \
    #   --tv_path D:/project-cirrus/cases/traffic_volumes_simplified.geojson \
    #   --output_path output/so6_occupancy_matrix.json
    main()
