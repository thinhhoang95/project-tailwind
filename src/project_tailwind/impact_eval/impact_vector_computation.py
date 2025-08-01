import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from geopy.distance import geodesic
import numpy as np
import datetime
import os
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import argparse
import json
from typing import List, Tuple, Dict, Any, Set

from project_tailwind.get4d.get_4d_trajectory import get_4d_trajectory
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.vnav.vnav_performance import Performance, get_eta_and_distance_climb, get_eta_and_distance_descent
from project_tailwind.vnav.vnav_profiles_rev1 import (
    NARROW_BODY_JET_CLIMB_PROFILE,
    NARROW_BODY_JET_DESCENT_PROFILE,
    NARROW_BODY_JET_CLIMB_VS_PROFILE,
    NARROW_BODY_JET_DESCENT_VS_PROFILE,
)

warnings.filterwarnings('ignore')

def load_waypoint_graph(graph_path: str = "D:/project-akrav/data/graphs/ats_fra_nodes_only.gml") -> nx.Graph:
    """
    Loads the waypoint graph from a GML file.
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Waypoint graph not found at: {graph_path}")
    return nx.read_gml(graph_path)

def load_traffic_volumes(tv_path: str) -> gpd.GeoDataFrame:
    """
    Loads traffic volumes from a GeoJSON file and prepares them for processing.
    """
    tv_gdf = gpd.read_file(tv_path)
    if 'traffic_volume_id' not in tv_gdf.columns:
        raise ValueError("GeoJSON file must have a 'traffic_volume_id' property in each feature.")
    if 'min_fl' not in tv_gdf.columns or 'max_fl' not in tv_gdf.columns:
        raise ValueError("GeoJSON file must have 'min_fl' and 'max_fl' properties in each feature.")
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


def compute_impact_vector_for_route(
    route: str,
    nodes_graph: nx.Graph,
    tv_gdf: gpd.GeoDataFrame,
    tvtw_indexer: TVTWIndexer,
    perf_model: Performance,
    climb_table: list,
    descent_table: list
) -> Tuple[str, List[int]]:
    """
    Wrapper function to compute impact vector for a single route.
    """
    # For simplicity, we assume a fixed takeoff time for all routes.
    # This could be randomized or varied in a more advanced simulation.
    takeoff_time = datetime.datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)

    impact_vector = compute_impact_vector(
        route=route,
        nodes_graph=nodes_graph,
        tv_gdf=tv_gdf,
        tvtw_indexer=tvtw_indexer,
        takeoff_time=takeoff_time,
        perf_model=perf_model,
        climb_table=climb_table,
        descent_table=descent_table,
    )
    return (route, impact_vector)


def compute_impact_vector(
    route: str,
    nodes_graph: nx.Graph,
    tv_gdf: gpd.GeoDataFrame,
    tvtw_indexer: TVTWIndexer,
    takeoff_time: datetime.datetime,
    perf_model: Performance,
    climb_table: list,
    descent_table: list,
    cruise_ground_speed_kts: float = 460.0,
    cruise_altitude_ft: float = 35000.0,
    sampling_dist_nm: float = 5.0
) -> List[int]:
    """
    Computes the impact vector (a sequence of TVTW indices) for a single route.
    """
    try:
        altitudes_ft, elapsed_times_s, final_waypoints, _, _ = get_4d_trajectory(
            route=route,
            nodes_graph=nodes_graph,
            eta_and_distance_climb_table=climb_table,
            eta_and_distance_descend_table=descent_table,
            cruise_ground_speed_kts=cruise_ground_speed_kts,
            cruise_altitude_ft=cruise_altitude_ft,
        )

        if not altitudes_ft or not elapsed_times_s:
            return []

        # The new get_4d_trajectory returns the full list of waypoints including TOC/TOD
        # No more simplification is needed here.
        lats = [wp['lat'] for wp in final_waypoints]
        lons = [wp['lon'] for wp in final_waypoints]
        
        # Ensure the lengths of all lists match
        if not (len(lats) == len(lons) == len(altitudes_ft) == len(elapsed_times_s)):
            # This case should ideally not happen with the new implementation
            # but as a safeguard, we'll skip the route.
            return []

        trajectory_df = pd.DataFrame({
            'latitude': lats,
            'longitude': lons,
            'altitude_ft': altitudes_ft,
            'timestamp': [takeoff_time + datetime.timedelta(seconds=s) for s in elapsed_times_s]
        })

        impact_vector = []
        
        for i in range(len(trajectory_df) - 1):
            p1 = trajectory_df.iloc[i]
            p2 = trajectory_df.iloc[i+1]

            p1_geom = Point(p1['longitude'], p1['latitude'])
            p2_geom = Point(p2['longitude'], p2['latitude'])

            if p1_geom.distance(p2_geom) == 0:
                continue

            # Check initial point
            tvs_at_p1 = set(find_tvs_for_point(p1_geom, p1['altitude_ft'], tv_gdf))
            for tv_id in tvs_at_p1:
                minute_of_day = p1['timestamp'].hour * 60 + p1['timestamp'].minute
                time_window_idx = minute_of_day // tvtw_indexer.time_bin_minutes
                tvtw_index = tvtw_indexer.get_tvtw_index(tv_id, time_window_idx)
                if tvtw_index is not None:
                    impact_vector.append(tvtw_index)

            # Sample along segment
            line = LineString([p1_geom, p2_geom])
            dist_nm = geodesic((p1_geom.y, p1_geom.x), (p2_geom.y, p2_geom.x)).nm
            num_samples = int(dist_nm / sampling_dist_nm)

            for j in range(1, num_samples + 1):
                fraction = j / num_samples
                sample_point = line.interpolate(fraction, normalized=True)
                sample_alt = p1['altitude_ft'] + fraction * (p2['altitude_ft'] - p1['altitude_ft'])
                
                time_delta = (p2['timestamp'] - p1['timestamp']) * fraction
                sample_time = p1['timestamp'] + time_delta

                tvs_at_sample = set(find_tvs_for_point(sample_point, sample_alt, tv_gdf))
                for tv_id in tvs_at_sample:
                    minute_of_day = sample_time.hour * 60 + sample_time.minute
                    time_window_idx = minute_of_day // tvtw_indexer.time_bin_minutes
                    tvtw_index = tvtw_indexer.get_tvtw_index(tv_id, time_window_idx)
                    if tvtw_index is not None:
                        impact_vector.append(tvtw_index)

        return sorted(list(set(impact_vector)))

    except (ValueError, nx.NetworkXError, NotImplementedError, KeyError) as e:
        # print(f"Could not process route '{route}': {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Compute impact vectors for representative routes.")
    parser.add_argument("--routes_dir", type=str, required=False, default="output/city_pairs/representatives_filtered", help="Directory containing representative route CSV files.")
    parser.add_argument("--tv_path", type=str, required=False, default="D:/project-cirrus/cases/traffic_volumes_simplified.geojson", help="Path to the traffic volumes GeoJSON file.")
    parser.add_argument("--graph_path", type=str, default="D:/project-akrav/data/graphs/ats_fra_nodes_only.gml", help="Path to the waypoint graph GML file.")
    parser.add_argument("--output_path", type=str, required=False, default="output/impact_vectors.json", help="Path to save the output JSON file with impact vectors.")
    parser.add_argument("--tvtw_indexer_path", type=str, required=False, default="output/tvtw_indexer.json", help="Path to save or load the TVTW indexer.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs. Defaults to CPU count.")
    args = parser.parse_args()

    # Setup
    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
    
    # 1. Load data
    print("Loading data...")
    nodes_graph = load_waypoint_graph(args.graph_path)
    tv_gdf = load_traffic_volumes(args.tv_path)
    
    all_routes = []
    for filename in os.listdir(args.routes_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(args.routes_dir, filename)
            df = pd.read_csv(file_path)
            # Assuming the route is in a column named 'route'
            if 'route' in df.columns:
                all_routes.extend(df['route'].tolist())
    
    print(f"Loaded {len(all_routes)} routes to process.")

    # 2. Setup TVTW Indexer
    if os.path.exists(args.tvtw_indexer_path):
        print("Loading existing TVTW indexer...")
        tvtw_indexer = TVTWIndexer.load(args.tvtw_indexer_path)
    else:
        print("Building new TVTW indexer...")
        tvtw_indexer = TVTWIndexer(time_bin_minutes=30)
        tvtw_indexer.build_from_tv_geojson(args.tv_path)
        tvtw_indexer.save(args.tvtw_indexer_path)
        print(f"TVTW indexer saved to {args.tvtw_indexer_path}")

    # 3. Setup Performance Model
    perf_model = Performance(
        climb_speed_profile=NARROW_BODY_JET_CLIMB_PROFILE,
        descent_speed_profile=NARROW_BODY_JET_DESCENT_PROFILE,
        climb_vertical_speed_profile=NARROW_BODY_JET_CLIMB_VS_PROFILE,
        descent_vertical_speed_profile=[(alt, -vs) for alt, vs in NARROW_BODY_JET_DESCENT_VS_PROFILE],
        cruise_altitude_ft=35000.0,
        cruise_speed_kts=460.0,
    )
    climb_table = get_eta_and_distance_climb(perf_model, origin_airport_elevation_ft=0.0)
    descent_table = get_eta_and_distance_descent(perf_model, destination_airport_elevation_ft=0.0)

    # 4. Process routes in parallel
    print(f"Processing routes with {n_jobs} workers...")
    
    # Using partial to pass fixed arguments to the worker function
    process_func = partial(compute_impact_vector_for_route,
                           nodes_graph=nodes_graph,
                           tv_gdf=tv_gdf,
                           tvtw_indexer=tvtw_indexer,
                           perf_model=perf_model,
                           climb_table=climb_table,
                           descent_table=descent_table)

    results = {}
    with Pool(processes=n_jobs) as pool:
        with tqdm(total=len(all_routes)) as pbar:
            for route, impact_vector in pool.imap_unordered(process_func, all_routes):
                if impact_vector:
                    results[route] = impact_vector
                pbar.update()
    
    # 5. Save results
    print(f"Saving {len(results)} impact vectors to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Done.")

if __name__ == '__main__':
    # Example usage:
    # python src/project_tailwind/impact_eval/impact_vector_computation.py \
    #   --routes_dir output/city_pairs/representatives_filtered \
    #   --tv_path cases/traffic_volumes_simplified.geojson \
    #   --output_path output/impact_vectors.json \
    #   --tvtw_indexer_path output/tvtw_indexer.json
    main()
