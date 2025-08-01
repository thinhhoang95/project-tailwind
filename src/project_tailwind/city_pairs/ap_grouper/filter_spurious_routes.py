#!/usr/bin/env python3
"""
Filter spurious routes from city pair representatives CSV files.

This script processes CSV files containing flight routes and filters out
routes that are more than 150% of the great circle distance between
origin and destination airports.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from geopy.distance import great_circle
import warnings
warnings.filterwarnings('ignore')


def load_waypoints_graph(graph_path):
    """Load waypoint graph and return node coordinates dictionary."""
    G = nx.read_gml(graph_path)
    coords = {}
    for node, data in G.nodes(data=True):
        coords[node] = (float(data['lat']), float(data['lon']))
    return coords


def calculate_great_circle_distance(origin, destination, coords):
    """Calculate great circle distance between two waypoints."""
    if origin not in coords or destination not in coords:
        return None
    
    origin_coords = coords[origin]
    dest_coords = coords[destination]
    return great_circle(origin_coords, dest_coords).kilometers


def calculate_route_distance(route_waypoints, coords):
    """Calculate total distance for a route through waypoints."""
    waypoints = route_waypoints.split()
    total_distance = 0
    
    for i in range(len(waypoints) - 1):
        wp1, wp2 = waypoints[i], waypoints[i + 1]
        if wp1 in coords and wp2 in coords:
            dist = great_circle(coords[wp1], coords[wp2]).kilometers
            total_distance += dist
        else:
            # If any waypoint is missing, we can't calculate the route
            return None
    
    return total_distance


def process_csv_file(csv_path, coords, output_dir):
    """Process a single CSV file and filter spurious routes."""
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"Empty file: {csv_path}")
            return
        
        # Calculate great circle distances vectorized
        gc_distances = []
        route_distances = []
        
        for _, row in df.iterrows():
            origin = row['origin']
            destination = row['destination']
            route = row['route']
            
            # Calculate great circle distance
            gc_dist = calculate_great_circle_distance(origin, destination, coords)
            # Calculate route distance
            route_dist = calculate_route_distance(route, coords)
            
            gc_distances.append(gc_dist)
            route_distances.append(route_dist)
        
        # Add distance columns
        df['great_circle_distance'] = gc_distances
        df['route_distance'] = route_distances
        
        # Filter out rows where we couldn't calculate distances
        valid_mask = (df['great_circle_distance'].notna()) & (df['route_distance'].notna())
        df_valid = df[valid_mask].copy()
        
        if df_valid.empty:
            print(f"No valid routes in: {csv_path}")
            return
        
        # Calculate ratio
        df_valid['distance_ratio'] = df_valid['route_distance'] / df_valid['great_circle_distance']
        
        # Filter spurious routes (> 150% of great circle distance)
        threshold = 1.5
        df_filtered = df_valid[df_valid['distance_ratio'] <= threshold].copy()
        
        # Drop the temporary columns
        df_output = df_filtered[['origin', 'destination', 'route']].copy()
        
        # Save filtered data
        output_path = output_dir / csv_path.name
        df_output.to_csv(output_path, index=False)
        
        removed_count = len(df_valid) - len(df_filtered)
        print(f"Processed {csv_path.name}: {len(df_valid)} routes -> {len(df_filtered)} routes ({removed_count} spurious removed)")
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")


def main():
    # Paths
    csv_dir = Path("output/city_pairs/representatives")
    waypoints_graph_path = Path("../project-akrav/data/graphs/ats_fra_nodes_only.gml")
    output_dir = Path("output/city_pairs/representatives_filtered")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load waypoints coordinates
    print("Loading waypoints graph...")
    coords = load_waypoints_graph(waypoints_graph_path)
    print(f"Loaded {len(coords)} waypoints")
    
    # Get all CSV files
    csv_files = list(csv_dir.glob("*_representatives.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Processing {csv_file.name}...")
        process_csv_file(csv_file, coords, output_dir)
    
    print("\nFiltering complete!")


if __name__ == "__main__":
    main()