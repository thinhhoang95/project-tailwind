import json
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
import warnings

# Suppress networkx warnings for better output
warnings.filterwarnings("ignore")


def load_route_graph(gml_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Load the route graph and create a mapping from waypoint names to coordinates.
    
    Args:
        gml_path: Path to the GML file containing nodes with lat/lon
        
    Returns:
        Dictionary mapping waypoint names to (lat, lon) tuples
    """
    graph = nx.read_gml(gml_path)
    waypoint_coords = {}
    
    for node_id, data in graph.nodes(data=True):
        waypoint_name = node_id
        lat = float(data['lat'])
        lon = float(data['lon'])
        waypoint_coords[waypoint_name] = (lat, lon)
    
    return waypoint_coords


def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray, 
                        lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance calculation.
    
    Args:
        lat1, lon1: Arrays of latitude and longitude for first points
        lat2, lon2: Arrays of latitude and longitude for second points
        
    Returns:
        Array of distances in nautical miles
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (np.sin(dlat / 2) ** 2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in nautical miles
    R = 3440.065
    
    return R * c


def compute_route_distance(waypoints: List[str], 
                          waypoint_coords: Dict[str, Tuple[float, float]]) -> float:
    """
    Compute total distance for a route given a sequence of waypoints.
    
    Args:
        waypoints: List of waypoint names in order
        waypoint_coords: Dictionary mapping waypoint names to coordinates
        
    Returns:
        Total route distance in nautical miles, or -1 if any waypoint is missing
    """
    if len(waypoints) < 2:
        return 0.0
    
    # Check if all waypoints exist in the graph
    missing_waypoints = [wp for wp in waypoints if wp not in waypoint_coords]
    if missing_waypoints:
        print(f"Warning: Missing waypoints {missing_waypoints}")
        return -1.0
    
    # Extract coordinates
    coords = [waypoint_coords[wp] for wp in waypoints]
    lats = np.array([coord[0] for coord in coords])
    lons = np.array([coord[1] for coord in coords])
    
    # Compute distances between consecutive waypoints
    if len(lats) == 1:
        return 0.0
    
    distances = haversine_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:])
    
    return float(np.sum(distances))


def process_all_routes(impact_vectors_path: str, gml_path: str) -> Dict[str, Dict]:
    """
    Process all routes and compute their distances.
    
    Args:
        impact_vectors_path: Path to the impact_vectors.json file
        gml_path: Path to the GML graph file
        
    Returns:
        Dictionary with same structure as input but adding distance property
    """
    print("Loading route graph...")
    waypoint_coords = load_route_graph(gml_path)
    print(f"Loaded {len(waypoint_coords)} waypoints")
    
    print("Loading impact vectors...")
    with open(impact_vectors_path, 'r') as f:
        impact_vectors = json.load(f)
    
    print(f"Processing {len(impact_vectors)} routes...")
    
    route_data = {}
    processed = 0
    missing_count = 0
    
    for route_str, impact_vector_list in impact_vectors.items():
        # Parse waypoints from route string
        waypoints = route_str.split()
        
        # Compute distance
        distance = compute_route_distance(waypoints, waypoint_coords)
        
        # Create new structure with original data plus distance
        route_data[route_str] = {
            "impact_vectors": impact_vector_list,
            "distance": distance
        }
        
        if distance == -1:
            missing_count += 1
        
        processed += 1
        if processed % 10000 == 0:
            print(f"Processed {processed}/{len(impact_vectors)} routes...")
    
    print(f"Completed processing. {missing_count} routes had missing waypoints.")
    
    return route_data


def main():
    """Main function to compute route distances."""
    impact_vectors_path = "D:/project-tailwind/output/impact_vectors.json"
    gml_path = "D:/project-akrav/data/graphs/ats_fra_nodes_only.gml"
    output_path = "D:/project-tailwind/output/route_distances.json"
    
    print("Starting route distance computation...")
    
    try:
        route_data = process_all_routes(impact_vectors_path, gml_path)
        
        # Save results
        print(f"Saving results to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(route_data, f, indent=2)
        
        # Print statistics
        valid_distances = [route["distance"] for route in route_data.values() if route["distance"] >= 0]
        if valid_distances:
            print(f"\nStatistics:")
            print(f"Total routes: {len(route_data)}")
            print(f"Valid routes: {len(valid_distances)}")
            print(f"Average distance: {np.mean(valid_distances):.2f} nm")
            print(f"Min distance: {np.min(valid_distances):.2f} nm")
            print(f"Max distance: {np.max(valid_distances):.2f} nm")
        
        print("Route distance computation completed successfully!")
        
    except Exception as e:
        print(f"Error during computation: {e}")
        raise


if __name__ == "__main__":
    main()