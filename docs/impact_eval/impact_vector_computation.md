# Documentation for `impact_vector_computation_with_entry_time.py`

## 1. Overview

This script is designed to determine the precise interaction between flight routes and predefined 4D airspaces. For each flight route, it computes detailed "impact data," which includes not only which airspaces the flight enters but also the exact entry and exit times for each crossing, measured in seconds from takeoff.

This is crucial for advanced analysis of airspace congestion and flight interactions. By capturing the temporal dynamics of a flight's path, we can perform more accurate simulations and evaluations of air traffic scenarios.

## 2. Core Concepts

### Impact Data
The **Impact Data** is a JSON object that provides a detailed record of a flight's interaction with the airspace structure. For each route, it contains:
-   **`impact_vector`**: A chronologically sorted list of integer IDs. Each ID corresponds to a specific "Traffic Volume Time Window" (TVTW) that the flight's trajectory intersects.
-   **`entry_times_s`**: A list of floating-point values representing the entry time (in seconds from takeoff) for each corresponding TVTW in the impact vector.
-   **`exit_times_s`**: A list of floating-point values representing the exit time (in seconds from takeoff) for each corresponding TVTW.

### Traffic Volume Time Window (TVTW)
A **Traffic Volume (TV)** is a defined 3D airspace, typically represented as a polygon with a minimum and maximum flight level.

To account for the temporal dimension, the day is divided into discrete time bins (e.g., 15 minutes each). A **TVTW** is the combination of a specific Traffic Volume and one of these time bins.

The `TVTWIndexer` utility class is responsible for assigning a unique, persistent integer index to every possible TVTW pair.

## 3. Algorithm Description

The script executes the following steps to compute the impact data for a collection of routes:

1.  **Load Inputs**: The script begins by loading the necessary data:
    *   **Waypoint Graph**: A NetworkX graph (`.gml` file) containing all possible waypoints and their geographic coordinates.
    *   **Traffic Volumes (TVs)**: A GeoJSON file defining the 3D traffic volumes.
    *   **Representative Routes**: A collection of `.csv` files, each containing flight routes defined as strings of waypoints.

2.  **Initialize TVTW Indexer**: It creates or loads a `TVTWIndexer` to ensure consistent mapping of TVTWs to integer IDs.

3.  **Generate 4D Trajectory**: For each route, the `get_4d_trajectory` function simulates the flight using an aircraft performance model, generating a detailed 4D trajectory (latitude, longitude, altitude, and elapsed time in seconds).

4.  **Reconstruct and Sample Trajectory**: The script creates a comprehensive, time-sorted list of points representing the flight's path. This includes the original waypoints and additional points sampled at a fixed distance (e.g., every 5 nautical miles) along the trajectory segments to ensure high-resolution analysis.

5.  **Detect Entry and Exit Events**: The script iterates through the sorted list of trajectory points. It maintains a set of the TVTWs the flight currently occupies.
    *   **Entry**: When the flight moves from a location outside a TVTW to one inside it, a new "occupancy" is opened, and the entry time (in elapsed seconds) is recorded.
    *   **Exit**: When the flight moves from being inside a TVTW to outside it, the occupancy is closed. An interval `(tvtw_index, entry_time_s, exit_time_s)` is created and stored.
    *   This stateful tracking ensures that each continuous crossing of a TVTW is captured as a single, precise event.

6.  **Construct Final Impact Data**: After processing the entire trajectory, the collected intervals are sorted by their entry times. The data is then separated into three lists: `impact_vector`, `entry_times_s`, and `exit_times_s`.

7.  **Parallel Processing**: To accelerate the computation, the script uses Python's `multiprocessing` library to process multiple routes in parallel.

8.  **Save Results**: The final output is a JSON file containing a dictionary that maps each route string to its computed impact data object.

## 4. Inputs

-   **Routes Directory (`--routes_dir`)**: Path to a directory containing `.csv` files. Each file should have a 'route' column with waypoint strings.
    -   *Default*: `output/city_pairs/representatives_filtered`
-   **Traffic Volumes Path (`--tv_path`)**: Path to a GeoJSON file defining the 3D traffic volumes.
    -   *Default*: `D:/project-cirrus/cases/traffic_volumes_simplified.geojson`
-   **Graph Path (`--graph_path`)**: Path to the waypoint network graph in GML format.
    -   *Default*: `D:/project-akrav/data/graphs/ats_fra_nodes_only.gml`

## 5. Outputs

-   **Impact Data (`--output_path`)**: A JSON file mapping each route string to its impact data object.
    -   *Default*: `output/impact_vectors_with_times.json`
-   **TVTW Indexer (`--tvtw_indexer_path`)**: A JSON file that stores the `TVTWIndexer` object's state.
    -   *Default*: `output/tvtw_indexer.json`

### Example Output Format
```json
{
    "ROUTE_STRING_1": {
        "impact_vector": [101, 250, 251],
        "entry_times_s": [120.5, 350.2, 580.9],
        "exit_times_s": [350.2, 580.9, 720.0]
    },
    "ROUTE_STRING_2": {
        ...
    }
}
```

## 6. Usage Example

To run the script, use the following command structure, adjusting the paths as necessary.

```bash
python src/project_tailwind/impact_eval/impact_vector_computation_with_entry_time.py \
  --routes_dir output/city_pairs/representatives_filtered \
  --tv_path cases/traffic_volumes_simplified.geojson \
  --output_path output/impact_vectors_with_times.json \
  --tvtw_indexer_path output/tvtw_indexer.json
```
