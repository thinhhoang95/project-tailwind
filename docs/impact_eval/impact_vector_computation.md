# Impact Vector Computation

## Overview

This script is designed to determine the interaction between flight routes and predefined, three-dimensional airspaces over time. For each flight route, it computes an "impact vector," which is a compact representation of all the specific airspaces and time windows the flight passes through.

This is crucial for understanding the potential congestion or interaction effects of a set of routes on the air traffic control system. By converting a 4D trajectory (latitude, longitude, altitude, time) into a standardized vector, we can easily compare, group, and analyze the system-wide effects of different routes or traffic patterns.

## Core Concepts

### Impact Vector
An **Impact Vector** is a sorted list of unique integer IDs. Each ID corresponds to a specific "Traffic Volume Time Window" (TVTW) that a flight's trajectory intersects. If a flight passes through three specific airspaces during certain time intervals, its impact vector will contain the three corresponding unique IDs. This provides a standardized "fingerprint" of the route's interaction with the airspace structure.

### Traffic Volume Time Window (TVTW)
A **Traffic Volume (TV)** is a defined 3D airspace, typically represented as a polygon with a minimum and maximum flight level.

To account for the temporal dimension, the day is divided into discrete time bins (e.g., 15 minutes each). A **TVTW** is the combination of a specific Traffic Volume and one of these time bins.

The `TVTWIndexer` utility class is responsible for assigning a unique, persistent integer index to every possible TVTW pair. This allows the impact vector to be a simple, efficient list of integers.

## Algorithm Description

The script executes the following steps to compute the impact vectors for a collection of routes:

1.  **Load Inputs**: The script begins by loading the necessary data:
    *   **Waypoint Graph**: A NetworkX graph (`.gml` file) containing all possible waypoints and their geographic coordinates.
    *   **Traffic Volumes (TVs)**: A GeoJSON file where each feature is a polygon representing an airspace volume, with properties for minimum and maximum flight levels (`min_fl`, `max_fl`).
    *   **Representative Routes**: A collection of CSV files, where each file contains a list of flight routes defined as a string of waypoints (e.g., "WP1 WP2 WP3...").

2.  **Initialize TVTW Indexer**: It creates or loads a `TVTWIndexer`. If an indexer file doesn't exist, it builds a new one by cataloging all `traffic_volume_id`s from the TV GeoJSON and creating indices for each one across all time bins in a 24-hour period.

3.  **Generate 4D Trajectory**: For each individual route string, the script calls the `get_4d_trajectory` function. This function uses the waypoint graph and an aircraft performance model to simulate the flight, generating a detailed 4D trajectory. The output is a sequence of points, each with a latitude, longitude, altitude, and timestamp.

4.  **Sample the Trajectory**: The 4D trajectory is a series of connected line segments. To ensure all interactions are captured, the algorithm samples points at a fixed distance (e.g., every 5 nautical miles) along each of these segments. For each sample point, it interpolates the altitude and timestamp.

5.  **Identify TV Intersections**: At each sample point, the script performs a check:
    *   It finds all Traffic Volume polygons that contain the point's latitude and longitude.
    *   It filters this list to keep only the volumes where the point's altitude falls between the volume's `min_fl` and `max_fl`.

6.  **Map to TVTW Index**: For each matching TV found in the previous step:
    *   The sample point's timestamp is used to determine the corresponding time window index (e.g., 2:17 PM falls into the 14:15-14:30 time window).
    *   The `TVTWIndexer` is used to retrieve the unique integer index for that specific (TV, time window) combination.

7.  **Construct the Vector**: All unique TVTW indices collected along the entire trajectory are gathered into a list. This list is then sorted to create the final, canonical impact vector for the route.

8.  **Parallel Processing**: To accelerate the computation for a large number of routes, the script uses Python's `multiprocessing` library to perform these calculations in parallel across multiple CPU cores.

9.  **Save Results**: The final output is a JSON file containing a dictionary that maps each route string to its computed impact vector.

## Inputs

-   **Routes Directory (`--routes_dir`)**: Path to a directory containing `.csv` files. Each file should have a 'route' column with waypoint strings.
    -   *Default*: `output/city_pairs/representatives_filtered`
-   **Traffic Volumes Path (`--tv_path`)**: Path to a GeoJSON file defining the 3D traffic volumes.
    -   *Default*: `D:/project-cirrus/cases/traffic_volumes_simplified.geojson`
-   **Graph Path (`--graph_path`)**: Path to the waypoint network graph in GML format.
    -   *Default*: `D:/project-akrav/data/graphs/ats_fra_nodes_only.gml`

## Outputs

-   **Impact Vectors (`--output_path`)**: A JSON file mapping each route string to its impact vector (a list of integers).
    -   *Default*: `output/impact_vectors.json`
-   **TVTW Indexer (`--tvtw_indexer_path`)**: A JSON file that stores the `TVTWIndexer` object's state. This allows for consistent indexing across different runs.
    -   *Default*: `output/tvtw_indexer.json`

## Usage Example

To run the script, use the following command structure, adjusting the paths as necessary.

```bash
python src/project_tailwind/impact_eval/impact_vector_computation.py \
  --routes_dir output/city_pairs/representatives_filtered \
  --tv_path cases/traffic_volumes_simplified.geojson \
  --output_path output/impact_vectors.json \
  --tvtw_indexer_path output/tvtw_indexer.json
```
