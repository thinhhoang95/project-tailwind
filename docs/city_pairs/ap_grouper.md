# `ap_grouper` Module Documentation

The `ap_grouper` module is a comprehensive suite of tools designed for trajectory analysis, specifically focusing on grouping flight trajectories into meaningful communities. It leverages various techniques such as distance matrix computation, community detection, and representative trajectory selection to identify common flight paths from a given dataset.

The entry point for this module is `grouper_auto.py`, which automates the process of reading flight data, processing it by airport pairs, and extracting representative trajectories.

## Core Concepts

The module operates on the following core concepts:

- **Trajectory:** A flight path represented as a sequence of waypoints.
- **Distance Metric:** The Fréchet distance is used to measure the similarity between two trajectories.
- **Community Detection:** The Leiden algorithm is employed to group similar trajectories into communities.
- **Representative Trajectory:** For each community, a representative trajectory is selected, which is the most central trajectory within that community.

## Workflow

The overall workflow of the `ap_grouper` module is as follows:

1.  **Load Data:** Flight data is loaded from a CSV file.
2.  **Process by Airport Pair:** The data is grouped by origin-destination airport pairs.
3.  **Compute Distance Matrix:** For each airport pair, a pairwise distance matrix is computed for all trajectories using the Fréchet distance.
4.  **Build Graph:** A graph is constructed from the distance matrix, where an edge is created between two trajectories if their distance is below a certain threshold.
5.  **Detect Communities:** The Leiden algorithm is applied to the graph to detect communities of trajectories.
6.  **Find Representatives:** For each community, a representative trajectory is identified.
7.  **Filter Spurious Routes:** A final filtering step removes routes that are significantly longer than the great-circle distance between the origin and destination.

## Modules and Functions

### `grouper_auto.py`

This is the main script that orchestrates the entire trajectory grouping process.

#### `main()`

The main function that:
- Defines input and output directories.
- Iterates through all CSV files in the input directory.
- For each file, it loads the flight data and calls `process_all_pairs_and_get_representatives` to perform the analysis.

#### `process_all_pairs_and_get_representatives(df, output_path, pool)`

- Takes a DataFrame of flights, an output path, and a multiprocessing pool as input.
- Finds all unique airport pairs in the DataFrame.
- Iterates through each pair and processes them to find representative trajectories.
- If a pair has fewer than `MIN_FLIGHTS`, it keeps the unique trajectories.
- Otherwise, it calls `process_airport_pair` to perform community detection.
- Saves the representative trajectories to the output CSV file.

**Usage Example:**

```python
import pandas as pd
import multiprocessing as mp

# Assuming flights_df is a pandas DataFrame with flight data
# and WorkerInitializer is properly defined and initialized
with mp.Pool(processes=N_PROCESSES, initializer=WorkerInitializer.initialize) as pool:
    process_all_pairs_and_get_representatives(flights_df, "output/representatives.csv", pool)
```

### `grouper.py`

This module contains the core logic for processing a single airport pair and performing community detection.

#### `process_airport_pair(df, origin, destination, pool, ...)`

- Extracts all trajectories for a given origin-destination pair.
- If the number of trajectories is sufficient, it computes the distance matrix using `compute_trajectory_distances`.
- It then calls `group_trajectories_into_communities` to perform community detection.
- Finally, it formats the output into a DataFrame.

**Input:**

- `df`: A pandas DataFrame with flight data.
- `origin`: The origin airport code (e.g., "EGLF").
- `destination`: The destination airport code (e.g., "LFPB").
- `pool`: A multiprocessing pool.

**Output:**

A pandas DataFrame with the following columns:
- `trajectory`: The trajectory string.
- `community_id`: The ID of the community the trajectory belongs to.
- `is_representative`: A boolean flag indicating if the trajectory is a representative.
- `representative_trajectory`: The representative trajectory for the community.

### `compute_distance_frdist.py`

This module is responsible for computing the distance between trajectories.

#### `compute_distance_matrix(trajectories, distance_metric, resampling_n)`

- Computes the pairwise distance matrix for a list of trajectories.
- It uses the specified `distance_metric` (default is "frechet").
- Trajectories are resampled to `resampling_n` points before computing the distance.

**Input:**

- `trajectories`: A list of trajectory strings.
- `distance_metric`: The distance metric to use ("frechet" or "great_circle").
- `resampling_n`: The number of points to resample each trajectory to.

**Output:**

A NumPy array representing the pairwise distance matrix.

#### `frechet_distance(path1, path2, resampling_n)`

- Computes the Fréchet distance between two paths.
- The paths are first resampled to `resampling_n` points.

**Input:**

- `path1`: A list of (latitude, longitude) tuples.
- `path2`: A list of (latitude, longitude) tuples.
- `resampling_n`: The number of points to resample each trajectory to.

**Output:**

The Fréchet distance as a float.

### `community_detection.py`

This module handles the community detection part of the workflow.

#### `build_graph_from_distance_matrix(distance_matrix, threshold)`

- Builds a graph from a distance matrix.
- An edge is created between two nodes (trajectories) if their distance is below the `threshold`.
- The edge weight is the inverse of the distance.

**Input:**

- `distance_matrix`: A NumPy array representing the pairwise distance matrix.
- `threshold`: The distance threshold for creating an edge.

**Output:**

An `igraph.Graph` object.

#### `detect_communities(graph)`

- Detects communities in a graph using the Leiden algorithm.

**Input:**

- `graph`: An `igraph.Graph` object.

**Output:**

A tuple containing the partition and a dictionary of communities.

#### `get_representative_trajectories(communities, distance_matrix)`

- Finds the most representative trajectory for each community.
- The representative is the one with the minimum total distance to all other trajectories in the same community.

**Input:**

- `communities`: A dictionary of communities.
- `distance_matrix`: The pairwise distance matrix.

**Output:**

A dictionary mapping each community ID to the index of its representative trajectory.

### `filter_spurious_routes.py`

This module provides functionality to filter out spurious routes from the results.

#### `main()`

- The main function that loads the representative routes and filters them.
- It calculates the great-circle distance and the actual route distance for each representative.
- If the route distance is more than 150% of the great-circle distance, the route is considered spurious and removed.

**Usage:**

This script is intended to be run after the `grouper_auto.py` script to clean up the results.

```bash
python src/project_tailwind/city_pairs/ap_grouper/filter_spurious_routes.py
```

This will read the representative files from `output/city_pairs/representatives` and save the filtered results to `output/city_pairs/representatives_filtered`.
