# NetworkEvaluator Module

The `network_evaluator.py` module provides the `NetworkEvaluator` class, which is used to assess air traffic network congestion. It works by comparing the flight demand (occupancy) from a `FlightList` object against the defined capacities of traffic volumes. Its primary output is an "excess traffic vector," which quantifies the overload in each Traffic Volume Time Window (TVTW).

## Core Concepts

-   **Excess Traffic**: The number of flights exceeding the capacity of a specific TVTW. `Excess = Occupancy - Capacity`. A value greater than zero indicates an overload.
-   **Capacity Data**: The module expects capacity information for traffic volumes, typically defined on an hourly basis (e.g., "6:00-7:00": 23 flights). It automatically converts these hourly capacities to the more granular TVTW time bins used by `FlightList`.
-   **Vectorized Operations**: The evaluator heavily relies on NumPy for vectorized calculations, ensuring high performance when dealing with large datasets.

## The `NetworkEvaluator` Class

The `NetworkEvaluator` class orchestrates the entire evaluation process. It integrates flight data with capacity data to produce actionable insights about network hotspots.

### Initialization

To create an instance of the `NetworkEvaluator`, you need a GeoDataFrame containing traffic volume information (including capacities) and an initialized `FlightList` object.

```python
import geopandas as gpd
from project_tailwind.optimize.eval import FlightList, NetworkEvaluator

# 1. Load FlightList
flight_list = FlightList(
    occupancy_file_path='path/to/so6_occupancy_matrix_with_times.json',
    tvtw_indexer_path='path/to/tvtw_indexer.json'
)

# 2. Load Traffic Volume data
tv_gdf = gpd.read_file('path/to/traffic_volumes.geojson')

# 3. Initialize Evaluator
network_evaluator = NetworkEvaluator(
    traffic_volumes_gdf=tv_gdf,
    flight_list=flight_list
)
```

### Input Data

1.  **`traffic_volumes_gdf`**: A GeoDataFrame where each row represents a traffic volume. It must contain the following columns:
    -   `traffic_volume_id`: A unique identifier for the traffic volume.
    -   `capacity`: A dictionary (or a string representation of one) defining hourly capacities. Example: `{'6:00-7:00': 25, '7:00-8:00': 30}`.
    -   `geometry`: The geometric representation of the traffic volume.

2.  **`flight_list`**: An instance of the `FlightList` class, which has already loaded the flight occupancy data.

### Key Methods and Usage

#### Computing Excess Traffic

-   **`compute_excess_traffic_vector()`**: This is the core method of the evaluator. It returns a 1D NumPy array where each element represents the excess traffic for the corresponding TVTW.

    ```python
    excess_vector = network_evaluator.compute_excess_traffic_vector()
    print(excess_vector)
    # Output: [0. 0. 2. 5. 0. ... ]
    # (Indicates an overload of 2 flights in TVTW 2, and 5 in TVTW 3)
    ```

#### Identifying Overloaded Sectors

-   **`get_overloaded_tvtws(threshold=0.0)`**: Returns a list of dictionaries, each detailing an overloaded TVTW. This is more user-friendly than the raw excess vector.

    ```python
    overloaded_sectors = network_evaluator.get_overloaded_tvtws()
    for sector in overloaded_sectors:
        print(sector)
    # Output:
    # {
    #     'tvtw_index': 1234,
    #     'traffic_volume_id': 'ZAB_14',
    #     'occupancy': 15.0,
    #     'capacity': 12.0,
    #     'excess': 3.0,
    #     'utilization_ratio': 1.25
    # }
    ```

#### Horizon-Based Metrics

-   **`compute_horizon_metrics(horizon_time_windows)`**: Calculates `z_max` (the maximum excess traffic) and `z_sum` (the total excess traffic) within a specified time horizon from the start. These are key performance indicators for network health.

    ```python
    # Analyze the first 4 hours (assuming 15-min bins, so 16 windows)
    metrics = network_evaluator.compute_horizon_metrics(horizon_time_windows=16)
    print(metrics)
    # Output: {'z_max': 5.0, 'z_sum': 22.0, 'horizon_windows': 16}
    ```

#### System-Wide Statistics

-   **`get_capacity_utilization_stats()`**: Provides a dictionary of statistics about the overall system utilization, including mean/max utilization, total demand vs. capacity, and the percentage of overloaded TVTWs.

    ```python
    stats = network_evaluator.get_capacity_utilization_stats()
    print(stats)
    # Output:
    # {
    #     'mean_utilization': 0.85,
    #     'max_utilization': 1.75,
    #     'overloaded_tvtws': 42,
    #     ...
    # }
    ```

## Algorithm: Excess Traffic Calculation

The `compute_excess_traffic_vector` method follows these steps:

1.  **Get Total Occupancy**: It calls `flight_list.get_total_occupancy_by_tvtw()` to get a vector of flight counts for every TVTW.
2.  **Build Total Capacity Vector**:
    -   It initializes a NumPy array `total_capacity` with zeros, with the same length as the number of TVTWs.
    -   It iterates through the `capacity_by_tvtw` dictionary (which was prepared during initialization).
    -   For each traffic volume, it adds its capacity array to the `total_capacity` vector. This aggregates capacities from different traffic volumes that might apply to the same time windows.
3.  **Calculate Excess**: It performs a vectorized subtraction: `excess_vector = total_occupancy - total_capacity`.
4.  **Filter and Clean**:
    -   It sets the excess to `0` for any TVTW where capacity is not defined (i.e., `total_capacity == 0`). This prevents flagging occupancy as "excess" in unmanaged airspace.
    -   It applies `np.maximum(0, excess_vector)` to clamp all negative values (where occupancy is below capacity) to zero, as we are only interested in overloads.

This vectorized approach is highly efficient and avoids explicit loops over the thousands of TVTWs, making the evaluation process fast even for large-scale scenarios.
