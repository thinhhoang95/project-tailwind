# NetworkEvaluator Module

The `network_evaluator.py` module provides the `NetworkEvaluator` class, which is used to assess air traffic network congestion. It works by comparing the flight demand (occupancy) from a `FlightList` object against the defined capacities of traffic volumes. Its primary output is an "excess traffic vector," which quantifies the overload in each Traffic Volume Time Window (TVTW) after comparing hourly demand to hourly capacity and redistributing any hourly excess back to constituent bins proportionally to per-bin occupancy (with equal split fallback if all bins are zero).

## Core Concepts

-   **Excess Traffic**: The overload computed from hourly comparisons of demand vs capacity, redistributed to the finer TVTW bins that comprise each hour. In the core evaluator, redistribution is proportional to per-bin occupancy within the hour (equal split is used only if all per-bin occupancies are zero).
-   **Capacity Data**: The module expects capacity information for traffic volumes, typically defined on an hourly basis (e.g., "6:00-7:00": 23 flights). Internally, capacities are compared at the hourly level; no per-bin capacities are used for overload calculation.
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

Notes:
- Capacities may be provided as a JSON string or dictionary in the GeoDataFrame. They are parsed and stored as `float` hourly capacities per traffic volume.
- Time-bin resolution is inferred from `flight_list.time_bin_minutes`. The evaluator supports any divisor of 60 minutes (e.g., 5, 10, 15, 30, 60). The expected bins per TV per day are computed as `24 * (60 // time_bin_minutes)`.

### Key Methods and Usage

#### Computing Excess Traffic

-   **`compute_excess_traffic_vector()`**: This is the core method of the evaluator. It returns a 1D NumPy array where each element represents the excess traffic for the corresponding TVTW, after computing hourly excess and redistributing it proportionally to the per-bin occupancy within that hour for the corresponding traffic volume (equal split if all per-bin occupancies are zero).

    ```python
    excess_vector = network_evaluator.compute_excess_traffic_vector()
    print(excess_vector)
    # Example: if hourly excess is 5 and bin occupancies within the hour are [10, 0, 0, 0],
    # the proportional redistribution yields [5, 0, 0, 0] for that hour's bins.
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

-   **`compute_horizon_metrics(horizon_time_windows, percentile_for_z_max=95)`**: Calculates percentile-based peak overload (`z_95`, with the percentile configurable) and the total overload (`z_sum`) within a specified horizon from the start of day. The current implementation applies the horizon by slicing the first `K` bins of the flattened TVTW vector.

    ```python
    # Analyze the first 4 hours (assuming 15-min bins, so 16 windows)
    metrics = network_evaluator.compute_horizon_metrics(horizon_time_windows=16)
    print(metrics)
    # Example output: {'z_95': 3.0, 'z_sum': 22.0, 'horizon_windows': 16}
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

1.  **Get Total Occupancy (per TVTW)**: Call `flight_list.get_total_occupancy_by_tvtw()` to get a vector of flight counts for every TVTW. Internally, the evaluator also builds an hourly occupancy matrix of shape `(num_tvs, 24)` by summing the relevant bins per hour.
2.  **Compare Hourly Demand vs Capacity**: For each traffic volume and each hour present in the capacity table, compute `hourly_excess = max(0, hourly_occupancy - hourly_capacity)`.
3.  **Redistribute Hourly Excess to Bins**: For any positive `hourly_excess`, redistribute it proportionally across the TVTW bins of that hour based on each bin's occupancy within that hour (use equal split if all per-bin occupancies are zero). This preserves the total excess (`z_sum`) while assigning bin-level excess values that better reflect within-hour hotspots.
4.  **Return Per-Bin Excess Vector**: The result is a 1D vector aligned with the TVTW indexing used by `FlightList`.

Notes:
- The redistribution strategy in the core evaluator is proportional to per-bin occupancy (equal split only if all per-bin occupancies are zero). The API-facing evaluator in `src/server_tailwind/airspace/network_evaluator_for_api.py` currently uses equal split.
- The evaluator caches the last computed hourly occupancy matrix in `last_hourly_occupancy_matrix` for reporting.

This approach is efficient and aligns hourly capacity definitions with fine-grained TVTW analysis.

## Additional Utilities

-   **`get_overloaded_tvtws(threshold=0.0)`**: Returns detailed overload records per bin, including hourly occupancy/capacity context and utilization ratios.
-   **`get_hotspot_flights(threshold=0.0, mode='bin'|'hour')`**: Retrieves flight identifiers associated with hotspots, either per bin or grouped per (traffic_volume_id, hour).
-   **`get_capacity_utilization_stats()`**: Provides system-wide utilization metrics over bins with defined capacity.
-   **`compute_delay_stats()`**: Computes delay statistics by comparing current vs original takeoff times captured at evaluator initialization.
