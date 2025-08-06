# FlightList Module

The `flight_list.py` module provides the `FlightList` class, which is designed to load, manage, and process flight occupancy data from SO6 flight data files. It represents the flight data as a sparse matrix, where each row corresponds to a flight and each column represents a specific Traffic Volume Time Window (TVTW). This sparse representation is highly efficient for handling large-scale aviation datasets.

## Core Concepts

-   **Occupancy Matrix**: A matrix where `matrix[i, j] = 1` if flight `i` occupies TVTW `j`, and `0` otherwise. The matrix is implemented as a `scipy.sparse.csr_matrix` for memory and computational efficiency.
-   **TVTW (Traffic Volume Time Window)**: A specific traffic volume (a segment of airspace) during a discrete time interval (e.g., 15 minutes).
-   **TVTW Indexer**: A mapping that provides information about the TVTWs, such as the time bin duration and the mapping from traffic volume IDs to indices in the occupancy matrix.

## The `FlightList` Class

The `FlightList` class is the primary component of this module. It handles the loading of data, construction of the sparse occupancy matrix, and provides methods to query and manipulate the flight data.

### Initialization

To create an instance of the `FlightList` class, you need to provide paths to two JSON files: the occupancy data and the TVTW indexer.

```python
from project_tailwind.optimize.eval.flight_list import FlightList

occupancy_file = 'path/to/so6_occupancy_matrix_with_times.json'
tvtw_indexer_file = 'path/to/tvtw_indexer.json'

flight_list = FlightList(
    occupancy_file_path=occupancy_file,
    tvtw_indexer_path=tvtw_indexer_file
)
```

### Input Files

1.  **`so6_occupancy_matrix_with_times.json`**: This JSON file contains the core flight data. It's a dictionary where keys are flight IDs. Each value is another dictionary containing:
    -   `takeoff_time`: The flight's takeoff time in ISO format.
    -   `origin`: The origin airport code.
    -   `destination`: The destination airport code.
    -   `distance`: The flight distance.
    -   `occupancy_intervals`: A list of intervals where the flight occupies a TVTW. Each interval includes `tvtw_index`, `entry_time_s`, and `exit_time_s`.

2.  **`tvtw_indexer.json`**: This file provides metadata for the TVTWs. It contains:
    -   `time_bin_minutes`: The duration of each time bin in minutes (e.g., 15).
    -   `tv_id_to_idx`: A mapping from traffic volume IDs to their base index in the occupancy matrix.

### Key Methods and Usage

#### Getting Occupancy Data

-   **`get_occupancy_vector(flight_id)`**: Returns a dense 1D NumPy array representing the occupancy for a single flight across all TVTWs.

    ```python
    flight_id = 'UAL123'
    occupancy_vector = flight_list.get_occupancy_vector(flight_id)
    print(occupancy_vector)
    # Output: [0. 0. 1. 1. 1. 0. ... ]
    ```

-   **`get_total_occupancy_by_tvtw()`**: Returns a 1D NumPy array showing the total number of flights occupying each TVTW. This is useful for calculating network load.

    ```python
    total_load = flight_list.get_total_occupancy_by_tvtw()
    print(total_load)
    # Output: [ 5.  8. 12. 15. 11. ... ]
    ```

-   **`get_flights_in_tvtw(tvtw_index)`**: Returns a list of flight IDs that occupy a specific TVTW.

    ```python
    tvtw_idx = 1024
    flights = flight_list.get_flights_in_tvtw(tvtw_idx)
    print(flights)
    # Output: ['AAL456', 'DAL789', ...]
    ```

#### Accessing Flight Metadata

-   **`get_flight_metadata(flight_id)`**: Retrieves a dictionary of metadata for a specific flight, including takeoff time, origin, destination, and detailed occupancy intervals.

    ```python
    metadata = flight_list.get_flight_metadata('SWA789')
    print(metadata)
    # Output:
    # {
    #     'takeoff_time': datetime.datetime(...),
    #     'origin': 'LAX',
    #     'destination': 'JFK',
    #     'distance': 2475,
    #     'occupancy_intervals': [...]
    # }
    ```

#### Simulating Delays

-   **`shift_flight_occupancy(flight_id, delay_minutes)`**: Simulates a flight delay by shifting its occupancy vector forward in time. This is a powerful tool for "what-if" analysis. The method returns a new, shifted occupancy vector but does not modify the original matrix.

    ```python
    # Simulate a 30-minute delay for a flight
    original_vector = flight_list.get_occupancy_vector('JBU101')
    delayed_vector = flight_list.shift_flight_occupancy('JBU101', delay_minutes=30)
    ```

-   **`update_flight_occupancy(flight_id, new_occupancy_vector)`**: Updates the main occupancy matrix with a new vector for a flight. This can be used to apply a simulated delay.

    ```python
    flight_list.update_flight_occupancy('JBU101', delayed_vector)
    ```

#### Summary Statistics

-   **`get_summary_stats()`**: Provides a dictionary of summary statistics about the dataset, including the number of flights, TVTWs, matrix sparsity, and average/max occupancy.

    ```python
    stats = flight_list.get_summary_stats()
    print(stats)
    # Output:
    # {
    #     'num_flights': 5000,
    #     'num_tvtws': 25000,
    #     'matrix_sparsity': 0.998,
    #     ...
    # }
    ```

## Algorithm: Sparse Matrix Construction

The core of the `FlightList` class is the `_build_occupancy_matrix` method. It constructs the `scipy.sparse.csr_matrix` efficiently.

1.  It initializes three empty lists: `row_indices`, `col_indices`, and `data`.
2.  It iterates through each flight in the input data. Each flight is assigned a unique row index.
3.  For each flight, it iterates through its `occupancy_intervals`.
4.  For each interval, it appends the flight's row index to `row_indices`, the `tvtw_index` to `col_indices`, and the value `1.0` to `data`.
5.  Finally, it creates the CSR matrix from these lists: `sparse.csr_matrix((data, (row_indices, col_indices)), ...)`.

This "coordinate" format (COO) initialization is an efficient way to build a sparse matrix when the locations of the non-zero elements are known. The CSR (Compressed Sparse Row) format is then used for efficient row slicing and matrix-vector products, which are common operations in the evaluator.
