# CASA Module Documentation

## Overview

The `casa` module provides a re-adapted implementation of the C-CASA (Constrained Collaborative Airspace Slot Allocation) algorithm. This simulation tool is designed to model and calculate delays for a set of flights subject to a single traffic volume regulation at a specific airspace volume (referred to as the `reference_location`).

The core of the algorithm is a rate-limiting queue model. It processes flights scheduled to enter the regulated airspace, counts them in rolling time windows, and if the count exceeds the defined capacity (e.g., an hourly rate), it "pushes" the excess flights to enter at a later time, thereby assigning delay.

## Core Algorithm

The simulation follows a multi-step process to calculate delays.

### 1. Flight Eligibility and Event Generation

- **Data Loading**: The process begins by loading flight trajectory data from an SO6 (Schedule of Operations, version 6) occupancy JSON file. This file contains detailed 4D trajectory information for each flight, including which Traffic Volume/Time Windows (TVTWs) they occupy and their entry/exit times for each.
- **Filtering Flights**: The algorithm filters the master list of flights (`identifier_list`) to find those that are "eligible" for the regulation. A flight is considered eligible if its trajectory intersects with the specified `reference_location` (a Traffic Volume, or TV) during one of the `active_time_windows`.
- **Determining Entry Time**: For each eligible flight, the algorithm finds its *first entry time* into the `reference_location`. This is the absolute timestamp when the flight first crosses the boundary of the regulated airspace volume.
- **Creating CASA Events**: Each eligible flight is converted into a `casa_event` dictionary. This structure holds the flight's ID, its original entry time (`t_entry_orig`), and a revised entry time (`t_entry_star`) which is initially the same as the original time but will be updated if a delay is assigned.

```
casa_event = {
    "flight_id": "FL123",
    "t_entry_orig": datetime.datetime(...), // Original calculated entry time
    "t_entry_star": datetime.datetime(...), // Revised entry time, subject to delay
    "tv_id": "ZNY_50"                     // The regulated airspace volume
}
```

### 2. Rolling Window Generation

- The simulation timeline, defined by the start of the first active time window and the end of the last one, is divided into a series of overlapping "C-CASA windows".
- The function `generate_casa_windows` creates these windows based on a specified `window_length_min` and `window_stride_min`. For example, with a length of 20 minutes and a stride of 10 minutes, the windows would be [00:00, 00:20], [00:10, 00:30], [00:20, 00:40], etc.

### 3. Capacity and Delay Assignment Loop

This is the core of the simulation where delays are calculated. The algorithm iterates through each C-CASA window in chronological order.

- **Sorting**: Before each window is processed, the entire list of `casa_events` is sorted by their current revised entry time (`t_entry_star`). This is crucial because a flight delayed from a previous window might now fall into the current one.
- **Identifying Entrants**: The algorithm identifies all flights whose `t_entry_star` falls within the current window's start and end times.
- **Calculating Window Capacity**: The capacity for the window is calculated based on the `hourly_rate`. A fractional component (`carry`) is maintained between windows to ensure that, over time, the total capacity accurately reflects the hourly rate, even with short window lengths.
  ```python
  # Example logic
  capacity_per_window_fractional = hourly_rate * (window_length_min / 60.0)
  carry += capacity_per_window_fractional
  capacity = int(carry) # The whole number of flights allowed
  carry -= capacity     # The remainder is carried to the next window
  ```
- **Assigning Delays**: The `assign_delays` function is called with the list of `entrants` and the calculated `capacity`.
  - If `len(entrants) <= capacity`, no action is needed. All flights can enter as scheduled.
  - If `len(entrants) > capacity`, the flights are processed in a First-In, First-Out (FIFO) manner. The first `capacity` number of flights are allowed. The remaining (`excess`) flights are delayed.
  - The `t_entry_star` for each excess flight is updated to be slightly after the current window's end time (`window_end`). This "pushes" them out of the current window, ensuring they will be re-evaluated in a subsequent window.

### 4. Final Delay Calculation

- After all windows have been processed, the final delay for each flight is calculated as the difference between its final revised entry time (`t_entry_star`) and its original entry time (`t_entry_orig`).
- The result is returned as a dictionary mapping each flight ID to its calculated delay in minutes.

## Functions

### `run_readapted_casa(...)`

This is the main entry point for the C-CASA simulation. It orchestrates the entire process from data loading to final delay calculation.

**Parameters:**
- `so6_occupancy_path` (str): The file path to the JSON file containing SO6 flight occupancy data.
- `identifier_list` (List[str]): A list of flight IDs to be considered in the simulation.
- `reference_location` (str): The identifier of the Traffic Volume (TV) where the regulation is applied.
- `tvtw_indexer` (TVTWIndexer): An instance of the `TVTWIndexer` class, used to decode TVTW indices from the occupancy data.
- `hourly_rate` (float): The maximum number of flights allowed to enter the `reference_location` per hour.
- `active_time_windows` (List[int]): A list of integer indices representing the time windows (bins) during which the regulation is active.
- `ccasa_window_length_min` (int, optional): The duration of each rolling C-CASA window in minutes. Defaults to 20.
- `window_stride_min` (int, optional): The time step between the start of consecutive C-CASA windows in minutes. Defaults to 10.

**Returns:**
- `Dict[str, float]`: A dictionary where keys are flight IDs and values are the calculated delays in minutes. Flights not eligible or not delayed will have a value of 0.0.

---

### `assign_delays(...)`

This function implements the core delay logic for a single C-CASA window. It modifies the entry times of flights that exceed the window's capacity.

**Parameters:**
- `entrants` (List[Dict[str, Any]]): A list of CASA event dictionaries for flights scheduled to enter in the current window, sorted by their `t_entry_star`.
- `capacity` (int): The maximum number of flights allowed in this window.
- `window_end` (datetime.datetime): The end time of the current window.
- `epsilon_s` (int, optional): A small buffer in seconds to push delayed flights beyond the window's end time, preventing them from landing exactly on the boundary. Defaults to 1.

**Returns:**
- `List[Dict[str, Any]]`: The input list of `entrants` with `t_entry_star` updated for delayed flights. The list is modified in-place.

---

### `generate_casa_windows(...)`

A helper function to generate the list of rolling time windows for the simulation.

**Parameters:**
- `start_time` (datetime.datetime): The absolute start time for the first window.
- `end_time` (datetime.datetime): The simulation end time; windows will not be generated beyond this point.
- `window_length_min` (int): The length of each window in minutes.
- `window_stride_min` (int): The stride between consecutive windows in minutes.

**Returns:**
- `List[Tuple[datetime.datetime, datetime.datetime]]`: A list of tuples, where each tuple represents the start and end time of a window.

## Input Data Format

### SO6 Occupancy JSON (`so6_occupancy_path`)

The primary input is a JSON file where each key is a flight ID. The value is an object containing flight details, most importantly the `occupancy_intervals`.

```json
{
  "FL123": {
    "takeoff_time": "2023-01-01T10:00:00Z",
    "occupancy_intervals": [
      {
        "tvtw_index": 12345, // Encoded TV and Time Window index
        "entry_time_s": 600, // Seconds from takeoff to enter this TV
        "exit_time_s": 900   // Seconds from takeoff to exit this TV
      },
      // ... more intervals
    ]
  },
  "FL456": {
    // ...
  }
}
```

## Output Format

The simulation returns a dictionary mapping every flight ID from the initial `identifier_list` to a float representing the calculated delay in minutes.

```json
{
  "FL123": 15.5, // 15.5 minutes of delay
  "FL456": 0.0,  // No delay
  "FL789": 0.0   // Not eligible for the regulation, no delay
}
```

## Usage Example

```python
import datetime
import pandas as pd
from project_tailwind.casa.casa import run_readapted_casa
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

# 1. Initialize the TVTWIndexer
# This typically requires loading TV and airport metadata.
# (Assuming tv_meta_df and airport_meta_df are loaded pandas DataFrames)
# tvtw_indexer = TVTWIndexer(tv_meta_df, airport_meta_df, time_bin_minutes=15)

# For demonstration, we can mock it if its structure is known
class MockTVTWIndexer:
    def __init__(self, time_bin_minutes=15):
        self.time_bin_minutes = time_bin_minutes
    def get_tvtw_index(self, tv_id, tw_idx):
        # Dummy implementation
        return f"{tv_id}_{tw_idx}"
    def get_tvtw_from_index(self, index):
        # Dummy implementation
        parts = index.split('_')
        return parts[0], int(parts[1])

tvtw_indexer = MockTVTWIndexer(time_bin_minutes=15)


# 2. Define simulation parameters
so6_file = "path/to/your/so6_occupancy.json"
flights = ["FL123", "FL456", "FL789", "FL001", "FL002"]
regulated_tv = "ZNY_50"
rate = 10.0  # flights per hour
active_windows = list(range(40, 48)) # e.g., from 10:00 to 12:00 if time_bin is 15 min

# 3. Run the simulation
delays = run_readapted_casa(
    so6_occupancy_path=so6_file,
    identifier_list=flights,
    reference_location=regulated_tv,
    tvtw_indexer=tvtw_indexer,
    hourly_rate=rate,
    active_time_windows=active_windows,
    ccasa_window_length_min=20,
    window_stride_min=10
)

# 4. Print results
print(delays)
```
