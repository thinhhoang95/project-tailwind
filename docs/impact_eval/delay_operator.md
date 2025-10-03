# Delay Operator Documentation

The `delay_operator` function is a core component of the impact evaluation system. It simulates the effect of a ground delay on a flight's trajectory by adjusting its space-time representation, known as the occupancy vector.

## Algorithm Details

The operator works by shifting the time component of a flight's trajectory. A flight's path is represented as an "occupancy vector," which is a list of indices. Each index corresponds to a unique Traffic Volume/Time Window (TVTW) that the flight passes through. A TVTW represents a specific segment of airspace (a Traffic Volume, or TV) during a specific interval of time (a Time Window).

The algorithm performs the following steps:

1.  **Calculate Time Bin Shift**: The input `delay_min` (delay in minutes) is converted into an equivalent number of "time bins". This is done by dividing the delay by the duration of a single time bin (e.g., 15 minutes). This result, `time_bin_shift`, represents how many time slots forward the trajectory needs to be moved.

2.  **Handle No-Op Cases**: If the delay is zero or smaller than the resolution of a single time bin, no change is made, and the original occupancy vector is returned.

3.  **Iterate and Shift**: The function iterates through each TVTW index in the original `occupancy_vector`.
    *   For each index, it uses the `TVTWIndexer` to resolve the index into its constituent parts: a `tv_id` (the identifier for the airspace volume) and a `time_window_idx` (the index for the time interval).
    *   The `time_window_idx` is shifted forward by adding the `time_bin_shift`.
    *   The calculation includes a modulo operation (`% num_time_bins`) to handle cases where the delay pushes the time window into the next day, causing it to wrap around.

4.  **Re-assemble Vector**: The `tv_id` (which remains unchanged) and the `new_time_window_idx` are used to look up a new global TVTW index via the `indexer`.

5.  **Return New Vector**: These new indices are collected into a `new_occupancy_vector`, which represents the trajectory of the delayed flight. This new vector is then returned.

## Inputs and Outputs

### Inputs

-   `occupancy_vector` (`List[int]`): The original flight trajectory represented as a list of TVTW indices.
-   `delay_min` (`int`): The ground delay to be applied, in minutes.
-   `indexer` (`TVTWIndexer`): An initialized instance of the `TVTWIndexer` class, which is used to resolve and create TVTW indices.

### Outputs

-   `List[int]`: A new occupancy vector representing the delayed trajectory.

### Raises

-   `ValueError`: If a TVTW index from the input vector or a newly computed one cannot be resolved by the `indexer`, indicating a data inconsistency.

## Usage Example

```python
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.impact_eval.operators.delay import delay_operator

# 1. Assume an initialized TVTWIndexer
# For this example, let's say time bins are 15 minutes each.
# indexer = TVTWIndexer(...)
# indexer.time_bin_minutes = 15
# indexer.num_time_bins = 96 # 24 * (60 / 15)

# Let's mock the indexer for a runnable example
class MockIndexer:
    def __init__(self, time_bin_minutes, num_time_bins):
        self.time_bin_minutes = time_bin_minutes
        self.num_time_bins = num_time_bins

    def get_tvtw_from_index(self, index):
        # Mock implementation: tv_id is the index // 96, time_window is index % 96
        tv_id = index // self.num_time_bins
        time_window_idx = index % self.num_time_bins
        return tv_id, time_window_idx

    def get_tvtw_index(self, tv_id, time_window_idx):
        # Mock implementation
        return tv_id * self.num_time_bins + time_window_idx

indexer = MockIndexer(time_bin_minutes=15, num_time_bins=96)


# 2. Original flight trajectory (occupancy vector)
# Represents a flight passing through TV 1 at time 10, TV 2 at time 11, TV 3 at time 12
original_vector = [
    indexer.get_tvtw_index(1, 10), # 1*96 + 10 = 106
    indexer.get_tvtw_index(2, 11), # 2*96 + 11 = 203
    indexer.get_tvtw_index(3, 12)  # 3*96 + 12 = 300
]
print(f"Original Vector: {original_vector}")

# 3. Apply a 30-minute delay
delay_minutes = 30
# This corresponds to a shift of 30 / 15 = 2 time bins.

delayed_vector = delay_operator(
    occupancy_vector=original_vector,
    delay_min=delay_minutes,
    indexer=indexer
)

# The new vector should have the time indices shifted by 2
# Expected: [ (1, 12), (2, 13), (3, 14) ]
# Indices:  [ 1*96+12=108, 2*96+13=205, 3*96+14=302 ]
print(f"Delayed Vector: {delayed_vector}")

# Expected output:
# Original Vector: [106, 203, 300]
# Delayed Vector: [108, 205, 302]
```
