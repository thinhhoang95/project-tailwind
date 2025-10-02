Given the computed entry/exit times and occupancy counts for all the flights, the goal is to implement several new data structures like `delay_assignment_table`, a `delta_occupancy_view` and an inherited class FlightListWithDelta to store the "incremental changes" to support rapid recomputation of occupancy and entry/exit time values.

# Original so6_occupancy_matrix format:
```json
"263854795": { // flight identifier
        "occupancy_intervals": [
            {
                "tvtw_index": 33504, // traffic volume - time slot (called a cell, or tvtw for traffic volume time window)
                "entry_time_s": 0.0, // since takeoff
                "exit_time_s": 38.0 // since takeoff
            },
            {
                "tvtw_index": 33408,
                "entry_time_s": 0.0,
                "exit_time_s": 38.0
            },
            {
                "tvtw_index": 30720,
                "entry_time_s": 0.0,
                "exit_time_s": 307.0
            },
            {
                "tvtw_index": 33120,
                "entry_time_s": 0.0,
                "exit_time_s": 307.0
            },...
}
```

The traffic volume time window index could be looked up using the `tvtw_indexer` module, using the corresponding traffic volume to index `json` file:

```python
for tv_name, tv_idx in self._tv_id_to_idx.items(): # how to compute the tvtw index from the traffic volume index
    for time_idx in range(self.num_time_bins):
        # The global index is calculated based on the traffic volume's index and the time bin's index.
        global_idx = tv_idx * self.num_time_bins + time_idx
        tvtw_tuple = (tv_name, time_idx)
        self._tvtw_to_idx[tvtw_tuple] = global_idx
        self._idx_to_tvtw[global_idx] = tvtw_tuple
```

# Delay application

Following the application of a regulation (essentially a tuple of control traffic volume, active regulation time frame (e.g., 11:00-12:15), the flight list, and the allowed entry rate (rolling-hour entrance count), the slot allocation algorithm will assign concrete delay values in minutes to each flight in the flight list. There could be zero delay assignment, but we will only admit a minimum delay of 1 minute as meaningful.

For this, a simple data structure could be proposed (for example: a class based on dictionary) that stores the flight identifier and the concrete delay value (in minutes). The minimum delay value is 1. We should also add some methods to allow import/export this from/to a file.

# The occupancy view

In the second step, we need to create another class to describe the `delta_occupancy_view`. This object will contain the following information:

- For each flight: the new entry/exit time (in seconds since takeoff). Store the new, concrete time values (not the delta seconds since the old value).

- We also store a **sparse** vector for all TVTWs, the change in occupancy compared to the old matrix. To obtain this, we first have to compute the new occupancy counts based on the delayed flights, then we subtract the old occupancy counts.

# The FlightListWithDelta class
1. The FlightListWithDelta class will inherit the `flight_list.py` class `FlightList`, with a new method called `step_by_delay`. When this method is called with one or several `delta_occupancy_view`, it will update using vectorization, the occupancy matrix, the metadata, so that the existing methods function as if the whole flight list was precomputed. 

2. There will be additional properties that could be retrieved quickly: the number of regulations (number of delta_occupancy_view stored), the number of flights get delayed, the total delay assigned (in minutes), and the histogram of delay distribution for statistical insights that will be used by downstream applications. 

> Ideally, these properties/variables should be updated after the last `delta_occupancy_view` had been incorporated. 

## RegulationHistory class
We also need to implement a Helper class called `RegulationHistory` in order to keep track with past submitted regulation. Each regulation will have an identifier so we can cross-refer to the corresponding `delta_occupancy_view` that it creates. Please just implement this as a placeholder class for now (the history part), with just the identifier, we will wire in what a "regulation" looks like later from the code.

## Requirements for the FlightListWithDelta class
- Efficiency is of paramount importance, we have to massively leverage vectorization to speed up this process. If this "step" is slow, it will ruin the optimization loop on top of this.

- Be careful to maintain backward compatibility with the current `FlightList` class: do not alter the other methods, if you need similar methods but cannot immediately reuse what are already implemented, please create new methods and use them instead.

- Remember to leave comments throughout the code.