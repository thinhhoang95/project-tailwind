Please help me implement a simplified version of the CASA algorithm described in detail below. For reference, this algorithm will be called Continuous Computer Assisted Slot Allocation, or C-CASA.

# Algorithm Details
## Inputs
- `window_length_min`: the size of the window, in minutes. Typically `20` minutes.
- `window_stride_min`: the stride of the rolling window, typically set to `10` minutes.
- `rate`: the flow rate allowed to the traffic volume.

## Outputs
- A `revised_takeoff_time` for each flight.

## Steps
We will keep track of each takeoff time for each flight.

1. It starts with a rolling window starting from midnight of the day, of length `window_length_min`.
2. It will count the flights in each the window, if the number of flight exceeds the given `rate`, it will keep the first `rate` number of flights, and try to move the whole "train of flights" starting from the order of `rate+1` to after the end of the current rolling window, by adding to the initial provisional takeoff time the **remaining time in the rolling window at flight order `rate`**
3. The the window slides to the right an amount of `window_stride_min` and the whole process restarts.

## Instructions
1. First compute the window counts by writing the `window_count.py` module, based on `flight_counter_tv.py` (which is an orchestrator for `flight_counter_tv_large_segment.py`) **for each rolling window of size `window_length_min` and stride `window_stride_min`, for each traffic volume**, we initialize the delay assignment process.
2. After each window counts are computed, we flag the traffic-volume-time-window as hotspot if the count **exceeds capacity value divided by 60/(window_length_min)**.
3. If the traffic-volume-time-window is flagged as a hotspot, we assign the delay to the flight according to the *Steps* section above. We override the delay when the new delay is bigger than the previous one. The unit for delay is minute.
4. Finally, you save everything in one csv file, with two columns: `flight_identifier`, and `delay_min`.

# Requirements
- You may need to read the `flight_counter_tv.py` first to know how to call the `flight_counter` module.
- You first plan to implement the rolling window.
- You also plan how to keep track of the states and variables. Use appropriate data structure.
- You can use todo tool to keep track of your progress.
- As the world's best software developer, you are required to write high-quality code with no bugs. You also implement everything faithfully, without simplifying anything without request.

# Context
### Assumptions
We will assume that C-CASA is always active. This is a deviation from the real world implementation:
- In real world, the network managers ought to create a regulation first.
- A regulation will set the rate on specific traffic flows, not on the traffic volume itself.

However, analyzing the traffic flows require deep expertise about operational situations. For research, we will implement a simpler version to demonstrate the effectiveness of the approach, and this version is more in line with the flow management models typically found in literature.