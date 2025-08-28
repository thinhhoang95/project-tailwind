Please plan for the implementation of `RegulationPlanSimulation` API endpoint. 

# Instructions

Goal: Given the *regulation plan*, which is a set of *regulations* that comprise of three parts: a reference (usually the hotspot) traffic volume, the list of flights being targeted, and the **hourly** rate for our FCFS queue. Then obtain the "post-regulation" results, detailed below:

- Use the `PlanEvaluator` to compute the delay for each targetted flight given the plan and retrieve the new evaluation results, including the excess vector, delay stats, objective and objective components.

- Compute the pre-regulation and post-regulation occupancy count for 25 busiest traffic volumes pre-regulation (busy-ness defined as count - hourly capacity)

- Similarly, compute the pre-regulation and post-regulation occupancy count for 25 busiest traffic volumes post-regulation (busy-ness defined as count - hourly capacity).

# Outputs

- The delay assignment for each flight.
- Extend the `plan_evaluator` if needed to (e.g., computation of post regulation occupancy count), but the extension needs to base on delta view to prevent excessive recomputation. 
- Use vectorization for efficiency.
- For hours that don't have capacity, skip over them rather than assuming a value.
- Return rolling-hour occupancy arrays for top-K busiest traffic volumes:
  - Busiest TVs are selected by the highest max of (pre_rolling_count - hourly_capacity) over the active regulation time windows.
  - For each selected TV, return full-length arrays (size = bins_per_tv, e.g., 96 for 15-min bins) for both pre- and post-regulation rolling-hour counts, plus the hourly capacity per bin and the set of active time windows.
  - Rolling-hour count at bin i is the sum of occupancies over the next W bins where W = 60 / time_bin_minutes (forward-looking window). Bins whose starting hour has no capacity are excluded from the ranking (do not impute).