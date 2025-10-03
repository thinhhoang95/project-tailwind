# The Ultimate goal

The ultimate goal is to extend the current code written in `regen_second_order.py` to be able to successfully derive the second order regulation proposals.

In the current code context, we were given a hotspot cell (traffic volume plus time window), and we managed to derive a set of regulation proposals thanks to the `regen` module. After the implementation, we must be able to pick the best regulation (in terms of objective improvement), apply that regulation, then the state of the network changed as a result of regulation (delay) application, then we run to get proposals again. The whole point is to ensure the whole chain from proposal to transition clicks together.

# Instructions

In the extension, I would like to implement these changes:

1. We use `hotspot_segment_extractor.py` module to extract the hotspots. Pick the **third ranked hotspot** in terms of `max_excess` (instead of pre-choosing the hotspot like in the current code).

2. Run `regen`, pick the best regulation in terms of objective improvement. The regulation plan is then converted into the form of a `DFRegulationPlan` (in `src/parrhesia/actions/regulations.py`).

3. We apply the `DFRegulationPlan` by calling `step_by_delay` of `FlightListWithDelta` in `resources.py`. This will result in the resources' flight list get updated.

4. The capacities_by_tv remains unchanged, the flight list is changed. We may need to recompute the hotspots, the flows and re-run `regen` one more time.

# Pitfalls

- After implementation, you shall need to verify whether the "delay_step" in the flight list might need to invalidate/rebuild cache anywhere in the code.

- Keep commenting your code thoroughly.

- Add a debug_verbose flag, if Active then:

    - After the first hotspot extraction, show the top 5 hotspots (in terms of TVID, time window and max exceedances).
    - Print the delay assignment table after the DFRegulationPlan had been applied. 
    - Print out the effects of the first few flights (say 3 flights): for each flight, show the first 3 traffic volumes and show how entry/exit times have shifted.
    - Check that there is indeed change in the occupancy matrix (print some statistics to verify this).
    - After the second hotspot extraction, show the top 5 hotspots. 