The objective value is indeed scoped, rather than being a "whole network value".

1.  **Filtering by Traffic Volume:** The process starts in `RateFinder._find_rates_locked`. A `tv_filter` is created that contains only the `control_volume_id` (the hotspot) being analyzed.

    ```290:290:src/parrhesia/flow_agent/rate_finder.py
    // ... existing code ...
                    tv_filter=[control_volume_id],
    // ... existing code ...
    ```

2.  **Scoped Context:** This `tv_filter` is used to build a `ScoreContext` object. This context, which is used for all subsequent calculations, is therefore scoped to just that single traffic volume. This happens within `_ensure_context_and_baseline` when it calls `build_score_context`.

    ```660:660:src/parrhesia/flow_agent/rate_finder.py
    // ... existing code ...
                        tv_filter=tv_filter,
    // ... existing code ...
    ```

3.  **Scoped Occupancy Calculation:** The scoring function `score_with_context` in `src/parrhesia/flow_agent/safespill_objective.py` uses this scoped context. It computes flight occupancy, which is a key input to the objective function, but *only* for the traffic volumes specified in `context.tvs_of_interest`.

    You can see this filtering in a few places:
    *   The comment at the top of the block:
        ```43:43:src/parrhesia/flow_agent/safespill_objective.py
        // ... existing code ...
        # Occupancy only for TVs of interest
        // ... existing code ...
        ```
    *   The loop that builds the final occupancy map:
        ```74:83:src/parrhesia/flow_agent/safespill_objective.py
        // ... existing code ...
            for tv in context.tvs_of_interest:
                tvs = str(tv)
                base_all = context.base_occ_all_by_tv.get(tvs, zeros)
                base_sched_zero = context.base_occ_sched_zero_by_tv.get(tvs, zeros)
                sched_cur = occ_sched.get(tvs, zeros)
                occ_by_tv[tvs] = (
                    base_all.astype(np.int64)
                    + sched_cur.astype(np.int64)
                    - base_sched_zero.astype(np.int64)
                )
        ```

4.  **Scoped Cost Components:**
    *   The capacity cost (`J_cap`) is then calculated using this scoped `occ_by_tv` map, so it only considers capacity violations within the specified hotspot.
    *   Other components of the objective function, like `J_delay`, are calculated based on the delays of a specific subset of flights—those that are entrants to the `control_volume_id` for the flows being regulated. The delay calculation for a flight might consider its entire trajectory, but the objective function only sums the delays for this relevant subset of flights.

In summary, the objective function returned by `rate_finder` is specifically tailored to the hotspot (`control_volume_id`) and flows under consideration, not the entire network.