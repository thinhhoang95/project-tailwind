### Plan to switch RateFinder to use objective.py (rolling-hour objective)

- Define the inputs the scorer needs
  - Build `flights_by_flow` from current flows restricted to the `control_volume_id` and `active_windows`:
    - For each `flow_id`, include only flights that cross the `control_volume_id` within `active_windows`, using entry time at that TV as `requested_dt`.
  - Build `capacities_by_tv` for the `control_volume_id`:
    - Length `T=indexer.num_time_bins` array with the per-hour throughput replicated over the hour’s bins from `evaluator.hourly_capacity_by_tv[control_volume_id]`.
  - Use `target_cells` to emphasize the chosen window: `{(control_volume_id, t) for t in active_windows}`; no ripple cells.
  - Use `tv_filter=[control_volume_id]` to restrict occupancy/alpha to the TV of interest.
  - Map `config.objective_weights` into `ObjectiveWeights` if provided, else use defaults.

- Build and cache objective context once per call
  - Introduce a small cache in `RateFinder` for an `objective.ScoreContext` keyed by:
    - `(plan_key, control_volume_id, window_bins, sorted(flow_ids))`
  - Build with `build_score_context(flights_by_flow, indexer, capacities_by_tv, target_cells, ripple_cells=None, flight_list, weights, tv_filter=[control_volume_id])`.

- Establish a correct baseline
  - Baseline schedule must be “no regulation”: `n_f_t == d_f(t)` for scheduled flights, so `J_reg=0` and delays are zero.
  - Compute once via `score_with_context(n=d, ..., context)` and cache `(J_baseline, components_baseline)` keyed with the same context key.

- Replace candidate evaluation to use the scorer
  - Remove use of `NetworkEvaluator.compute_excess_traffic_vector()` and `compute_delay_stats()`.
  - Do not call `parrhesia.fcfs.scheduler.assign_delays` for evaluation; instead:
    - Convert candidate rate(s) into per-bin integer release schedules `n_f_t`:
      - For rate = ∞, use `n=d` (baseline).
      - For finite rate R:
        - Compute per-bin quota `q = round(R * bin_minutes / 60)`.
        - For each flow, greedily allocate `min(q, ready - released)` at each active bin (in ascending time), where “ready” counts flights with `requested_bin <= t` that haven’t been released; put any remainder into overflow bin `T`.
    - Evaluate `J_total, components, artifacts = score_with_context(n_f_t, ..., context)`.
    - Use `delta_j = J_total - J_baseline` for comparisons.
    - Cache candidate results keyed by the existing `signature` tuple (extend if needed) to avoid recomputation.

- Coordinate-descent loop wiring
  - Per-flow mode:
    - Maintain `best_rates[flow_id]`.
    - For each flow, iterate candidate rates, build `n_f_t` from the current rate map (others fixed at their current best or baseline), evaluate with the scorer, and update if `delta_j` improves.
    - After all passes (or early stop), run one final scorer call with the best rate map to produce the final `objective`, `components`, and `artifacts["delays_min"]`.
  - Blanket mode:
    - Treat as a single “union” schedule by either:
      - Applying the same rate to each flow and summing via the scorer, or
      - Constructing a single synthetic flow “__blanket__” with the union flights and one schedule (prefer this to mimic original blanket behavior).
    - Evaluate as above.

- Diagnostics and outputs
  - Keep API stable: return `(rates_out, best_delta, diagnostics)`.
  - Update `diagnostics`:
    - Include `baseline_objective`, `baseline_components`, `final_objective`, and `final_components`.
    - Keep `per_flow_history` (rate -> `delta_j`) and `rate_grid`.
    - Replace `aggregate_delays_size` by the length of `artifacts["delays_min"]` from the final evaluation; optionally also include the full delays mapping for debugging.
    - Maintain `eval_calls`, `cache_hits`, timing, and `stopped_early`.

- Caching adjustments
  - Baseline cache should be keyed by context (plan_key + TV + window + flows) rather than just `plan_key`.
  - Candidate cache remains keyed by `(plan_key, control_volume_id, window_bins, flow_ids, mode, rates_tuple)`.

- Imports and minor plumbing
  - Add: `from parrhesia.optim.objective import ObjectiveWeights, build_score_context, score_with_context`.
  - Add small helpers in `RateFinder`:
    - `_build_flights_by_flow(...)`
    - `_build_capacities_for_tv(...)`
    - `_build_schedule_from_rates(...)`
    - `_ensure_context_and_baseline(...)` to centralize context/baseline caching.

- Testing notes
  - Update/extend tests to assert:
    - Baseline `n=d` yields `J_reg=0` and no delays.
    - Finite rate reduces `J_cap` while potentially increasing `J_reg`/`J_tv`.
    - Per-flow vs blanket mode behaviors remain deterministic across the same `rate_grid` and `passes`.

- Performance considerations
  - Reuse `ScoreContext` across all candidates in a `find_rates` call.
  - Keep candidate result caching; avoid recomputing identical `n` schedules.
  - If needed, add a lightweight schedule-hash in the candidate signature to ensure stable cache hits.