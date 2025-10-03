**RateFinder**

- Purpose: For a chosen hotspot and set of flows, find hourly control rates that maximize objective improvement.
- Location: `src/parrhesia/flow_agent/rate_finder.py`

Config
- `RateFinderConfig` fields:
  - `rate_grid`: Candidate hourly rates (first position usually `inf` for “no control”).
  - `passes`, `epsilon`: Coordinate descent passes and early-stop tolerance.
  - `max_eval_calls`: Budget for objective evaluations per call.
  - `cache_size`: LRU size for candidate/context caches.
  - `objective_weights`: Optional weights for the objective.
  - `use_adaptive_grid`, `max_adaptive_rate`, `max_adaptive_candidates`: Build a data-driven grid around entrants and capacity.
  - `fast_scorer_enabled`, `verbose`: Speed and logging knobs.

Flow
1) Entrants and rate grid
- Restricts to selected flights per flow; computes entries (`flight_id`, `entry_dt?`, `time_bin`) for the hotspot window.
- Resolves a rate grid by merging the configured grid with an adaptive grid derived from entrants pressure and capacity snapshots.

2) Context and baseline
- Builds `flights_by_flow` and `capacities_by_tv` for the controlled TV and window; creates or reuses a `ScoreContext`.
- Computes the baseline objective (no regulation) once and caches it by context signature.

3) Coordinate descent
- Per-flow mode: Iterates flows ordered by entrants, sweeping the rate grid to find the best improvement for each; repeats for a few passes until improvement is small (`epsilon`) or budget is consumed.
- Blanket mode: Sweeps a single rate across the union of flights, otherwise identical logic.
- Schedules are built deterministically: hourly rate → per-bin quota derived from `time_bin_minutes`, ensuring unserved demand goes into the last “overflow” bin.

4) Evaluation and caching
- Evaluates candidates via `score_with_context(...)` (or a precomputed-occupancy fast path), memoized by a signature of plan, hotspot, flows, mode, and rate tuple.
- Returns `(rates, delta_j, info)` where `rates` is a map per flow (`per_flow`) or a single value (`blanket`), and `delta_j` is the improvement vs. baseline.

Diagnostics
- The `info` payload includes keys like `control_volume_id`, `window_bins`, `mode`, `entrants_by_flow`, `history` (per-flow/blanket improvement by rate) and `components`/`artifacts` for the best candidate.

Examples
- Per-flow search with adaptive grid
```python
from parrhesia.flow_agent.rate_finder import RateFinder, RateFinderConfig

rf = RateFinder(evaluator=evalr, flight_list=fl, indexer=idx,
                config=RateFinderConfig(use_adaptive_grid=True, max_eval_calls=128))

flows = {"F1": ("AA10", "AA20"), "F2": ("BA30",)}
rates, delta_j, info = rf.find_rates(
    plan_state=state,
    control_volume_id="TV123",
    window_bins=(30, 36),
    flows=flows,
    mode="per_flow",
)
print(rates)         # e.g., {"F1": 24.0, "F2": inf}
print(delta_j)       # negative improvements are better
```

- Scheduling from rates (intuition)
```python
# With 30-minute bins (bin_minutes=30) and rate=24 flights/hour, quota per bin = 12 flights.
# The schedule releases up to 12 flights each active bin, carrying spillover into the last overflow bin.
```

Tips
- If you hit the `max_eval_calls` budget, enable `use_adaptive_grid` to shrink the candidate set intelligently.
- Rates of `inf` are treated as unconstrained (no regulation) and may be dropped by MCTS sanitization.

