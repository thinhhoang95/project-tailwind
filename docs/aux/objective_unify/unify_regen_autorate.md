The ultimate goal is to ensure that the same objective calculation process is used across the `regen` module and the simulated annealing code for rate finding employed in `automatic_rate_adjustment.py` are consistent and should return the same value when the inputs are the same.

Below is a detailed plan.

# Coding style instructions
- Put detailed comments in your code.

### Key mismatches to fix
- Scope
  - Regen’s objective evaluates over `tv_filter` but only the hotspot is tagged “target”; all other TVs are treated as context (no ripple), affecting J_reg/J_tv bin classification.
  - Predicted deficit reduction sums capacity across K bins, while J_cap and hotspot exceedance subtract a single hourly cap per rolling window.
- Target vs ripple
  - Hotspot is target; other TVs in `tv_filter` should be ripple to match the reference classification scheme.
- Weights
  - Regen passes `weights=None` (defaults). Reference path allows overriding, but defaults match. We should allow explicit `ObjectiveWeights` pass-through for parity.

### Plan and precise edits

- Align scopes (TV filter and time bins)
  - Keep regen’s `tv_filter` as the hotspot + flows’ control TVs + TVs traversed by candidate flows.
  - Build `target_cells` as now: hotspot × window bins.
  - Add `ripple_cells` = (each non-hotspot TV in `tv_filter`) × (the same window bins). This marks non-hotspot TVs as ripple only within the proposal window; classification dilates by ±w via `class_tolerance_w` just like the reference.

- Distinguish ripple TVs in context
  - Extend `build_local_context` in `src/parrhesia/flow_agent35/regen/predict.py` to accept and forward `ripple_cells` to `build_score_context`.
  - Build ripple cells in `src/parrhesia/flow_agent35/regen/engine.py` and pass them into the context.

Code edits

1) `src/parrhesia/flow_agent35/regen/predict.py` — add `ripple_cells` to `build_local_context` and pass it through.
```15:36:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/predict.py
def build_local_context(
    *,
    indexer,
    flight_list,
    capacities_by_tv: Mapping[str, np.ndarray],
    target_cells: Sequence[Tuple[str, int]],
    flights_by_flow: Mapping[int, Sequence[Mapping[str, Any]]],
    weights: Optional[ObjectiveWeights] = None,
    tv_filter: Optional[Iterable[str]] = None,
    ripple_cells: Optional[Iterable[Tuple[str, int]]] = None,
) -> ScoreContext:
    """Build a localized :class:`ScoreContext` for evaluation."""

    return build_score_context(
        flights_by_flow,
        indexer=indexer,
        capacities_by_tv=capacities_by_tv,
        target_cells=list(target_cells),
        ripple_cells=ripple_cells,
        flight_list=flight_list,
        weights=weights,
        tv_filter=tv_filter,
    )
```

2) `src/parrhesia/flow_agent35/regen/engine.py` — add a ripple builder and pass `ripple_cells` into the context.

Add helper after `_build_target_cells`:
```71:115:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/engine.py
def _build_target_cells(hotspot_tv: str, timebins_h: Sequence[int]) -> List[Tuple[str, int]]:
    return [(str(hotspot_tv), int(b)) for b in timebins_h]


def _build_ripple_cells(
    hotspot_tv: str,
    timebins_h: Sequence[int],
    tv_filter: Sequence[str],
) -> List[Tuple[str, int]]:
    """
    Mark all non-hotspot TVs in tv_filter as ripple over the same time window.
    """
    cells: List[Tuple[str, int]] = []
    hset = {str(hotspot_tv)}
    for tv in tv_filter:
        stv = str(tv)
        if stv in hset:
            continue
        for b in timebins_h:
            cells.append((stv, int(b)))
    return cells
```

Use it when building the scoring context:
```246:264:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/engine.py
    target_cells = _build_target_cells(hotspot_tv, timebins_seq)
    tv_filter = _build_tv_filter(
        hotspot_tv,
        bundles,
        flights_by_flow=flights_by_flow,
        flight_list=flight_list,
        indexer=indexer,
    )
    ripple_cells = _build_ripple_cells(hotspot_tv, timebins_seq, tv_filter)

    # Context uses hotspot as target and all other TVs in tv_filter as ripple
    context = build_local_context(
        indexer=indexer,
        flight_list=flight_list,
        capacities_by_tv=capacities_by_tv,
        target_cells=target_cells,
        flights_by_flow=flights_by_flow,
        weights=None,
        tv_filter=tv_filter,
        ripple_cells=ripple_cells,
    )
```

- Fix the hotspot “predicted deficit” calculator to match rolling-hour semantics (subtract one hourly capacity per bin, not sum across K bins). This aligns with `J_cap` and `compute_hotspot_exceedance`.

3) `src/parrhesia/flow_agent35/regen/predict.py` — update `compute_delta_deficit_per_hour`:
```157:168:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/predict.py
        for t in range(start, min(end, T - 1) + 1):
            t_end = min(T, t + bins_per_hour)
            occ_before_sum = float(occ_before_vec[t:t_end].sum())
            occ_after_sum = float(occ_after_vec[t:t_end].sum())
            # Use a single hourly capacity for the rolling window anchored at t
            hour_idx = int(t) // bins_per_hour
            h_start = hour_idx * bins_per_hour
            h_end = min(h_start + bins_per_hour, T)
            # Robust to non-constant per-bin caps within the hour
            slice_caps = cap_vec[h_start:h_end]
            hourly_cap = float(np.median(slice_caps)) if slice_caps.size else float(cap_vec[t] if t < cap_vec.size else 0.0)
            ex_before = max(0.0, occ_before_sum - hourly_cap)
            ex_after = max(0.0, occ_after_sum - hourly_cap)
            total_delta += ex_before - ex_after
            count += 1
```

- Weights parity
  - Defaults already match the reference. To allow explicit parity when desired, optionally thread `ObjectiveWeights` into regen’s API and example runner. Minimal change: add an optional `objective_weights: Optional[ObjectiveWeights] = None` argument to `propose_regulations_for_hotspot(...)`, pass it into `build_local_context(weights=objective_weights)`, and add a pass-through from `examples/regen/regen_test_bench_custom_tvtw.py` if you want to experiment with non-default weights. Not required if you’re fine with defaults.

Example change (proposed new code, not yet present):
```python
from parrhesia.optim.objective import ObjectiveWeights

def propose_regulations_for_hotspot(..., objective_weights: Optional[ObjectiveWeights] = None, ...) -> List[Proposal]:
    ...
    context = build_local_context(..., weights=objective_weights, ripple_cells=ripple_cells, ...)
```

### Testing and acceptance
- With the above edits, both paths use:
  - Same TV scope: target ∪ ripple (hotspot + traversed/control TVs), occupancy/J_cap computed only there.
  - Same target/ripple distinction: hotspot marked target; other TVs in that set marked ripple on the proposal window; classification uses identical ±w dilation.
  - Same weights: defaults unless explicitly overridden to match.
  - Same rolling-hour semantics in the hotspot deficit metric.

- Quick equivalence check (script/notebook - for coding agents: do not implement this)
  - Build `flights_by_flow`, `indexer`, `flight_list`, `capacities_by_tv` once.
  - Construct `target_cells` and `ripple_cells` as in the regen path above; use the same `tv_filter`.
  - Evaluate a baseline schedule `n0` with both scorers:
    - Reference: `parrhesia.optim.objective.build_score_context(...)` + `score_with_context(... spill_mode="dump_to_next_bin")`.
    - Regen: `parrhesia.flow_agent.safespill_objective.score_with_context(...)` with the `context` built by regen.
  - Compare `J_total` and the component dicts; expect close agreement (small numeric differences may arise from minor implementation differences but should be negligible).
  - Also compare the hotspot exceedance and “predicted deficit” changes after applying a regulation: with the fix, the deficit metric will now move consistently with J_cap’s hotspot slice.

### Should automatic_rate_adjustment switch to safespill_objective?
- Not necessary. The current automatic rate path already:
  - Uses the same components via `parrhesia.optim.objective` and default `spill_mode="dump_to_next_bin"`.
  - Computes occupancy with the same “all-flights base + scheduled-current − scheduled-zero” composition.
- If you want absolute uniformity and to remove duplication, a future cleanup could make `safespill_objective.score_with_context(...)` a thin wrapper around `parrhesia.optim.objective.score_with_context(..., spill_mode="dump_to_next_bin")`. That’s optional and low risk once your tests above show the numerics line up.

### Notes
- The doc snippet mentioning a 100× factor on J_cap in `safespill_objective` is outdated; the current code computes `J_total = J_cap + J_delay + J_reg + J_tv (+ optional)`, matching the reference scaling.

- If you later want ripple TVs tagged beyond the proposal window (e.g., to emphasize downstream spill effects), expand `ripple_cells` to include an enlarged time window or adopt an auto-ripple builder analogous to `automatic_rate_adjustment._auto_ripple_cells_from_flows`.

Summary
- Added `ripple_cells` handling in `predict.build_local_context` and engine to mark all non-hotspot TVs as ripple within the window.
- Fixed `compute_delta_deficit_per_hour` to subtract a single hourly cap per rolling window, matching J_cap/exceedance semantics.
- Kept weights consistent (defaults); optional pass-through for explicit `ObjectiveWeights`.
- No need to switch `automatic_rate_adjustment` to `safespill_objective`; both are consistent under dump-to-next-bin semantics.