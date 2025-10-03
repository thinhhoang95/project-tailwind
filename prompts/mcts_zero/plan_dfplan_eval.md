### Goal
Design a small module that takes a `DFRegulationPlan` and returns:
- concrete per-flight delays (minutes) under the plan
- objective components (non-legacy), and the overall objective

### Key integration points from your codebase
- Use the TVTW indexer/time-bin grid and flight list like the simulation path in `run_regulation_plan_simulation(...)`.
- Reuse the flow objective/scoring path (build n_f_t schedules per flow, compute ripple cells, call `flow_score(...)`).
 - Reuse the flow objective/scoring path (build n_f_t schedules per flow, compute ripple cells, build a shared `ScoreContext` and call `score_with_context(...)` twice for baseline and regulated).
- Prefer flight lists captured directly from the plan’s `flights` field instead of parsing DSL.

### Module shape
- File: `src/parrhesia/actions/dfplan_evaluator.py`
- Public API:
  - `evaluate_df_regulation_plan(plan, *, indexer_path, flights_path, capacities_path=None, weights=None, include_excess_vector=False) -> DFPlanEvalResult`

### Inputs
- `plan`: `DFRegulationPlan` (from `src/parrhesia/actions/regulations.py`)
- `indexer_path`: path to a TVTW indexer artifact
- `flights_path`: path to baseline occupancy (same file your parser/evaluator expects)
- `capacities_path` (optional): path to capacities; if omitted, pull from existing resources like in the wrapper
- `weights` (optional): objective weights dict
- `include_excess_vector` (optional): pass-through to optionally compute a full excess vector if you decide to expose it

### Outputs
- `delays_by_flight`: dict[str, int] (minutes)
- `objective_components`: dict[str, float] (non-legacy, e.g., J_cap, J_delay, J_reg, J_tv)
- `objective`: float (post-plan objective under the plan)
- `pre_objective`: float (baseline objective with n=d)
- `delta_objective`: float (`pre_objective - objective`)

### Core steps
1) Load indexer and parser context (like the wrapper does)
- Load `TVTWIndexer` from `indexer_path`.
- Create a `RegulationParser` with `flights_path` and the indexer.

2) Convert `DFRegulationPlan` to internal “Regulation” objects
- For each `DFRegulation` in `plan.regulations`:
  - Map `window_from`/`window_to` to bin indices for the current `time_bin_minutes`.
  - Build `Regulation.from_components(location=tv_id, rate=allowed_rate_per_hour, time_windows=[bin indices], target_flight_ids=list(reg.flights))`.
  - Note: this bypasses DSL and fixes the flows to the flights explicitly listed in the plan.

3) Evaluate the plan to obtain post-regulation view
- `NetworkPlan(regs)` → `PlanEvaluator(...).evaluate_plan(...)` → `plan_result`.
- No need to keep `delta_view` for post-occupancy nor fetch pre-occupancy from baseline (these are not needed).

4) Build per-TV capacity vectors
- Prefer the precomputed capacity-per-bin matrix if available; otherwise derive from hourly capacities (exactly like the wrapper logic).
- Normalize capacities with your `normalize_capacities(...)`.

5) Build flows (one per regulation) and map flights to flows
- Assign each regulation a stable flow index (order in the list).
- Map each flight in `reg.flights` uniquely to its flow (first-come, first-assigned).
- Use `prepare_flow_scheduling_inputs(...)` to get `flights_by_flow` with per-flight requested bins.

6) Build per-flow demand and scheduled allowance arrays
- Demand: histogram requested-bin counts per flow over all T bins.
- Baseline schedule: copy demand (no caps).
- Regulated schedule: apply per-bin allowance computed from each regulation’s hourly rate distributed over its time-window bins (`distribute_hourly_rate_to_bins(...)`), clamp per-bin demand to allowance.

7) Ripple and TV filter for scoring
- Determine ripple cells from the plan’s metadata using the same precedence as `DFRegulationPlan._resolve_ripples(...)` (e.g., via `plan.to_base_eval_payload(...)` to read either `ripples` or `auto_ripple_time_bins` and translate to cells with `compute_auto_ripple_cells(...)`).
- TV filter is the union of target TVs and ripple TVs.

8) Score
- Build a shared `ScoreContext` with `build_score_context(...)` using `flights_by_flow`, `indexer`, normalized capacities, `target_cells`, `ripple_cells`, `flight_list`, `weights_obj`, and `tv_filter`.
- Baseline objective (n=d): `J_before, comps_before, arts_before = score_with_context(context.d_by_flow, ..., context=context, spill_mode="dump_to_next_bin")`.
- Regulated objective: `J_after, comps_after, arts_after = score_with_context(n_f_t, ..., context=context, spill_mode="dump_to_next_bin")`.
- Use `arts_after["delays_min"]` as authoritative per-flight delay minutes.
- Set `objective = J_after`, `objective_components = comps_after`, `pre_objective = J_before`, `delta_objective = J_before - J_after`.

Drop-in reference snippet:
```python
# Shared scoring context
context = build_score_context(
    flights_by_flow,
    indexer=tvtw_indexer,
    capacities_by_tv=cap_by_tv,
    target_cells=target_cells,
    ripple_cells=ripple_cells,
    flight_list=parser.flight_list,
    weights=weights_obj,
    tv_filter=tv_filter,
)

J_before, comps_before, arts_before = score_with_context(
    context.d_by_flow,
    flights_by_flow=flights_by_flow,
    capacities_by_tv=cap_by_tv,
    flight_list=parser.flight_list,
    context=context,
    spill_mode="dump_to_next_bin",
)

J_after, comps_after, arts_after = score_with_context(
    n_f_t,
    flights_by_flow=flights_by_flow,
    capacities_by_tv=cap_by_tv,
    flight_list=parser.flight_list,
    context=context,
    spill_mode="dump_to_next_bin",
)

delta_objective = float(J_before - J_after)
delays_by_flight = {str(fid): int(v) for fid, v in (arts_after.get("delays_min") or {}).items()}
```

9) Return results
- Prefer `artifacts["delays_min"]` for `delays_by_flight`.
- Return `objective_components` and `objective` (optionally `pre_objective`/`delta_objective` if you compute both).

### Edge cases
- Empty flows or no flights → return empty delays and zeroed components.
- Windows crossing day-end: clamp to [0, T-1] and use “24:00” as exclusive end.
- Invalid inputs (missing targets, bad times) should raise `ValueError` (align with `regulations.py` validations).

### Minimal skeleton
```python
# src/parrhesia/actions/dfplan_evaluator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from parrhesia.actions.regulations import DFRegulationPlan, DEFAULT_AUTO_RIPPLE_TIME_BINS

# Imports below should mirror what's already used in your wrapper:
# - TVTWIndexer, RegulationParser, Regulation, NetworkPlan, PlanEvaluator
# - prepare_flow_scheduling_inputs, distribute_hourly_rate_to_bins
# - compute_auto_ripple_cells, normalize_capacities
# - ObjectiveWeights, build_score_context, score_with_context

@dataclass(frozen=True)
class DFPlanEvalResult:
    delays_by_flight: Dict[str, int]
    objective_components: Dict[str, float]
    objective: float
    pre_objective: float
    delta_objective: float

def _time_to_bin(hhmm: str, *, time_bin_minutes: int) -> int:
    hh, mm = hhmm.split(":")[:2]
    minutes = int(hh) * 60 + int(mm)
    return minutes // int(time_bin_minutes)

def _window_to_bins(start: str, end: str, *, time_bin_minutes: int, T: int) -> List[int]:
    # end is exclusive (support "24:00")
    start_bin = max(0, _time_to_bin(start, time_bin_minutes=time_bin_minutes))
    end_bin_excl = min(T, _time_to_bin(end, time_bin_minutes=time_bin_minutes))
    if end.endswith(":00") and end.startswith("24"):
        end_bin_excl = T
    return list(range(start_bin, max(start_bin, end_bin_excl)))

def _dfplan_to_regulations(plan: DFRegulationPlan, *, indexer, T: int) -> List[Any]:
    regs = []
    tbm = int(indexer.time_bin_minutes)
    for r in plan.regulations:
        wins = _window_to_bins(r.window_from, r.window_to, time_bin_minutes=tbm, T=T)
        reg = Regulation.from_components(
            location=r.tv_id,
            rate=int(r.allowed_rate_per_hour),
            time_windows=[int(b) for b in wins],
            filter_type="IC",
            filter_value="__",
            target_flight_ids=list(r.flights),
        )
        regs.append(reg)
    return regs

def evaluate_df_regulation_plan(
    plan: DFRegulationPlan,
    *,
    indexer_path: str,
    flights_path: str,
    capacities_path: Optional[str] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> DFPlanEvalResult:
    tvtw_indexer = TVTWIndexer.load(indexer_path)
    parser = RegulationParser(flights_file=flights_path, tvtw_indexer=tvtw_indexer)

    # Build regs and evaluate plan
    T = int(tvtw_indexer.num_time_bins)
    regs = _dfplan_to_regulations(plan, indexer=tvtw_indexer, T=T)
    network_plan = NetworkPlan(regs)
    evaluator = PlanEvaluator(traffic_volumes_gdf=None, parser=parser, tvtw_indexer=tvtw_indexer)
    plan_result = evaluator.evaluate_plan(network_plan, flight_list=parser.flight_list, weights=weights)

    # Pre/post occupancy vectors
    pre_total = parser.flight_list.get_total_occupancy_by_tvtw().astype(np.float32, copy=False)
    post_total = plan_result["delta_view"].get_total_occupancy_by_tvtw().astype(np.float32, copy=False)

    # Capacities per TV per bin (same logic as wrapper)
    cap_by_tv = ...  # derive or load and normalize via normalize_capacities(...)

    # Flows and mapping: one flow per regulation
    flow_to_reg: Dict[int, Any] = {}
    flow_map: Dict[str, int] = {}
    for idx, reg in enumerate(regs):
        flow_to_reg[idx] = reg
        for fid in (reg.target_flight_ids or []):
            s = str(fid)
            if s not in flow_map:
                flow_map[s] = idx

    flights_by_flow_raw, _ = prepare_flow_scheduling_inputs(
        flight_list=parser.flight_list,
        flow_map=flow_map,
        hotspot_ids=[r.location for r in regs],
        flight_ids=list(flow_map.keys()),
    )
    flights_by_flow = {int(k): v for k, v in flights_by_flow_raw.items() if v}

    # Build demand and regulated schedule per flow
    bins_per_hour = 60 // int(tvtw_indexer.time_bin_minutes)
    n_f_t: Dict[int, List[int]] = {}
    n0_f_t: Dict[int, List[int]] = {}
    for fid, specs in flights_by_flow.items():
        demand = np.zeros(T + 1, dtype=np.int64)
        for spec in specs:
            rb = spec.get("requested_bin")
            if rb is None:
                continue
            b = int(rb)
            if 0 <= b < T:
                demand[b] += 1
        baseline = demand.copy()
        baseline_total = int(np.sum(demand[:T], dtype=np.int64))
        baseline[T] = max(0, baseline_total - int(np.sum(baseline[:T], dtype=np.int64)))

        schedule = demand.copy()
        reg = flow_to_reg.get(int(fid))
        if reg:
            wins = [int(w) for w in getattr(reg, "time_windows", []) or []]
            if wins:
                start_bin, end_bin = min(wins), max(wins)
                allowance = distribute_hourly_rate_to_bins(int(reg.rate), bins_per_hour, start_bin, end_bin)
                for offset, b in enumerate(range(start_bin, end_bin + 1)):
                    if 0 <= b < T:
                        allow_val = int(allowance[offset]) if offset < allowance.size else 0
                        schedule[b] = int(min(schedule[b], allow_val))
        schedule[T] = max(0, int(np.sum(demand[:T], dtype=np.int64)) - int(np.sum(schedule[:T], dtype=np.int64)))
        n_f_t[int(fid)] = schedule.astype(int).tolist()
        n0_f_t[int(fid)] = baseline.astype(int).tolist()

    # Ripple handling via plan metadata precedence
    base_payload = plan.to_base_eval_payload(warn_on_auto_fallback=False)
    ripple_cells = ...
    target_cells = ...
    tv_filter = ...

    # Weights
    weight_kwargs = {}
    if isinstance(weights, Mapping):
        for k, v in weights.items():
            if k in ObjectiveWeights.__dataclass_fields__:
                weight_kwargs[k] = v
    weights_obj = ObjectiveWeights(**weight_kwargs) if weight_kwargs else ObjectiveWeights()

    # Score (shared context; baseline and regulated)
    context = build_score_context(
        flights_by_flow,
        indexer=tvtw_indexer,
        capacities_by_tv=cap_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        flight_list=parser.flight_list,
        weights=weights_obj,
        tv_filter=tv_filter,
    )

    J_before, comps_before, arts_before = score_with_context(
        context.d_by_flow,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=cap_by_tv,
        flight_list=parser.flight_list,
        context=context,
        spill_mode="dump_to_next_bin",
    )
    J_after, comps_after, arts_after = score_with_context(
        n_f_t,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=cap_by_tv,
        flight_list=parser.flight_list,
        context=context,
        spill_mode="dump_to_next_bin",
    )

    delays_min = arts_after.get("delays_min", {}) or {}
    delays_by_flight = {str(fid): int(val) for fid, val in delays_min.items()}
    objective_components = {str(k): float(v) for k, v in (comps_after or {}).items()}
    pre_objective = float(J_before)
    objective = float(J_after)
    delta_objective = float(J_before - J_after)

    return DFPlanEvalResult(
        delays_by_flight=delays_by_flight,
        objective_components=objective_components,
        objective=objective,
        pre_objective=pre_objective,
        delta_objective=delta_objective,
    )
```

### Notes/decisions
- Time window binning: treat `window_to` as exclusive; allow “24:00”.
- Flows are defined by the plan’s regulations; flights come from each regulation’s `flights`.
- Ripple strategy: reuse `plan.to_base_eval_payload(...)` to retrieve either explicit `ripples` or `auto_ripple_time_bins` and translate to `ripple_cells` with your existing helper.

### Minimal tests
- One TV, one regulation, small window, tiny rate, 3 flights in that window → delays sum > 0; objective_components keys present.
- Explicit `ripples` vs `auto_ripple_time_bins` both exercised.
- Windows at boundaries (“00:00”, “24:00”) and empty flights.

- Built an end-to-end plan for a new `dfplan_evaluator.py` module that converts a `DFRegulationPlan` into `Regulation` objects, evaluates the plan, constructs per-flow schedules, and computes per-flight delays and objective components using your existing scoring pipeline.