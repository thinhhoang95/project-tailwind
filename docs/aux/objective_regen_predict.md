I traced how the regen entry computes objectives and where scopes/time windows differ. Below are the concrete call paths, with code references, and a focused comparison of scopes, rolling-hour handling, and component/scale differences.

### Trace from entry point to objective calculations
- examples runner sets up inputs and calls the hotspot proposal entry:
```407:416:examples/regen/regen_test_bench_custom_tvtw.py
def main() -> None:
    print("[regen] Resolving artifacts ...")
    occ_path, idx_path, caps_path = load_artifacts()
    ...
    flight_list, indexer, evaluator = build_data(occ_path, idx_path, caps_path)
```

- Hourly capacities are expanded to per-bin arrays by repeating the hourly value across all bins within the hour (note: repeated, not divided):
```157:179:examples/regen/regen_test_bench_custom_tvtw.py
def _capacities_per_bin(
    evaluator: NetworkEvaluator,
    indexer: TVTWIndexer,
) -> Dict[str, np.ndarray]:
    per_hour = getattr(evaluator, "hourly_capacity_by_tv", {}) or {}
    bins_per_hour = int(indexer.rolling_window_size())
    T = int(indexer.num_time_bins)
    capacities: Dict[str, np.ndarray] = {}
    for tv_id in indexer.tv_id_to_idx.keys():
        arr = np.zeros(T, dtype=np.float64)
        hours = per_hour.get(tv_id, {}) or {}
        for h, cap in hours.items():
            ...
            start = hour * bins_per_hour
            ...
            end = min(start + bins_per_hour, T)
            arr[start:end] = float(cap)
        capacities[str(tv_id)] = arr
    return capacities
```

- It then calls into the regen engine:
```428:434:examples/regen/regen_test_bench_custom_tvtw.py
proposals = propose_regulations(
    hotspot_payload=hotspot_payload,
    evaluator=evaluator,
    indexer=indexer,
)
```

- The engine computes hotspot exceedance for the designated TV and bins:
```178:186:src/parrhesia/flow_agent35/regen/engine.py
exceedance_stats = compute_hotspot_exceedance(
    indexer=indexer,
    flight_list=flight_list,
    capacities_by_tv=capacities_by_tv,
    hotspot_tv=hotspot_tv,
    timebins_h=timebins_seq,
    caches=extractor.caches,
)
```

- Exceedance uses rolling-hour occupancy by bin and subtracts an hourly capacity value (not summed across K bins):
```31:49:src/parrhesia/flow_agent35/regen/exceedance.py
rolling_occ = np.asarray(caches_local.get("rolling_occ_by_bin"))
hourly_capacity_matrix = np.asarray(caches_local.get("hourly_capacity_matrix"))
...
row = int(row_map[str(hotspot_tv)])
occ_row = rolling_occ[row]
cap_row = hourly_capacity_matrix[row]
...
hour_idx = min(cap_row.shape[0] - 1, max(0, b // bins_per_hour))
hourly_cap = float(cap_row[hour_idx])
exceed = max(0.0, float(occ_row[b]) - hourly_cap)
```

- The engine builds target cells for the hotspot/timebins, and a tv filter that includes the hotspot + all flows’ control TVs + all TVs traversed by those flows:
```71:72:src/parrhesia/flow_agent35/regen/engine.py
def _build_target_cells(hotspot_tv: str, timebins_h: Sequence[int]) -> List[Tuple[str, int]]:
    return [(str(hotspot_tv), int(b)) for b in timebins_h]
```

```118:146:src/parrhesia/flow_agent35/regen/engine.py
def _build_tv_filter(
    hotspot_tv: str,
    bundles: Sequence[Bundle],
    *,
    flights_by_flow: Mapping[int, Sequence[Mapping[str, Any]]],
    flight_list: Any,
    indexer,
) -> List[str]:
    ...
    tvs = {str(hotspot_tv)}
    for bundle in bundles:
        for fs in bundle.flows:
            if fs.control_tv_id:
                tvs.add(str(fs.control_tv_id))
    traversed = _tvs_traversed_by_flows(
        flights_by_flow=flights_by_flow,
        flight_list=flight_list,
        indexer=indexer,
    )
    tvs.update(traversed)
    return sorted(tvs)
```

- The scoring context is created with that tv_filter and hotspot target cells:
```254:263:src/parrhesia/flow_agent35/regen/engine.py
context = build_local_context(
    indexer=indexer,
    flight_list=flight_list,
    capacities_by_tv=capacities_by_tv,
    target_cells=target_cells,
    flights_by_flow=flights_by_flow,
    weights=None,
    tv_filter=tv_filter,
)
```

```27:36:src/parrhesia/flow_agent35/regen/predict.py
return build_score_context(
    flights_by_flow,
    indexer=indexer,
    capacities_by_tv=capacities_by_tv,
    target_cells=list(target_cells),
    ripple_cells=None,
    flight_list=flight_list,
    weights=weights,
    tv_filter=tv_filter,
)
```

- Baseline schedule is the per-flow histogram of requested control times:
```42:45:src/parrhesia/flow_agent35/regen/predict.py
out: Dict[Any, np.ndarray] = {}
for flow_id, arr in context.d_by_flow.items():
    out[flow_id] = np.asarray(arr, dtype=np.int64).copy()
```

- Regulation applies per-flow hourly rate constraints inside the chosen window, distributing hourly allowances to bins and clamping demand:
```68:83:src/parrhesia/flow_agent35/regen/predict.py
allowances = distribute_hourly_rate_to_bins(
    rate.allowed_rate_R,
    bins_per_hour=bph,
    start_bin=start,
    end_bin=end,
)
...
for offset, bin_idx in enumerate(range(start, end + 1)):
    ...
    allowed = allowances[offset] if offset < allowances.size else allowances[-1]
    demand_val = demand_arr[bin_idx] if bin_idx < demand_arr.shape[0] else 0
    baseline_arr[bin_idx] = min(int(demand_val), int(allowed))
```

- The pairwise objective is computed using the “safe-spill” scorer (not the generic one):
```105:120:src/parrhesia/flow_agent35/regen/predict.py
score_before, components_before, artifacts_before = score_with_context(
    baseline,
    flights_by_flow=flights_by_flow,
    capacities_by_tv=capacities_by_tv,
    flight_list=flight_list,
    context=context,
    audit_exceedances=True,
)
score_after, components_after, artifacts_after = score_with_context(
    regulated,
    flights_by_flow=flights_by_flow,
    capacities_by_tv=capacities_by_tv,
    flight_list=flight_list,
    context=context,
    audit_exceedances=True,
)
```

- The “predicted deficit reduction” re-computes rolling-hour exceedance over only the hotspot TV and only inside the proposal window:
```158:166:src/parrhesia/flow_agent35/regen/predict.py
t_end = min(T, t + bins_per_hour)
occ_before_sum = float(occ_before_vec[t:t_end].sum())
occ_after_sum = float(occ_after_vec[t:t_end].sum())
cap_sum = float(cap_vec[t:t_end].sum())
ex_before = max(0.0, occ_before_sum - cap_sum)
ex_after = max(0.0, occ_after_sum - cap_sum)
```

### Scope of TVs and time bins
- Objective (J) scope during scoring:
  - TV scope: all TVs in `tv_filter` (hotspot + each flow’s control TV + any TV traversed by flights in candidate flows).
  - Time scope: all bins 0..T−1. Target cells are only the hotspot at the specified bins and only affect α-weights, not the occupancy range.
  - Evidence:
    - tv_filter creation:
      ```118:146:src/parrhesia/flow_agent35/regen/engine.py
      ```
    - context building with tv_filter and hotspot target cells:
      ```254:263:src/parrhesia/flow_agent35/regen/engine.py
      ```
- Hotspot exceedance (E_target) scope:
  - TV scope: hotspot TV only.
  - Time scope: only the designated `timebins_h`.
  - Evidence:
    ```31:49:src/parrhesia/flow_agent35/regen/exceedance.py
    ```
- Predicted deficit reduction scope:
  - TV scope: hotspot TV(s) only (from `target_cells`).
  - Time scope: only the selected proposal window `[start_bin..end_bin]`.
  - Evidence:
    ```144:152:src/parrhesia/flow_agent35/regen/predict.py
    ```
    ```158:166:src/parrhesia/flow_agent35/regen/predict.py
    ```

Implication: ΔJ (objective difference) aggregates effects over all TVs in `tv_filter` and over the full horizon, while the “delta_deficit_per_hour” and hotspot exceedance are restricted to the hotspot and its specified window. They are not scoped the same.

### Rolling-hour vs non-rolling-hour handling
- Objective J_cap:
  - Rolling-hour occupancy computed (internally via rolling sums), subtracts capacity at each bin.
  - Evidence:
    ```1045:1062:src/parrhesia/optim/objective.py
    rh = rolling_hour_sum(occ.astype(np.int64, copy=False), int(K)).astype(np.float64)
    exceed = np.maximum(0.0, rh - cap)
    ```
- Hotspot exceedance (input to E_target):
  - Uses cached rolling-hour occupancy-by-bin, subtracting one hourly cap value per bin.
  - Evidence:
    ```31:49:src/parrhesia/flow_agent35/regen/exceedance.py
    exceed = max(0.0, float(occ_row[b]) - hourly_cap)
    ```
- Predicted deficit reduction:
  - Recomputes rolling sums by summing occ[t..t+K−1], but also sums capacity across the same K bins: `cap_sum = sum(cap_vec[t..t+K−1])`.
  - With capacities built as “hourly value repeated per bin” (see `_capacities_per_bin`), `cap_sum` equals K × hourly_cap, whereas the other two (J_cap and exceedance) subtract a single hourly cap for the rolling window.
  - Evidence:
    - Capacity construction (hourly value repeated across bins):
      ```157:179:examples/regen/regen_test_bench_custom_tvtw.py
      ```
    - Predicted deficit computation:
      ```158:166:src/parrhesia/flow_agent35/regen/predict.py
      ```
  - This makes “delta_deficit_per_hour” numerically not directly comparable to J_cap or the hotspot exceedance unless capacity arrays are per-bin capacities (hourly_cap/K). As constructed in the bench, they are hourly values repeated, not per-bin. So `cap_sum` is K× too large here.

### Component and scaling differences that affect comparability
- Safe-spill scorer vs generic scorer:
  - Regen uses `parrhesia.flow_agent.safespill_objective.score_with_context`, not `parrhesia.optim.objective.score_with_context`.
  - The safe-spill variant:
    - Always uses `spill_mode="dump_to_next_bin"` when computing delays.
      ```36:41:src/parrhesia/flow_agent/safespill_objective.py
      delays_min, realised_start = assign_delays_flowful_preparsed(
          context.flights_sorted_by_flow,
          n_by_flow,
          indexer,
          spill_mode="dump_to_next_bin",
      )
      ```
    - Scales J_total such that J_cap is multiplied by 100, changing J’s absolute scale:
      ```120:131:src/parrhesia/flow_agent/safespill_objective.py
      J_total = 100.0 * J_cap + J_delay + J_reg + J_tv + J_share + J_spill
      ```
  - The generic scorer (not used here) calculates:
    ```376:376:src/parrhesia/optim/objective.py
    J_total = J_cap + J_delay + J_reg + J_tv + J_share + J_spill
    ```
    (No 100× factor; default spill_mode differs.)
- Alpha weighting and attention:
  - J_cap is weighted by α per TV/bin: target cells (hotspot at specified bins) get α_gt, ripple α_rip (unused here), all others α_ctx. This emphasizes hotspot bins but still includes contributions from all TVs/time under `tv_filter`.
  - The hotspot exceedance and predicted deficit metrics are unweighted raw exceedance measures.
- Objective terms present vs absent:
  - J includes capacity exceedance, delay, regularisation, total variation, and possibly fairness/spill components.
  - The hotspot exceedance and predicted deficit are capacity-only measures.

### Bottom line: when these metrics are not directly comparable
- TV scope mismatch: ΔJ covers all TVs in `tv_filter`; hotspot metrics cover only the hotspot.
- Time scope mismatch: ΔJ can change outside the window (delays/spill ripple); hotspot metrics are window-limited.
- Rolling-hour subtraction mismatch: predicted deficit currently subtracts K× the hourly capacity (given the bench’s capacity arrays), whereas J_cap and hotspot exceedance subtract the hourly cap once per rolling window.
- Component/scale mismatch:
  - ΔJ includes multiple terms beyond capacity exceedance; hotspot metrics do not.
  - Safe-spill J scales J_cap by 100 and uses “dump_to_next_bin”; the generic scorer doesn’t.

If you want the metrics to be comparable, either:
- Convert capacity arrays to per-bin values (hourly_cap/K) for `compute_delta_deficit_per_hour`, or change it to subtract a single hourly cap per rolling window (not sum over K bins), and
- Align TV/time scopes (e.g., compute J only over hotspot/time window) and/or report J_cap-only deltas on the hotspot to match the exceedance scope, and
- Use the same scorer variant and spill semantics (safe-spill vs generic) and same J_total scaling.

- Engine scoring and scopes:
```254:263:src/parrhesia/flow_agent35/regen/engine.py
context = build_local_context(... tv_filter=tv_filter, target_cells=target_cells ...)
```
- Pairwise scoring and delta-deficit computation:
```311:327:src/parrhesia/flow_agent35/regen/engine.py
) = score_pair(...)
delta_deficit = compute_delta_deficit_per_hour(
    occ_before,
    occ_after,
    capacities_by_tv=capacities_by_tv,
    target_cells=target_cells,
    indexer=indexer,
    window=window,
)
```

Summary
- ΔJ uses safe-spill objective across all TVs in `tv_filter` and entire horizon with α-weighted J_cap (scaled by 100), delay, regularisation, and TV smoothness; spills use “dump_to_next_bin”.
- Hotspot exceedance and predicted deficit are hotspot-only, window-only capacity measures.
- Predicted deficit currently subtracts K× hourly capacity due to how capacities are built in the bench; J_cap and exceedance subtract the hourly cap once. This and the scope/scale differences make these numbers non-comparable without adjustments.


# Remarks
- The objective’s scoring runs over all time bins in the horizon (e.g., full day), not just the hotspot window.
- The hotspot’s bins (those in `target_cells`) are given higher weight in the capacity-exceedance term (J_cap) via α-weights (e.g., α_gt > α_ctx). Other terms (delay, regularization, TV smoothness) are not time-window weighted.
- Separate metrics like hotspot exceedance/“delta_deficit_per_hour” are computed only on the hotspot and only over its selected window (unweighted).