### Goal
Implement a script in `src/project_tailwind/flow_x/cache_flow_extract.py` that:
- Exports all hotspot bins to a CSV (tvtw_index, traffic_volume_id, time_bin).
- For each hotspot, extracts flows (groups) and writes a second CSV with one row per flow: reference TV, hotspot TV, avg similarity, score, flight IDs, plus cached metrics CountOver(t), SumOver(t), MinSlack(t) for t=0..15.

### Assumptions and grounding
- Hotspot items and flight lists come from `NetworkEvaluator` and `FlightList`.
- Flow extraction uses `FlowXExtractor.find_groups_from_evaluator_item(...)` returning multiple groups per hotspot.
- Excess logic compares hourly capacity against rolling-hour occupancy per bin.

Key APIs to rely on:
```310:388:src/project_tailwind/optimize/eval/network_evaluator.py
def get_hotspot_flights(...):
    ...
    if mode == "bin":
        return [{"tvtw_index": int(tvtw_idx),
                 "flight_ids": [...],
                 "hourly_capacity": float(hourly_capacity),
                 "capacity_per_bin": float(capacity_per_bin)}]
    if mode == "hour":
        return [{"traffic_volume_id": str, "hour": int, "flight_ids": [...],
                 "hourly_occupancy": float, "unique_flights": int}]
```

```827:850:src/project_tailwind/flow_x/flow_extractor.py
def find_groups_from_evaluator_item(self, hotspot_item: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
    # returns list of groups with keys:
    # "reference_sector", "group_flights", "score", "avg_pairwise_similarity",
    # "group_size", "mean_path_length", "hotspot": {"traffic_volume_id","hour"}
```

```195:229:src/project_tailwind/optimize/eval/flight_list.py
def shift_flight_occupancy(self, flight_id: str, delay_minutes: int) -> np.ndarray:
    # shifts per 15-min bin; forward shifts zero-fill on the left
```

### Inputs
- Required files:
  - `output/so6_occupancy_matrix_with_times.json`
  - `output/tvtw_indexer.json`
  - TV GeoJSON with capacities (path provided by user; default to the test path if not given).
- CLI options (proposed):
  - `--mode {bin,hour}` (default: hour for flow extraction; hotspots CSV uses bin mode)
  - `--threshold` (default: 0.0)
  - `--output-dir` (default: `output/flow_dump`)
  - FlowX knobs: `--max-groups`, `--k-max`, `--alpha`, `--avg-objective`, `--group-size-lam`, `--path-length-gamma`
  - Scoping: `--only-tv`, `--only-hour`, `--only-tvtw`, `--limit-hotspots`
  - Cache horizon: fixed t=0..15 (per prompt)

### Outputs
- CSV 1: `hotspots.csv` (per overloaded bin)
  - Required: `tvtw_index`, `traffic_volume_id`, `time_bin`
  - Recommended extras: `hour`, `hourly_capacity`, `hourly_occupancy`, `capacity_per_bin`, `num_flights`
- CSV 2: `flows_with_cache.csv` (one row per flow)
  - `hotspot_traffic_volume_id`, `hotspot_hour` (and/or `hotspot_tvtw_index` if bin-mode)
  - `reference_sector`, `group_size`, `avg_pairwise_similarity`, `score`, `mean_path_length`
  - `flight_ids` (whitespace-separated)
  - Cache columns:
    - `count_over_0..15`
    - `sum_over_0..15`
    - `min_slack_0..15`

### High-level flow
1. Load `FlightList` and TV GeoJSON; init `NetworkEvaluator`.
2. Build hotspots list in bin mode; export CSV 1 with derived `traffic_volume_id` and `time_bin` from `tvtw_index`.
3. For flow extraction, select hotspots (hour- or bin-mode) per CLI scope.
4. For each hotspot, run `FlowXExtractor.find_groups_from_evaluator_item(...)` to get groups.
5. Precompute global structures once:
   - `occ_base = flight_list.get_total_occupancy_by_tvtw()`
   - `bins_per_hour`, `num_time_bins_per_tv`, `tv_row_of_tvtw`, `hour_of_tvtw`
   - `cap_per_bin` via evaluator’s per-bin distribution logic
   - `hourly_capacity_matrix` shape `[num_tvs, 24]`
   - `hourly_occ_base` (from evaluator cache or computed from `occ_base`)
6. For each group:
   - Build `g0` = sum of per-flight occupancy vectors (counts), and `g0_bool = (g0 > 0)`.
   - For each t ∈ [0..15], compute caches (vectorized; see below).
7. Append one CSV row per flow with metadata and all cached values.

### Detailed cache computation (vectorized; no heavy recomputation per t)
- Notation:
  - g0: base group occupancy counts (1D, length num_tvtws)
  - g_t: g0 shifted right by t bins (in time) globally with zero-fill
  - delta_t = g_t - g0
  - occ_t = occ_base + delta_t
  - For per-hour aggregation, define for TV row r and hour h:
    - indices(r,h) = contiguous block of `bins_per_hour` tvtws
- Hourly occupancy and excess:
  - Precompute `hourly_occ_base[r,h] = sum(occ_base[indices(r,h)])`.
  - Precompute `group_hourly_counts(g) = sum(g[indices(r,h)])` (counts, not boolean).
  - Then `hourly_occ_t = hourly_occ_base + group_hourly_counts(g_t) - group_hourly_counts(g0)`.
  - `hourly_excess_t = maximum(hourly_occ_t - hourly_capacity_matrix, 0)`.
- CountOver(t):
  - Let `present_bins_t[r,h] = number of tvtw bins in indices(r,h) where g_t > 0`.
  - `CountOver(t) = sum over (r,h) where hourly_excess_t > 0 of present_bins_t[r,h]`.
- SumOver(t):
  - `excess_per_bin(r,h) = hourly_excess_t[r,h] / bins_per_hour` for hours with excess > 0.
  - `SumOver(t) = sum over (r,h) of present_bins_t[r,h] * excess_per_bin(r,h)`.
- MinSlack(t):
  - `slack_per_bin = maximum(cap_per_bin - occ_t, 0)`.
  - Mask to group-occupied bins: `mask_t = (g_t > 0)`.
  - `MinSlack(t) = min(slack_per_bin[mask_t])` (if mask empty, treat as None or NaN).

Performance notes:
- Build `g0` once per group; compute `g_t` by slicing/rolling with fill zeros (no per-flight loops).
- Use precomputed `tv_row_of_tvtw` and `hour_of_tvtw` to aggregate counts per (tv,h) with `np.unique`/`np.add.at` without Python loops.
- Avoid calling `NetworkEvaluator.compute_excess_traffic_vector()` in the t-loop; compute hourly_excess_t via the aggregator math above. This mirrors evaluator logic exactly.
- If a correctness-first baseline is desired, implement a fallback that uses `DeltaFlightList` and evaluator calls per t (slower but simpler), behind a `--slow` flag.

### Edge cases
- Groups with size < 2: skip (extractor already enforces min).
- Missing capacity hours: treat as 0 (no positive slack), consistent with evaluator’s usage.
- Shifts beyond day-end: dropped bins are discarded (zero-fill).
- Duplicated flights across groups: acceptable as separate rows.
- Empty flows for a hotspot: write none for that hotspot.

### Testing
- Smoke test: Run against `tests/test_hotspot_flow_retrieval.py`’s setup to pick a known hotspot (e.g., tv=MASB5KL, hour=6), extract flows, compute caches t=0..15, and verify:
  - CSV 1 has expected columns and non-empty rows.
  - CSV 2 rows match group metadata from extractor and contain 16 values per cache metric.
  - Monotonic sanity: `MinSlack(t)` typically non-decreasing for larger t if delays reduce overload.
- Validate that for t=0, `CountOver(0)` equals the count of overloaded bins in baseline where group is present.

### Implementation outline
- CLI + I/O scaffolding
- Hotspot discovery/export (bin-mode)
- Flow extraction loop
- Precompute base arrays and capacity mappings
- Group-level cache computation (vectorized)
- CSV writing

### Deliverables
- `src/project_tailwind/flow_x/cache_flow_extract.py`
- Two CSVs under `output/flow_dump/`: `hotspots.csv`, `flows_with_cache.csv`
- Optional: a short README section in `docs/flow_x/` describing usage

- Ensured plan aligns with existing APIs and data model, with a clear vectorized approach for per-flow cache metrics.
- Be extra careful about tvtw indexing when you access the elements.
