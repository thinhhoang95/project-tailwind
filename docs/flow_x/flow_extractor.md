## FlowX Extractor

Finds spatially coherent groups of flights feeding a hotspot (traffic volume at a given hour/bin) using a spectral relaxation over a flight–flight similarity graph. Operates directly on precomputed occupancy metadata from `FlightList` and is compatible with `NetworkEvaluator.get_hotspot_flights` outputs.

## Goals
- Identify upstream flows of flights that likely contribute to a hotspot.
- Be robust across sector granularity and time discretization.
- Return top group(s) with clear, comparable scores.

## Data Model
- Sector universe `S`: TV rows (traffic volume rows) indexed by integer row id.
- Time bins: fixed-size minutes per bin (`FlightList.time_bin_minutes`), `bins_per_hour = 60 // time_bin_minutes`.
- `A[f][t]`: for each flight `f`, set of TV rows occupied at time-of-day bin `t`.
- Hotspot `H`: identified by `traffic_volume_id` (or `tvtw_index`) and an hour; `H_tv_row` is the row index for the hotspot’s TV.
- `H_entry_by_flight[f]`: earliest time bin within the hotspot hour when flight `f` occupies `H_tv_row`.

`FlightList` requirements (duck-typed):
- Attributes: `time_bin_minutes`, `num_tvtws`, `tv_id_to_idx: Dict[str,int]`, `flight_metadata`.
- Each `flight_metadata[fid]["occupancy_intervals"]` contains items with `{"tvtw_index": int}`.

## Algorithm Overview
1. Build per-flight time-binned sets `A[f][t]` from `FlightList.flight_metadata`.
2. Determine earliest entry into the hotspot sector during the hotspot hour for each candidate flight to get `H_entry_by_flight`.
3. Collect upstream candidate references `R`: TVs seen before hotspot entry among the candidate flights; filter by minimum supporting flights and optional cap. By default, reference collection allows the hotspot bin to be included to account for time-bin quantization.
4. For each reference `r` in `R`:
   - Align sequences: for each flight that passes `r` before (or at the same bin as) `H`, take the slice `[t_r, t_H)` of per-bin TV-row sets to form aligned sequences `A_r[f]`. Require at least 2 aligned bins.
   - Build a flight–flight similarity matrix `W` using unweighted Jaccard over the union of TVs across the aligned window (time-agnostic); optionally sparsify by `tau` (absolute) or `sparsification_alpha` (adaptive).
   - Spectral relaxation: take top eigenvector of `W` (optionally degree-normalized), order flights by its values, and sweep group sizes to optimize an objective.
   - Keep the best-scoring group for this `r`.
5. Choose top group(s) across all `r` (with optional collapsing of duplicate groups by membership) and return up to `max_groups`.

## Similarity Function
- Unweighted Jaccard over the union of TVs across the aligned window between `r` and `H` for each flight.
- Optional sparsification:
  - `tau`: zero out `W[i,j] < tau`.
  - `sparsification_alpha`: compute mean/std of off-diagonals; keep entries `>= mean - sparsification_alpha*std`.
- Note: weighted or per-bin Jaccard has been removed; weighted Jaccard is deprecated in the implementation.

## Spectral Relaxation + Threshold Sweep
- Matrix choice: `W` or `D^{-1/2} W D^{-1/2}` if `normalize_by_degree=True`.
- Compute top eigenvector and order flights descending by its entries.
- Sweep `k = 1..k_max_trajectories_per_group` (default `m`) over the ordered list; maintain cumulative pairwise sum within the top-`k` subset.
- Objective options:
  - Average objective (default): maximize average pairwise similarity within the set, i.e., `pair_sum / (k*(k-1)/2)`.
  - Linear penalty: maximize `pair_sum - group_size_lam*k` with `group_size_lam >= 0`.
- Enforce `min_group_size` when ranking/returning groups.
- Optional score shaping: `path_length_gamma > 0` adds `path_length_gamma * mean_path_length` to the spectral score, favoring longer aligned paths.

## Peeling and Ranking
- `find_groups_from_hotspot_*` evaluates all candidate references and maintains the top `max_groups` by score.
- With `auto_collapse_group_output=True` (default), groups with identical membership are collapsed to the highest-scoring instance before ranking.
- Groups are returned with `group_rank` assigned by descending score.

## Public API
- Constructor: `FlowXExtractor(flight_list, debug_verbose_path: Optional[str] = "output/flow_extractor")`
  - Validates presence of required `FlightList` attributes.

- `find_group_from_hotspot_hour(hotspot_tv_id, hotspot_hour, candidate_flight_ids, *, auto_collapse_group_output=True, min_flights_per_ref=3, max_references: Optional[int]=500, tau=None, alpha_sparsification: Optional[float]=0.0, group_size_lam=0.0, normalize_by_degree=False, average_objective=True, k_max_trajectories_per_group=None, max_groups=None, path_length_gamma: float=None, debug_verbose_path: Optional[str]=None)` → returns list of groups when found, otherwise an empty-result dict.
  - Thin wrapper over `find_groups_from_hotspot_hour`. Note: current implementation returns the full list of groups (not just the first) when any are found.

- `find_groups_from_hotspot_hour(hotspot_tv_id, hotspot_hour, candidate_flight_ids, *, auto_collapse_group_output=True, min_flights_per_ref=3, max_references=500, tau=None, sparsification_alpha: Optional[float]=0.0, group_size_lam=0.0, normalize_by_degree=False, average_objective=True, k_max_trajectories_per_group=None, max_groups=3, min_group_size=2, path_length_gamma=0.0, debug_verbose_path: Optional[str]=None)` → `List[Dict]`
  - Multi-reference evaluation with optional collapsing and score shaping.

- `find_group_from_hotspot_bin(hotspot_tvtw_index, candidate_flight_ids, **kwargs)` / `find_groups_from_hotspot_bin(...)`
  - Convenience wrappers that derive `(tv_row, hour)` from `tvtw_index` and delegate to the hour variants.

- `find_group_from_evaluator_item(hotspot_item, **kwargs)` / `find_groups_from_evaluator_item(...)`
  - Accepts a single item from `NetworkEvaluator.get_hotspot_flights`:
    - Hour-mode: `{ "traffic_volume_id", "hour", "flight_ids" }`.
    - Bin-mode: `{ "tvtw_index", "flight_ids" }`.

### Parameters (key ones)
- `hotspot_tv_id` / `hotspot_tvtw_index`: identifies the hotspot sector and time.
- `candidate_flight_ids`: flights to consider upstream of the hotspot.
- `min_flights_per_ref`: minimum distinct flights that must pass a reference sector `r` before `H`.
- `max_references`: cap the number of candidate reference sectors (default 500 in `find_groups_from_hotspot_hour`).
- `tau`, `sparsification_alpha`: sparsify the similarity matrix (`tau` absolute threshold; `sparsification_alpha` adaptive thresholding). In `find_group_from_hotspot_hour` the argument name is `alpha_sparsification`.
- `group_size_lam`: linear size penalty coefficient used when `average_objective=False`.
- `normalize_by_degree`: use `D^{-1/2} W D^{-1/2}` prior to eigen decomposition.
- `average_objective`: if true, optimize average pairwise similarity; otherwise use linear penalty with `group_size_lam`.
- `k_max_trajectories_per_group`: cap the maximum group size considered by the sweep.
- `max_groups`, `min_group_size`: ranking and minimum-size controls.
- `auto_collapse_group_output`: collapse duplicate groups by membership before ranking.
- `path_length_gamma`: optional positive weight to favor longer aligned paths.

### Return Schema (per group)
- `reference_sector: str | None`
- `group_flights: List[str]`
- `score: float` (spectral score optionally adjusted by `path_length_gamma`)
- `avg_pairwise_similarity: float` (within selected group)
- `group_size: int`
- `mean_path_length: float` (average number of aligned bins per group flight)
- `hotspot: { "traffic_volume_id": str, "hour": int }`
- `group_rank: int` (present when ranking multiple groups)

## Example Usage

### From hotspot hour
```
from project_tailwind.flow_x.flow_extractor import FlowXExtractor
# Assume you already built `flight_list` and have candidate flights for a hotspot
fx = FlowXExtractor(flight_list)
groups_or_result = fx.find_group_from_hotspot_hour(
    hotspot_tv_id="TV_ABC123",
    hotspot_hour=14,
    candidate_flight_ids=candidate_fids,
    min_flights_per_ref=3,
    max_references=500,
    tau=None,                 # or e.g., 0.3
    alpha_sparsification=1.0, # adaptive sparsification; use one of tau or alpha_sparsification
    group_size_lam=0.0,
    normalize_by_degree=False,
    average_objective=True,
    k_max_trajectories_per_group=None,
    path_length_gamma=0.0,
)
```

### Multiple ranked groups
```
groups = fx.find_groups_from_hotspot_hour(
    hotspot_tv_id="TV_ABC123",
    hotspot_hour=14,
    candidate_flight_ids=candidate_fids,
    max_groups=3,
    min_group_size=2,
    sparsification_alpha=1.0,
    auto_collapse_group_output=True,
)
for g in groups:
    print(g["group_rank"], g["reference_sector"], g["group_size"])  # noqa
```

### Using `tvtw_index` or evaluator item
```
# Bin-mode
g1 = fx.find_group_from_hotspot_bin(
    hotspot_tvtw_index=123456,
    candidate_flight_ids=candidate_fids,
    sparsification_alpha=1.0,
)

# Directly from evaluator item
item = {"traffic_volume_id": "TV_ABC123", "hour": 14, "flight_ids": candidate_fids}
g2 = fx.find_group_from_evaluator_item(item, sparsification_alpha=1.0)
```

## Practical Tips
- Prefer `sparsification_alpha ~ 1.0` to prune weak similarities when similarity scales vary across references.
- Use `average_objective=True` for scale-invariant grouping; use `group_size_lam > 0` only when you want to bias against large groups with a linear penalty.
- Increase `min_flights_per_ref` to ignore noisy references; adjust `max_references` (default 500) to limit compute.
- Ensure `FlightList.flight_metadata[fid]["occupancy_intervals"]` is populated; missing or empty intervals will skip flights.

## Complexity Notes
- Building `A` is linear in total occupancy intervals.
- For each reference `r`, similarity is `O(m^2 * T_r)` where `m` is flights passing `r` and `T_r` is aligned length; spectral step uses dense `eigh` (with power-iteration fallback).
- Ranking runs across many references; only the top `max_groups` are retained.

## Edge Cases
- Unknown `traffic_volume_id` → `ValueError`.
- If too few flights enter the hotspot hour or pass a reference, methods return empty results.
- With fewer than `min_group_size` aligned flights, groups are discarded.
