# FlowX Extractor

Finds spatially coherent groups of flights feeding a hotspot (traffic volume at a given hour/bin) using a spectral relaxation over a flight–flight similarity graph. It operates directly on precomputed occupancy metadata from `FlightList` and is compatible with `NetworkEvaluator.get_hotspot_flights` outputs.

## Goals
- Identify upstream flows of flights that likely contribute to a hotspot.
- Do so robustly across sector granularity and time discretization.
- Provide single-best group or peel multiple disjoint groups with clear scores.

## Data Model
- Sector universe `S`: TV rows (traffic volume rows) indexed by integer row id.
- Time bins: fixed-size minutes per bin (`FlightList.time_bin_minutes`), `bins_per_hour = 60 // time_bin_minutes`.
- `A[f][t]`: for each flight `f`, set of TV rows occupied at time-of-day bin `t`.
- Hotspot `H`: given by `traffic_volume_id` (or `tvtw_index`) and an hour; `H_tv_row` is the row index for the hotspot’s TV.
- `H_entry_by_flight[f]`: earliest time bin within the hotspot hour when flight `f` occupies `H_tv_row`.

`FlightList` requirements (duck-typed):
- Attributes: `time_bin_minutes`, `num_tvtws`, `tv_id_to_idx: Dict[str,int]`, `flight_metadata`.
- Each `flight_metadata[fid]["occupancy_intervals"]` contains items with `{"tvtw_index": int}`.

## Algorithm Overview
1. Build per-flight time-binned sets `A[f][t]` from `FlightList.flight_metadata`.
2. Determine each candidate flight’s earliest entry into the hotspot sector during the hotspot hour to get `H_entry_by_flight`.
3. Collect upstream candidate references `R`: TVs seen before hotspot entry among the candidate flights; filter by minimum number of supporting flights and optional cap.
4. For each reference `r ∈ R`:
   - Align sequences: for each flight that passes `r` before `H`, take the slice `[t_r, t_H)` of per-bin TV-row sets to form aligned sequences `A_r[f]`.
   - Build a flight–flight similarity matrix `W` using per-bin weighted Jaccard, aggregated by mean across bins; optionally sparsify by `tau` (absolute) or `alpha` (adaptive).
   - Spectral relaxation: take top eigenvector of `W` (or degree-normalized), order flights by its values, and sweep group sizes to optimize an objective.
   - Keep the best-scoring group for this `r`.
5. Choose the best group over all `r` (or peel multiple groups iteratively, removing selected flights and repeating until limits are reached).

## Similarity Function
- Per-bin weighted Jaccard over sets of TV rows:
  - Weights: down-weight ubiquitous sectors by `w[s] = 1 / log(2 + freq[s])`, where `freq[s]` counts appearances across all aligned sequences and bins for the current `r`.
  - For flights `a,b`, compute similarity at bin `t` as `J_w(A_r[a][t], A_r[b][t])`; overall pairwise similarity is the mean over aligned bins.
- Optional sparsification:
  - `tau`: zero out `W[i,j] < tau`.
  - `alpha`: compute mean/std of off-diagonals; keep entries `>= mean - alpha*std`.

## Spectral Relaxation + Threshold Sweep
- Matrix choice: `W` or `D^{-1/2} W D^{-1/2}` if `normalize_by_degree=True`.
- Compute top eigenvector and order flights descending by its entries.
- Sweep `k = 1..k_max` (default `m`) over the ordered list; maintain cumulative pairwise sum within the top-`k` subset.
- Objective options:
  - Average objective (default): maximize average pairwise similarity within the set, i.e., `pair_sum / (k*(k-1)/2)` for `k ≥ 2`.
  - Linear penalty: maximize `pair_sum - lam*k` with `lam ≥ 0`.
- Select the `k` with the best score; enforce `min_group_size` (defaults applied at call sites).

## Peeling Procedure (Multi-Group)
- `find_groups_from_hotspot_*` repeats extraction up to `max_groups`.
- After selecting a group, remove its flights from the candidate set and recompute from step 1.
- Returns groups in extraction order, each with `group_rank`.

## Public API
- Constructor: `FlowXExtractor(flight_list)`
  - Validates presence of required `FlightList` attributes; no heavy imports at runtime.

- `find_group_from_hotspot_hour(hotspot_tv_id, hotspot_hour, candidate_flight_ids, *, min_flights_per_ref=3, max_references=20, tau=None, alpha=0.0, lam=0.0, normalize_by_degree=False, average_objective=True, k_max=None, max_groups=None)` → `Dict`
  - Computes the best single group; thin wrapper over `find_groups_from_hotspot_hour` returning the first/best group or an empty result.

- `find_groups_from_hotspot_hour(hotspot_tv_id, hotspot_hour, candidate_flight_ids, *, min_flights_per_ref=3, max_references=20, tau=None, alpha=0.0, lam=0.0, normalize_by_degree=False, average_objective=True, k_max=None, max_groups=3, min_group_size=2)` → `List[Dict]`
  - Multi-group peeling. See parameters below.

- `find_group_from_hotspot_bin(hotspot_tvtw_index, candidate_flight_ids, **kwargs)` / `find_groups_from_hotspot_bin(...)`
  - Convenience wrappers that derive `(tv_row, hour)` from `tvtw_index` and delegate to the “hour” variants.

- `find_group_from_evaluator_item(hotspot_item, **kwargs)` / `find_groups_from_evaluator_item(...)`
  - Accepts a single item from `NetworkEvaluator.get_hotspot_flights`:
    - Hour-mode: `{ "traffic_volume_id", "hour", "flight_ids" }`.
    - Bin-mode: `{ "tvtw_index", "flight_ids" }`.

### Parameters (key ones)
- `hotspot_tv_id` / `hotspot_tvtw_index`: identifies the hotspot sector and time.
- `candidate_flight_ids`: flights to consider upstream of the hotspot.
- `min_flights_per_ref`: minimum distinct flights that must pass a reference sector `r` before `H` to keep `r` as a candidate.
- `max_references`: cap the number of candidate reference sectors (by frequency).
- `tau`, `alpha`: sparsification of similarity matrix (`tau` absolute threshold, `alpha` adaptive thresholding).
- `lam`: linear size penalty (only if `average_objective=False`).
- `normalize_by_degree`: use `D^{-1/2} W D^{-1/2}` prior to eigen decomposition.
- `average_objective`: if true, optimize average pairwise similarity; otherwise use linear penalty.
- `k_max`: cap the maximum group size considered by the sweep.
- `max_groups`, `min_group_size`: peeling controls in the multi-group variant.

### Return Schema (per group)
- `reference_sector: str | None`
- `group_flights: List[str]`
- `score: float` (best objective value from the sweep)
- `avg_pairwise_similarity: float` (within selected group)
- `group_size: int`
- `hotspot: { "traffic_volume_id": str, "hour": int }`
- `group_rank: int` (only in multi-group results)

## Example Usage

### From hotspot hour
```python
from project_tailwind.flow_x.flow_extractor import FlowXExtractor
# Assume you already built `flight_list` and have candidate flights for a hotspot
fx = FlowXExtractor(flight_list)
result = fx.find_group_from_hotspot_hour(
    hotspot_tv_id="TV_ABC123",
    hotspot_hour=14,
    candidate_flight_ids=candidate_fids,
    min_flights_per_ref=3,
    max_references=20,
    tau=None,           # or e.g., 0.3
    alpha=1.0,          # adaptive sparsification; use one of tau or alpha
    lam=0.0,
    normalize_by_degree=False,
    average_objective=True,
    k_max=None,
)
print(result["reference_sector"], result["group_size"], result["avg_pairwise_similarity"])  # noqa
```

### Peel multiple groups
```python
groups = fx.find_groups_from_hotspot_hour(
    hotspot_tv_id="TV_ABC123",
    hotspot_hour=14,
    candidate_flight_ids=candidate_fids,
    max_groups=3,
    min_group_size=2,
    alpha=1.0,
)
for g in groups:
    print(g["group_rank"], g["reference_sector"], g["group_size"])  # noqa
```

### Using `tvtw_index` or evaluator item
```python
# Bin-mode
g1 = fx.find_group_from_hotspot_bin(hotspot_tvtw_index=123456, candidate_flight_ids=candidate_fids)

# Directly from evaluator item
item = {"traffic_volume_id": "TV_ABC123", "hour": 14, "flight_ids": candidate_fids}
g2 = fx.find_group_from_evaluator_item(item, alpha=1.0)
```

## Practical Tips
- Start with `alpha ≈ 1.0` to sparsify weak similarities; prefer `alpha` over `tau` when scales vary.
- Use `average_objective=True` for scale-invariant grouping; switch to `lam > 0` only when you want to bias against large groups.
- Increase `min_flights_per_ref` to ignore noisy references; adjust `max_references` to limit compute.
- Ensure `FlightList.flight_metadata[fid]["occupancy_intervals"]` is populated; missing or empty intervals will skip flights.

## Complexity Notes
- Building `A` is linear in total occupancy intervals.
- For each reference `r`, similarity is `O(m^2 · T_r)` where `m` is flights passing `r` and `T_r` is aligned length; spectral step is dense `eigh` on `m×m` (fallback to power iteration on failure).
- Peeling runs multiple passes, each on a shrinking set of flights.

## Edge Cases
- Unknown `traffic_volume_id` → `ValueError`.
- If too few flights enter the hotspot hour or pass a reference, methods return empty results.
- With `m < 2` aligned flights, similarity matrices are empty; groups below `min_group_size` are discarded.

