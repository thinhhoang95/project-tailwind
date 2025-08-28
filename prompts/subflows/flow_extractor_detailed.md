### Plan overview
- Implement a focused submodule in `src/project_tailwind/subflows/flow_extractor.py` with:
  - Footprint extraction helpers (leveraging `FlightList`)
  - Vectorized Jaccard similarity computation
  - Graph construction and Leiden community detection
  - A single public wrapper to run the full flow for a hotspot item
- Add small, surgical extensions to `src/project_tailwind/optimize/eval/flight_list.py` to expose efficient traffic-volume footprints.
- Write a synthetic unit test validating footprints, Jaccard, and clustering.

### Inputs and expected outputs
- **Inputs**
  - `FlightList` instance (existing)
  - One hotspot item from `NetworkEvaluator.get_hotspot_flights(...)`:
    - mode "bin": `{"tvtw_index": int, "flight_ids": List[str], ...}`
    - mode "hour": `{"traffic_volume_id": str, "hour": int, "flight_ids": List[str], ...}`
  - `threshold` for graph construction from similarity, optional `resolution` for Leiden.
- **Output**
  - Dict mapping `flight_identifier -> community_index` (0-based consecutive).

### Step 1 — Extend `FlightList` for footprints
Add minimal, vectorized helpers without altering existing behavior.

- **New derived attributes (computed once on init or lazily cached):**
  - `num_time_bins_per_tv: int = 1440 // self.time_bin_minutes`
  - `idx_to_tv_id: Dict[int, str]` built from `tv_id_to_idx` (inverse map)
- **Helpers**
  - `get_flight_tv_sequence_indices(flight_id: str) -> np.ndarray`
    - Build from `flight_metadata[flight_id]["occupancy_intervals"]` sorted by `entry_time_s`.
    - Map each `tvtw_index -> tv_idx = tvtw_index // num_time_bins_per_tv` via vectorized integer division.
    - Optionally compress consecutive duplicates (stay within same TV over multiple bins).
  - `get_flight_tv_footprint_indices(flight_id: str, hotspot_tv_index: int | None = None) -> np.ndarray`
    - Compute sequence; if `hotspot_tv_index` is given, prefix-trim sequence up to and including the first occurrence of that TV index.
    - Return unique TV indices in that prefix (as a sorted or insertion-order-unique set; order is irrelevant for Jaccard).
  - `get_footprints_for_flights(flight_ids: Sequence[str], hotspot_tv_index: int | None) -> List[np.ndarray]`
    - Batch convenience; vectorize mapping and deduplication where possible.
- **Performance/caching**
  - Cache per-flight TV sequence arrays; computing them once allows fast prefix cuts later.
  - Use NumPy arrays for sequence and set ops; avoid Python loops where possible.

### Step 2 — Jaccard similarity (vectorized)
Implement `compute_jaccard_similarity(footprints: List[np.ndarray]) -> np.ndarray` with a sparse-friendly path:

- Compress TV indices to a local contiguous space:
  - Build `all_tv = sorted(set().union(*[set(fp) for fp in footprints]))`
  - Map `global_tv_idx -> local_tv_idx` to reduce matrix width.
- Build a binary CSR matrix `X` of shape `(n_flights, n_local_tvs)` with ones at footprint positions. This is fast to construct from `indptr/indices` arrays.
- Compute pairwise intersection counts as `M = X @ X.T` (sparse-sparse matmul).
- Row sums `s = X.sum(axis=1).A1` give set sizes. For each nonzero entry `(i, j)` in `M`:
  - `intersection = M[i, j]`
  - `union = s[i] + s[j] - intersection`
  - `similarity = intersection / union` (define 0 if union == 0)
- Produce a dense `S` only if `n_flights` is small; otherwise keep edges in COO and materialize only needed entries later. Always set `S[i, i] = 1.0`.

Notes:
- This matches the spec’s “similarity” (Jaccard distance is `1 - similarity` but we use similarity as edge weights for Leiden).
- Vectorization: CSR build + sparse matmul + vectorized union/ratio on nonzero pairs avoids nested Python loops.

### Step 3 — Build graph and run Leiden
Implement `run_leiden_from_similarity(S or edges, threshold: float = 0.1, resolution: float = 1.0, seed: int | None = None) -> List[int]`:

- If working with a dense `S` (small n), build edges `i<j` where `S[i, j] >= threshold`, weight = `S[i, j]`.
- If using sparse COO from Step 2, filter nonzero pairs by threshold directly; no densification.
- Create an undirected `igraph.Graph(n=n_flights, edges=edges, directed=False)` and set edge attribute `weight`.
- Partition via `leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=resolution, seed=seed)` and return `membership`.

Edge cases:
- `n_flights <= 1`: return single community.
- If no edges after thresholding: fallback to lower threshold or treat as all singletons (document behavior; default to singletons for deterministic output).

### Step 4 — Public wrapper
Implement `assign_communities_for_hotspot(flight_list: FlightList, hotspot_item: Dict[str, Any], mode: Literal["bin","hour"], threshold: float = 0.1, resolution: float = 1.0, seed: int | None = None) -> Dict[str, int]`:

- Extract `flight_ids = hotspot_item["flight_ids"]`.
- Determine `hotspot_tv_index`:
  - mode "hour": `tv_id = hotspot_item["traffic_volume_id"]` -> `hotspot_tv_index = flight_list.tv_id_to_idx[tv_id]`
  - mode "bin": `tvtw_index = hotspot_item["tvtw_index"]` -> `hotspot_tv_index = tvtw_index // flight_list.num_time_bins_per_tv`
- Build footprints with prefix trimming: `fps = flight_list.get_footprints_for_flights(flight_ids, hotspot_tv_index)`.
- Compute Jaccard similarity; run Leiden; map membership back to the input `flight_ids` in order:
  - Return `{flight_id: community_index}`.

### API surface (proposed signatures)
- In `flight_list.py`:
  - `get_flight_tv_sequence_indices(flight_id: str) -> np.ndarray`
  - `get_flight_tv_footprint_indices(flight_id: str, hotspot_tv_index: int | None = None) -> np.ndarray`
  - `get_footprints_for_flights(flight_ids: Sequence[str], hotspot_tv_index: int | None) -> List[np.ndarray]`
  - `num_time_bins_per_tv: int` and `idx_to_tv_id: Dict[int, str]` properties
- In `subflows/flow_extractor.py`:
  - `compute_jaccard_similarity(footprints: List[np.ndarray]) -> np.ndarray | Tuple[np.ndarray, Any]`
  - `run_leiden_from_similarity(S_or_edges, threshold: float = 0.1, resolution: float = 1.0, seed: int | None = None) -> List[int]`
  - `assign_communities_for_hotspot(flight_list: FlightList, hotspot_item: Dict[str, Any], mode: Literal["bin","hour"], threshold: float = 0.1, resolution: float = 1.0, seed: int | None = None) -> Dict[str, int]`

### Unit test (synthetic)
Add `tests/test_subflows_flow_extractor.py`:

- Build tiny synthetic files on tmpfs:
  - `tvtw_indexer.json`: `time_bin_minutes=15`, `tv_id_to_idx={"TVA":0,"TVB":1,"TVC":2,"TVD":3}`
  - `so6_occupancy_matrix_with_times.json`: 6 flights with clear route patterns:
    - Group 1 (A1,A2,A3): TVA→TVB→TVC before TVB
    - Group 2 (B1,B2,B3): TVA→TVD before TVB
  - Ensure at least one hour/bin corresponds to TVB being overloaded so hotspot selection includes those flights.
- Create `FlightList`, synthesize a hotspot item (mode "hour" with `traffic_volume_id="TVB"` and `flight_ids`).
- Assertions:
  - `get_flight_tv_sequence_indices` order is correct and dedupes consecutive TVs.
  - Prefix slicing with `hotspot_tv_index=tv_id_to_idx["TVB"]` yields expected footprints.
  - Jaccard similarities: within-group > between-group (check a few pairwise values).
  - Leiden partitions two communities; output maps `flight_id -> {0,1}` with consistent grouping.
  - Edge-case: single-flight hotspot returns `{fid: 0}`.

### Performance and vectorization
- Use NumPy for tvtw→tv mapping and deduplication; avoid per-element Python loops.
- Build CSR directly for footprints; use sparse matmul for intersections.
- Avoid densifying similarity for large `n_flights`; filter edges from sparse `M` to honor `threshold`.
- Cache per-flight TV sequences in `FlightList` to reuse across hotspots.

### Error handling and validation
- Validate `flight_ids` exist in `FlightList`.
- Validate `threshold ∈ [0,1]`.
- If union size is zero for any pair, set similarity to 0 (skip edge).
- Robust behavior when no edges survive threshold (return singletons).

- Summary:
  - Implemented a detailed plan to add small helpers to `FlightList`, compute vectorized Jaccard similarities over TV footprints, threshold into a weighted graph, and run Leiden via `igraph`/`leidenalg`.
  - The public API `assign_communities_for_hotspot(...)` returns the required `flight_id -> community_index` mapping and supports both hotspot modes with correct `hotspot_tv_index` handling.
  - A synthetic unit test validates each step (footprints, similarity, clustering) and edge cases.