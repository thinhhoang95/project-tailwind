Let me quickly scan the relevant resource and flow code to verify how the shared `FlightList` is constructed, how caches are managed, and whether applying deltas would keep compute and regen in sync. I'll open the resources, base `FlightList`, delta view, and flows APIs in parallel.

[5 tools called]

I'll search for how flows are collected/built to see which FlightList APIs and caches they use, and check if any additional caches need invalidation when applying a delta. I'll scan flow_pipeline and related modules in parallel.

[12 tools called]

I looked through the shared resources, flight list, flows, and regen engine to see how deltas propagate and where caches live.

### Short answer
- Yes, applying a delta to the shared `FlightListWithDelta` (from `AppResources`) will work as expected.
- No mixed old/new state as long as you finalize the updates before invoking flows/regen.
- Caches that matter for regen are rebuilt per run, so they’ll reflect the updated state. The only persistent caches in `AppResources` don’t depend on occupancy and don’t need refresh.

### Why it’s safe
- The delta path updates both the sparse matrix and per-flight metadata, and invalidates the per‑flight sequence cache:
```51:82:/mnt/d/project-tailwind/src/project_tailwind/stateman/flight_list_with_delta.py
def _apply_single_view(self, view: DeltaOccupancyView) -> None:
    dense_delta = view.as_dense_delta(np.int64)
    ...
    self._applied_views.append(view)
    ...
    for flight_id, intervals in view.per_flight_new_intervals.items():
        ...
        self.clear_flight_occupancy(flight_id)
        ...
        self.add_flight_occupancy(flight_id, column_indices)
        ...
        self.flight_metadata[flight_id]["occupancy_intervals"] = canonical_intervals
        ...
        self._flight_tv_sequence_cache.pop(flight_id, None)
```
- If you call `step_by_delay(..., finalize=True)`, the read-optimized CSR gets synced:
```29:44:/mnt/d/project-tailwind/src/project_tailwind/stateman/flight_list_with_delta.py
def step_by_delay(self, *views: DeltaOccupancyView, finalize: bool = True) -> None:
    ...
    if finalize:
        self.finalize_occupancy_updates()
```
```516:522:/mnt/d/project-tailwind/src/project_tailwind/optimize/eval/flight_list.py
def finalize_occupancy_updates(self):
    self._sync_occupancy_matrix()
```
- Flow computation uses the in-memory global resources you set, so it sees the updated `FlightListWithDelta`:
```60:63:/mnt/d/project-tailwind/src/parrhesia/api/flows.py
g_idx, g_fl = get_global_resources()
if g_idx is not None and g_fl is not None:
    return g_idx, g_fl
```
- The regen engine builds a fresh feature extractor (and base caches) on each call, so it won’t reuse stale occupancy:
```212:219:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/engine.py
extractor = build_flow_features_extractor(
    indexer=indexer,
    flight_list=flight_list,
    capacities_by_tv=capacities_by_tv,
    travel_minutes_map=travel_minutes_map,
    ...
)
```
```101:109:/mnt/d/project-tailwind/src/parrhesia/metaopt/feats/flow_features.py
self.caches: Dict[str, Any] = build_base_caches(
    flight_list=self.flight_list,
    capacities_by_tv=self.capacities_by_tv,
    indexer=self.indexer,
)
```
- Those base caches pull the aggregate occupancy from the updated CSR:
```53:58:/mnt/d/project-tailwind/src/parrhesia/metaopt/base_caches.py
occ_base = np.asarray(flight_list.get_total_occupancy_by_tvtw(), dtype=np.float64)
if occ_base.size != num_tvs * T:
    raise ValueError("Occupancy vector size mismatch with indexer bins")
```
- Exceedance is computed using the extractor’s caches (i.e., the fresh ones):
```226:233:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/engine.py
exceedance_stats = compute_hotspot_exceedance(
    indexer=indexer,
    flight_list=flight_list,
    capacities_by_tv=capacities_by_tv,
    hotspot_tv=hotspot_tv,
    timebins_h=timebins_seq,
    caches=extractor.caches,
)
```

### Pitfalls to avoid
- Always finalize before any reads:
  - `get_total_occupancy_by_tvtw()` reads `occupancy_matrix` and does not auto-sync. Use `step_by_delay(..., finalize=True)` or call `finalize_occupancy_updates()` yourself.
```367:375:/mnt/d/project-tailwind/src/project_tailwind/optimize/eval/flight_list.py
def get_total_occupancy_by_tvtw(self) -> np.ndarray:
    return np.array(self.occupancy_matrix.sum(axis=0)).flatten()
```
- Don’t reuse a `FlowFeaturesExtractor` across deltas. It snapshots caches at construction. The engine recreates it, so you’re fine as-is.
- If you compute and hold your own `flow_to_flights` metadata before applying a delta, recompute it afterward if you intend to rely on it without passing a fresh `flows_payload`. In this script, `flows_payload` is recomputed inside `propose_regulations`, so you’re safe.

### How to apply a delta safely (shared resources)
```python
res = get_resources().preload_all()
# build a DeltaOccupancyView somehow (not shown)
res.flight_list.step_by_delay(view, finalize=True)  # ensures CSR + metadata are current
# then call compute_flows / propose_regulations
```

- `AppResources` caches like `capacity_per_bin_matrix`, `tv_centroids`, and `travel_minutes()` don’t depend on occupancy and need no refresh.

- The width-alignment in `regen_test_bench_custom_tvtw.py` ensures `num_tvtws == V*T`, preventing base-cache size mismatches when building features/exceedance.

- `compute_flows` uses `set_global_resources(...)` as shown in the script before it runs, so it uses the same updated `FlightListWithDelta` object.

Summary
- Applying deltas to the shared `FlightListWithDelta` works normally with no stale/mixture issues if you finalize before running flows/regen.
- Per-flight and aggregate caches are either invalidated (per-flight sequence) or rebuilt fresh (features/exceedance).
- Persistent `AppResources` caches don’t require refresh since they don’t depend on occupancy.