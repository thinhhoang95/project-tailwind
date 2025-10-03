I'll scan the codebase for the cache refresh path and resource lifecycles to validate whether the plan's cache invalidation is sufficient. Then I'll inspect `AppResources`, `FlightList.step_by_delay`, and the hotspot extraction to ensure no stale caches persist after apply.

[5 tools called]

I'll locate the DF/regen APIs and view/delay utilities to confirm signatures and data flow, then verify that applying delays updates matrices and that our calls match expectations.

[5 tools called]

Let me check whether `_build_capacities_by_tv` used in the plan exists with the expected signature and confirm if `examples/regen/regen_second_order.py` defines or imports it properly. I'll also confirm `AppResources.flight_list` is `FlightListWithDelta` in our codebase.

[2 tools called]

Let me search for the specific symbols used in your plan to verify they exist and check wrapper invalidation hooks.

[4 tools called]

Let me open the relevant wrapper classes to confirm their cache invalidation methods and ensure `QueryAPIWrapper.refresh_flight_list` properly clears cached state.

[2 tools called]

Let me open the `FlightListWithDelta` implementation to confirm that `step_by_delay` rebuilds occupancy matrices and invalidates any internal caches, ensuring downstream views see the new state.

[1 tool called]

I reviewed the whole chain and the caches involved. Short answer: the workflow is sound for the example script context, but if you expect server wrappers to see the updated state, you must pass those wrappers into the refresh call; otherwise only the global resources are re-registered.

### Verdict
- For the example runner (`examples/regen/regen_second_order.py`), applying delays then calling `refresh_after_state_update(res)` is sufficient to keep the next computations consistent. Hotspot re-extraction reads from the mutated `res.flight_list`, and flows/evaluation reuse the in-memory resources/global registration.
- For the FastAPI server, if you mutate the shared `FlightList`, you must refresh the wrapper caches explicitly (airspace, counts, query), otherwise endpoints will serve stale data.

### Why it’s safe in the script
- Hotspot discovery reads the current in-memory `FlightList` occupancy and shared capacities; it doesn’t rely on stale globals:
```55:67:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/hotspot_segment_extractor.py
res = (resources or get_resources()).preload_all()
fl = res.flight_list

num_tvs = len(fl.tv_id_to_idx)
...
total_occ = fl.get_total_occupancy_by_tvtw().astype(np.float32, copy=False)
```
- Flow computation prefers globally-registered resources; your code sets them before computing flows, and `refresh_after_state_update` re-registers them after apply:
```59:66:/mnt/d/project-tailwind/src/parrhesia/api/flows.py
g_idx, g_fl = get_global_resources()
if g_idx is not None and g_fl is not None:
    return g_idx, g_fl  # type: ignore[return-value]
```
- DF plan evaluation uses the same in-memory `AppResources` (ignores path args), so it sees the current `flight_list`:
```242:261:/mnt/d/project-tailwind/src/parrhesia/actions/dfplan_evaluator.py
# ... Uses process-wide AppResources ...
res = get_resources()
tvtw_indexer = res.indexer
flight_list = res.flight_list  # FlightListWithDelta (subclass of FlightList)
...
T_idx = int(getattr(tvtw_indexer, "num_time_bins"))
T_fl = int(getattr(flight_list, "num_time_bins_per_tv"))
```
- The `FlightListWithDelta` apply path is designed for in-place mutation:
```45:69:/mnt/d/project-tailwind/src/project_tailwind/stateman/flight_list_with_delta.py
def step_by_delay(self, *views: DeltaOccupancyView, finalize: bool = True) -> None:
    ...
    for view in views:
        ...
        self._apply_single_view(view)
    if finalize:
        self.finalize_occupancy_updates()
```

### Cache invalidation details (script vs server)
- The helper can invalidate wrappers, but only if you pass them. With just `refresh_after_state_update(res)`, it only re-registers global resources:
```13:32:/mnt/d/project_tailwind/src/server_tailwind/core/cache_refresh.py
def refresh_after_state_update(
    resources: Any,
    *,
    airspace_wrapper: Optional[Any] = None,
    count_wrapper: Optional[Any] = None,
    query_wrapper: Optional[Any] = None,
) -> None:
    if airspace_wrapper is not None and hasattr(airspace_wrapper, "invalidate_caches"):
        airspace_wrapper.invalidate_caches()
    if count_wrapper is not None and hasattr(count_wrapper, "invalidate_caches"):
        count_wrapper.invalidate_caches()
    if query_wrapper is not None and hasattr(query_wrapper, "refresh_flight_list"):
        query_wrapper.refresh_flight_list(resources.flight_list)

    if _parrhesia_resources is not None:
        try:
            _parrhesia_resources.set_global_resources(resources.indexer, resources.flight_list)
```
- Wrappers keep their own caches and must be refreshed to avoid stale views:
```66:71:/mnt/d/project-tailwind/src/server_tailwind/airspace/airspace_api_wrapper.py
def invalidate_caches(self) -> None:
    with self._slack_lock:
        self._slack_vector = None
        self._total_occupancy_vector = None
```
```46:49:/mnt/d/project-tailwind/src/server_tailwind/CountAPIWrapper.py
def invalidate_caches(self) -> None:
    """Drop cached occupancy aggregates so subsequent queries recompute from the flight list."""
    self._total_occupancy_vector = None
```
```74:113:/mnt/d/project_tailwind/src/server_tailwind/query/QueryAPIWrapper.py
def refresh_flight_list(self, flight_list) -> None:
    """Rebind the flight list and rebuild cached sparse structures and lookups."""
    self.flight_list = flight_list
    ...
    self.occupancy_csr = self.flight_list.occupancy_matrix
    ...
    # Reset cached structures so they repopulate lazily with fresh data
    self._origin_to_rows = {}
    ...
    self._total_occupancy_vector = None
```
- The server registers global resources at startup so parrhesia modules reuse the same indexer/flight_list:
```30:36:/mnt/d/project_tailwind/src/server_tailwind/main.py
_res = get_resources().preload_all()
if parr_res is not None:
    try:
        parr_res.set_global_resources(_res.indexer, _res.flight_list)
```

### One potential blocker to fix
The `DeltaOccupancyView` implementation in your repo appears to have syntax issues that would break `from_delay_table` at runtime (stray lines and missing variables). Example snippets:
```27:35:/mnt/d/project-tailwind/src/project_tailwind/stateman/delta_view.py
def _compute_shift_bins(entry_seconds: np.ndarray, delay_seconds: int, bin_seconds: int) -> np.ndarray:
    ...
    entry_bins = np.floor(entry_seconds / bin_seconds).astype(np.int64)
astype(np.int64)
    return new_entry_bins - entry_bins
```
```102:111:/mnt/d/project-tailwind/src/project_tailwind/stateman/delta_view.py
new_entry_seconds = entry_seconds[valid_mask] + delay_seconds
new_exit_seconds = exit_seconds[valid_mask] + delay_seconds
per_flight_new_intervals[flight_id] =
    {
        "tvtw_index": int(col),
        "entry_time_s": float(new_entry),
        "exit_time_s": float(new_exit),
    }
    for col, new_entry, new_exit in zip(new_cols, new_entry_seconds, new_exit_seconds)
```
If this file is as shown, the plan will fail when building the view. I recommend fixing these lines before running the second-order workflow.

### Recommendations
- Script context (as in your plan): keep `refresh_after_state_update(res)`; it’s sufficient for the follow-up hotspot extraction. You’re already calling `set_global_resources` for flows.
- Server context: when mutating the shared state, call:
  - `refresh_after_state_update(res, airspace_wrapper, count_wrapper, query_wrapper)` so all wrapper caches are cleared and the query wrapper rebinds the latest CSR.
- Fix the `DeltaOccupancyView` syntax issues before running. Then run a quick smoke: compute pre/post hotspot list around the regulated TV to confirm state changed.

- Optional: In `_evaluate_and_apply_plan`, you already snapshot pre/post occupancy and print L1 delta; that’s a good invariant check.

Summary
- Plan is structurally sound; cache handling is correct for the example runner.
- For server usage, you must pass wrappers into `refresh_after_state_update(...)` or call their invalidation methods explicitly; otherwise wrappers will serve stale data.
- Fix `src/project_tailwind/stateman/delta_view.py` before executing; it currently contains syntax errors that would block the plan.