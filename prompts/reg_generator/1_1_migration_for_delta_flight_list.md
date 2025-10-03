- **Stale caches/objects**: If you mutate `res.flight_list` after creating things that cache state (e.g., `NetworkEvaluator`, any features extractor), they may read the pre-delta snapshot.
  - Fix: apply deltas first, call `finalize_occupancy_updates()`, then build `NetworkEvaluator`/extractors; or recreate them after deltas.

- **Flow metadata drift**: If you compute `flows_payload`/`flow_to_flights` before deltas and then mutate the flight list, those inputs can be inconsistent with the post-delta state.
  - Fix: re-run `compute_flows` and rebuild `flow_to_flights`/proxies after applying deltas; use the same global `(indexer, flight_list)`.

- **Index bounds/day wrap**: Deltas that shift occupancy past [0, T) can produce out-of-range `tvtw_index` and raise on `add_flight_occupancy`.
  - Fix: clip/guard shifts; ensure `DeltaOccupancyView` stays within [0, T); if needed, pad to `expected_tvtws` before applying.

- **Takeoff-time semantics**: `FlightListWithDelta` updates `occupancy_intervals` (and their `entry_time_s`), but does not shift `takeoff_time`. Any logic that uses `takeoff_time + entry_time_s` (e.g., `iter_hotspot_crossings`) will be correct only if your delta updates `entry_time_s` accordingly.
  - Fix: ensure your deltas adjust `entry_time_s`; if you truly need takeoff shifts, consider `DeltaFlightList` (overlay adjusts `takeoff_time`) instead of in-place mutation.

- **Global resource sync (server parity)**: If you mutate the shared flight list and other layers cache derived views, they may need a refresh.
  - Fix: in server contexts, call `refresh_after_state_update(...)`; in the bench, just recreate the evaluator or re-run builders after deltas.

- **Performance churn**: Many small deltas with frequent finalize calls can be slow due to LIL→CSR sync and cache rebuilds.
  - Fix: batch deltas (`finalize=False`), then `finalize_occupancy_updates()` once.

Recommended order when using deltas:
- Apply deltas to `res.flight_list` → `finalize_occupancy_updates()`.
- Recompute `flows_payload` and rebuild hotspot metadata.
- Recreate `NetworkEvaluator` (or rebuild any dependent caches).
- Run `propose_regulations`.

- Reusing the same object graph across deltas without re-init is the main way things get “screwed up”; recomputing flows and rebuilding evaluator avoids it.