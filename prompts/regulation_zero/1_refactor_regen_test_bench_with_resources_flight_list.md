### Instructions
- The goal is to reuse resources flight list instead of re-reading from scratch. `FlightListWithDelta` subclasses your `FlightList`, so it’s API-compatible with `compute_flows`, the flow pipeline, and the regen engine.
- Keep your existing defensive occupancy padding; it’s safe to run on `FlightListWithDelta` and helps guard against rare mismatches.
- Nothing changes unless you actually apply deltas; by default it behaves like the baseline `FlightList`.

### Minimal edits

- Add this import:
```25:33:examples/regen/regen_test_bench_custom_tvtw.py
# Flow regeneration engine
from parrhesia.api.flows import compute_flows
from parrhesia.api.resources import set_global_resources
from parrhesia.flow_agent35.regen.engine import propose_regulations_for_hotspot
from parrhesia.flow_agent35.regen.exceedance import compute_hotspot_exceedance
from parrhesia.flow_agent35.regen.rates import compute_e_target
from parrhesia.flow_agent35.regen.types import RegenConfig
from parrhesia.optim.capacity import build_bin_capacities, normalize_capacities
from server_tailwind.core.resources import AppResources, ResourcePaths  # NEW
```

- Replace `build_data` to source `indexer`/`flight_list`/`GDF` from resources (and keep your padding + evaluator wiring):
```134:198:examples/regen/regen_test_bench_custom_tvtw.py
def build_data(occupancy_path: Path, indexer_path: Path, caps_path: Path) -> Tuple[FlightList, TVTWIndexer, NetworkEvaluator]:
    """
    Loads core data structures from artifact paths.

    This function initializes the main objects required for network evaluation
    and regulation proposals using the shared resources flight list (delta-enabled)
    for consistency with the server.
    """
    # Use AppResources so flows/regen share the same flight list/indexer as the server
    res = AppResources(ResourcePaths(
        occupancy_file_path=occupancy_path,
        tvtw_indexer_path=indexer_path,
        traffic_volumes_path=caps_path,
    )).preload_all()

    indexer = res.indexer
    flight_list = res.flight_list

    # Defensive occupancy width alignment (keep as before)
    expected_tvtws = len(indexer.tv_id_to_idx) * indexer.num_time_bins
    if getattr(flight_list, "num_tvtws", 0) < expected_tvtws:
        from scipy import sparse  # type: ignore
        pad_cols = expected_tvtws - int(flight_list.num_tvtws)
        pad_matrix = sparse.lil_matrix((int(flight_list.num_flights), pad_cols))
        flight_list._occupancy_matrix_lil = sparse.hstack(  # type: ignore[attr-defined]
            [flight_list._occupancy_matrix_lil, pad_matrix], format="lil"  # type: ignore[attr-defined]
        )
        flight_list.num_tvtws = expected_tvtws  # type: ignore[assignment]
        flight_list._temp_occupancy_buffer = np.zeros(expected_tvtws, dtype=np.float32)  # type: ignore[attr-defined]
        flight_list._lil_matrix_dirty = True  # type: ignore[attr-defined]
        flight_list._sync_occupancy_matrix()  # type: ignore[attr-defined]

    # Capacities
    caps_gdf = res.traffic_volumes_gdf
    if caps_gdf.empty:
        raise SystemExit("Traffic volume capacity file is empty; cannot proceed.")

    evaluator = NetworkEvaluator(caps_gdf, flight_list)
    # Preserve the original GeoJSON path so the shared capacity builder can be used.
    try:
        evaluator._capacities_path = str(caps_path)  # type: ignore[attr-defined]
    except Exception:
        print(f"[warning] Failed to set capacities path on evaluator")
        pass

    return flight_list, indexer, evaluator
```

- Ensure the global resources are the ones from resources when calling flows (your code already calls `set_global_resources(indexer, flight_list)`; with the change above, those come from resources).

### Why this won’t “screw things up”
- **API compatibility**: `FlightListWithDelta` inherits your `FlightList`. The flow pipeline and regen engine use duck-typed methods that are implemented unchanged in the subclass.
- **Stable baseline**: Unless you call `step_by_delay(...)`, no delta is applied. Occupancy/metadata remain baseline.
- **Consistency**: Both the server and your bench now use the same artifacts and the same global `(indexer, flight_list)` for `compute_flows`, avoiding subtle divergence.
- **Safety net**: The existing occupancy padding remains in place to protect against rare indexer/occupancy width mismatches.

We will also opt for full reuse, so you swap the `_tv_centroids` and `_travel_minutes_from_centroids` with `res.tv_centroids` and `res.travel_minutes()`.

- Set global resources and run flows:
```python
from server_tailwind.core.resources import get_resources
res = get_resources().preload_all()
set_global_resources(res.indexer, res.flight_list)
```

Summary
- Switched `build_data` to use `server_tailwind.core.resources.AppResources` so the bench uses the delta-enabled `flight_list`.
- Kept occupancy padding and capacity path wiring; `compute_flows` now consumes the same global resources as the server.
- Behavior unchanged unless deltas are applied; this makes your bench consistent without introducing risk.