Multiple wrappers cache occupancy- and metadata-derived arrays, so they’ll keep serving old results unless you invalidate or refresh those caches after each state change.

### Why caches make replacement insufficient
These components cache data derived from the baseline `flight_list`:

```57:67:/mnt/d/project-tailwind/src/server_tailwind/query/QueryAPIWrapper.py
        self.occupancy_csr = self.flight_list.occupancy_matrix
        if not sparse.isspmatrix_csr(self.occupancy_csr):
            self.occupancy_csr = sparse.csr_matrix(self.occupancy_csr, dtype=np.float32)
        self.occupancy_csc = self.occupancy_csr.tocsc(copy=False)

        self.num_tvtws = int(self.flight_list.num_tvtws)

        # Precompute flight metadata lookups
        self._flight_row_lookup: Dict[str, int] = {fid: idx for idx, fid in enumerate(self.flight_list.flight_ids)}
        self._flight_meta = self.flight_list.flight_metadata
```

```167:172:/mnt/d/project-tailwind/src/server_tailwind/CountAPIWrapper.py
    def _get_or_build_total_vector(self) -> np.ndarray:
        if self._total_occupancy_vector is not None:
            return self._total_occupancy_vector
        vec = self._flight_list.get_total_occupancy_by_tvtw().astype(np.float32, copy=False)
        self._total_occupancy_vector = vec
        return vec
```

```96:103:/mnt/d/project-tailwind/src/server_tailwind/airspace/airspace_api_wrapper.py
            total_capacity_per_bin = self._evaluator._get_total_capacity_vector()
            total_occupancy = self._flight_list.get_total_occupancy_by_tvtw()
            slack = total_capacity_per_bin.astype(np.float32, copy=False) - total_occupancy.astype(np.float32, copy=False)

            self._total_occupancy_vector = total_occupancy.astype(np.float32, copy=False)
            self._slack_vector = slack
            return self._slack_vector
```

Because `FlightListWithDelta.step_by_delay(...)` rebuilds the CSR, any cached references and derived vectors above become stale unless you refresh them.

```29:44:/mnt/d/project-tailwind/src/project_tailwind/stateman/flight_list_with_delta.py
    def step_by_delay(self, *views: DeltaOccupancyView, finalize: bool = True) -> None:
        """Apply one or more delta views to the flight list."""

        if not views:
            if finalize:
                self.finalize_occupancy_updates()
            return

        for view in views:
            if not isinstance(view, DeltaOccupancyView):
                raise TypeError("All arguments to step_by_delay must be DeltaOccupancyView instances")
            self._apply_single_view(view)

        if finalize:
            self.finalize_occupancy_updates()
```

### Minimal changes to make it work consistently

1) Construct the shared flight list as `FlightListWithDelta` once
```12:20:/mnt/d/project-tailwind/src/server_tailwind/core/resources.py
from project_tailwind.optimize.eval.flight_list import FlightList
```
Change the import and instantiation:

```python
# add this import
from project_tailwind.stateman.flight_list_with_delta import FlightListWithDelta
```

```python
# inside AppResources.flight_list
self._flight_list = FlightListWithDelta(
    occupancy_file_path=str(self.paths.occupancy_file_path),
    tvtw_indexer_path=str(self.paths.tvtw_indexer_path),
)
```

Code reference of current factory:
```54:61:/mnt/d/project-tailwind/src/server_tailwind/core/resources.py
    def flight_list(self) -> FlightList:
        with self._lock:
            if self._flight_list is None:
                self._flight_list = FlightList(
                    occupancy_file_path=str(self.paths.occupancy_file_path),
                    tvtw_indexer_path=str(self.paths.tvtw_indexer_path),
                )
            return self._flight_list
```

2) Add tiny cache-reset hooks to wrappers
- In `AirspaceAPIWrapper`:
```python
def invalidate_caches(self):
    with self._slack_lock:
        self._slack_vector = None
        self._total_occupancy_vector = None
```

- In `CountAPIWrapper`:
```python
def invalidate_caches(self):
    self._total_occupancy_vector = None
```

- In `QueryAPIWrapper`:
```python
def refresh_flight_list(self, flight_list):
    # rebind fl + rebuild sparse views and clear all cached masks/vectors
    self.flight_list = flight_list
    self.time_bin_minutes = int(self.flight_list.time_bin_minutes)
    self.bins_per_tv = int(self.flight_list.num_time_bins_per_tv)
    self.tv_id_to_idx = {str(k): int(v) for k, v in self.flight_list.tv_id_to_idx.items()}
    self.idx_to_tv_id = {int(k): str(v) for k, v in self.flight_list.idx_to_tv_id.items()}
    self.num_flights = int(self.flight_list.num_flights)
    self.flight_ids = np.asarray(self.flight_list.flight_ids, dtype=object)
    self.occupancy_csr = self.flight_list.occupancy_matrix
    if not sparse.isspmatrix_csr(self.occupancy_csr):
        self.occupancy_csr = sparse.csr_matrix(self.occupancy_csr, dtype=np.float32)
    self.occupancy_csc = self.occupancy_csr.tocsc(copy=False)
    self.num_tvtws = int(self.flight_list.num_tvtws)
    self._flight_row_lookup = {fid: idx for idx, fid in enumerate(self.flight_list.flight_ids)}
    self._flight_meta = self.flight_list.flight_metadata
    self._takeoff_cache = {fid: meta.get("takeoff_time") for fid, meta in self._flight_meta.items()}
    self._origin_to_rows.clear()
    self._destination_to_rows.clear()
    self._tv_bin_to_rows.clear()
    self._tv_time_mask_cache.clear()
    self._flight_entries_cache.clear()
    self._arrival_time_cache.clear()
    self._node_cache.clear()
    self._capacity_state_cache.clear()
    self._total_occupancy_vector = None
```

3) After each “commit” to the network state (after calling `step_by_delay(...)`)
Run this sequence so all endpoints see the new state:
```python
# Apply the delta(s)
_res.flight_list.step_by_delay(view)  # view = DeltaOccupancyView

# Reset caches everywhere
airspace_wrapper.invalidate_caches()
count_wrapper.invalidate_caches()
query_wrapper.refresh_flight_list(_res.flight_list)

# Re-register in parrhesia (keeps external modules aligned)
if parr_res is not None:
    parr_res.set_global_resources(_res.indexer, _res.flight_list)
```

Current registration at startup:
```33:36:/mnt/d/project-tailwind/src/server_tailwind/main.py
if parr_res is not None:
    try:
        parr_res.set_global_resources(_res.indexer, _res.flight_list)
```

Notes
- You don’t need to rebuild `AppResources` or restart the server; just call the three cache refresh methods and re-register resources after each state change.
- If you prefer to never refresh, an alternative is to re-instantiate `AirspaceAPIWrapper`, `CountAPIWrapper`, and `QueryAPIWrapper` after each commit, but the small invalidation methods are cheaper and simpler.

Summary
- FlightListWithDelta is drop-in compatible.
- Add small cache-reset methods and call them after each applied delta, and re-register the shared resources with parrhesia so all endpoints stay consistent.