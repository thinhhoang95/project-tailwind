### Goal
Design an efficient, vectorized state transition layer under `src/project_tailwind/stateman` to:
- Apply per‑flight delay assignments
- Produce a delta occupancy view (sparse)
- Update a `FlightList`-compatible state incrementally
- Track applied regulations

Below is a concrete, in‑detail plan with file layout, APIs, algorithms, and test coverage.

### Constraints and key facts from the codebase
- `FlightList` (in `src/project_tailwind/optimize/eval/flight_list.py`) exposes:
  - Sparse matrix `occupancy_matrix` (CSR) and an updateable LIL mirror (`_occupancy_matrix_lil`)
  - `flight_metadata[flight_id]['occupancy_intervals']` with `tvtw_index`, `entry_time_s`, `exit_time_s`
  - Helpers like `get_total_occupancy_by_tvtw`, `update_flight_occupancy`, `clear_flight_occupancy`, `add_flight_occupancy`, `finalize_occupancy_updates`
- TVTW indexing is row-major: `global_idx = tv_idx * T + time_idx`, where `T = num_time_bins`
- Efficient re-aggregation exists in `parrhesia.optim.occupancy.compute_occupancy(delays)`, but for fast incremental updates we will compute per-flight column moves and aggregate deltas ourselves.
- We will not use `optimize/eval/delta_flight_list.py` (legacy).

### Module layout (new)
- `/mnt/d/project-tailwind/src/project_tailwind/stateman/__init__.py`
- `/mnt/d/project-tailwind/src/project_tailwind/stateman/types.py`
- `/mnt/d/project-tailwind/src/project_tailwind/stateman/delay_assignment.py`
- `/mnt/d/project-tailwind/src/project_tailwind/stateman/delta_view.py`
- `/mnt/d/project-tailwind/src/project_tailwind/stateman/flight_list_with_delta.py`
- `/mnt/d/project-tailwind/src/project_tailwind/stateman/regulation_history.py`
- `/mnt/d/project-tailwind/src/project_tailwind/stateman/utils.py`

### Public APIs (signatures)
- `DelayAssignmentTable`
  - `from_dict(d: dict[str, int]) -> DelayAssignmentTable`
  - `to_dict() -> dict[str, int]`
  - `load_json(path: str) -> DelayAssignmentTable`, `save_json(path: str) -> None`
  - `load_csv(path: str) -> DelayAssignmentTable`, `save_csv(path: str) -> None`
  - `merge(other: DelayAssignmentTable, *, policy: Literal["max","sum","overwrite"]="overwrite") -> DelayAssignmentTable`
  - `nonzero_items() -> Iterable[tuple[str,int]]` (min delay 1 enforced on write)
- `DeltaOccupancyView`
  - `from_delay_table(flights: FlightList, delays: DelayAssignmentTable) -> DeltaOccupancyView`
  - `num_tvtws: int`
  - `delta_counts_sparse: scipy.sparse.csr_matrix` shape (1, num_tvtws)
  - `per_flight_new_intervals: dict[str, list[dict]]` with concrete `tvtw_index`, `entry_time_s`, `exit_time_s`
  - `delays: DelayAssignmentTable`
  - `changed_flights() -> list[str]`
  - `as_dense_delta(dtype=np.int64) -> np.ndarray`  # 1D num_tvtws
  - `stats() -> dict[str, Any]`
- `FlightListWithDelta(FlightList)`
  - `step_by_delay(*views: DeltaOccupancyView, finalize: bool = True) -> None`
  - `applied_regulations: list[str]` (ids from history when available)
  - `delay_histogram: dict[int,int]`
  - `total_delay_assigned_min: int`
  - `num_delayed_flights: int`
  - `num_regulations: int`
  - `get_delta_aggregate() -> np.ndarray`  # optional, aggregate of all applied deltas (dense)
- `RegulationHistory` (placeholder)
  - `record(regulation_id: str, view: DeltaOccupancyView) -> None`
  - `get(regulation_id: str) -> Optional[DeltaOccupancyView]`
  - `list_ids() -> list[str]`

### Core data model decisions
- Interpret “new entry/exit time” as original seconds plus delay in seconds: `new_entry_time_s = old_entry_time_s + delay_s`, `new_exit_time_s = old_exit_time_s + delay_s`. This matches how bins shift in `compute_occupancy`.
- Delta occupancy is stored as a single sparse row vector of length `num_tvtws` with integer counts (−1, 0, +1, ...).
- Only delayed flights are rewritten; all other rows remain untouched.
- Backward compatibility: `FlightListWithDelta` preserves the base `FlightList` API and data shapes.

### Algorithms

#### A. Vectorized per-flight column moves (fast path)
Given:
- For each changed flight `f`, arrays:
  - `tvtw_indices_f` (int64) from metadata; shape `(Ni,)`
  - Decode to `(tv_idx_f, time_idx_f)` via integer division/modulo:
    - `tv_idx_f = tvtw_indices_f // T`
    - `time_idx_f = tvtw_indices_f % T`
  - Delay in seconds `D = delay_min * 60`
  - Bin size `Δ = time_bin_minutes * 60`
  - `shift_bins = floor((entry_s + D)/Δ) - floor(entry_s/Δ)` computed vectorized for the flight using `entry_time_s` array
  - `new_time_idx = time_idx_f + shift_bins` (drop items that move out of [0, T-1])
  - `new_cols = tv_idx_f * T + new_time_idx` (int64)
- Old row has 1s at `old_cols = tvtw_indices_f`
- Delta for aggregate counts:
  - add +1 at `new_cols`, add −1 at `old_cols` (use bincount or sparse COO->CSR combine across flights)
- Row update:
  - Clear row (fast LIL slice assign to 0)
  - Assign 1.0 at unique `new_cols` for that row

This avoids recomputing all flights and keeps updates sparse and localized.

#### B. Sparse delta vector construction
- Accumulate two flat integer arrays across all changed flights: `old_cols_all`, `new_cols_all`
- Dense accumulation with `np.bincount` and reshape to 1D `(num_tvtws,)`:
  - `delta = bincount(new_cols_all, minlength=num_tvtws) - bincount(old_cols_all, minlength=num_tvtws)`
- Convert to CSR row vector only if needed for storage/transmission

#### C. Per‑flight intervals recomputation
- For each changed flight’s interval list:
  - `new_entry_time_s = entry_time_s + delay_s`
  - `new_exit_time_s = exit_time_s + delay_s`
  - `new_tvtw_index = tv_idx * T + new_time_idx` when in-range; if out-of-range, drop that interval
- Persist these new interval dicts in `DeltaOccupancyView.per_flight_new_intervals[fid]`
- When applying to state, replace the flight’s `occupancy_intervals` with the new list

#### D. Stats and summaries
- `total_delay_assigned_min = sum(delays.values())`
- `num_delayed_flights = count of nonzero delays applied`
- `delay_histogram = {delay_min -> count}`
- `num_regulations = len(applied_regulations)` (or number of applied views)

### File-by-file plan

#### `types.py`
- Type aliases for readability:
  - `FlightId = str`
  - `TVTWIndex = int`
  - `DelayMinutes = int`
  - `Intervals = list[dict]`

#### `delay_assignment.py`
- Immutable mapping with validation (min 1 minute on write/serialize).
- Merge policies:
  - `overwrite`: right wins
  - `max`: per-flight max
  - `sum`: arithmetic sum
- Utilities for JSON/CSV IO (CSV columns: `flight_id,delay_min`).

Example skeleton:
```python
class DelayAssignmentTable:
    def __init__(self, delays: dict[str, int] | None = None): ...
    def to_dict(self) -> dict[str, int]: ...
    @classmethod
    def from_dict(cls, d: dict[str, int]) -> "DelayAssignmentTable": ...
    # IO
    @classmethod
    def load_json(cls, path: str) -> "DelayAssignmentTable": ...
    def save_json(self, path: str) -> None: ...
    @classmethod
    def load_csv(cls, path: str) -> "DelayAssignmentTable": ...
    def save_csv(self, path: str) -> None: ...
    # Ops
    def merge(self, other: "DelayAssignmentTable", *, policy="overwrite") -> "DelayAssignmentTable": ...
    def nonzero_items(self) -> list[tuple[str, int]]: ...
```

#### `utils.py`
- Vector helpers:
  - `decode_tvtw(indices: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]`
  - `compute_shift_bins(entry_s: np.ndarray, delay_s: int, bin_len_s: int) -> np.ndarray`
  - `safe_new_time_idx(time_idx: np.ndarray, shift: np.ndarray, T: int) -> np.ndarray` (returns mask + new_time_idx)
  - `accumulate_delta(old_cols: np.ndarray, new_cols: np.ndarray, num_tvtws: int) -> np.ndarray`

#### `delta_view.py`
- Builds a delta view from a `FlightList` and a `DelayAssignmentTable` using Algorithm A/B/C.
- Stores:
  - `delta_counts_sparse: csr_matrix` (1 × num_tvtws)
  - `per_flight_new_intervals: dict[str, list[dict]]`
  - `delays: DelayAssignmentTable`
  - Optional: `regulation_id: Optional[str]`
- Provides `as_dense_delta`, `changed_flights`, `stats`.

Example skeleton:
```python
class DeltaOccupancyView:
    @classmethod
    def from_delay_table(cls, flights: FlightList, delays: DelayAssignmentTable) -> "DeltaOccupancyView":
        # 1) collect changed flights, build arrays, compute shifts
        # 2) build old_cols_all, new_cols_all
        # 3) compute delta vector (dense), convert to CSR
        # 4) build per_flight_new_intervals
        # 5) return instance
        ...
```

#### `flight_list_with_delta.py`
- Subclass `FlightList` without altering its existing behavior for non-updated paths.
- Keeps metrics and applied views.
- `step_by_delay(*views, finalize=True)`:
  1. For each view:
     - For each `flight_id` in `view.changed_flights()`:
       - `clear_flight_occupancy(flight_id)`
       - `add_flight_occupancy(flight_id, np.array([iv["tvtw_index"] for iv in per_flight_new_intervals[fid]], dtype=np.int64))`
       - Replace `flight_metadata[fid]["occupancy_intervals"]` with `view.per_flight_new_intervals[fid]`
     - Update counters, histogram, regulation ids
  2. If `finalize`: `finalize_occupancy_updates()`
- Caches:
  - Optional: pre-decoded `(tv_idx, time_idx, entry_s, exit_s)` per flight to accelerate repeated steps
- Provide `.get_delta_aggregate()` that sums deltas of applied views (optional convenience).

Example skeleton:
```python
class FlightListWithDelta(FlightList):
    def __init__(...):
        super().__init__(...)
        self._metrics = {...}
        self._applied_views: list[DeltaOccupancyView] = []

    def step_by_delay(self, *views: DeltaOccupancyView, finalize: bool = True) -> None:
        for view in views:
            for fid, intervals in view.per_flight_new_intervals.items():
                self.clear_flight_occupancy(fid)
                cols = np.fromiter((int(iv["tvtw_index"]) for iv in intervals), dtype=np.int64)
                if cols.size:
                    self.add_flight_occupancy(fid, cols)
                self.flight_metadata[fid]["occupancy_intervals"] = intervals
            # update metrics/hist/applied list
        if finalize:
            self.finalize_occupancy_updates()
```

#### `regulation_history.py`
- Minimal placeholder as requested:
```python
class RegulationHistory:
    def __init__(self):
        self._by_id: dict[str, DeltaOccupancyView] = {}
        self._order: list[str] = []
    def record(self, regulation_id: str, view: DeltaOccupancyView) -> None: ...
    def get(self, regulation_id: str) -> DeltaOccupancyView | None: ...
    def list_ids(self) -> list[str]: ...
```

### Performance considerations
- Use numpy arrays for all per-flight computations; avoid Python loops except per flight boundary.
- Prefer `np.bincount` for aggregating delta counts; it’s extremely fast.
- Batch LIL row updates per flight; finalize once per batch to rebuild CSR.
- Drop intervals that shift outside day horizon [0, T−1] to avoid invalid indices.
- Maintain per‑flight decoded caches if multiple steps are expected; invalidate caches if time bin configuration changes.

### Edge cases
- Flights in delays that don’t exist in state: skip with warning.
- Delay < 1 minute: enforce 1 on serialize; treat 0 as no-op internally.
- Intervals shifting past end-of-day: drop those contributions.
- Duplicate intervals mapping to the same new column: row’s binary nature handles this; aggregation via bincount handles duplicates correctly.

### Integration points
- Consumers expecting a `FlightList` can use `FlightListWithDelta` transparently.
- For independent verification or when full recomputation is fine, you can cross-check `DeltaOccupancyView` with `parrhesia.optim.occupancy.compute_occupancy` and mapping back to global tvtw indices using `tv_id_to_idx` and `num_time_bins`.

### Testing plan
- Unit tests under `tests/stateman/`:
  - Build a tiny `FlightList` and `TVTWIndexer` stub; craft known intervals.
  - Verify `DelayAssignmentTable` IO and merge policies.
  - Verify vectorized shift math: integer bin shifts with different `entry_time_s` and `time_bin_minutes`.
  - Verify `DeltaOccupancyView.delta_counts_sparse` equals “+1 at new cols, −1 at old cols” for a few flights.
  - Verify `FlightListWithDelta.step_by_delay` mutates only affected rows and updates metadata.
  - Verify `get_total_occupancy_by_tvtw()` after step equals base + cumulative delta.
  - Large random tests: generate random intervals/delays; compare against ground truth from recomputing with `compute_occupancy` mapped to tvtw vector.

### Phased delivery
1) Scaffold `stateman` package and `DelayAssignmentTable`
2) Implement vector utilities and `DeltaOccupancyView`
3) Implement `FlightListWithDelta.step_by_delay`
4) Add `RegulationHistory` placeholder and wire optional recording
5) Tests, docs, examples

### Minimal example (usage)
```python
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.stateman.delay_assignment import DelayAssignmentTable
from project_tailwind.stateman.delta_view import DeltaOccupancyView
from project_tailwind.stateman.flight_list_with_delta import FlightListWithDelta

fl = FlightList(occupancy_file_path, tvtw_indexer_path)
state = FlightListWithDelta(occupancy_file_path, tvtw_indexer_path)

delays = DelayAssignmentTable.from_dict({"F123": 15, "F456": 5})
view = DeltaOccupancyView.from_delay_table(fl, delays)

state.step_by_delay(view, finalize=True)
```

- After step:
  - `state.occupancy_matrix` reflects delayed flights
  - `state.flight_metadata[fid]['occupancy_intervals']` updated for changed flights
  - `state.num_regulations`, `state.total_delay_assigned_min`, `state.delay_histogram` populated

- For aggregate deltas only (no state mutation), use `view.delta_counts_sparse`.