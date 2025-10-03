State Transition Layer (stateman)

This module provides fast, vectorized state transitions for flight occupancy using per‑flight delay assignments. It lets you compute a sparse delta for TVTW counts and incrementally update a FlightList-compatible state without full recomputation.

What you get
- DelayAssignmentTable: typed container with IO and merge helpers
- DeltaOccupancyView: sparse delta + per‑flight rewritten intervals
- FlightListWithDelta: in-place, vectorized application to occupancy and metadata
- RegulationHistory: placeholder mapping regulation ids -> views

Core Concepts
- TVTW: Traffic‑Volume Time‑Window column indices are contiguous: tv_index * T + time_index, where T is the number of time bins per day.
- Per‑flight shift: A delay shifts each interval’s entry/exit seconds forward; the time bin index moves by floor((entry + delay)/Δ) − floor(entry/Δ), with Δ = time_bin_minutes * 60.
- Delta vector: A single 1xN CSR row with integer counts. For changed flights, −1 at old columns, +1 at new columns.

Key APIs
- DelayAssignmentTable
  - Class: src/project_tailwind/stateman/delay_assignment.py:13
  - Factories/IO: `from_dict`, `load_json`, `save_json`, `load_csv`, `save_csv` (src/project_tailwind/stateman/delay_assignment.py:48, src/project_tailwind/stateman/delay_assignment.py:56, src/project_tailwind/stateman/delay_assignment.py:63, src/project_tailwind/stateman/delay_assignment.py:69, src/project_tailwind/stateman/delay_assignment.py:92)
  - Merge policies: `merge(policy="overwrite|max|sum")` (src/project_tailwind/stateman/delay_assignment.py:100)
  - Iteration: `nonzero_items()` yields flights with delay > 0 (src/project_tailwind/stateman/delay_assignment.py:117)

- DeltaOccupancyView
  - Class: src/project_tailwind/stateman/delta_view.py:37
  - Build from delays: `from_delay_table(flights, delays, regulation_id=None)` (src/project_tailwind/stateman/delta_view.py:55)
  - Data: `delta_counts_sparse` (CSR 1xN), `per_flight_new_intervals`, `delays`
  - Helpers: `changed_flights()` (src/project_tailwind/stateman/delta_view.py:134), `as_dense_delta()` (src/project_tailwind/stateman/delta_view.py:139), `stats()` (src/project_tailwind/stateman/delta_view.py:144)

- FlightListWithDelta
  - Class: src/project_tailwind/stateman/flight_list_with_delta.py:14
  - Apply views: `step_by_delay(*views, finalize=True)` (src/project_tailwind/stateman/flight_list_with_delta.py:29)
  - Metrics: `applied_regulations`, `delay_histogram`, `total_delay_assigned_min`, `num_delayed_flights`, `num_regulations` (src/project_tailwind/stateman/flight_list_with_delta.py:17)
  - Aggregate: `get_delta_aggregate()` dense delta (src/project_tailwind/stateman/flight_list_with_delta.py:45)

- RegulationHistory
  - Class: src/project_tailwind/stateman/regulation_history.py:10
  - Methods: `record`, `get`, `list_ids` (src/project_tailwind/stateman/regulation_history.py:17, src/project_tailwind/stateman/regulation_history.py:25, src/project_tailwind/stateman/regulation_history.py:28)

Usage Example
```python
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.stateman import DelayAssignmentTable, DeltaOccupancyView, FlightListWithDelta

fl = FlightList(occupancy_file_path, tvtw_indexer_path)
state = FlightListWithDelta(occupancy_file_path, tvtw_indexer_path)

delays = DelayAssignmentTable.from_dict({"F123": 15, "F456": 5})
view = DeltaOccupancyView.from_delay_table(fl, delays, regulation_id="REG-001")

# Mutate state in-place
state.step_by_delay(view, finalize=True)

# Inspect metrics
print(state.total_delay_assigned_min, state.num_delayed_flights, state.applied_regulations)

# Work with aggregate delta only (no mutation)
dense_delta = view.as_dense_delta()
```

How It Works
- Vectorized bin shift
  - For each changed flight, decode old TVTW columns into `(tv_idx, time_idx)`.
  - Compute `shift_bins` from entry times + delay; add to `time_idx` and drop any outside `[0, T)`.
  - Encode new columns, add −1 at old, +1 at new to the delta vector.

- Incremental state update
  - For each changed flight: clear its row in the LIL mirror, set 1.0 at new columns, replace `flight_metadata[fid]["occupancy_intervals"]`.
  - Call `finalize_occupancy_updates()` once per batch to rebuild CSR.

Performance Notes
- All per‑flight math uses numpy arrays; aggregation uses sparse CSR with duplicate summing.
- Finalization is batched via `finalize=True` in `step_by_delay` to avoid repeated CSR rebuilds.

Edge Cases
- Unknown flights in a delay table are skipped (warning only).
- Delays <= 0 are treated as no‑ops; serialization enforces minimum 1 minute.
- Intervals shifted out of day horizon are dropped.

Testing
- Focused tests live under `tests/stateman/` and verify:
  - Delay IO/merge; vectorized shift math; sparse delta construction; in‑place state mutation and metadata update; metrics.

Tips
- If you need only the delta counts, construct a `DeltaOccupancyView` and skip `FlightListWithDelta`.
- For repeated steps on the same state, prefer batching multiple views into a single `step_by_delay(..., finalize=True)` call.

