Can you help me implement the following changes in the code base?

### What we will change
- Goal: keep SA optimizing per-bin release rates, but let FIFO compute minute-precise delays by supplying each flight’s within-bin requested time at the controlled volume using `entry_time_s`.

### Source of within-bin times
- `FlightList.flight_metadata[fid]` has:
  - `takeoff_time`: naive UTC `datetime`
  - `occupancy_intervals`: list with `tvtw_index`, `entry_time_s`, `exit_time_s`
- We can decode `tvtw_index -> (tv_id, bin)` via `TVTWIndexer.get_tvtw_from_index`.

### Where to add the within-bin requested times
- File: `src/parrhesia/optim/sa_optimizer.py`
- Function: `prepare_flow_scheduling_inputs(...)`

Plan:
1) Extend precomputation in `prepare_flow_scheduling_inputs` to capture both the earliest crossing bin and its offset seconds, per flight and per hotspot TV.
   - Build `earliest_crossing_by_flight_by_tv: Dict[str, Dict[str, Tuple[int, float]]]` that stores, for each flight and TV, the pair `(earliest_bin, entry_time_s_at_that_bin)`.
   - Use the existing `decode` helper to map each interval’s `tvtw_index` to `(tv_id, bin)`.
   - When choosing earliest, compare `bin`; store the entry seconds for the selected interval.

2) For each flow, pick the controlled volume TV as already implemented.
   - For each flight in the flow:
     - Determine `requested_bin` exactly as today (controlled volume if possible; fallback to earliest across hotspots; last resort 0).
     - If we have a `takeoff_time` and the `(bin, entry_s)` for the chosen TV:
       - Compute within-bin requested time: `requested_dt = takeoff_time + timedelta(seconds=entry_s)`.
       - Build the flight spec including BOTH:
         - `requested_bin` (for demand histograms and compatibility), and
         - `requested_dt` (for minute-precise FIFO delays).
     - Else, keep the current behavior (just `requested_bin`).

3) Do not change SA variables or moves.
   - SA still optimizes `n_f(t)` per flow per bin.
   - Demands (`d_f(t)`) will still be computed from `requested_bin` (unchanged); if only `requested_dt` were present, the scheduler already derives its bin consistently, but we will keep `requested_bin` explicitly.

4) Leave FIFO scheduler as is.
   - `parrhesia/fcfs/flowful.py` already:
     - Accepts `requested_dt` or the pair `takeoff_time` + `entry_time_s`.
     - Computes delays with minute precision when a datetime exists; otherwise whole-bin multiples.

5) Leave objective/scoring wiring unchanged.
   - `score`/`score_with_context` call FIFO; with `requested_dt` present, delays become minute-precise automatically.
   - Occupancy after delays still operates on bins; sub-bin delays round up to bins inside `compute_occupancy` (as designed).

### Backward compatibility and fallbacks
- If a flight is missing `takeoff_time` or the relevant interval’s `entry_time_s`, we keep the current bin-only path.
- If a flight doesn’t cross the controlled volume, we fallback to earliest across the hotspot set; if that still fails, use bin 0, as today.

### Data correctness notes
- Keeping `requested_bin` in the spec guarantees consistency for existing demand and reporting.
- `requested_dt` will correspond to the same bin chosen for `requested_bin`, because it’s computed from that interval’s `entry_time_s`.
- All times remain naive UTC; `TVTWIndexer` derives the bin from the date of `requested_dt`.

### Incremental integration steps
- Edit `prepare_flow_scheduling_inputs`:
  - Augment the existing per-flight earliest-bin map to also store `entry_time_s`.
  - For each output spec, set `requested_bin` and, when available, `requested_dt` as described.
- No code changes in `fcfs/flowful.py`, `optim/objective.py`, or the API are required for the core logic.
- Optional: add a small debug log line when a spec is enriched with `requested_dt`, and when we fallback to bin-only.

### Testing plan
- Unit: `_normalize_flight_spec` already covers `requested_dt` and `(takeoff_time, entry_time_s)` inputs; add a minimal test asserting the returned `(flight_id, requested_dt, bin)` matches expectations given known `takeoff_time`, `entry_time_s`, and indexer.
- Unit: `preprocess_flights_for_scheduler` should sort within a bin by `requested_dt` (existing behavior). Add a test with two flights in same bin but different `entry_time_s`; verify order.
- FIFO behavior:
  - Set one flow with two flights whose `requested_dt` are 2 and 7 minutes into the same bin; set `n_f(bin) = 1` and `n_f(bin+1) = 1`.
  - Expect delays of 0 and ceil((start_of_next_bin - requested_dt)/60) minutes (not a 15-minute multiple).
- E2E via API:
  - Small payload with a single controlled TV window and two flights; ensure the returned `delays_min` include a non-15-minute value.
- Regression: Compare pre/post objective components on a small scenario; only `J_delay` should potentially change (lower or equal), others unchanged for the same `n`.

### Documentation updates
- Update `API_README_SA.md` and `API_README_FLOWS.md` to state:
  - SA still optimizes per-bin rates.
  - When flight specs include `requested_dt` (derived from `takeoff_time + entry_time_s`), FIFO returns minute-precision delays instead of bin multiples.
- Note in `prompts/datetime_rate_opt.md` that the SA path now passes within-bin times.

### Risks and mitigations
- Missing metadata: handled via fallback to bin-only.
- Date boundaries: `bin_of_datetime` uses the day of `requested_dt`; the chosen bin remains consistent with the decoded TVTW index.
- Performance: negligible overhead; all logic is linear in number of intervals per flight.

### Acceptance criteria
- For scenarios where `entry_time_s` is within a bin (not at bin start), the API’s `delays_min` includes values not divisible by the bin length.
- No change in SA variable shapes, API schema, or capacity/occupancy logic other than more precise delays.
- Existing bin-only datasets continue to work unchanged.

- Implementing this means only one function edit (`prepare_flow_scheduling_inputs`) plus tests and docs; the FIFO scheduler already supports the needed shapes.

- In short: we’ll enrich each flight spec with a real `requested_dt` using `takeoff_time + entry_time_s` of the earliest crossing at the chosen controlled volume, while preserving `requested_bin` for demand computations. This will make FIFO assign minute-level delays without changing SA’s optimization space.