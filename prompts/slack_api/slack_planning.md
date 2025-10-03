### What the endpoint does
- Input: `traffic_volume_id`, `ref_time_str`, `sign` (“plus” | “minus”).
- Output: for every traffic volume, the slack at the “query time bin” determined by shifting the reference bin by the nominal travel time (distance / 475 kts), with details.
- Slack is capacity_per_bin − occupancy_per_bin at that bin. Positive slack suggests good “room to push flights into.”

### API surface
- Add GET `"/slack_distribution"` in `src/server_tailwind/main.py`:
  - Query params: `traffic_volume_id: str`, `ref_time_str: str`, `sign: str`.
  - Delegates to `AirspaceAPIWrapper.get_slack_distribution(...)`.
- Add `get_slack_distribution(...)` method in `src/server_tailwind/airspace/airspace_api_wrapper.py`.

### Core data and precomputations (cached/amortized in `AirspaceAPIWrapper`)
- Slack vector (1D, length = num_tvtws):
  - Compute once and cache: `slack_vector = capacity_per_bin_vector − total_occupancy_by_bin`.
  - `capacity_per_bin_vector`: use `_evaluator._get_total_capacity_vector()`.
  - `total_occupancy_by_bin`: `self._flight_list.get_total_occupancy_by_tvtw()`.
  - Store in `self._slack_vector` with a lock `self._slack_lock`. Recompute on demand if cache missing; consider invalidation hooks if flight list evolves later.
- Pairwise TV travel times (minutes), persisted:
  - File: `output/tv_travel_minutes_475.json`.
  - If file exists, load; else compute and save.
  - Compute distances in nautical miles between TV centroids from `self._traffic_volumes_gdf` (ensure CRS is EPSG:4326). Use a haversine function (e.g., `project_tailwind.impact_eval.distance_computation.haversine_vectorized`) to generate an NxN matrix.
  - Convert distance to minutes: minutes = (nm / 475.0) * 60.0.
  - Store nested dict: `{"metadata": {"speed_kts": 475, ...}, "travel_minutes": {src: {dst: minutes}}}`.
  - Cache in `self._tv_travel_minutes` with `self._travel_lock`.

### Computing the query time bin
- Parse `ref_time_str` robustly:
  - Accept `HHMMSS`, `HHMM`, `HH:MM`, `HH:MM:SS`. Convert to seconds since midnight, then to bin index: `ref_bin = floor(seconds / (time_bin_minutes * 60))`.
- For each TV:
  - Get travel minutes `m` from `traffic_volume_id` (the hotspot/source) to the iterated TV.
  - Bin shift = round(m / time_bin_minutes).
  - If `sign == "plus"`: `query_bin = ref_bin + shift`. If `sign == "minus"`: `query_bin = ref_bin - shift`.
  - Clamp `query_bin` to `[0, bins_per_tv - 1]`. (Optionally track `clamped: bool`.)

### Looking up slack for each TV at the query bin
- Compute `tvtw_idx = tv_row * bins_per_tv + query_bin`.
- Read:
  - `slack = slack_vector[tvtw_idx]`
  - `occupancy = total_occupancy_vector[tvtw_idx]` (re-use cached vector built alongside slack)
  - `capacity_per_bin`:
    - `hour = query_bin // bins_per_hour` where `bins_per_hour = 60 // time_bin_minutes`.
    - `hourly_cap = _evaluator.hourly_capacity_by_tv.get(tv_id, {}).get(hour, -1)`.
    - `cap_bin = hourly_cap / bins_per_hour if hourly_cap > -1 else 0.0`.
- Build a time-window label like `"HH:MM-HH:MM"` for `query_bin` (same formatting used elsewhere).
- Package result per TV with fields:
  - `traffic_volume_id`, `time_window`, `slack`, `occupancy`, `capacity_per_bin`, `distance_nm`, `travel_minutes`, `bin_offset`, `clamped`.
- Sort results by `slack` descending. (Optionally include all, even negatives; we’ll return all by default.)

### Data flow and threading
- Follow existing pattern: run heavy work in the wrapper’s thread pool using `loop.run_in_executor`.
- Guard cached structures with locks to avoid duplicate builds.

### Units and conversions
- Distance in nautical miles; speed in knots (nm/hr). Travel minutes = `(nm / 475) * 60`.
- Time bin width `time_bin_minutes` from `FlightList`.
- Bin offsets derived by `round(travel_minutes / time_bin_minutes)`.

### Edge cases and validation
- Validate `traffic_volume_id` exists; raise 404 via FastAPI handler.
- Validate `sign` in {“plus”, “minus”}; else 400.
- If a TV has no capacity for the query hour, `capacity_per_bin = 0` so slack may be negative; keep it.
- Clamp bins at edges (no wrap-around).
- If persisted travel-times file exists but `metadata.speed_kts != 475`, recompute.

### File and code changes
- `src/server_tailwind/main.py`: add `@app.get("/slack_distribution")` endpoint mapping to wrapper.
- `src/server_tailwind/airspace/airspace_api_wrapper.py`:
  - Add fields: `self._slack_vector`, `self._slack_lock`, `self._tv_travel_minutes`, `self._travel_lock`.
  - Add helpers:
    - `_get_or_build_slack_vector()`
    - `_format_time_window(bin_offset: int) -> str`
    - `_ensure_travel_minutes(speed_kts: float = 475.0) -> Dict[str, Dict[str, float]]`
    - `_compute_tv_centroid_latlon_map()`
  - Add public async method:
    - `get_slack_distribution(traffic_volume_id: str, ref_time_str: str, sign: str) -> Dict[str, Any]`
- Reuse `NetworkEvaluator` and `FlightList` already wired in wrapper; no changes required in `network_evaluator_for_api.py` or core `optimize` evaluator.

### Response shape (example)
- Top-level:
  - `traffic_volume_id`, `ref_time_str`, `sign`, `time_bin_minutes`, `nominal_speed_kts`, `count`
  - `results`: list of objects sorted by slack desc:
    - `traffic_volume_id`, `time_window`, `slack`, `occupancy`, `capacity_per_bin`, `distance_nm`, `travel_minutes`, `bin_offset`, `clamped`

- Optional future knobs (not required now): `only_positive=true`, `top_k`, `wrap_24h=true`.

- I’m ready to implement the wrapper method, caching, persistence, and the FastAPI endpoint next.