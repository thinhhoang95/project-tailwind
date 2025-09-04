### GET `/flows`

Compute flow partitions for flights crossing one or more traffic volumes (TVs) and return per-flow demand plus per-flight details.

#### Query parameters
- **tvs** (required, string): Comma-separated traffic volume IDs. Example: `TV1,TV2`.
- **timebins** (optional, string): Comma-separated global time-bin indices to filter crossings (0 = start of day). Example: `18,19`.
- **threshold** (optional, float in [0,1], default 0.1): Jaccard similarity cutoff used before Leiden clustering.
- **resolution** (optional, float > 0, default 1.0): Leiden resolution; larger values yield more clusters.

Validation errors (HTTP 400) are returned if:
- **tvs** is empty
- **timebins** contains a non-integer value
- **threshold** is not a float in [0,1]
- **resolution** is not a positive float

#### 200 OK response
Top-level object:
- **num_time_bins** (int): Number of bins in the day (from the TVTW indexer).
- **tvs** (string[]): Echo of requested TVs.
- **timebins** (int[]): Echo of provided time bins; empty if none provided.
- **flows** (Flow[]): List of flow objects.

Flow object:
- **flow_id** (int): Stable 0-based identifier.
- **controlled_volume** (string|null): Selected controlled TV for this flow, if any.
- **demand** (int[]): Length `num_time_bins` array; counts of flights per requested bin.
- **flights** (Flight[]): Per-flight details.

Flight object:
- **flight_id** (string)
- **requested_bin** (int)
- **earliest_crossing_time** (string|null): Earliest crossing at the controlled volume (or among requested TVs if none selected), formatted as `YYYY-MM-DD HH:MM:SS`.

#### Example
Request:
```bash
curl -G \
  --data-urlencode "tvs=TV1,TV2" \
  --data-urlencode "timebins=18,19" \
  --data-urlencode "threshold=0.1" \
  --data-urlencode "resolution=1.0" \
  http://localhost:8000/flows
```

Response (truncated):
```json
{
  "num_time_bins": 48,
  "tvs": ["TV1", "TV2"],
  "timebins": [18, 19],
  "flows": [
    {
      "flow_id": 0,
      "controlled_volume": "TV1",
      "demand": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, ...],
      "flights": [
        {"flight_id": "F123", "requested_bin": 18, "earliest_crossing_time": "2025-01-01 09:00:10"},
        {"flight_id": "F456", "requested_bin": 19, "earliest_crossing_time": "2025-01-01 09:45:00"}
      ]
    }
  ]
}
```

#### Notes
- Flows are built by clustering TV footprints using Jaccard similarity and Leiden; default parameters are `threshold=0.1`, `resolution=1.0`, and a fixed seed of 0.
- If `timebins` is omitted, crossings across all bins are considered for the requested TVs, and the response's `timebins` will be an empty array.
- Data is loaded from `data/tailwind/tvtw_indexer.json` and `data/tailwind/so6_occupancy_matrix_with_times.json` inside the repository.
- `earliest_crossing_time` is derived from flight `takeoff_time` plus segment `entry_time_s` for the specified `requested_bin` and may be `null` if not available.

### POST `/base_evaluation`

Compute the baseline schedule and objective for a set of user-provided flows and time windows over target TVs. This endpoint prepares scheduling inputs using the existing policy (controlled-volume selection among targets, requested bins per flight), builds the baseline schedule `n0` (equal to demand), and evaluates the objective with current capacities and weights.

Additionally, for each flow it returns per-TV demand vectors for both target and ripple TVs.

#### JSON body
- **flows** (required, object): Mapping of flow-id -> list of flight IDs. Flow IDs may be strings or numeric; they are coerced to integers deterministically.
- **targets** (required, object): Mapping `TV_ID -> {"from": "HH:MM[:SS]", "to": "HH:MM[:SS]"}`. Defines attention cells for target TVs.
- **ripples** (optional, object): Same schema as `targets`. Defines secondary attention cells.
- **auto_ripple_time_bins** (optional, integer, default 0): If greater than 0, ripple cells are computed automatically as the union of TVTW footprints of all flights in all flows, dilated by ±`auto_ripple_time_bins` along time. When provided and > 0, this overrides `ripples`.
- **indexer_path** (optional, string): Override path to `tvtw_indexer.json`. Default: `data/tailwind/tvtw_indexer.json`.
- **flights_path** (optional, string): Override path to `so6_occupancy_matrix_with_times.json`. Default: `data/tailwind/so6_occupancy_matrix_with_times.json`.
- **capacities_path** (optional, string): Override path to capacities GeoJSON. Default: `data/cirrus/wxm_sm_ih_maxpool.geojson`.
- **weights** (optional, object): Partial overrides for `ObjectiveWeights` (e.g., `{"alpha_gt": 10.0, "lambda_delay": 0.1}`).

Validation errors (HTTP 400) are returned if:
- **flows** is missing or not an object
- **targets** is missing or empty
- Time ranges are malformed (HH:MM or HH:MM:SS required)

Unknown items are ignored gracefully:
- Unknown TV IDs in `targets`/`ripples` are dropped
- Unknown flight IDs in `flows` are ignored

#### 200 OK response
Top-level object:
- **num_time_bins** (int): Number of bins in the day.
- **tvs** (string[]): List of target TV IDs considered for control.
- **target_cells** (Array<[string, int]>): Explicit (tv, bin) pairs from `targets`.
- **ripple_cells** (Array<[string, int]>): Explicit (tv, bin) pairs from `ripples`.
- **flows** (FlowEval[]): Evaluation per flow.
- **objective** (object): `{"score": number, "components": {"J_cap": number, "J_delay": number, "J_reg": number, "J_tv": number, ...}}`.
- **weights_used** (object): Effective weights after overrides.

FlowEval object:
- **flow_id** (int)
- **controlled_volume** (string|null)
- **n0** (int[]): Length `T+1` array; counts by requested bin including overflow at index `T`.
- **demand** (int[]): Length `T` array; `n0` without overflow.
 - **target_demands** (object): Mapping `TV_ID -> int[]` (length `T`) giving earliest-crossing demand per time bin for each target TV.
 - **ripple_demands** (object): Mapping `TV_ID -> int[]` (length `T`) giving earliest-crossing demand per time bin for each ripple TV.

#### Example
Request:
```bash
curl -X POST http://localhost:8000/base_evaluation \
  -H 'Content-Type: application/json' \
  -d '{
    "flows": {"0": ["FLIGHT_1", "FLIGHT_2"], "1": ["FLIGHT_3"]},
    "targets": {"TV_A": {"from": "08:00", "to": "09:00"}},
    "auto_ripple_time_bins": 2,
    "weights": {"alpha_gt": 10.0, "lambda_delay": 0.1}
  }'
```

Response (truncated):
```json
{
  "num_time_bins": 48,
  "tvs": ["TV_A"],
  "target_cells": [["TV_A", 16], ["TV_A", 17]],
  "ripple_cells": [["TV_X", 10], ["TV_X", 11], ["TV_Y", 22], ...],
  "flows": [
    {
      "flow_id": 0,
      "controlled_volume": "TV_A",
      "n0": [0,0,0,1,0, ...],
      "demand": [0,0,0,1,0, ...],
      "target_demands": {"TV_A": [0,0,0,1,0, ...]},
      "ripple_demands": {"TV_B": [0,0,1,0,0, ...]}
    }
  ],
  "objective": {
    "score": 9546.9,
    "components": {"J_cap": 9056.5, "J_delay": 441.0, "J_reg": 39.6, "J_tv": 9.8}
  },
  "weights_used": {"alpha_gt": 10.0, "alpha_rip": 3.0, "alpha_ctx": 0.5, "beta_gt": 0.1, ...}
}
```

#### Notes
- Controlled volume selection is restricted to TVs provided in `targets`.
- Requested bins per flight are the earliest crossing at the controlled volume; if absent, earliest among targets; else 0.
- Time windows map to bins via half-open intervals `[from, to)`. If `to <= from`, the window is ignored.
- Capacities are loaded per TV from the GeoJSON and used in `J_cap` via rolling-hour exceedance.
- `n0` includes an overflow bin at index `T` by design; `demand` excludes overflow.
