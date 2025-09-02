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

