### Plan for the occupancy counting tool API

### Unified input format (Data Source)
- **Option A (file-based, preferred for large data)**:
  - `occupancy_file_path`: path to `so6_occupancy_matrix_with_times.json`
  - `tvtw_indexer_path`: path to `tvtw_indexer.json`
- **Option B (in-memory object)**:
  - `flights`: dict keyed by `flight_id`, values contain `occupancy_intervals` and metadata
  - `indexer`: dict with `time_bin_minutes` and `tv_id_to_idx`

The flight-level occupancy JSON already in the repo matches what we need:

```70:94:docs/impact_eval/impact_vector_so6_with_entry_time_documentation.md
    {
        "FLIGHT123": {
            "occupancy_intervals": [
                {
                    "tvtw_index": 101,
                    "entry_time_s": 120.5,
                    "exit_time_s": 350.2
                },
                {
                    "tvtw_index": 250,
                    "entry_time_s": 350.2,
                    "exit_time_s": 680.0
                }
            ],
            "distance": 150.7,
            "takeoff_time": "2023-08-01T08:05:00",
            "origin": "JFK",
            "destination": "BOS"
        },
        "FLIGHT456": {
            ...
        }
    }
```

The TVTW indexer JSON shape we rely on:

```115:125:src/project_tailwind/impact_eval/tvtw_indexer.py
    def save(self, file_path: str):
        """
        Saves the indexer's state to a JSON file.
        """
        state = {
            'time_bin_minutes': self.time_bin_minutes,
            'tv_id_to_idx': self._tv_id_to_idx,
        }
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)
```

Notes:
- `tv_id_to_idx` must be contiguous 0..T-1. Number of bins per TV is `num_time_bins = 1440 // time_bin_minutes`.
- For performance, load once and keep a `FlightList` instance in memory; sum counts with CSR ops.

### Unified output format
- For a given request window and TVs:
  - `time_bin_minutes`: int
  - `timebins`:
    - `start_bin`: int (offset within a TV)
    - `end_bin`: int (inclusive)
    - `labels`: list[str] of “HH:MM-HH:MM” for each returned bin, in order
  - `counts`: dict[tv_id -> list[int]] overall counts per bin
  - Optional `by_category`: dict[category_id -> dict[tv_id -> list[int]]] if categories are provided
  - `metadata`: { `num_tvs`, `num_bins`, `total_flights_considered`, `missing_flight_ids` (if any) }

### API contract
- Method: POST
- Path: `/original_counts`
- Request body:
  - `traffic_volume_ids` (list[str], required)
  - `from_time_str` (str, optional; formats: HHMM, HHMMSS, HH:MM, HH:MM:SS)
  - `to_time_str` (str, optional; same formats; required if `from_time_str` is provided)
  - `categories` (optional dict[str -> list[str]], e.g., `{"flowA": ["F1","F2"], "flowB": ["F3"]}`)
  - `include_overall` (bool, default true)
- Response: as in “Unified output format” above

### Examples

- Example request (no categories; full day):
```json
{
  "traffic_volume_ids": ["TV_A", "TV_B"]
}
```

- Example response (15-min bins; truncated):
```json
{
  "time_bin_minutes": 15,
  "timebins": {
    "start_bin": 0,
    "end_bin": 95,
    "labels": ["00:00-00:15", "00:15-00:30", "..."]
  },
  "counts": {
    "TV_A": [4, 7, 12, "..."],
    "TV_B": [1, 3, 5, "..."]
  },
  "metadata": {
    "num_tvs": 2,
    "num_bins": 96,
    "total_flights_considered": 12345
  }
}
```

- Example request (with time window and categories):
```json
{
  "traffic_volume_ids": ["TV_A"],
  "from_time_str": "06:00",
  "to_time_str": "07:30",
  "categories": {
    "flow_1": ["F001", "F002", "F003"],
    "flow_2": ["F010", "F011"]
  }
}
```

- Example response (bins 06:00–07:30, 15-min bins → N=7):
```json
{
  "time_bin_minutes": 15,
  "timebins": {
    "start_bin": 24,
    "end_bin": 30,
    "labels": ["06:00-06:15", "06:15-06:30", "06:30-06:45", "06:45-07:00", "07:00-07:15", "07:15-07:30", "07:15-07:30"]
  },
  "counts": {
    "TV_A": [8, 11, 9, 12, 10, 7, 6]
  },
  "by_category": {
    "flow_1": { "TV_A": [3, 5, 4, 6, 5, 3, 2] },
    "flow_2": { "TV_A": [2, 2, 1, 3, 2, 2, 1] }
  },
  "metadata": {
    "num_tvs": 1,
    "num_bins": 7,
    "total_flights_considered": 5,
    "missing_flight_ids": []
  }
}
```

### Implementation outline (server-side)
- Load once (on process start):
  - `FlightList(occupancy_file_path="output/so6_occupancy_matrix_with_times.json", tvtw_indexer_path="output/tvtw_indexer.json")`
- Time parsing:
  - Accept HHMM, HHMMSS, HH:MM, HH:MM:SS; compute `start_seconds` and `end_seconds`
  - `bin = seconds // (time_bin_minutes*60)`; clamp to `[0, num_bins_per_tv-1]`
- Overall counts (fast path):
  - `total = flight_list.get_total_occupancy_by_tvtw()` returns 1D vector over all TVTWs
  - For each `tv_id`: `base = tv_idx * num_bins_per_tv`, slice `total[base+start_bin:base+end_bin+1]`
- By-category counts:
  - Map flight IDs to row indices; build boolean row mask per category
  - For each TV, slice columns for that TV/time-range and sum over masked rows
- Time labels:
  - Format `HH:MM-HH:MM` from offset, derived solely from `time_bin_minutes`
- Validations:
  - Unknown `traffic_volume_ids` → 404
  - If only one of `from_time_str` or `to_time_str` is provided → 400
  - If `to < from` (wrap-around) → 400 (simple first version)
  - Unknown flight IDs in categories: ignore, return them in `missing_flight_ids`
