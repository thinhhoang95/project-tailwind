# Counts APIs

This document covers two related endpoints:
- Original Counts over TVs and time windows
- Autorate Occupancy aggregation from a prior optimization result

The server holds a single in‑memory FlightList, so requests reuse loaded data without re-reading large JSON files.

---

## Original Counts Endpoint

- Method: POST
- Path: `/original_counts`
- Content-Type: `application/json`
- Auth: none

---

## Request Body

- `traffic_volume_ids` (list[string], optional):
  - If provided, a dedicated `mentioned_counts` section will be returned for these TVs. All must be known (404 on any unknown ID).

- `from_time_str` (string, optional) and `to_time_str` (string, optional):
  - Accepts formats: `HHMM`, `HHMMSS`, `HH:MM`, `HH:MM:SS`.
  - If one is provided, the other is required; otherwise the full day is returned.
  - Time strings map to inclusive bin indices via: `bin = floor(seconds / (time_bin_minutes * 60))`, then clamped to `[0, num_bins_per_tv - 1]`.
  - `to_time_str` must be at least `from_time_str` (no wrap-around in this version); otherwise 400.

- `categories` (object, optional):
  - A mapping `{ category_id: [flight_id, ...] }`.
  - For each category, counts are computed only over the specified flights.
  - Unknown flight IDs are ignored and returned in `metadata.missing_flight_ids`.

- `flight_ids` (list[string], optional):
  - When provided and `categories` is absent, only these flights are admitted to the counting process (acts as a filter).
  - Unknown flight IDs are ignored and returned in `metadata.missing_flight_ids`.
  - Ignored if `categories` is present.

- `rank_by` (string, optional; default `"total_count"`):
  - Supported values:
    - `"total_count"`: Ranks by total count over the selected time range.
    - `"total_excess"`: Ranks by total overload over the selected time range, consistent with `/hotspots` semantics.
      - `excess_per_bin = max(count_per_bin − capacity_per_bin, 0)`; bins with `capacity_per_bin = -1` are ignored (contribute 0).
      - When `rolling_hour=true` (default), `count_per_bin` is the forward-looking 60‑minute rolling sum per bin; otherwise raw per-bin counts are used.

- `rolling_hour` (boolean, optional; default `true`):
  - When true, each bin’s count is the sum over the next hour from that bin (non-wrapping).

- `include_overall` (boolean, optional):
  - Accepted for backward compatibility; the endpoint always returns `counts` for the ranked top‑50 TVs.

---

## Response Body

- `time_bin_minutes` (int): Size of each time bin in minutes (e.g., 15).
- `timebins` (object):
  - `start_bin` (int): First returned bin index within a TV.
  - `end_bin` (int): Last returned bin index within a TV (inclusive).
  - `labels` (list[string]): Human-readable window labels for each returned bin, formatted `HH:MM-HH:MM`.
- `counts` (object): `{ tv_id: [int, ...] }` counts per bin for the ranked top‑50 TVs (ranking may include TVs not in `traffic_volume_ids`).
- `capacity` (object): `{ tv_id: [float, ...] }` capacity per bin aligned to the same bins as `counts` for the ranked top‑50 TVs. Values are the hourly capacity value repeated across bins in that hour; `-1` indicates capacity not available.
- `mentioned_counts` (object, optional): `{ tv_id: [int, ...] }` counts per bin for TVs explicitly provided in `traffic_volume_ids` (if any). These TVs can also appear in `counts`.
- `mentioned_capacity` (object, optional): `{ tv_id: [float, ...] }` capacity per bin for TVs explicitly provided in `traffic_volume_ids`.
- `by_category` (object, optional): `{ category_id: { tv_id: [int, ...] } }` per-category counts per bin for the same set of TVs as in `counts` (top‑50).
- `by_category_mentioned` (object, optional): present when `traffic_volume_ids` are provided; `{ category_id: { tv_id: [int, ...] } }` per-category counts per bin for only the mentioned TVs.
- `metadata` (object):
  - `num_tvs` (int): Number of TVs included in `counts` (typically 50).
  - `num_mentioned` (int, optional): Number of TVs included in `mentioned_counts` when `traffic_volume_ids` is provided.
  - `num_bins` (int): Number of bins returned (`end_bin - start_bin + 1`).
  - `total_flights_considered` (int): Total unique flights considered; if categories are provided, counts flights in the union of all category lists, else all flights.
  - `rank_by` (string): Ranking criterion used.
  - `top_k` (int): Fixed at 50.
  - `rolling_hour` (boolean): Whether rolling-hour accumulation was applied.
  - `rolling_window_minutes` (int): Rolling window size in minutes (60).
  - `ranked_tv_ids` (list[string]): TV IDs in ranked order matching the keys in `counts`.
  - `missing_flight_ids` (list[string], optional): Any unknown flight IDs encountered in `categories` or `flight_ids`.

Notes
- The binning and number of time bins per TV derive from the preloaded TVTW indexer (`time_bin_minutes`).
- When `include_overall` is false, `counts` is omitted and only `by_category` is returned (if provided).

---

## Examples

### 1) Full day, no requested TVs (top‑50 returned)

Request
```json
{}
```

Response (truncated)
```json
{
  "time_bin_minutes": 15,
  "timebins": {
    "start_bin": 0,
    "end_bin": 95,
    "labels": ["00:00-00:15", "00:15-00:30", "..."]
  },
  "counts": {
    "TV_123": [4, 7, 12, "..."],
    "TV_456": [3, 5, 9, "..."],
    "...": ["..."]
  },
  "capacity": {
    "TV_123": [20, 20, 20, "..."],
    "TV_456": [18, 18, 18, "..."]
  },
  "metadata": {
    "num_tvs": 50,
    "num_bins": 96,
    "total_flights_considered": 12345,
    "rank_by": "total_count",
    "top_k": 50,
    "rolling_hour": true,
    "rolling_window_minutes": 60,
    "ranked_tv_ids": ["TV_123", "TV_456", "..."]
  }
}
```

### 2) Time window with requested TVs and mentioned_counts

Request
```json
{
  "traffic_volume_ids": ["TV_A"],
  "from_time_str": "06:00",
  "to_time_str": "07:30"
}
```

Response (bins 06:00–07:30 → N=7 for 15-min bins)
```json
{
  "time_bin_minutes": 15,
  "timebins": {
    "start_bin": 24,
    "end_bin": 30,
    "labels": [
      "06:00-06:15", "06:15-06:30", "06:30-06:45",
      "06:45-07:00", "07:00-07:15", "07:15-07:30", "07:15-07:30"
    ]
  },
  "counts": {
    "TV_A": [8, 11, 9, 12, 10, 7, 6],
    "TV_X": [7, 9, 8, 11, 9, 6, 5],
    "...": ["..."]
  },
  "capacity": {
    "TV_A": [22, 22, 22, 22, 22, 22, 22],
    "TV_X": [18, 18, 18, 18, 18, 18, 18]
  },
  "mentioned_counts": {
    "TV_A": [8, 11, 9, 12, 10, 7, 6]
  },
  "mentioned_capacity": {
    "TV_A": [22, 22, 22, 22, 22, 22, 22]
  },
  "metadata": {
    "num_tvs": 50,
    "num_mentioned": 1,
    "num_bins": 7,
    "total_flights_considered": 12345
  }
}
```

### 3) Time window with categories (flows)

Request
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

Response (truncated)
```json
{
  "time_bin_minutes": 15,
  "timebins": {
    "start_bin": 24,
    "end_bin": 30,
    "labels": ["06:00-06:15", "06:15-06:30", "..."]
  },
  "counts": {
    "TV_A": [8, 11, 9, 12, 10, 7, 6],
    "TV_B": [4, 6, 5, 7, 6, 4, 3],
    "...": ["..."]
  },
  "by_category": {
    "flow_1": { "TV_A": [3, 5, 4, 6, 5, 3, 2], "TV_B": [2, 3, 3, 4, 3, 2, 1] },
    "flow_2": { "TV_A": [2, 2, 1, 3, 2, 2, 1], "TV_B": [1, 1, 1, 2, 1, 1, 1] }
  },
  "by_category_mentioned": {
    "flow_1": { "TV_A": [3, 5, 4, 6, 5, 3, 2] },
    "flow_2": { "TV_A": [2, 2, 1, 3, 2, 2, 1] }
  },
  "metadata": {
    "num_tvs": 50,
    "num_bins": 7,
    "total_flights_considered": 5,
    "missing_flight_ids": []
  }
}
```

### 4) Filter by explicit flight list (no categories)

Request
```json
{
  "traffic_volume_ids": ["TV_A", "TV_B"],
  "from_time_str": "08:00",
  "to_time_str": "09:00",
  "flight_ids": ["F001", "F010", "F999"]
}
```

Response (truncated; note `missing_flight_ids` for unknown entries)
```json
{
  "time_bin_minutes": 15,
  "timebins": { "start_bin": 32, "end_bin": 36, "labels": ["08:00-08:15", "..."] },
  "counts": {
    "TV_A": [1, 2, 1, 0, 0],
    "TV_B": [0, 1, 1, 0, 0],
    "...": ["..."]
  },
  "mentioned_counts": { "TV_A": [1, 2, 1, 0, 0], "TV_B": [0, 1, 1, 0, 0] },
  "metadata": {
    "num_tvs": 50,
    "num_mentioned": 2,
    "num_bins": 5,
    "total_flights_considered": 2,
    "missing_flight_ids": ["F999"]
  }
}
```

---

## Error Handling

- 400 Bad Request:
  - Only one of `from_time_str` / `to_time_str` provided.
  - Invalid time format or components.
  - `to_time_str` earlier than `from_time_str`.
  - Wrong data types (e.g., `traffic_volume_ids` not a list).

- 404 Not Found:
  - Any unknown `traffic_volume_ids` (when provided).

- 500 Internal Server Error:
  - Unexpected server-side failures.

Example error response
```json
{
  "detail": "Unknown traffic_volume_ids: [\"TV_UNKNOWN\"]"
}
```

---

## Original Flight Contribution Counts Endpoint

- Method: POST
- Path: `/original_flight_contrib_counts`
- Content-Type: `application/json`
- Auth: none

---

## Request Body

- `traffic_volume_ids` (list[string], optional):
  - If provided, they do not change ranking but can be used by clients to focus on specific TVs. All must be known (404 on any unknown ID).

- `from_time_str` (string, optional) and `to_time_str` (string, optional):
  - Same formats and rules as `/original_counts`.

- `flight_ids` (list[string], required):
  - The flight list whose contribution is measured. Unknown flights are ignored and reported in `metadata.missing_flight_ids`.

- `rank_by` (string, optional; default `"total_count"`):
  - Supported values:
    - `"total_count"`: Ranks TVs by total rolling-hour count over the selected time range using `total_counts`.
    - `"total_excess"`: Ranks by total rolling-hour overload over the selected time range using `total_counts` and capacity; bins with `capacity=-1` are ignored.
    - `"flight_list_count"`: Ranks by the total rolling-hour count contributed by the provided `flight_ids` over the selected time range using `flight_list_counts`.
    - `"flight_list_relative"`: Ranks by the share of rolling-hour counts contributed by the provided `flight_ids` relative to the total (`flight_list_counts / total_counts`). TVs with zero total counts receive a ratio of 0.

- `rolling_hour` (boolean, optional; default `true`):
  - When true, both `total_counts` and `flight_list_counts` use forward-looking 60‑minute sums per bin (no wrap).

Notes
- Ranking uses the metric specified by `rank_by` (default `total_counts`) over the requested time range.
- Capacity arrays repeat hourly capacity across bins within that hour; `-1` denotes missing capacity.

---

## Response Body

- `time_bin_minutes` (int): Size of each time bin in minutes.
- `timebins` (object):
  - `start_bin` (int), `end_bin` (int), `labels` (list[string]) as in `/original_counts`.
- `total_counts` (object): `{ tv_id: [int, ...] }` total rolling-hour counts across all flights for top‑50 TVs (ranked set).
- `flight_list_counts` (object): `{ tv_id: [int, ...] }` rolling-hour counts attributed only to the provided `flight_ids` for the same TVs.
- `capacity` (object): `{ tv_id: [float, ...] }` capacity per bin aligned to the returned bins for the same TVs.
- `metadata` (object):
  - `num_tvs` (int): Number of TVs included (typically 50).
  - `num_bins` (int): Number of bins returned (`end_bin - start_bin + 1`).
  - `total_flights_considered` (int): Number of recognized flights from `flight_ids` (deduplicated).
  - `rank_by` (string), `top_k` (int), `rolling_hour` (boolean), `rolling_window_minutes` (int = 60).
  - `ranked_tv_ids` (list[string]): Ranked order matching keys in `total_counts` and `flight_list_counts`.
  - `missing_flight_ids` (list[string], optional): Any unknown flight IDs from `flight_ids`.

---

## Examples

### A) Full day, required flight list

Request
```json
{
  "flight_ids": ["F001", "F002", "F003"]
}
```

Response (truncated)
```json
{
  "time_bin_minutes": 15,
  "timebins": { "start_bin": 0, "end_bin": 95, "labels": ["00:00-00:15", "..."] },
  "total_counts": {
    "TV_123": [5, 9, 12, "..."],
    "TV_456": [3, 6, 8, "..."]
  },
  "flight_list_counts": {
    "TV_123": [1, 2, 3, "..."],
    "TV_456": [0, 1, 2, "..."]
  },
  "capacity": {
    "TV_123": [20, 20, 20, "..."],
    "TV_456": [18, 18, 18, "..."]
  },
  "metadata": {
    "num_tvs": 50,
    "num_bins": 96,
    "total_flights_considered": 3,
    "rank_by": "total_count",
    "top_k": 50,
    "rolling_hour": true,
    "rolling_window_minutes": 60,
    "ranked_tv_ids": ["TV_123", "TV_456", "..."]
  }
}
```

### B) Time window and rank by total_excess

Request
```json
{
  "from_time_str": "06:00",
  "to_time_str": "07:30",
  "flight_ids": ["F001", "F002", "F999"],
  "rank_by": "total_excess"
}
```

Response (shape, truncated; note `missing_flight_ids`)
```json
{
  "time_bin_minutes": 15,
  "timebins": { "start_bin": 24, "end_bin": 30, "labels": ["06:00-06:15", "..."] },
  "total_counts": { "TV_A": [8, 11, 9, 12, 10, 7, 6] },
  "flight_list_counts": { "TV_A": [2, 3, 2, 2, 1, 1, 1] },
  "capacity": { "TV_A": [22, 22, 22, 22, 22, 22, 22] },
  "metadata": {
    "num_tvs": 50,
    "num_bins": 7,
    "total_flights_considered": 2,
    "rank_by": "total_excess",
    "top_k": 50,
    "rolling_hour": true,
    "rolling_window_minutes": 60,
    "ranked_tv_ids": ["TV_A", "..."]
  }
}
```

---

## Implementation Notes

- The server loads `FlightList` once at startup from:
  - `output/so6_occupancy_matrix_with_times.json`
  - `output/tvtw_indexer.json`
- Overall counts reuse a cached total occupancy vector for performance when no flight filter is applied.
- Ranking is performed over the selected time range using the (optionally) rolling-hour transformed counts; the top‑50 TVs are returned.
- Rolling-hour uses a forward-looking window of size `ceil(60 / time_bin_minutes)` bins (no wrap at day boundary).
- Category counts sum over the selected flight rows of the sparse occupancy matrix and are returned for the same top‑50 TVs.
- Time labels are generated purely from `time_bin_minutes` and bin indices.

---

## Autorate Occupancy Aggregation

Aggregate pre/post occupancy counts per traffic volume across flows, using only the data already present in a prior `/automatic_rate_adjustment` response. No optimization is run here.

### Endpoint

- Method: POST
- Path: `/autorate_occupancy`
- Content-Type: `application/json`
- Auth: none

### Request Body

- `autorate_result` (object, required): The exact JSON object returned by `/automatic_rate_adjustment`.
- `include_capacity` (boolean, optional, default `true`): Whether to include per-bin capacity arrays.
- `rolling_hour` (boolean, optional, default `true`): When true, `pre_counts` and `post_counts` are transformed into rolling-hour sums (forward-looking window of 60 minutes), consistent with `/original_counts`.

Notes
- TV set adheres strictly to the prior result:
  - Targets: `autorate_result.tvs` (order preserved)
  - Ripples: unique TV order from `autorate_result.ripple_cells` (append after targets; dedup)
- Binning is taken from the autorate result (`num_time_bins`) and the app indexer (`time_bin_minutes`).
- Capacity per bin uses the server’s preloaded matrix; unknown TVs or missing capacity return `-1` per bin.

### Response Body

- `time_bin_minutes` (int): Bin size in minutes.
- `num_bins` (int): Number of bins per TV.
- `tv_ids_order` (list[string]): Targets first in given order, then ripples (deduped).
- `timebins.labels` (list[string]): Labels `HH:MM-HH:MM` for all bins `[0..T-1]`.
- `pre_counts` (object): `{ tv_id: int[T] }` baseline occupancy across all flights for each TV. When `rolling_hour` is true, counts are rolling-hour aggregates.
- `post_counts` (object): `{ tv_id: int[T] }` baseline adjusted by the optimized delays (applied only to affected flights). When `rolling_hour` is true, counts are rolling-hour aggregates.
- `capacity` (object, optional): `{ tv_id: float[T] }` capacity per bin (hourly value repeated; `-1` if unknown).

### Example

Request
```json
{
  "autorate_result": { /* the full object returned by /automatic_rate_adjustment */ },
  "include_capacity": true
}
```

Response (shape, truncated)
```json
{
  "time_bin_minutes": 15,
  "num_bins": 96,
  "tv_ids_order": ["TV_TARGET_A", "TV_TARGET_B", "TV_RIPPLE_X"],
  "timebins": { "labels": ["00:00-00:15", "00:15-00:30", "..."] },
  "pre_counts": {
    "TV_TARGET_A": [3, 5, 4, "..."],
    "TV_TARGET_B": [1, 2, 3, "..."],
    "TV_RIPPLE_X": [0, 1, 1, "..."]
  },
  "post_counts": {
    "TV_TARGET_A": [2, 4, 3, "..."],
    "TV_TARGET_B": [1, 1, 2, "..."],
    "TV_RIPPLE_X": [0, 1, 1, "..."]
  },
  "capacity": {
    "TV_TARGET_A": [20, 20, 20, "..."],
    "TV_TARGET_B": [18, 18, 18, "..."],
    "TV_RIPPLE_X": [15, 15, 15, "..."]
  }
}
```
# Counts APIs

This document covers two related endpoints:
- Original Counts over TVs and time windows
- Autorate Occupancy aggregation from a prior optimization result

The server holds a single in‑memory FlightList, so requests reuse loaded data without re-reading large JSON files.

---

## Original Counts Endpoint

- Method: POST
- Path: `/original_counts`
- Content-Type: `application/json`
- Auth: none

---

## Request Body

- `traffic_volume_ids` (list[string], optional):
  - If provided, a dedicated `mentioned_counts` section will be returned for these TVs. All must be known (404 on any unknown ID).

- `from_time_str` (string, optional) and `to_time_str` (string, optional):
  - Accepts formats: `HHMM`, `HHMMSS`, `HH:MM`, `HH:MM:SS`.
  - If one is provided, the other is required; otherwise the full day is returned.
  - Time strings map to inclusive bin indices via: `bin = floor(seconds / (time_bin_minutes * 60))`, then clamped to `[0, num_bins_per_tv - 1]`.
  - `to_time_str` must be at least `from_time_str` (no wrap-around in this version); otherwise 400.

- `categories` (object, optional):
  - A mapping `{ category_id: [flight_id, ...] }`.
  - For each category, counts are computed only over the specified flights.
  - Unknown flight IDs are ignored and returned in `metadata.missing_flight_ids`.

- `flight_ids` (list[string], optional):
  - When provided and `categories` is absent, only these flights are admitted to the counting process (acts as a filter).
  - Unknown flight IDs are ignored and returned in `metadata.missing_flight_ids`.
  - Ignored if `categories` is present.

- `rank_by` (string, optional; default `"total_count"`):
  - Supported values:
    - `"total_count"`: Ranks by total count over the selected time range.
    - `"total_excess"`: Ranks by total overload over the selected time range, consistent with `/hotspots` semantics.
      - `excess_per_bin = max(count_per_bin − capacity_per_bin, 0)`; bins with `capacity_per_bin = -1` are ignored (contribute 0).
      - When `rolling_hour=true` (default), `count_per_bin` is the forward-looking 60‑minute rolling sum per bin; otherwise raw per-bin counts are used.

- `rolling_hour` (boolean, optional; default `true`):
  - When true, each bin’s count is the sum over the next hour from that bin (non-wrapping).

- `include_overall` (boolean, optional):
  - Accepted for backward compatibility; the endpoint always returns `counts` for the ranked top‑50 TVs.

---

## Response Body

- `time_bin_minutes` (int): Size of each time bin in minutes (e.g., 15).
- `timebins` (object):
  - `start_bin` (int): First returned bin index within a TV.
  - `end_bin` (int): Last returned bin index within a TV (inclusive).
  - `labels` (list[string]): Human-readable window labels for each returned bin, formatted `HH:MM-HH:MM`.
- `counts` (object): `{ tv_id: [int, ...] }` counts per bin for the ranked top‑50 TVs (ranking may include TVs not in `traffic_volume_ids`).
- `capacity` (object): `{ tv_id: [float, ...] }` capacity per bin aligned to the same bins as `counts` for the ranked top‑50 TVs. Values are the hourly capacity value repeated across bins in that hour; `-1` indicates capacity not available.
- `mentioned_counts` (object, optional): `{ tv_id: [int, ...] }` counts per bin for TVs explicitly provided in `traffic_volume_ids` (if any). These TVs can also appear in `counts`.
- `mentioned_capacity` (object, optional): `{ tv_id: [float, ...] }` capacity per bin for TVs explicitly provided in `traffic_volume_ids`.
- `by_category` (object, optional): `{ category_id: { tv_id: [int, ...] } }` per-category counts per bin for the same set of TVs as in `counts` (top‑50).
- `by_category_mentioned` (object, optional): present when `traffic_volume_ids` are provided; `{ category_id: { tv_id: [int, ...] } }` per-category counts per bin for only the mentioned TVs.
- `metadata` (object):
  - `num_tvs` (int): Number of TVs included in `counts` (typically 50).
  - `num_mentioned` (int, optional): Number of TVs included in `mentioned_counts` when `traffic_volume_ids` is provided.
  - `num_bins` (int): Number of bins returned (`end_bin - start_bin + 1`).
  - `total_flights_considered` (int): Total unique flights considered; if categories are provided, counts flights in the union of all category lists, else all flights.
  - `rank_by` (string): Ranking criterion used.
  - `top_k` (int): Fixed at 50.
  - `rolling_hour` (boolean): Whether rolling-hour accumulation was applied.
  - `rolling_window_minutes` (int): Rolling window size in minutes (60).
  - `ranked_tv_ids` (list[string]): TV IDs in ranked order matching the keys in `counts`.
  - `missing_flight_ids` (list[string], optional): Any unknown flight IDs encountered in `categories` or `flight_ids`.

Notes
- The binning and number of time bins per TV derive from the preloaded TVTW indexer (`time_bin_minutes`).
- When `include_overall` is false, `counts` is omitted and only `by_category` is returned (if provided).

---

## Examples

### 1) Full day, no requested TVs (top‑50 returned)

Request
```json
{}
```

Response (truncated)
```json
{
  "time_bin_minutes": 15,
  "timebins": {
    "start_bin": 0,
    "end_bin": 95,
    "labels": ["00:00-00:15", "00:15-00:30", "..."]
  },
  "counts": {
    "TV_123": [4, 7, 12, "..."],
    "TV_456": [3, 5, 9, "..."],
    "...": ["..."]
  },
  "capacity": {
    "TV_123": [20, 20, 20, "..."],
    "TV_456": [18, 18, 18, "..."]
  },
  "metadata": {
    "num_tvs": 50,
    "num_bins": 96,
    "total_flights_considered": 12345,
    "rank_by": "total_count",
    "top_k": 50,
    "rolling_hour": true,
    "rolling_window_minutes": 60,
    "ranked_tv_ids": ["TV_123", "TV_456", "..."]
  }
}
```

### 2) Time window with requested TVs and mentioned_counts

Request
```json
{
  "traffic_volume_ids": ["TV_A"],
  "from_time_str": "06:00",
  "to_time_str": "07:30"
}
```

Response (bins 06:00–07:30 → N=7 for 15-min bins)
```json
{
  "time_bin_minutes": 15,
  "timebins": {
    "start_bin": 24,
    "end_bin": 30,
    "labels": [
      "06:00-06:15", "06:15-06:30", "06:30-06:45",
      "06:45-07:00", "07:00-07:15", "07:15-07:30", "07:15-07:30"
    ]
  },
  "counts": {
    "TV_A": [8, 11, 9, 12, 10, 7, 6],
    "TV_X": [7, 9, 8, 11, 9, 6, 5],
    "...": ["..."]
  },
  "capacity": {
    "TV_A": [22, 22, 22, 22, 22, 22, 22],
    "TV_X": [18, 18, 18, 18, 18, 18, 18]
  },
  "mentioned_counts": {
    "TV_A": [8, 11, 9, 12, 10, 7, 6]
  },
  "mentioned_capacity": {
    "TV_A": [22, 22, 22, 22, 22, 22, 22]
  },
  "metadata": {
    "num_tvs": 50,
    "num_mentioned": 1,
    "num_bins": 7,
    "total_flights_considered": 12345
  }
}
```

### 3) Time window with categories (flows)

Request
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

Response (truncated)
```json
{
  "time_bin_minutes": 15,
  "timebins": {
    "start_bin": 24,
    "end_bin": 30,
    "labels": ["06:00-06:15", "06:15-06:30", "..."]
  },
  "counts": {
    "TV_A": [8, 11, 9, 12, 10, 7, 6],
    "TV_B": [4, 6, 5, 7, 6, 4, 3],
    "...": ["..."]
  },
  "by_category": {
    "flow_1": { "TV_A": [3, 5, 4, 6, 5, 3, 2], "TV_B": [2, 3, 3, 4, 3, 2, 1] },
    "flow_2": { "TV_A": [2, 2, 1, 3, 2, 2, 1], "TV_B": [1, 1, 1, 2, 1, 1, 1] }
  },
  "by_category_mentioned": {
    "flow_1": { "TV_A": [3, 5, 4, 6, 5, 3, 2] },
    "flow_2": { "TV_A": [2, 2, 1, 3, 2, 2, 1] }
  },
  "metadata": {
    "num_tvs": 50,
    "num_bins": 7,
    "total_flights_considered": 5,
    "missing_flight_ids": []
  }
}
```

### 4) Filter by explicit flight list (no categories)

Request
```json
{
  "traffic_volume_ids": ["TV_A", "TV_B"],
  "from_time_str": "08:00",
  "to_time_str": "09:00",
  "flight_ids": ["F001", "F010", "F999"]
}
```

Response (truncated; note `missing_flight_ids` for unknown entries)
```json
{
  "time_bin_minutes": 15,
  "timebins": { "start_bin": 32, "end_bin": 36, "labels": ["08:00-08:15", "..."] },
  "counts": {
    "TV_A": [1, 2, 1, 0, 0],
    "TV_B": [0, 1, 1, 0, 0],
    "...": ["..."]
  },
  "mentioned_counts": { "TV_A": [1, 2, 1, 0, 0], "TV_B": [0, 1, 1, 0, 0] },
  "metadata": {
    "num_tvs": 50,
    "num_mentioned": 2,
    "num_bins": 5,
    "total_flights_considered": 2,
    "missing_flight_ids": ["F999"]
  }
}
```

---

## Error Handling

- 400 Bad Request:
  - Only one of `from_time_str` / `to_time_str` provided.
  - Invalid time format or components.
  - `to_time_str` earlier than `from_time_str`.
  - Wrong data types (e.g., `traffic_volume_ids` not a list).

- 404 Not Found:
  - Any unknown `traffic_volume_ids` (when provided).

- 500 Internal Server Error:
  - Unexpected server-side failures.

Example error response
```json
{
  "detail": "Unknown traffic_volume_ids: [\"TV_UNKNOWN\"]"
}
```

---

## Implementation Notes

- The server loads `FlightList` once at startup from:
  - `output/so6_occupancy_matrix_with_times.json`
  - `output/tvtw_indexer.json`
- Overall counts reuse a cached total occupancy vector for performance when no flight filter is applied.
- Ranking is performed over the selected time range using the (optionally) rolling-hour transformed counts; the top‑50 TVs are returned.
- Rolling-hour uses a forward-looking window of size `ceil(60 / time_bin_minutes)` bins (no wrap at day boundary).
- Category counts sum over the selected flight rows of the sparse occupancy matrix and are returned for the same top‑50 TVs.
- Time labels are generated purely from `time_bin_minutes` and bin indices.

---

## Autorate Occupancy Aggregation

Aggregate pre/post occupancy counts per traffic volume across flows, using only the data already present in a prior `/automatic_rate_adjustment` response. No optimization is run here.

### Endpoint

- Method: POST
- Path: `/autorate_occupancy`
- Content-Type: `application/json`
- Auth: none

### Request Body

- `autorate_result` (object, required): The exact JSON object returned by `/automatic_rate_adjustment`.
- `include_capacity` (boolean, optional, default `true`): Whether to include per-bin capacity arrays.
- `rolling_hour` (boolean, optional, default `true`): When true, `pre_counts` and `post_counts` are transformed into rolling-hour sums (forward-looking window of 60 minutes), consistent with `/original_counts`.

Notes
- TV set adheres strictly to the prior result:
  - Targets: `autorate_result.tvs` (order preserved)
  - Ripples: unique TV order from `autorate_result.ripple_cells` (append after targets; dedup)
- Binning is taken from the autorate result (`num_time_bins`) and the app indexer (`time_bin_minutes`).
- Capacity per bin uses the server’s preloaded matrix; unknown TVs or missing capacity return `-1` per bin.

### Response Body

- `time_bin_minutes` (int): Bin size in minutes.
- `num_bins` (int): Number of bins per TV.
- `tv_ids_order` (list[string]): Targets first in given order, then ripples (deduped).
- `timebins.labels` (list[string]): Labels `HH:MM-HH:MM` for all bins `[0..T-1]`.
- `pre_counts` (object): `{ tv_id: int[T] }` baseline occupancy across all flights for each TV. When `rolling_hour` is true, counts are rolling-hour aggregates.
- `post_counts` (object): `{ tv_id: int[T] }` baseline adjusted by the optimized delays (applied only to affected flights). When `rolling_hour` is true, counts are rolling-hour aggregates.
- `capacity` (object, optional): `{ tv_id: float[T] }` capacity per bin (hourly value repeated; `-1` if unknown).

### Example

Request
```json
{
  "autorate_result": { /* the full object returned by /automatic_rate_adjustment */ },
  "include_capacity": true
}
```

Response (shape, truncated)
```json
{
  "time_bin_minutes": 15,
  "num_bins": 96,
  "tv_ids_order": ["TV_TARGET_A", "TV_TARGET_B", "TV_RIPPLE_X"],
  "timebins": { "labels": ["00:00-00:15", "00:15-00:30", "..."] },
  "pre_counts": {
    "TV_TARGET_A": [3, 5, 4, "..."],
    "TV_TARGET_B": [1, 2, 3, "..."],
    "TV_RIPPLE_X": [0, 1, 1, "..."]
  },
  "post_counts": {
    "TV_TARGET_A": [2, 4, 3, "..."],
    "TV_TARGET_B": [1, 1, 2, "..."],
    "TV_RIPPLE_X": [0, 1, 1, "..."]
  },
  "capacity": {
    "TV_TARGET_A": [20, 20, 20, "..."],
    "TV_TARGET_B": [18, 18, 18, "..."],
    "TV_RIPPLE_X": [15, 15, 15, "..."]
  }
}
```
