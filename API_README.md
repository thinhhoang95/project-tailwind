# Airspace Traffic Analysis API

This FastAPI server provides endpoints for analyzing traffic volume occupancy data in airspace management systems.

## Authentication Quickstart

All endpoints (except `POST /token`) require a JWT bearer token.

- Get a token:
  - `POST /token` with form fields `username` and `password` (`application/x-www-form-urlencoded`).
  - Demo users (for local dev):
    - `nm@intuelle.com` / `nm123`
    - `thinh.hoangdinh@enac.fr` / `Vy011195`
- Use the token: add header `Authorization: Bearer <token>` to every request.

Examples:
```bash
# Obtain a token
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=nm@intuelle.com&password=nm123' | \
  python -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')

# Call a protected endpoint
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/traffic_volumes"
```

## Features

- **`/tv_count`** - Get occupancy counts for all time windows of a specific traffic volume
- **`/tv_flights`** - Get flight identifiers grouped by time window for a specific traffic volume
- **`/tv_flights_ordered`** - Get all flights for a traffic volume ordered by proximity to a reference time
- **`/regulation_ranking_tv_flights_ordered`** - Ordered TV flights near ref time (no scores/components)
- **`/flow_extraction`** - Compute community assignments for flights near a reference time using Jaccard similarity and Leiden clustering
- **`/traffic_volumes`** - List all available traffic volume IDs
- **`/tv_count_with_capacity`** - Get occupancy counts along with hourly capacity for a traffic volume
- **`/hotspots`** - Get list of hotspots detected via sliding rolling-hour counts (contiguous overloaded segments per TV) with detailed statistics
- **`/slack_distribution`** - For a source TV and reference time, returns per-TV slack at the query bin shifted by nominal travel time (475 kts), with an optional additional shift `delta_min` (minutes)
- **`/regulation_plan_simulation`** - Simulate a regulation plan to get per-flight delays, objective metrics, and rolling-hour occupancy for all TVs that changed (pre/post); no server-side ranking
- **`/common_traffic_volumes`** - Given a list of flight identifiers, returns the list of unique traffic volumes that any of these flights pass through (union)
- **`/flight_query_ast`** - Evaluate composable JSON AST queries over flights (crossings, sequences, time windows, capacity checks)
- **`/flight_query_nlp`** - Convert natural language prompts into flight query ASTs, then evaluate them using the same engine as `/flight_query_ast`
- **Authentication** - OAuth2 password flow with JWT access tokens
- **`/token`** - Issue access token and user info (`display_name`, `organization`) (demo users: `nm@intuelle.com` / `nm123`, `thinh.hoangdinh@enac.fr` / `Vy011195`)
- **`/protected`** - Example protected endpoint requiring `Authorization: Bearer <token>`
- **Data Science Integration** - Uses `NetworkEvaluator` for computational analysis
- **Network Abstraction** - `AirspaceAPIWrapper` handles network layer and JSON serialization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-api.txt
```

### 2. Start the Server

```bash
python run_server.py
```

The server will start on `http://localhost:8000`

### 3. Access Interactive Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation

## API Endpoints

### Authentication

JWT bearer authentication is available using the OAuth2 password flow. In development, a demo user is provided.

Environment variables (optional):
- `TAILWIND_SECRET_KEY`: Secret key used to sign JWTs (default: `change-me`).
- `TAILWIND_ACCESS_TOKEN_MINUTES`: Token expiry in minutes (default: `30`).

Demo credentials:
- `username`: `nm@intuelle.com`, `password`: `nm123`
- `username`: `thinh.hoangdinh@enac.fr`, `password`: `Vy011195`

#### POST `/token`

Obtain an access token using form data.

Content type: `application/x-www-form-urlencoded`

Form fields:
- `username`: user name
- `password`: user password

Response:
```json
{
  "access_token": "<jwt>",
  "token_type": "bearer",
  "display_name": "<display name>",
  "organization": "<organization>"
}
```

Example:
```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=nm@intuelle.com&password=nm123"
```

#### GET `/protected`

Example protected route. Requires header `Authorization: Bearer <token>`.

Example:
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=nm@intuelle.com&password=nm123' | python -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/protected
```

### GET `/tv_count?traffic_volume_id={id}`

Returns occupancy counts for all time windows of a specific traffic volume.

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "occupancy_counts": {
    "06:00-06:15": 42,
    "06:15-06:30": 35,
    "06:30-06:45": 28,
    ...
  },
  "metadata": {
    "time_bin_minutes": 15,
    "total_time_windows": 96,
    "total_flights_in_tv": 1234
  }
}
```

### GET `/tv_count_with_capacity?traffic_volume_id={id}`

Returns occupancy counts for all time windows of a specific traffic volume, plus hourly capacity sourced from the traffic volumes GeoJSON.

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "occupancy_counts": {
    "06:00-06:15": 42,
    "06:15-06:30": 35
  },
  "hourly_capacity": {
    "06:00-07:00": 23,
    "07:00-08:00": 25
  },
  "metadata": {
    "time_bin_minutes": 15,
    "total_time_windows": 96,
    "total_flights_in_tv": 1234
  }
}
```

### GET `/tv_flights?traffic_volume_id={id}`

Returns flight identifiers for each time window of a specific traffic volume.

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze

**Response:**
```json
{
  "06:00-06:15": ["0200AFRAM650E", "3944E1AFR96RF"],
  "06:15-06:30": ["1234XYZZY", "5678ABCDE"],
  "06:30-06:45": [],
  ...
}
```

### GET `/tv_flights_ordered?traffic_volume_id={id}&ref_time_str={HHMMSS}`

Returns all flights that pass through the specified traffic volume, ordered by how close their arrival time at that traffic volume is to the reference time.

Use `ref_time_str` in numeric `HHMMSS` format (e.g., `084510` for 08:45:10). `HHMM` is also accepted (seconds assumed 00).

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze
- `ref_time_str` (string): Reference time for ordering flights by proximity

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "ref_time_str": "080010",
  "ordered_flights": [
    "0200AFRAM650E",
    "3944E1AFR96RF",
    "1234XYZZY"
  ],
  "details": [
    {
      "flight_id": "0200AFRAM650E",
      "arrival_time": "07:59:58",
      "arrival_seconds": 28798,
      "delta_seconds": 12,
      "time_window": "07:45-08:00"
    },
    {
      "flight_id": "3944E1AFR96RF",
      "arrival_time": "08:00:12",
      "arrival_seconds": 28812,
      "delta_seconds": 2,
      "time_window": "08:00-08:15"
    }
  ]
}
```

### GET `/regulation_ranking_tv_flights_ordered?traffic_volume_id={id}&ref_time_str={HHMMSS}&seed_flight_ids={csv}&duration_min={m}&top_k={n}`

Returns flights that pass through the specified traffic volume near a reference time, ordered by proximity to the reference time. No ranking scores or component breakdown are computed (heavy `FlightFeatures` computations are skipped). If `duration_min` is provided, results are filtered to include only flights whose entry time into the TV lies between the reference time and `ref_time_str + duration_min` minutes. Results retain the proximity ordering.

Use `ref_time_str` in numeric `HHMMSS` format (e.g., `084510` for 08:45:10). `HHMM` is also accepted (seconds assumed 00). `seed_flight_ids` is accepted (comma-separated) but not used.

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze
- `ref_time_str` (string): Reference time for ordering flights by proximity (numeric `HHMMSS` or `HHMM`)
- `seed_flight_ids` (string): Comma-separated seed flight IDs used to build the footprint
- `duration_min` (integer, optional): Positive minutes window; after ranking, keep only flights whose `arrival_time` into the TV is within `[ref_time_str, ref_time_str + duration_min]`
- `top_k` (integer, optional): Limit number of ranked flights returned

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "ref_time_str": "080010",
  "seed_flight_ids": ["0200AFRAM650E", "3944E1AFR96RF"],
  "ranked_flights": [
    {
      "flight_id": "0200AFRAM650E",
      "arrival_time": "08:00:12",
      "time_window": "08:00-08:15",
      "delta_seconds": 2
    },
    {
      "flight_id": "1234XYZZY",
      "arrival_time": "08:02:30",
      "time_window": "08:00-08:15",
      "delta_seconds": 140
    }
  ],
  "metadata": {
    "num_candidates": 120,
    "num_ranked": 20,
    "time_bin_minutes": 15,
    "duration_min": 20
  }
}
```

### GET `/flow_extraction?traffic_volume_id={id}&ref_time_str={time}&threshold={t}&resolution={r}&seed={n}&limit={k}`

Computes community assignments for flights that pass through the specified traffic volume near a reference time. It reuses the ordered flights list and then applies the same Jaccard + Leiden pipeline used by `/flows` via `parrhesia.flows.flow_pipeline.build_global_flows`. Internally, the pipeline trims footprints up to the earliest crossing of the requested TV and clusters using the provided `threshold` and `resolution`. Direction-aware reweighting uses TV centroids when available.

Accepts flexible time formats for `ref_time_str`: `HHMMSS`, `HHMM`, `HH:MM`, `HH:MM:SS`.

**Parameters:**
- `traffic_volume_id` (string): Source traffic volume ID
- `ref_time_str` (string): Reference time string
- `threshold` (float, optional): Similarity threshold for edges (default: `0.8`)
- `resolution` (float, optional): Leiden resolution parameter (default: `1.0`)
- `seed` (integer, optional): Random seed for Leiden
- `limit` (integer, optional): Limit number of closest flights to include

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "ref_time_str": "080010",
  "flight_ids": ["0200AFRAM650E", "3944E1AFR96RF", "1234XYZZY"],
  "communities": {"0200AFRAM650E": 0, "3944E1AFR96RF": 0, "1234XYZZY": 1},
  "groups": {"0": ["0200AFRAM650E", "3944E1AFR96RF"], "1": ["1234XYZZY"]},
  "metadata": {
    "num_flights": 3,
    "time_bin_minutes": 15,
    "threshold": 0.8,
    "resolution": 1.0
  }
}
```

**Notes:**
- The underlying flow extraction trims each candidate flight's footprint up to its first occurrence of the specified traffic volume and computes Jaccard similarity across these footprints, unified with `/flows`.
- There is no bin/hour "mode" parameter. Grouping is determined solely by the provided `traffic_volume_id` and candidate `flight_ids`.
- If the Leiden libraries are unavailable, a graph connected-components fallback is used.

### GET `/flow_extraction_legacy`

Provides the legacy flow extraction behavior using the previous subflows implementation (`project_tailwind.subflows.flow_extractor.assign_communities_for_hotspot`).

Query parameters and response format are identical to `/flow_extraction`; only the underlying clustering code path differs. This endpoint is retained for regression comparison and backward compatibility.

### GET `/traffic_volumes`

Returns list of available traffic volume IDs.

**Response:**
```json
{
  "available_traffic_volumes": ["TV001", "TV002", "MASB5KL", ...],
  "count": 150,
  "metadata": {
    "time_bin_minutes": 15,
    "total_tvtws": 14400,
    "total_flights": 50000
  }
}
```

### GET `/hotspots?threshold={value}`

Returns hotspots detected using a sliding rolling-hour window at each time bin (stride = `time_bin_minutes`).
- A bin is overloaded when `rolling_count(bin) − capacity_per_bin(bin) > threshold` and capacity is defined.
- Consecutive overloaded bins for the same TV are merged into one contiguous segment.

**Parameters:**
- `threshold` (float, optional): Minimum excess traffic to consider as overloaded (default: 0.0)

**Response:**
```json
{
  "hotspots": [
    {
      "traffic_volume_id": "MASB5KL",
      "time_bin": "08:15-08:30",
      "z_max": 12.5,
      "z_sum": 22.0,
      "hourly_occupancy": 67.0,
      "hourly_capacity": 55.0,
      "is_overloaded": true
    },
    {
      "traffic_volume_id": "TV001",
      "time_bin": "09:00-09:45",
      "z_max": 8.3,
      "z_sum": 18.1,
      "hourly_occupancy": 43.0,
      "hourly_capacity": 35.0,
      "is_overloaded": true
    }
  ],
  "count": 2,
  "metadata": {
    "threshold": 0.0,
    "time_bin_minutes": 15,
    "analysis_type": "rolling_hour_sliding"
  }
}
```

### GET `/slack_distribution?traffic_volume_id={id}&ref_time_str={time}&sign={plus|minus}&delta_min={minutes}`

Returns a slack distribution across all traffic volumes at the “query bin” computed by shifting the source reference bin by the nominal travel time (distance at 475 kts). You can optionally apply an additional shift of `delta_min` minutes (positive or negative) after the travel-time shift. Useful for finding TVs with spare capacity to absorb demand.

Accepts flexible time formats for `ref_time_str`: `HHMMSS`, `HHMM`, `HH:MM`, `HH:MM:SS`.

**Parameters:**
- `traffic_volume_id` (string): Source traffic volume ID
- `ref_time_str` (string): Reference time string
- `sign` (string): Either `plus` or `minus` (shift direction)
- `delta_min` (float, optional): Extra shift in minutes applied after travel-time shift; can be negative; default `0.0`

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "ref_time_str": "08:30",
  "sign": "plus",
  "delta_min": 0.0,
  "time_bin_minutes": 15,
  "nominal_speed_kts": 475.0,
  "count": 3,
  "results": [
    {
      "traffic_volume_id": "TV001",
      "time_window": "08:45-09:00",
      "slack": 12.0,
      "occupancy": 3.0,
      "capacity_per_bin": 15.0,
      "distance_nm": 120.5,
      "travel_minutes": 15.2,
      "bin_offset": 1,
      "clamped": false
    },
    {
      "traffic_volume_id": "TV002",
      "time_window": "08:30-08:45",
      "slack": 8.0,
      "occupancy": 5.0,
      "capacity_per_bin": 13.0,
      "distance_nm": 90.0,
      "travel_minutes": 11.4,
      "bin_offset": 1,
      "clamped": false
    }
  ]
}
```

**Result Fields:**
- `traffic_volume_id`: Destination TV ID
- `time_window`: Query time window label `HH:MM-HH:MM`
- `slack`: `capacity_per_bin - occupancy`
- `occupancy`: Occupancy at the query bin
- `capacity_per_bin`: Hourly capacity distributed evenly across bins in the hour
- `distance_nm`: Great-circle distance between TV centroids
- `travel_minutes`: Nominal travel time at 475 kts
- `bin_offset`: Signed bin shift applied from the reference bin
- `clamped`: Whether the query bin was clamped to the day edges
- `delta_min` (top-level field): The additional shift, in minutes, that was applied

**Response Fields:**
- `traffic_volume_id`: String identifier for the traffic volume
- `time_bin`: Contiguous overloaded period labeled by bin starts, formatted `HH:MM-HH:MM` (e.g., `08:15-08:30`). This represents the union of overloaded bins detected by sliding a 60‑minute window across `time_bin_minutes` steps.
- `z_max`: Maximum excess `(rolling_count − capacity_per_bin)` within the segment
- `z_sum`: Sum of excess over all bins within the segment
- `hourly_occupancy`: Peak rolling-hour occupancy within the segment (compatibility name)
- `hourly_capacity`: Capacity baseline across the segment (minimum hourly capacity encountered within the segment)
- `is_overloaded`: Always true for returned entries

Notes
- Capacity alignment is consistent with `/original_counts`: hourly capacity values are repeated across all bins within that hour, and bins without capacity use `-1` (not considered in overload detection and break segments).

### POST `/regulation_plan_simulation`

Simulates a regulation plan and returns per-flight delays, evaluation metrics, and rolling-hour occupancy for the top-K busiest TVs across all traffic volumes. TVs are ranked by max(pre_rolling_count − hourly_capacity) computed over the union of all active time windows provided in the plan. Also returns `pre_flight_context` with baseline takeoff and TV-arrival times for flights present in `delays_by_flight`.

You can provide regulations as raw strings in the `Regulation` DSL or as structured objects.

**Request (JSON):**
```json
{
  "regulations": [
    "TV_TVA IC__ 3 32",
    {
      "location": "TVA",
      "rate": 1,
      "time_windows": [32],
      "filter_type": "IC",
      "filter_value": "__",
      "target_flight_ids": ["F1", "F2", "F3"]
    }
  ],
  "weights": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
  
  "include_excess_vector": false
}
```

**Notes:**
- `regulations`: List of either raw strings like `TV_<LOC> <FILTER> <RATE> <TW>` or objects with `location`, `rate`, `time_windows`, and optional `filter_type`, `filter_value`, `target_flight_ids`.
- `weights`: Optional objective weights for combining overload and delay components.
- Deprecated: `top_k` is accepted but ignored for one release (for backward compatibility). The response includes all changed TVs in stable row order (no ranking). If `top_k` is present, a deprecation note is added under `metadata.deprecated`.
- `include_excess_vector`: If true, returns the full post-regulation excess vector; otherwise returns compact stats.
- Hours/bins with no capacity are skipped when computing busy-ness.
- Ranking metric: `max(pre_rolling_count - hourly_capacity)` over the union mask of active regulation windows.

**Response:**
```json
{
  "delays_by_flight": {"F1": 5, "F2": 0},
  "pre_flight_context": {
    "F1": {"takeoff_time": "07:12:05", "tv_arrival_time": "08:00:12"}
  },
  "delay_stats": {
    "total_delay_seconds": 300.0,
    "mean_delay_seconds": 150.0,
    "max_delay_seconds": 300.0,
    "min_delay_seconds": 0.0,
    "delayed_flights_count": 1,
    "num_flights": 2
  },
  "objective": 12.0,
  "objective_components": {
    "z_sum": 10.0,
    "z_max": 5.0,
    "delay_min": 5.0,
    "num_regs": 2,
    "alpha": 1.0,
    "beta": 2.0,
    "gamma": 0.1,
    "delta": 25.0
  },
  "rolling_changed_tvs": [
    {
      "traffic_volume_id": "TVA",
      "pre_rolling_counts": [3.0, 4.0, 5.0],
      "post_rolling_counts": [2.0, 3.0, 4.0],
      "capacity_per_bin": [1.0, 1.0, 1.0],
      "active_time_windows": [32]
    }
  ],
  "rolling_top_tvs": [ ... same as rolling_changed_tvs ... ],
  "excess_vector_stats": {"sum": 10.0, "max": 3.0, "mean": 0.1, "count": 9600},
  "metadata": {
    "time_bin_minutes": 15,
    "bins_per_tv": 384,
    "bins_per_hour": 4,
    "num_traffic_volumes": 1,
    "num_changed_tvs": 1,
    "deprecated": {
      "top_k": "accepted but ignored; will be removed in next release",
      "rolling_top_tvs": "alias of rolling_changed_tvs for one release"
    }
  }
}
```

Fields:
- `pre_flight_context`: Map of flight ID → `{takeoff_time, tv_arrival_time}` strings (HH:MM:SS). `tv_arrival_time` can be `null` if the flight does not enter any regulated traffic volume.

Fields:
- `pre_flight_context`: Map of flight ID → `{takeoff_time, tv_arrival_time}` strings (HH:MM:SS). `tv_arrival_time` can be `null` if the flight does not enter any regulated traffic volume.
- `rolling_changed_tvs`: All TVs where any raw occupancy bin changed due to the plan; arrays are full-length per TV. Results are returned in stable TV row order (no ranking or top-k selection).
- `rolling_top_tvs`: Deprecated alias, equal to `rolling_changed_tvs` for one release.
- `metadata.num_changed_tvs`: Count of changed TVs.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/regulation_plan_simulation" \
  -H "Content-Type: application/json" \
  -d '{
    "regulations": ["TV_TVA IC__ 3 32"],
    "weights": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
    "include_excess_vector": false
  }'
```

```bash
curl "http://localhost:8000/flow_extraction?traffic_volume_id=MASB5KL&ref_time_str=080010&threshold=0.8&resolution=1.0&limit=100"
```

## Architecture

### Components

1. **`main.py`** - FastAPI application with endpoint definitions
2. **`airspace_api_wrapper.py`** - Network layer handling HTTP requests and JSON serialization
3. **`network_evaluator_for_api.py`** - Data science logic for traffic analysis

### Data Flow

```
HTTP Request → FastAPI → AirspaceAPIWrapper → NetworkEvaluator → Data Processing → JSON Response
```

## Testing

### Direct API Testing
```bash
python test_api.py
```

### HTTP Client Testing
With server running (and a valid token in `$TOKEN`):
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=nm@intuelle.com&password=nm123' | python -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/tv_count?traffic_volume_id=MASB5KL"
```

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/tv_flights?traffic_volume_id=MASB5KL"
```

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/tv_flights_ordered?traffic_volume_id=MASB5KL&ref_time_str=080010"
```

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/regulation_ranking_tv_flights_ordered?traffic_volume_id=MASB5KL&ref_time_str=080010&seed_flight_ids=0200AFRAM650E,3944E1AFR96RF&top_k=20"
```

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/hotspots?threshold=0.0"
```

### POST `/flight_query_ast`

Evaluate composable JSON AST queries against the flight occupancy dataset. The evaluator supports logical composition (`and`, `or`, `not`) and atomic predicates such as `cross`, `sequence`, `origin`, `destination`, `arrival_window`, `takeoff_window`, `capacity_state`, `duration_between`, `count_crossings`, and `geo_region_cross`.

**Request (JSON):**
```json
{
  "query": { "type": "cross", "tv": "TVA", "time": { "clock": { "from": "11:15", "to": "11:45" } } },
  "options": {
    "select": "flight_ids",
    "order_by": "takeoff_time",
    "limit": 200,
    "deduplicate": true,
    "debug": false
  }
}
```

Fields:
- `query`: required root node of the AST (see plan for node types). Logical nodes combine children; atomic nodes filter flights via TV/time/airport/capacity predicates.
- `options` (optional): overrides for result shape.
  - `select`: `"flight_ids"` (default), `"count"`, or `"ids_and_times"`.
  - `order_by`: `"first_crossing_time"`, `"last_crossing_time"`, `"takeoff_time"`, `"dest"`, or omitted for natural order.
  - `limit`: cap number of returned flights (defaults to 50k hard limit when omitted).
  - `flight_ids`: restrict evaluation to a provided list of flight identifiers; unknown IDs trigger a 400.
  - `deduplicate`: keep/skip deduplication (default `true`).
  - `debug`: include cache diagnostics in `metadata.explain`.

**Response:**
- When `select = "flight_ids"` (default):
  ```json
  {
    "flight_ids": ["F1", "F2"],
    "metadata": {
      "time_bin_minutes": 15,
      "bins_per_tv": 96,
      "evaluation_ms": 12.4,
      "node_cache_hits": 3,
      "total_matches": 128,
      "result_size": 50,
      "truncated": true
    }
  }
  ```
- `select = "count"`: returns `{ "count": <int>, "metadata": {...} }`.
- `select = "ids_and_times"`: returns `{ "ids_and_times": [{"flight_id": ..., "first_crossing_time": "HH:MM:SS", "last_crossing_time": "HH:MM:SS"}, ...], "metadata": {...} }`.

**Example Queries:**
1. Flights crossing TVA between 11:15–11:45 (default response):
   ```json
   {
     "query": {
       "type": "cross",
       "tv": "TVA",
       "time": { "clock": { "from": "11:15", "to": "11:45" } }
     }
   }
   ```
2. Flights that cross TVA then TVB within 45 minutes and arrive at LFPO between 11:15–12:15:
   ```json
   {
     "query": {
       "type": "and",
       "children": [
         {
           "type": "sequence",
           "steps": [
             { "type": "cross", "tv": "TVA" },
             { "type": "cross", "tv": "TVB" }
           ],
           "within": { "minutes": 45 }
         },
         { "type": "destination", "airport": "LFPO" },
         {
           "type": "arrival_window",
           "clock": { "from": "11:15", "to": "12:15" },
           "method": "last_crossing"
         }
       ]
     }
   }
   ```
3. Count flights crossing any of `["TVX", "TVY", "TVZ"]` while overloaded:
   ```json
   {
     "query": {
       "type": "capacity_state",
       "tv": { "anyOf": ["TVX", "TVY", "TVZ"] },
       "time": { "bins": { "from": 32, "to": 44 } },
       "condition": "overloaded"
     },
     "options": { "select": "count" }
   }
   ```
4. Return ordered IDs with first/last crossing time summaries:
   ```json
   {
     "query": {
       "type": "cross",
       "tv": "TVHOT",
       "select": "ids_and_times"
     },
     "options": { "order_by": "first_crossing_time", "limit": 25 }
   }
   ```
5. Restrict evaluation to a provided flight list:
   ```json
   {
     "query": { "type": "cross", "tv": "TVA" },
     "options": { "flight_ids": ["0200AFRAM650E", "3944E1AFR96RF"] }
   }
   ```

Errors:
- 400 for malformed ASTs or invalid time specifications.
- 404 when referencing unknown traffic volume IDs.

### POST `/flight_query_nlp`

Natural language interface for `/flight_query_ast`. The server calls OpenAI with a deterministic system prompt that produces a strict JSON AST, merges any allowed `options`, and evaluates the AST via `QueryAPIWrapper`.

**Request (JSON):**
```json
{
  "prompt": "Flights that cross TVA between 11:15 and 11:45 ordered by takeoff time, limit 50",
  "options": {
    "order_by": "takeoff_time",
    "limit": 50,
    "select": "flight_ids",
    "debug": false
  },
  "model": "gpt-4o-mini"
}
```

Fields:
- `prompt` (**required**): natural language request. Must be non-empty.
- `options` (optional): same whitelist as `/flight_query_ast` (`select`, `order_by`, `limit`, `deduplicate`, `flight_ids`, `debug`). Unknown keys are rejected.
- `model` (optional): override the default model configured via `FLIGHT_QUERY_NLP_MODEL` (fallback `gpt-4o-mini`).

Configuration:
- `OPENAI_API_KEY` (**required**): used by the server to call OpenAI's Chat Completions API.
- `FLIGHT_QUERY_NLP_MODEL` (optional): default model name (default `gpt-4o-mini`).
- `FLIGHT_QUERY_NLP_TIMEOUT_S` (optional): request timeout in seconds (default `20`).

**Response:** Matches `/flight_query_ast` for the chosen `select` mode. When `options.debug = true`, the response also includes:
- top-level `ast`: raw JSON returned by the LLM (`{"query": {...}}`).
- `metadata.llm`: `{ "model": "…", "parse_ms": <number>, "prompt_tokens"?, "completion_tokens"? }`.

**Example (`select = "flight_ids"` with debug):**
```json
{
  "flight_ids": ["F123", "F456"],
  "metadata": {
    "time_bin_minutes": 15,
    "bins_per_tv": 96,
    "evaluation_ms": 10.8,
    "node_cache_hits": 2,
    "total_matches": 24,
    "result_size": 24,
    "llm": {
      "model": "gpt-4o-mini",
      "parse_ms": 812.3,
      "prompt_tokens": 642,
      "completion_tokens": 138
    }
  },
  "ast": {
    "query": {
      "type": "cross",
      "tv": "TVA",
      "time": { "clock": { "from": "11:15", "to": "11:45" } }
    }
  }
}
```

Errors:
- 400: missing/blank `prompt`, invalid option values, malformed LLM JSON, or AST validation failures.
- 404: surfaced when the evaluated AST references unknown traffic volume IDs.
- 502: upstream LLM timeouts or API errors.
- 500: unexpected server issues (including missing OpenAI credentials).
- 500 for unexpected server errors during evaluation.

## Configuration

The API expects the following data files:
- `output/so6_occupancy_matrix_with_times.json` - Flight occupancy data
- `output/tvtw_indexer.json` - Time window indexer
- Traffic volumes GeoJSON file (path configured in `airspace_api_wrapper.py`)

## Error Handling

- **404** - Traffic volume ID not found
- **500** - Internal server error (data loading issues, computation errors)

## Development

The server runs with auto-reload enabled for development. Modify the code and the server will automatically restart.

For production deployment, consider:
- Setting `reload=False`
- Using a production WSGI server like Gunicorn
- Adding authentication and rate limiting
- Implementing proper logging

### POST `/common_traffic_volumes`

Given a list of flight identifiers, returns the unique traffic volumes that any of the provided flights pass through (union across flights). The result is sorted by the stable TV row order used internally.

**Request (JSON):**
```json
{
  "flight_ids": ["0200AFRAM650E", "3944E1AFR96RF", "1234XYZZY"]
}
```

**Response:**
```json
{
  "flight_ids": ["0200AFRAM650E", "3944E1AFR96RF", "1234XYZZY"],
  "traffic_volumes": ["MASB5KL", "TV001"],
  "count": 2,
  "metadata": {
    "time_bin_minutes": 15,
    "num_input_flights": 3
  }
}
```

Errors:
- 400 if `flight_ids` is missing/invalid or contains unknown flight IDs# Airspace Traffic Analysis API

This FastAPI server provides endpoints for analyzing traffic volume occupancy data in airspace management systems.

## Authentication Quickstart

All endpoints (except `POST /token`) require a JWT bearer token.

- Get a token:
  - `POST /token` with form fields `username` and `password` (`application/x-www-form-urlencoded`).
  - Demo users (for local dev):
    - `nm@intuelle.com` / `nm123`
    - `thinh.hoangdinh@enac.fr` / `Vy011195`
- Use the token: add header `Authorization: Bearer <token>` to every request.

Examples:
```bash
# Obtain a token
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=nm@intuelle.com&password=nm123' | \
  python -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')

# Call a protected endpoint
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/traffic_volumes"
```

## Features

- **`/tv_count`** - Get occupancy counts for all time windows of a specific traffic volume
- **`/tv_flights`** - Get flight identifiers grouped by time window for a specific traffic volume
- **`/tv_flights_ordered`** - Get all flights for a traffic volume ordered by proximity to a reference time
- **`/regulation_ranking_tv_flights_ordered`** - Ordered TV flights near ref time (no scores/components)
- **`/flow_extraction`** - Compute community assignments for flights near a reference time using Jaccard similarity and Leiden clustering
- **`/traffic_volumes`** - List all available traffic volume IDs
- **`/tv_count_with_capacity`** - Get occupancy counts along with hourly capacity for a traffic volume
- **`/hotspots`** - Get list of hotspots detected via sliding rolling-hour counts (contiguous overloaded segments per TV) with detailed statistics
- **`/slack_distribution`** - For a source TV and reference time, returns per-TV slack at the query bin shifted by nominal travel time (475 kts), with an optional additional shift `delta_min` (minutes)
- **`/regulation_plan_simulation`** - Simulate a regulation plan to get per-flight delays, objective metrics, and rolling-hour occupancy for all TVs that changed (pre/post); no server-side ranking
- **Authentication** - OAuth2 password flow with JWT access tokens
- **`/token`** - Issue access token and user info (`display_name`, `organization`) (demo users: `nm@intuelle.com` / `nm123`, `thinh.hoangdinh@enac.fr` / `Vy011195`)
- **`/protected`** - Example protected endpoint requiring `Authorization: Bearer <token>`
- **Data Science Integration** - Uses `NetworkEvaluator` for computational analysis
- **Network Abstraction** - `AirspaceAPIWrapper` handles network layer and JSON serialization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-api.txt
```

### 2. Start the Server

```bash
python run_server.py
```

The server will start on `http://localhost:8000`

### 3. Access Interactive Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation

## API Endpoints

### Authentication

JWT bearer authentication is available using the OAuth2 password flow. In development, a demo user is provided.

Environment variables (optional):
- `TAILWIND_SECRET_KEY`: Secret key used to sign JWTs (default: `change-me`).
- `TAILWIND_ACCESS_TOKEN_MINUTES`: Token expiry in minutes (default: `30`).

Demo credentials:
- `username`: `nm@intuelle.com`, `password`: `nm123`
- `username`: `thinh.hoangdinh@enac.fr`, `password`: `Vy011195`

#### POST `/token`

Obtain an access token using form data.

Content type: `application/x-www-form-urlencoded`

Form fields:
- `username`: user name
- `password`: user password

Response:
```json
{
  "access_token": "<jwt>",
  "token_type": "bearer",
  "display_name": "<display name>",
  "organization": "<organization>"
}
```

Example:
```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=nm@intuelle.com&password=nm123"
```

#### GET `/protected`

Example protected route. Requires header `Authorization: Bearer <token>`.

Example:
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=nm@intuelle.com&password=nm123' | python -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/protected
```

### GET `/tv_count?traffic_volume_id={id}`

Returns occupancy counts for all time windows of a specific traffic volume.

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "occupancy_counts": {
    "06:00-06:15": 42,
    "06:15-06:30": 35,
    "06:30-06:45": 28,
    ...
  },
  "metadata": {
    "time_bin_minutes": 15,
    "total_time_windows": 96,
    "total_flights_in_tv": 1234
  }
}
```

### GET `/tv_count_with_capacity?traffic_volume_id={id}`

Returns occupancy counts for all time windows of a specific traffic volume, plus hourly capacity sourced from the traffic volumes GeoJSON.

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "occupancy_counts": {
    "06:00-06:15": 42,
    "06:15-06:30": 35
  },
  "hourly_capacity": {
    "06:00-07:00": 23,
    "07:00-08:00": 25
  },
  "metadata": {
    "time_bin_minutes": 15,
    "total_time_windows": 96,
    "total_flights_in_tv": 1234
  }
}
```

### GET `/tv_flights?traffic_volume_id={id}`

Returns flight identifiers for each time window of a specific traffic volume.

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze

**Response:**
```json
{
  "06:00-06:15": ["0200AFRAM650E", "3944E1AFR96RF"],
  "06:15-06:30": ["1234XYZZY", "5678ABCDE"],
  "06:30-06:45": [],
  ...
}
```

### GET `/tv_flights_ordered?traffic_volume_id={id}&ref_time_str={HHMMSS}`

Returns all flights that pass through the specified traffic volume, ordered by how close their arrival time at that traffic volume is to the reference time.

Use `ref_time_str` in numeric `HHMMSS` format (e.g., `084510` for 08:45:10). `HHMM` is also accepted (seconds assumed 00).

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze
- `ref_time_str` (string): Reference time for ordering flights by proximity

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "ref_time_str": "080010",
  "ordered_flights": [
    "0200AFRAM650E",
    "3944E1AFR96RF",
    "1234XYZZY"
  ],
  "details": [
    {
      "flight_id": "0200AFRAM650E",
      "arrival_time": "07:59:58",
      "arrival_seconds": 28798,
      "delta_seconds": 12,
      "time_window": "07:45-08:00"
    },
    {
      "flight_id": "3944E1AFR96RF",
      "arrival_time": "08:00:12",
      "arrival_seconds": 28812,
      "delta_seconds": 2,
      "time_window": "08:00-08:15"
    }
  ]
}
```

### GET `/regulation_ranking_tv_flights_ordered?traffic_volume_id={id}&ref_time_str={HHMMSS}&seed_flight_ids={csv}&duration_min={m}&top_k={n}`

Returns flights that pass through the specified traffic volume near a reference time, ordered by proximity to the reference time. No ranking scores or component breakdown are computed (heavy `FlightFeatures` computations are skipped). If `duration_min` is provided, results are filtered to include only flights whose entry time into the TV lies between the reference time and `ref_time_str + duration_min` minutes. Results retain the proximity ordering.

Use `ref_time_str` in numeric `HHMMSS` format (e.g., `084510` for 08:45:10). `HHMM` is also accepted (seconds assumed 00). `seed_flight_ids` is accepted (comma-separated) but not used.

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze
- `ref_time_str` (string): Reference time for ordering flights by proximity (numeric `HHMMSS` or `HHMM`)
- `seed_flight_ids` (string): Comma-separated seed flight IDs used to build the footprint
- `duration_min` (integer, optional): Positive minutes window; after ranking, keep only flights whose `arrival_time` into the TV is within `[ref_time_str, ref_time_str + duration_min]`
- `top_k` (integer, optional): Limit number of ranked flights returned

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "ref_time_str": "080010",
  "seed_flight_ids": ["0200AFRAM650E", "3944E1AFR96RF"],
  "ranked_flights": [
    {
      "flight_id": "0200AFRAM650E",
      "arrival_time": "08:00:12",
      "time_window": "08:00-08:15",
      "delta_seconds": 2
    },
    {
      "flight_id": "1234XYZZY",
      "arrival_time": "08:02:30",
      "time_window": "08:00-08:15",
      "delta_seconds": 140
    }
  ],
  "metadata": {
    "num_candidates": 120,
    "num_ranked": 20,
    "time_bin_minutes": 15,
    "duration_min": 20
  }
}
```

### GET `/flow_extraction?traffic_volume_id={id}&ref_time_str={time}&threshold={t}&resolution={r}&seed={n}&limit={k}`

Computes community assignments for flights that pass through the specified traffic volume near a reference time. It reuses the ordered flights list and then applies the same Jaccard + Leiden pipeline used by `/flows` via `parrhesia.flows.flow_pipeline.build_global_flows`. Internally, the pipeline trims footprints up to the earliest crossing of the requested TV and clusters using the provided `threshold` and `resolution`. Direction-aware reweighting uses TV centroids when available.

Accepts flexible time formats for `ref_time_str`: `HHMMSS`, `HHMM`, `HH:MM`, `HH:MM:SS`.

**Parameters:**
- `traffic_volume_id` (string): Source traffic volume ID
- `ref_time_str` (string): Reference time string
- `threshold` (float, optional): Similarity threshold for edges (default: `0.8`)
- `resolution` (float, optional): Leiden resolution parameter (default: `1.0`)
- `seed` (integer, optional): Random seed for Leiden
- `limit` (integer, optional): Limit number of closest flights to include

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "ref_time_str": "080010",
  "flight_ids": ["0200AFRAM650E", "3944E1AFR96RF", "1234XYZZY"],
  "communities": {"0200AFRAM650E": 0, "3944E1AFR96RF": 0, "1234XYZZY": 1},
  "groups": {"0": ["0200AFRAM650E", "3944E1AFR96RF"], "1": ["1234XYZZY"]},
  "metadata": {
    "num_flights": 3,
    "time_bin_minutes": 15,
    "threshold": 0.8,
    "resolution": 1.0
  }
}
```

**Notes:**
- The underlying flow extraction trims each candidate flight's footprint up to its first occurrence of the specified traffic volume and computes Jaccard similarity across these footprints, unified with `/flows`.
- There is no bin/hour "mode" parameter. Grouping is determined solely by the provided `traffic_volume_id` and candidate `flight_ids`.
- If the Leiden libraries are unavailable, a graph connected-components fallback is used.

### GET `/flow_extraction_legacy`

Provides the legacy flow extraction behavior using the previous subflows implementation (`project_tailwind.subflows.flow_extractor.assign_communities_for_hotspot`).

Query parameters and response format are identical to `/flow_extraction`; only the underlying clustering code path differs. This endpoint is retained for regression comparison and backward compatibility.

### GET `/traffic_volumes`

Returns list of available traffic volume IDs.

**Response:**
```json
{
  "available_traffic_volumes": ["TV001", "TV002", "MASB5KL", ...],
  "count": 150,
  "metadata": {
    "time_bin_minutes": 15,
    "total_tvtws": 14400,
    "total_flights": 50000
  }
}
```

### GET `/hotspots?threshold={value}`

Returns hotspots detected using a sliding rolling-hour window at each time bin (stride = `time_bin_minutes`).
- A bin is overloaded when `rolling_count(bin) − capacity_per_bin(bin) > threshold` and capacity is defined.
- Consecutive overloaded bins for the same TV are merged into one contiguous segment.

**Parameters:**
- `threshold` (float, optional): Minimum excess traffic to consider as overloaded (default: 0.0)

**Response:**
```json
{
  "hotspots": [
    {
      "traffic_volume_id": "MASB5KL",
      "time_bin": "08:15-08:30",
      "z_max": 12.5,
      "z_sum": 22.0,
      "hourly_occupancy": 67.0,
      "hourly_capacity": 55.0,
      "is_overloaded": true
    },
    {
      "traffic_volume_id": "TV001",
      "time_bin": "09:00-09:45",
      "z_max": 8.3,
      "z_sum": 18.1,
      "hourly_occupancy": 43.0,
      "hourly_capacity": 35.0,
      "is_overloaded": true
    }
  ],
  "count": 2,
  "metadata": {
    "threshold": 0.0,
    "time_bin_minutes": 15,
    "analysis_type": "rolling_hour_sliding"
  }
}
```

### GET `/slack_distribution?traffic_volume_id={id}&ref_time_str={time}&sign={plus|minus}&delta_min={minutes}`

Returns a slack distribution across all traffic volumes at the “query bin” computed by shifting the source reference bin by the nominal travel time (distance at 475 kts). You can optionally apply an additional shift of `delta_min` minutes (positive or negative) after the travel-time shift. Useful for finding TVs with spare capacity to absorb demand.

Accepts flexible time formats for `ref_time_str`: `HHMMSS`, `HHMM`, `HH:MM`, `HH:MM:SS`.

**Parameters:**
- `traffic_volume_id` (string): Source traffic volume ID
- `ref_time_str` (string): Reference time string
- `sign` (string): Either `plus` or `minus` (shift direction)
- `delta_min` (float, optional): Extra shift in minutes applied after travel-time shift; can be negative; default `0.0`

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "ref_time_str": "08:30",
  "sign": "plus",
  "delta_min": 0.0,
  "time_bin_minutes": 15,
  "nominal_speed_kts": 475.0,
  "count": 3,
  "results": [
    {
      "traffic_volume_id": "TV001",
      "time_window": "08:45-09:00",
      "slack": 12.0,
      "occupancy": 3.0,
      "capacity_per_bin": 15.0,
      "distance_nm": 120.5,
      "travel_minutes": 15.2,
      "bin_offset": 1,
      "clamped": false
    },
    {
      "traffic_volume_id": "TV002",
      "time_window": "08:30-08:45",
      "slack": 8.0,
      "occupancy": 5.0,
      "capacity_per_bin": 13.0,
      "distance_nm": 90.0,
      "travel_minutes": 11.4,
      "bin_offset": 1,
      "clamped": false
    }
  ]
}
```

**Result Fields:**
- `traffic_volume_id`: Destination TV ID
- `time_window`: Query time window label `HH:MM-HH:MM`
- `slack`: `capacity_per_bin - occupancy`
- `occupancy`: Occupancy at the query bin
- `capacity_per_bin`: Hourly capacity distributed evenly across bins in the hour
- `distance_nm`: Great-circle distance between TV centroids
- `travel_minutes`: Nominal travel time at 475 kts
- `bin_offset`: Signed bin shift applied from the reference bin
- `clamped`: Whether the query bin was clamped to the day edges
- `delta_min` (top-level field): The additional shift, in minutes, that was applied

**Response Fields:**
- `traffic_volume_id`: String identifier for the traffic volume
- `time_bin`: Contiguous overloaded period labeled by bin starts, formatted `HH:MM-HH:MM` (e.g., `08:15-08:30`). This represents the union of overloaded bins detected by sliding a 60‑minute window across `time_bin_minutes` steps.
- `z_max`: Maximum excess `(rolling_count − capacity_per_bin)` within the segment
- `z_sum`: Sum of excess over all bins within the segment
- `hourly_occupancy`: Peak rolling-hour occupancy within the segment (compatibility name)
- `hourly_capacity`: Capacity baseline across the segment (minimum hourly capacity encountered within the segment)
- `is_overloaded`: Always true for returned entries

Notes
- Capacity alignment is consistent with `/original_counts`: hourly capacity values are repeated across all bins within that hour, and bins without capacity use `-1` (not considered in overload detection and break segments).

### POST `/regulation_plan_simulation`

Simulates a regulation plan and returns per-flight delays, evaluation metrics, and rolling-hour occupancy for the top-K busiest TVs across all traffic volumes. TVs are ranked by max(pre_rolling_count − hourly_capacity) computed over the union of all active time windows provided in the plan. Also returns `pre_flight_context` with baseline takeoff and TV-arrival times for flights present in `delays_by_flight`.

You can provide regulations as raw strings in the `Regulation` DSL or as structured objects.

**Request (JSON):**
```json
{
  "regulations": [
    "TV_TVA IC__ 3 32",
    {
      "location": "TVA",
      "rate": 1,
      "time_windows": [32],
      "filter_type": "IC",
      "filter_value": "__",
      "target_flight_ids": ["F1", "F2", "F3"]
    }
  ],
  "weights": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
  
  "include_excess_vector": false
}
```

**Notes:**
- `regulations`: List of either raw strings like `TV_<LOC> <FILTER> <RATE> <TW>` or objects with `location`, `rate`, `time_windows`, and optional `filter_type`, `filter_value`, `target_flight_ids`.
- `weights`: Optional objective weights for combining overload and delay components.
- Deprecated: `top_k` is accepted but ignored for one release (for backward compatibility). The response includes all changed TVs in stable row order (no ranking). If `top_k` is present, a deprecation note is added under `metadata.deprecated`.
- `include_excess_vector`: If true, returns the full post-regulation excess vector; otherwise returns compact stats.
- Hours/bins with no capacity are skipped when computing busy-ness.
- Ranking metric: `max(pre_rolling_count - hourly_capacity)` over the union mask of active regulation windows.

**Response:**
```json
{
  "delays_by_flight": {"F1": 5, "F2": 0},
  "pre_flight_context": {
    "F1": {"takeoff_time": "07:12:05", "tv_arrival_time": "08:00:12"}
  },
  "delay_stats": {
    "total_delay_seconds": 300.0,
    "mean_delay_seconds": 150.0,
    "max_delay_seconds": 300.0,
    "min_delay_seconds": 0.0,
    "delayed_flights_count": 1,
    "num_flights": 2
  },
  "objective": 12.0,
  "objective_components": {
    "z_sum": 10.0,
    "z_max": 5.0,
    "delay_min": 5.0,
    "num_regs": 2,
    "alpha": 1.0,
    "beta": 2.0,
    "gamma": 0.1,
    "delta": 25.0
  },
  "rolling_changed_tvs": [
    {
      "traffic_volume_id": "TVA",
      "pre_rolling_counts": [3.0, 4.0, 5.0],
      "post_rolling_counts": [2.0, 3.0, 4.0],
      "capacity_per_bin": [1.0, 1.0, 1.0],
      "active_time_windows": [32]
    }
  ],
  "rolling_top_tvs": [ ... same as rolling_changed_tvs ... ],
  "excess_vector_stats": {"sum": 10.0, "max": 3.0, "mean": 0.1, "count": 9600},
  "metadata": {
    "time_bin_minutes": 15,
    "bins_per_tv": 384,
    "bins_per_hour": 4,
    "num_traffic_volumes": 1,
    "num_changed_tvs": 1,
    "deprecated": {
      "top_k": "accepted but ignored; will be removed in next release",
      "rolling_top_tvs": "alias of rolling_changed_tvs for one release"
    }
  }
}
```

Fields:
- `pre_flight_context`: Map of flight ID → `{takeoff_time, tv_arrival_time}` strings (HH:MM:SS). `tv_arrival_time` can be `null` if the flight does not enter any regulated traffic volume.

Fields:
- `pre_flight_context`: Map of flight ID → `{takeoff_time, tv_arrival_time}` strings (HH:MM:SS). `tv_arrival_time` can be `null` if the flight does not enter any regulated traffic volume.
- `rolling_changed_tvs`: All TVs where any raw occupancy bin changed due to the plan; arrays are full-length per TV. Results are returned in stable TV row order (no ranking or top-k selection).
- `rolling_top_tvs`: Deprecated alias, equal to `rolling_changed_tvs` for one release.
- `metadata.num_changed_tvs`: Count of changed TVs.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/regulation_plan_simulation" \
  -H "Content-Type: application/json" \
  -d '{
    "regulations": ["TV_TVA IC__ 3 32"],
    "weights": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
    "include_excess_vector": false
  }'
```

```bash
curl "http://localhost:8000/flow_extraction?traffic_volume_id=MASB5KL&ref_time_str=080010&threshold=0.8&resolution=1.0&limit=100"
```

## Architecture

### Components

1. **`main.py`** - FastAPI application with endpoint definitions
2. **`airspace_api_wrapper.py`** - Network layer handling HTTP requests and JSON serialization
3. **`network_evaluator_for_api.py`** - Data science logic for traffic analysis

### Data Flow

```
HTTP Request → FastAPI → AirspaceAPIWrapper → NetworkEvaluator → Data Processing → JSON Response
```

## Testing

### Direct API Testing
```bash
python test_api.py
```

### HTTP Client Testing
With server running (and a valid token in `$TOKEN`):
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=nm@intuelle.com&password=nm123' | python -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/tv_count?traffic_volume_id=MASB5KL"
```

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/tv_flights?traffic_volume_id=MASB5KL"
```

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/tv_flights_ordered?traffic_volume_id=MASB5KL&ref_time_str=080010"
```

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/regulation_ranking_tv_flights_ordered?traffic_volume_id=MASB5KL&ref_time_str=080010&seed_flight_ids=0200AFRAM650E,3944E1AFR96RF&top_k=20"
```

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/hotspots?threshold=0.0"
```

## Configuration

The API expects the following data files:
- `output/so6_occupancy_matrix_with_times.json` - Flight occupancy data
- `output/tvtw_indexer.json` - Time window indexer
- Traffic volumes GeoJSON file (path configured in `airspace_api_wrapper.py`)

## Error Handling

- **404** - Traffic volume ID not found
- **500** - Internal server error (data loading issues, computation errors)

## Development

The server runs with auto-reload enabled for development. Modify the code and the server will automatically restart.

For production deployment, consider:
- Setting `reload=False`
- Using a production WSGI server like Gunicorn
- Adding authentication and rate limiting
- Implementing proper logging
