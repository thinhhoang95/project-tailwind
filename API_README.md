# Airspace Traffic Analysis API

This FastAPI server provides endpoints for analyzing traffic volume occupancy data in airspace management systems.

## Features

- **`/tv_count`** - Get occupancy counts for all time windows of a specific traffic volume
- **`/tv_flights`** - Get flight identifiers grouped by time window for a specific traffic volume
- **`/tv_flights_ordered`** - Get all flights for a traffic volume ordered by proximity to a reference time
- **`/regulation_ranking_tv_flights_ordered`** - Rank ordered TV flights by heuristic features with scores and components
- **`/traffic_volumes`** - List all available traffic volume IDs
- **`/tv_count_with_capacity`** - Get occupancy counts along with hourly capacity for a traffic volume
- **`/hotspots`** - Get list of hotspots where traffic volume exceeds capacity with detailed statistics
- **`/slack_distribution`** - For a source TV and reference time, returns per-TV slack at the query bin shifted by nominal travel time (475 kts)
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

### GET `/regulation_ranking_tv_flights_ordered?traffic_volume_id={id}&ref_time_str={HHMMSS}&seed_flight_ids={csv}&top_k={n}`

Ranks flights that pass through the specified traffic volume near a reference time using heuristic features from `FlightFeatures`. It reuses the ordered flights list and augments with a score and component breakdown.

Use `ref_time_str` in numeric `HHMMSS` format (e.g., `084510` for 08:45:10). `HHMM` is also accepted (seconds assumed 00). Provide `seed_flight_ids` as a comma-separated list.

**Parameters:**
- `traffic_volume_id` (string): The traffic volume ID to analyze
- `ref_time_str` (string): Reference time for ordering flights by proximity
- `seed_flight_ids` (string): Comma-separated seed flight IDs used to build the footprint
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
      "delta_seconds": 2,
      "score": 0.873,
      "components": {
        "multiplicity": 1.0,
        "similarity": 0.75,
        "slack": 0.62
      }
    },
    {
      "flight_id": "1234XYZZY",
      "arrival_time": "08:02:30",
      "time_window": "08:00-08:15",
      "delta_seconds": 140,
      "score": 0.721,
      "components": {
        "multiplicity": 0.6,
        "similarity": 0.50,
        "slack": 0.58
      }
    }
  ],
  "metadata": {
    "num_candidates": 120,
    "num_ranked": 20,
    "time_bin_minutes": 15
  }
}
```

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

Returns list of hotspots (traffic volume and time bin combinations where capacity exceeds demands) with detailed statistics including z_max and z_sum metrics.

**Parameters:**
- `threshold` (float, optional): Minimum excess traffic to consider as overloaded (default: 0.0)

**Response:**
```json
{
  "hotspots": [
    {
      "traffic_volume_id": "MASB5KL",
      "time_bin": "06:00-07:00",
      "z_max": 12.5,
      "z_sum": 45.2,
      "hourly_occupancy": 67.0,
      "hourly_capacity": 55.0,
      "is_overloaded": true
    },
    {
      "traffic_volume_id": "TV001",
      "time_bin": "08:00-09:00", 
      "z_max": 8.3,
      "z_sum": 32.1,
      "hourly_occupancy": 43.0,
      "hourly_capacity": 35.0,
      "is_overloaded": true
    }
  ],
  "count": 2,
  "metadata": {
    "threshold": 0.0,
    "time_bin_minutes": 15,
    "analysis_type": "hourly_excess_capacity"
  }
}
```

### GET `/slack_distribution?traffic_volume_id={id}&ref_time_str={time}&sign={plus|minus}`

Returns a slack distribution across all traffic volumes at the “query bin” computed by shifting the source reference bin by the nominal travel time (distance at 475 kts). Useful for finding TVs with spare capacity to absorb demand.

Accepts flexible time formats for `ref_time_str`: `HHMMSS`, `HHMM`, `HH:MM`, `HH:MM:SS`.

**Parameters:**
- `traffic_volume_id` (string): Source traffic volume ID
- `ref_time_str` (string): Reference time string
- `sign` (string): Either `plus` or `minus` (shift direction)

**Response:**
```json
{
  "traffic_volume_id": "MASB5KL",
  "ref_time_str": "08:30",
  "sign": "plus",
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

**Response Fields:**
- `traffic_volume_id`: String identifier for the traffic volume
- `time_bin`: Hourly time window in format "HH:00-HH+1:00"
- `z_max`: Maximum excess traffic within the time bin
- `z_sum`: Total excess traffic within the time bin  
- `hourly_occupancy`: Actual traffic volume for the hour
- `hourly_capacity`: Maximum capacity for the hour
- `is_overloaded`: Boolean indicating if occupancy exceeds capacity

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
With server running:
```bash
curl "http://localhost:8000/tv_count?traffic_volume_id=MASB5KL"
```

```bash
curl "http://localhost:8000/tv_flights?traffic_volume_id=MASB5KL"
```

```bash
curl "http://localhost:8000/tv_flights_ordered?traffic_volume_id=MASB5KL&ref_time_str=080010"
```

```bash
curl "http://localhost:8000/regulation_ranking_tv_flights_ordered?traffic_volume_id=MASB5KL&ref_time_str=080010&seed_flight_ids=0200AFRAM650E,3944E1AFR96RF&top_k=20"
```

```bash
curl "http://localhost:8000/hotspots?threshold=0.0"
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