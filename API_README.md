# Airspace Traffic Analysis API

This FastAPI server provides endpoints for analyzing traffic volume occupancy data in airspace management systems.

## Features

- **`/tv_count`** - Get occupancy counts for all time windows of a specific traffic volume
- **`/tv_flights`** - Get flight identifiers grouped by time window for a specific traffic volume
- **`/tv_flights_ordered`** - Get all flights for a traffic volume ordered by proximity to a reference time
- **`/traffic_volumes`** - List all available traffic volume IDs
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