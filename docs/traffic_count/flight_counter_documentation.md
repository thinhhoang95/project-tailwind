# Flight Counter Documentation

## 1. Overview

This document describes the `flight_counter_tv_large_segment_window_len.py` script. The script is designed to count the number of flights passing through predefined 3D airspaces, known as Traffic Volumes (TVs), within specific time windows.

Its core purpose is to identify potential airspace congestion by comparing the flight counts against the declared capacity of each traffic volume. It processes flight trajectory data, determines which TVs are crossed, and aggregates these crossings into time-based bins.

A key feature of this script is its ability to handle long flight segments by sampling points along the trajectory. This ensures that TVs are not missed if a flight passes through them without the start or end point of a segment falling within the volume. The script is optimized for performance using parallel processing.

## 2. Inputs

The script requires two main input files: a flight data file and a traffic volume definition file.

### 2.1. Flights Data (CSV)

A CSV file containing flight trajectory data, where each row represents a segment of a flight.

-   **Path:** Provided via the `flights_path` argument.
-   **Format:** CSV
-   **Key Columns:**
    -   `flight_identifier`: A unique ID for each flight.
    -   `origin_aerodrome`: The flight's origin airport.
    -   `destination_aerodrome`: The flight's destination airport.
    -   `date_begin_segment`: The date of the segment's start point (e.g., `230801` for YYYYMMDD).
    -   `time_begin_segment`: The time of the segment's start point (e.g., `000000` for HHMMSS).
    -   `latitude_begin`: Latitude of the segment's start point.
    -   `longitude_begin`: Longitude of the segment's start point.
    -   `flight_level_begin`: Flight level (in hundreds of feet) at the start of the segment.
    -   `date_end_segment`: The date of the segment's end point.
    -   `time_end_segment`: The time of the segment's end point.
    -   `latitude_end`: Latitude of the segment's end point.
    -   `longitude_end`: Longitude of the segment's end point.

#### Example Input: `flights.csv`

```csv
flight_identifier,origin_aerodrome,destination_aerodrome,date_begin_segment,time_begin_segment,latitude_begin,longitude_begin,flight_level_begin,date_end_segment,time_end_segment,latitude_end,longitude_end
FLT123,EGLL,LFPG,230801,100000,51.47,-0.45,350,230801,100500,51.0,0.5,350
FLT123,EGLL,LFPG,230801,100500,51.0,0.5,350,230801,101000,50.5,1.5,350
FLT456,EDDF,LEMD,230801,120000,50.03,8.57,330,230801,121500,49.0,7.0,330
```

### 2.2. Traffic Volumes (GeoJSON)

A GeoJSON file defining the 3D geometry and properties of the traffic volumes.

-   **Path:** Provided via the `tv_path` argument.
-   **Format:** GeoJSON
-   **Structure:** A `FeatureCollection` where each `Feature` has:
    -   `geometry`: A `Polygon` defining the 2D horizontal shape of the TV.
    -   `properties`: An object containing metadata about the TV.
        -   `traffic_volume_id`: A unique name or identifier for the TV.
        -   `min_fl`: The minimum flight level (bottom) of the TV.
        -   `max_fl`: The maximum flight level (top) of the TV.
        -   `capacity`: A JSON string or object defining the hourly capacity of the TV. The keys are time ranges (e.g., "8:00-9:00") and values are the maximum number of flights allowed in that hour.

#### Example Input: `traffic_volumes.geojson`

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[2.0, 49.0], [3.0, 49.0], [3.0, 50.0], [2.0, 50.0], [2.0, 49.0]]]
      },
      "properties": {
        "traffic_volume_id": "TV_Paris_Approach",
        "min_fl": 100,
        "max_fl": 250,
        "capacity": "{\"00:00-01:00\": 50, \"01:00-02:00\": 50, ...}"
      }
    }
  ]
}
```

## 3. Outputs

The script generates one primary output file and one optional log file.

### 3.1. Flight Counts (CSV)

A CSV file containing the flight counts for each traffic volume, broken down into time bins. It also includes flags for hourly capacity overload.

-   **Path:** Provided via the `output_path` argument.
-   **Format:** CSV
-   **Columns:**
    -   `traffic_volume_id`: The unique ID of the traffic volume.
    -   Time Window Columns (e.g., `00:00-00:30`, `00:30-01:00`, ...): The number of unique flights that entered the TV during that time window. The size of the window is configurable (`time_bin_minutes`).
    -   `overload_{hh}` (e.g., `overload_00`, `overload_01`, ...): A binary flag (0 or 1) for each of the 24 hours of the day. It is set to `1` if the total flight count for that hour exceeded the TV's defined capacity.

#### Example Output: `flight_counts_tv.csv`

```csv
traffic_volume_id,00:00-00:30,00:30-01:00,...,23:30-24:00,overload_00,...,overload_23
TV_Paris_Approach,10,12,...,5,0,...,0
TV_London_TMA,25,22,...,8,1,...,0
```

### 3.2. Flight Log (CSV) (Optional)

If a `flight_log_path` is provided, the script generates a detailed CSV log of every time a flight enters and exits a traffic volume.

-   **Path:** Provided via the `flight_log_path` argument.
-   **Format:** CSV
-   **Columns:**
    -   `flight_identifier`: The flight's unique ID.
    -   `origin_aerodrome`: Flight origin.
    -   `destination_aerodrome`: Flight destination.
    -   `fl`: The flight level at the time of crossing.
    -   `traffic_volume_name`: The ID of the TV that was crossed.
    -   `entry_lon`, `entry_lat`, `entry_time`: The coordinates and timestamp of entry into the TV.
    -   `exit_lon`, `exit_lat`, `exit_time`: The coordinates and timestamp of exit from the TV.
    -   `time_window`: The time window bin corresponding to the entry time.

#### Example Output: `flight_log_tv.csv`

```csv
flight_identifier,origin_aerodrome,...,traffic_volume_name,entry_lon,entry_lat,entry_time,exit_lon,exit_lat,exit_time,time_window
FLT123,EGLL,...,TV_Paris_Approach,2.5,49.1,2023-08-01 10:08:15,2.8,49.5,2023-08-01 10:12:45,10:00-10:30
```

## 4. Algorithm

The script follows a multi-stage process involving data loading, parallel processing, result aggregation, and analysis.

### Step 1: Initialization
- The main function `flight_counter_tv` is invoked.
- It sets up parallel processing by determining the number of worker processes (`n_jobs`).
- It loads the Traffic Volume GeoJSON file using `load_traffic_volumes`. This function reads the file, creates a mapping from the TV name to a unique integer index, and prepares it for processing. A spatial index is built on the GeoDataFrame for efficient geometric queries.

### Step 2: Flight Processing
- The script loads the entire flights CSV into a pandas DataFrame.
- Unique flight identifiers are extracted and split into smaller batches. These batches are distributed among the worker processes for parallel execution.

### Step 3: Parallel Batch Processing (`process_flight_batch`)
- Each worker process receives a batch of flight IDs.
- For each flight ID, it filters the main DataFrame to get all segments belonging to that flight.
- It calls `process_flight` for each individual flight.
- The results (flight counts and log entries) from each flight are collected and returned to the main process.

### Step 4: Single Flight Analysis (`process_flight`)
This is the core logic where a single flight's trajectory is analyzed against the traffic volumes.
1.  **Segment Iteration:** The function iterates through each segment of the flight's trajectory.
2.  **Long Segment Sampling:**
    - For each segment, it calculates the geodesic distance between the start and end points.
    - If this distance exceeds a configurable threshold (`sampling_dist_nm`), the segment is considered "long."
    - The script then generates intermediate sample points along the segment at regular intervals (defined by `sampling_dist_nm`).
    - For each sample point, it interpolates the time and checks which TV, if any, contains this point at the segment's flight level. This prevents missing TVs that are crossed between the start and end points of a long segment.
3.  **TV Transition Detection:**
    - The script maintains a state of which TVs the flight is currently inside (`tv_entry_info`).
    - By comparing the set of TVs at the start of a segment (or sample point) with the set at the end, it detects entries and exits.
    - **Entry:** If a TV is present at the current point but was not at the previous one, a new entry is recorded with the current time and location.
    - **Exit:** If a TV was present at the previous point but is not at the current one, an exit is recorded. The script then calls `update_counts` to log the passage.
4.  **Point-in-TV Check (`find_tvs_for_point`):**
    - This helper function determines which TVs contain a given 3D point (lat, lon, flight level).
    - It first uses the highly efficient spatial index (`sindex`) of the GeoDataFrame to find all TVs whose 2D polygon intersects the point's location.
    - It then performs a precise check (`covers`) on this subset to confirm the point is truly inside the polygon.
    - Finally, it filters this list by checking if the flight's level is between the TV's `min_fl` and `max_fl`.

### Step 5: Updating Counts and Logs (`update_counts`)
- Once a flight's full passage through a TV is determined (from an entry point/time to an exit point/time), this function is called.
- It determines the time bin(s) the flight occupied the TV.
- It increments the count for the corresponding (TV index, time bin) in a sparse matrix. A `visited_tv_bins` set ensures that a flight is counted only once per TV per time bin, even if its trajectory causes it to enter and exit the same bin multiple times.
- If logging is enabled, it generates a detailed log entry with entry/exit coordinates and times.

### Step 6: Aggregation and Overload Calculation
- After all parallel processes are complete, the main process collects the sparse matrices of counts from all workers and sums them to get the final total counts.
- It converts the final sparse matrix into a pandas DataFrame, mapping TV indices back to their string IDs.
- **Overload Calculation:**
    - For each TV and for each hour of the day (0-23):
    - It sums the counts from the time bins that constitute that hour (e.g., for a 30-minute bin, it sums two bins).
    - It retrieves the hourly capacity from the TV's properties.
    - If the total count exceeds the capacity, the corresponding `overload_{hh}` flag is set to `1`.

### Step 7: Saving Results
- The final DataFrame with counts and overload flags is saved to the specified `output_path`.
- If logging was enabled, the collected log entries are saved to the `flight_log_path`.