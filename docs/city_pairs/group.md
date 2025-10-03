# Group by City Pairs Script (`group_by_city_pairs.py`)

## Overview

This script processes historical flight data from a collection of CSV files. It identifies flight routes, groups them based on their origin and destination "city pairs," and saves each group into a separate CSV file.

A "city pair" is determined by taking the first two characters of the origin airport ICAO code and the first two characters of the destination airport ICAO code. For example, a flight from `LFPG` (Paris) to `EGLL` (London) belongs to the city pair `LFEG`. The corresponding output file would be named `LFEG.csv`.

The script is designed to be robust, handling multiple input files, and logging its progress.

## How to Run

The script is executed from the command line. You need to provide the path to the directory containing the input CSV files and the path to the directory where the output files should be saved.

### Command-Line Usage

```bash
python -m project_tailwind.city_pairs.group_by_city_pairs --input-dir <path_to_your_input_data> --output-dir <path_to_your_output_data>
```

### Arguments

-   `--input-dir` (optional): The absolute or relative path to the directory containing the source CSV files. The script will search this directory recursively for any `.csv` file.
    -   **Default:** `D:\\project-akrav\\matched_filtered_data`
-   `--output-dir` (optional): The absolute or relative path to the directory where the grouped CSV files will be saved. If the directory does not exist, it will be created.
    -   **Default:** `output/city_pairs/grouped_flights_by_cpairs`

### Example

```bash
python -m project_tailwind.city_pairs.group_by_city_pairs --input-dir "C:\my_app\raw_flight_data" --output-dir "C:\my_app\grouped_flights"
```

## Input Data Format

The script expects input data in CSV format. The files can be nested in subdirectories within the `--input-dir`. Each CSV file should contain at least the following columns:

-   `flight_id`: A unique identifier for the flight.
-   `real_waypoints`: A space-separated string of ICAO airport codes representing the flight's path. The first code is the origin and the last is the destination.

### Example Input (`routes.csv`)

```csv
flight_id,real_waypoints,pass_times,speeds,alts,real_full_waypoints,full_pass_times,full_speeds,full_alts
flight1,LFPG EGLL,1,1,1,LFPG EGLL,1,1,1
flight2,LFPO KJFK,2,2,2,LFPO KJFK,2,2,2
flight3,EDDF OMDB,3,3,3,EDDF OMDB,3,3,3
flight4,LFPG EGLL EXTRA,4,4,4,LFPG EGLL EXTRA,4,4,4
```

## Output Data Format

The script generates multiple CSV files in the specified `--output-dir`, one for each city pair found in the input data. The filename is constructed as `<Origin_Prefix><Destination_Prefix>.csv`.

Each output file will have the following columns:

-   `flight_identifier`: The flight's unique ID (renamed from `flight_id`).
-   `route`: The full waypoint string (from `real_waypoints`).

### Example Output

Based on the example input above, the script would produce three files:

**1. `LFEG.csv`** (for flights from `LFPG` to `EGLL`)

```csv
flight_identifier,route
flight1,LFPG EGLL
```

**2. `LFKJ.csv`** (for flights from `LFPO` to `KJFK`)

```csv
flight_identifier,route
flight2,LFPO KJFK
```

**3. `EDOM.csv`** (for flights from `EDDF` to `OMDB`)

```csv
flight_identifier,route
flight3,EDDF OMDB
```

**4. `LFEX.csv`** (for flights from `LFPG` to `EXTRA`)

```csv
flight_identifier,route
flight4,LFPG EGLL EXTRA
```
