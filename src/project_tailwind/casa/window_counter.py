
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool, cpu_count
from functools import partial
import datetime
from rich.console import Console
from shapely.geometry import Point, LineString

from cirrus.traffic_counting.flight_counter_tv_large_segment_window_len import (
    find_tvs_for_point,
    is_in_ecac_bbox,
    ecac_polygon,
    process_flight as process_flight_trajectory,
)

console = Console()


def generate_rolling_windows(start_time, end_time, window_length_min, window_stride_min):
    """Generates rolling time windows."""
    windows = []
    current_time = start_time
    while current_time + datetime.timedelta(minutes=window_length_min) <= end_time:
        windows.append(
            (
                current_time,
                current_time + datetime.timedelta(minutes=window_length_min),
            )
        )
        current_time += datetime.timedelta(minutes=window_stride_min)
    return windows


def flight_intersects_window(flight, window_start, window_end):
    """Check if a flight's takeoff time is within the window."""
    takeoff_time = flight["initial_takeoff_time"]
    return window_start <= takeoff_time < window_end


def window_count(
    flights,
    tv_gdf,
    window_start,
    window_end,
    window_length_min,
    n_jobs=None,
):
    """
    Counts flights in traffic volumes for a single rolling window using trajectory-based counting.
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    # Filter flights that are in the current window based on their revised takeoff time
    window_flights = flights[
        (flights["revised_takeoff_time"] >= window_start)
        & (flights["revised_takeoff_time"] < window_end)
    ]

    unique_flight_ids = window_flights["flight_identifier"].unique()
    if len(unique_flight_ids) == 0:
        return {}, []

    flight_batches = [
        unique_flight_ids[i : i + 100]
        for i in range(0, len(unique_flight_ids), 100)
    ]

    # Process flight batches serially
    results = []
    for batch in flight_batches:
        result = process_flight_batch(
            batch,
            flights_df=flights,
            tv_gdf=tv_gdf,
            time_begin=window_start.strftime("%Y-%m-%d %H:%M:%S"),
            time_end=window_end.strftime("%Y-%m-%d %H:%M:%S"),
            window_length_min=window_length_min,
        )
        results.append(result)

    # Aggregate results
    counts = {tv["traffic_volume_id"]: 0 for _, tv in tv_gdf.iterrows()}
    flight_log = []
    for tv_counts, log_entries in results:
        for tv_name, num_flights in tv_counts.items():
            counts[tv_name] += num_flights
        flight_log.extend(log_entries)

    return counts, flight_log


def process_flight_batch(
    flight_ids_batch, flights_df, tv_gdf, time_begin, time_end, window_length_min
):
    """
    Process a batch of flights to count their intersections with traffic volumes.
    """
    from scipy.sparse import lil_matrix
    import datetime

    tv_counts = {}
    log_entries_batch = []

    # Initialize the sparse matrix for flight counts
    num_tvs = len(tv_gdf)
    if 1440 % window_length_min != 0:
        raise ValueError("window_length_min must be a divisor of 1440.")
    num_bins = 1440 // window_length_min
    flight_counts_matrix = lil_matrix((num_tvs, num_bins), dtype=int)

    # Parse time window boundaries
    time_begin_dt = datetime.datetime.strptime(time_begin, "%Y-%m-%d %H:%M:%S")
    time_end_dt = datetime.datetime.strptime(time_end, "%Y-%m-%d %H:%M:%S")
    
    # Calculate which time bin this window corresponds to
    window_start_minute = time_begin_dt.hour * 60 + time_begin_dt.minute
    window_bin = window_start_minute // window_length_min

    for flight_id in flight_ids_batch:
        flight_segments = flights_df[flights_df['flight_identifier'] == flight_id]
        
        if flight_segments.empty:
            continue
            
        # Process the flight trajectory using the existing logic
        visited_tv_bins = set()
        
        log_entries = process_flight_trajectory(
            flight_segments,
            tv_gdf,
            flight_counts_matrix,
            time_begin,
            time_end,
            flight_log_path="dummy",  # To trigger log generation
            time_bin_minutes=window_length_min
        )
        
        # Collect all log entries for this batch
        log_entries_batch.extend(log_entries)

    # Convert the sparse matrix to a dictionary of counts for this specific window
    # We only care about the counts in the window bin that corresponds to our time window
    for tv_idx in range(num_tvs):
        tv_name = tv_gdf.iloc[tv_idx]['traffic_volume_id']
        # Sum all counts for this TV across all time bins (since we're looking at a specific window)
        total_count = int(flight_counts_matrix[tv_idx, :].sum())
        if total_count > 0:
            tv_counts[tv_name] = total_count
    
    return tv_counts, log_entries_batch


def load_and_prepare_flights(flights_path):
    """
    Loads flights, adds provisional and revised takeoff times,
    and determines the actual flight time range from the data.
    """
    flights_df = pd.read_csv(flights_path)

    # Sort by flight_identifier and sequence to find the first segment
    flights_df = flights_df.sort_values(["flight_identifier", "sequence"])

    # Get the first segment for each flight
    first_segments = flights_df.drop_duplicates(
        subset="flight_identifier", keep="first"
    )

    # Function to create a datetime object from date and time columns
    def to_datetime(row):
        date_str = f"20{row['date_begin_segment']}"
        time_str = str(row["time_begin_segment"]).zfill(6)
        return pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M%S")

    # Calculate initial_takeoff_time for each flight
    takeoff_times = first_segments.apply(to_datetime, axis=1)

    # Determine actual time range from takeoff times
    actual_start_time = takeoff_times.min()
    actual_end_time = takeoff_times.max()

    # Create a mapping from flight_identifier to takeoff_time
    flight_id_to_takeoff = pd.Series(
        takeoff_times.values, index=first_segments["flight_identifier"]
    )

    # Map the takeoff times to the original dataframe
    flights_df["initial_takeoff_time"] = flights_df["flight_identifier"].map(
        flight_id_to_takeoff
    )

    flights_df["revised_takeoff_time"] = flights_df["initial_takeoff_time"]

    return flights_df, actual_start_time, actual_end_time
