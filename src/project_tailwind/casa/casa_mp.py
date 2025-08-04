

import pandas as pd
import geopandas as gpd
import json
from multiprocessing import Pool, cpu_count, Manager
from rich.console import Console
from rich.progress import track
import os
import datetime

from .window_counter import generate_rolling_windows, window_count, load_and_prepare_flights
from .delay_assigner import assign_delays
from cirrus.traffic_counting.flight_counter_tv_large_segment_window_len import (
    load_traffic_volumes,
)

console = Console()


def parse_capacity(capacity):
    """Parses the capacity information which can be a string or dict."""
    if isinstance(capacity, str):
        try:
            return json.loads(capacity)
        except json.JSONDecodeError:
            return {}
    elif isinstance(capacity, dict):
        return capacity
    return {}


def get_hourly_capacity(capacity_info, window_start):
    """
    Gets the hourly capacity for a given time from the capacity info dictionary.
    """
    hour = window_start.hour
    for key, value in capacity_info.items():
        # Capacity keys are strings like '8:00-9:00'
        if key.startswith(f"{hour}:"):
            return value
    return None


def process_window(args):
    """
    Worker function to process a single time window.
    Returns delays to be applied to flights.
    """
    window_start, window_end, flights_data, tv_data, window_length_min = args
    
    # Recreate DataFrames from serialized data
    flights_df = pd.DataFrame(flights_data)
    tv_gdf = gpd.GeoDataFrame(tv_data)
    
    # Convert datetime strings back to datetime objects
    flights_df['initial_takeoff_time'] = pd.to_datetime(flights_df['initial_takeoff_time'])
    flights_df['revised_takeoff_time'] = pd.to_datetime(flights_df['revised_takeoff_time'])
    
    flights_df.sort_values("revised_takeoff_time", inplace=True)
    
    counts, flight_log = window_count(
        flights_df, tv_gdf, window_start, window_end, window_length_min
    )
    
    delays_to_apply = []
    
    for tv_name, count in counts.items():
        tv_row = tv_gdf[tv_gdf['traffic_volume_id'] == tv_name]
        if tv_row.empty:
            continue
            
        capacity_info = parse_capacity(tv_row.iloc[0]['capacity'])
        hourly_capacity = get_hourly_capacity(capacity_info, window_start)
        
        if hourly_capacity is None:
            continue
            
        capacity_threshold = hourly_capacity / (60 / window_length_min)
        
        if count > capacity_threshold:
            console.print(
                f"[yellow]Hotspot detected in {tv_name} at "
                f"{window_start.strftime('%H:%M:%S')}: "
                f"Count {count} > Threshold {capacity_threshold:.2f}[/yellow]"
            )
            
            hotspot_flights = [
                entry["flight_identifier"]
                for entry in flight_log
                if entry["traffic_volume_name"] == tv_name
            ]
            
            rate_for_delay = int(capacity_threshold)
            
            delays_to_apply.append({
                'hotspot_flights': hotspot_flights,
                'window_end': window_end,
                'rate_for_delay': rate_for_delay
            })
    
    # Add window information to counts
    counts_with_window = {
        'window_start': window_start,
        'window_end': window_end,
        **counts
    }
    
    return delays_to_apply, counts_with_window, flight_log




def apply_sequential_delays(flights_df: pd.DataFrame, results):
    """Optimized sequential delay application that avoids expensive
    DataFrame operations inside the critical loop.

    The original implementation invoked ``assign_delays`` for every hotspot
    in every window, which repeatedly filtered and sorted a potentially large
    DataFrame. By working with plain Python dictionaries keyed by
    ``flight_identifier`` we reduce the per-iteration overhead drastically.
    """
    # Make sure we can index flights quickly by flight_identifier
    flights_df = flights_df.set_index("flight_identifier", drop=False)

    # Pre-materialise look-up dictionaries for constant-time access
    initial_times = flights_df["initial_takeoff_time"].to_dict()
    revised_times = flights_df["revised_takeoff_time"].to_dict()

    # Iterate over windows sequentially – ordering matters!
    for window_delays in track(results, description="Applying delays..."):
        for delay_info in window_delays:
            hotspot_ids = delay_info["hotspot_flights"]
            rate = delay_info["rate_for_delay"]

            if not hotspot_ids or len(hotspot_ids) <= rate:
                # Nothing to do for this hotspot
                continue

            # Sort the hotspot flights by their CURRENT revised take-off time
            hotspot_ids_sorted = sorted(hotspot_ids, key=lambda fid: revised_times[fid])

            reference_time = revised_times[hotspot_ids_sorted[rate - 1]]
            delay_minutes = (
                (delay_info["window_end"] - reference_time).total_seconds() / 60
            ) + 1  # small buffer to move the flight outside the window

            # Apply the same delay to all flights beyond the allowed rate
            for fid in hotspot_ids_sorted[rate:]:
                new_time = initial_times[fid] + datetime.timedelta(minutes=delay_minutes)
                if new_time > revised_times[fid]:
                    revised_times[fid] = new_time

    # Persist the updated revised_takeoff_time back into the DataFrame
    flights_df.loc[:, "revised_takeoff_time"] = flights_df["flight_identifier"].map(revised_times)

    # Restore the original integer index layout expected by downstream code
    return flights_df.reset_index(drop=True)


def run_casa(
    flights_path: str,
    tv_path: str,
    output_path: str,
    time_begin: str,
    time_end: str,
    window_length_min: int,
    window_stride_min: int,
):
    """
    Main orchestrator for the C-CASA algorithm using multiprocessing.
    """
    console.print("[bold cyan]Initiating C-CASA Simulation...[/bold cyan]")

    # Load data and determine the actual time range from flights
    flights_df, actual_start_time, actual_end_time = load_and_prepare_flights(
        flights_path
    )
    tv_gdf, tv_map = load_traffic_volumes(tv_path)

    # Use the provided time_begin and time_end if they are more restrictive
    sim_start_time = max(pd.to_datetime(time_begin), actual_start_time)
    sim_end_time = min(
        pd.to_datetime(time_end),
        actual_end_time + datetime.timedelta(hours=1),
    )

    console.print(
        f"[bold cyan]Effective simulation time window: "
        f"{sim_start_time.strftime('%Y-%m-%d %H:%M:%S')} to "
        f"{sim_end_time.strftime('%Y-%m-%d %H:%M:%S')}[/bold cyan]"
    )

    windows = generate_rolling_windows(
        sim_start_time, sim_end_time, window_length_min, window_stride_min
    )

    output_dir = os.path.dirname(output_path)
    counts_path = os.path.join(output_dir, "counts.csv")
    flight_log_path = os.path.join(output_dir, "flight_log.csv")

    if os.path.exists(counts_path) and os.path.exists(flight_log_path):
        console.print("[bold yellow]Loading existing counts and flight log from files...[/bold yellow]")
        counts_df = pd.read_csv(counts_path)
        all_flight_logs = pd.read_csv(flight_log_path).to_dict('records')

        all_counts = counts_df.to_dict('records')
        
        # Re-calculate delays based on loaded counts
        results = []
        for i, (window_start, window_end) in enumerate(windows):
            window_counts = all_counts[i]
            window_flight_log = [
                log for log in all_flight_logs 
                if pd.to_datetime(log['timestamp']) >= window_start and pd.to_datetime(log['timestamp']) < window_end
            ]
            
            delays_to_apply = []
            for tv_name, count in window_counts.items():
                if tv_name in ['window_start', 'window_end']:
                    continue

                tv_row = tv_gdf[tv_gdf['traffic_volume_id'] == tv_name]
                if tv_row.empty:
                    continue

                capacity_info = parse_capacity(tv_row.iloc[0]['capacity'])
                hourly_capacity = get_hourly_capacity(capacity_info, window_start)

                if hourly_capacity is None:
                    continue

                capacity_threshold = hourly_capacity / (60 / window_length_min)

                if count > capacity_threshold:
                    hotspot_flights = [
                        entry["flight_identifier"]
                        for entry in window_flight_log
                        if entry["traffic_volume_id"] == tv_name
                    ]

                    rate_for_delay = int(capacity_threshold)

                    delays_to_apply.append({
                        'hotspot_flights': hotspot_flights,
                        'window_end': window_end,
                        'rate_for_delay': rate_for_delay
                    })
            results.append(delays_to_apply)

    else:
        # Determine number of processes to use
        n_processes = max(1, cpu_count() - 1)
        console.print(f"[blue]Using {n_processes} processes for parallel window processing[/blue]")

        # Prepare data for multiprocessing (serialize DataFrames)
        flights_data = flights_df.to_dict('records')
        tv_data = tv_gdf.to_dict('records')
        
        # Prepare arguments for worker processes
        worker_args = []
        for window_start, window_end in windows:
            worker_args.append((
                window_start, 
                window_end, 
                flights_data, 
                tv_data, 
                window_length_min
            ))

        # Process windows in parallel
        console.print(f"[blue]Processing {len(windows)} windows in parallel...[/blue]")
        all_counts = []
        all_flight_logs = []
        with Pool(processes=n_processes) as pool:
            results = []
            for result, counts, flight_log in track(
                pool.imap(process_window, worker_args),
                total=len(worker_args),
                description="Processing windows..."
            ):
                results.append(result)
                all_counts.append(counts)
                all_flight_logs.extend(flight_log)

        # Save counts and flight log
        counts_df = pd.DataFrame(all_counts)
        counts_df.to_csv(counts_path, index=False)
        flight_log_df = pd.DataFrame(all_flight_logs)
        flight_log_df.to_csv(flight_log_path, index=False)
        console.print(f"[bold green]Counts and flight logs saved to {output_dir}[/bold green]")


    # Apply delays sequentially to maintain order and consistency using optimized approach
    console.print("[blue]Applying delays sequentially...[/blue]")
    flights_df = apply_sequential_delays(flights_df, results)

    # Final processing and output
    flights_df["delay_min"] = (
        flights_df["revised_takeoff_time"] - flights_df["initial_takeoff_time"]
    ).dt.total_seconds() / 60

    # Remove potential duplicate rows – one record per flight
    output_df = flights_df[["flight_identifier", "delay_min"]].drop_duplicates(
        subset=["flight_identifier"], keep="first"
    )
    output_df.to_csv(output_path, index=False)

    console.print(f"[bold green]C-CASA simulation complete. Results saved to {output_path}[/bold green]")
