
import pandas as pd
import json
from rich.console import Console
from rich.progress import track

from .window_counter import generate_rolling_windows, window_count, load_and_prepare_flights
from .delay_assigner import assign_delays
from cirrus.traffic_counting.flight_counter_tv_large_segment import (
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
    Main orchestrator for the C-CASA algorithm.
    """
    console.print("[bold cyan]Initiating C-CASA Simulation...[/bold cyan]")

    # Load data
    flights_df = load_and_prepare_flights(flights_path, time_begin)
    tv_gdf, tv_map = load_traffic_volumes(tv_path)
    


    start_time = pd.to_datetime(time_begin)
    end_time = pd.to_datetime(time_end)

    windows = generate_rolling_windows(
        start_time, end_time, window_length_min, window_stride_min
    )

    # Main loop over rolling windows
    for window_start, window_end in track(windows, description="Processing windows..."):
        flights_df.sort_values("revised_takeoff_time", inplace=True)

        counts, flight_log = window_count(
            flights_df, tv_gdf, window_start, window_end, window_length_min
        )

        for tv_name, count in counts.items():
            tv_row = tv_gdf[tv_gdf['traffic_volume_id'] == tv_name]
            if tv_row.empty:
                continue

            capacity_info = tv_row.iloc[0]['capacity']
            hourly_capacity = get_hourly_capacity(capacity_info, window_start)

            if hourly_capacity is None:
                continue

            # As per the instructions, the capacity value is divided by 60/(window_length_min)
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
                    if entry["traffic_volume"] == tv_name
                ]

                # The rate for delay assignment is the integer part of the threshold
                rate_for_delay = int(capacity_threshold)

                flights_df = assign_delays(
                    flights_df, hotspot_flights, window_end, rate_for_delay
                )

    # Final processing and output
    flights_df["delay_min"] = (
        flights_df["revised_takeoff_time"] - flights_df["initial_takeoff_time"]
    ).dt.total_seconds() / 60

    output_df = flights_df[["flight_identifier", "delay_min"]]
    output_df.to_csv(output_path, index=False)

    console.print(f"[bold green]C-CASA simulation complete. Results saved to {output_path}[/bold green]")
