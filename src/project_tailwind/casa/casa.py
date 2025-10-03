
import json
import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
from rich.console import Console

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from .delay_assigner import assign_delays

console = Console()


def generate_casa_windows(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    window_length_min: int,
    window_stride_min: int,
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """Generates rolling CASA windows."""
    windows = []
    current_time = start_time
    while current_time < end_time:
        window_end = current_time + datetime.timedelta(minutes=window_length_min)
        windows.append((current_time, window_end))
        current_time += datetime.timedelta(minutes=window_stride_min)
    return windows


def run_readapted_casa(
    so6_occupancy_path: str,
    identifier_list: List[str],
    reference_location: str,
    tvtw_indexer: TVTWIndexer,
    hourly_rate: float,
    active_time_windows: List[int],
    ccasa_window_length_min: int = 20,
    window_stride_min: int = 10,
) -> Dict[str, float]:
    """
    Re-adapted C-CASA implementation based on the new specification.

    This function calculates flight delays based on a single traffic volume
    regulation, using a rate-limiting queue model at the reference location.
    It counts entries in rolling windows and pushes excess entries to later windows.
    """
    console.print("[bold cyan]Initiating Re-adapted C-CASA Simulation...[/bold cyan]")

    with open(so6_occupancy_path, "r") as f:
        flight_data = json.load(f)

    eligible_tvtw_indices = {
        tvtw_indexer.get_tvtw_index(reference_location, tw_idx)
        for tw_idx in active_time_windows
    }

    casa_events = []
    for flight_id in identifier_list:
        flight_info = flight_data.get(flight_id)
        if not flight_info or "takeoff_time" not in flight_info:
            console.log(f"[yellow]Warning: Flight {flight_id} missing or malformed in occupancy data.[/yellow]")
            continue

        takeoff_time = pd.to_datetime(flight_info["takeoff_time"])
        first_entry_time = None

        for interval in flight_info.get("occupancy_intervals", []):
            if "tvtw_index" not in interval or "entry_time_s" not in interval:
                continue

            tv_id, _ = tvtw_indexer.get_tvtw_from_index(interval["tvtw_index"])

            if tv_id == reference_location:
                t_entry_abs = takeoff_time + datetime.timedelta(seconds=interval["entry_time_s"])

                minute_of_day = (t_entry_abs - t_entry_abs.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 60
                time_window_for_entry = int(minute_of_day / tvtw_indexer.time_bin_minutes)

                is_eligible = (
                    interval["tvtw_index"] in eligible_tvtw_indices or
                    time_window_for_entry in active_time_windows
                )

                if is_eligible:
                    if first_entry_time is None or t_entry_abs < first_entry_time:
                        first_entry_time = t_entry_abs

        if first_entry_time:
            casa_events.append({
                "flight_id": flight_id,
                "t_entry_orig": first_entry_time,
                "t_entry_star": first_entry_time,
                "tv_id": reference_location,
            })

    if not casa_events:
        console.print("[yellow]No eligible flights found for the given regulation.[/yellow]")
        return {flight_id: 0.0 for flight_id in identifier_list}

    ref_date = casa_events[0]["t_entry_orig"].date()
    day_start_datetime = datetime.datetime.combine(ref_date, datetime.time.min)

    active_start_abs = day_start_datetime + datetime.timedelta(minutes=min(active_time_windows) * tvtw_indexer.time_bin_minutes)
    active_end_abs = day_start_datetime + datetime.timedelta(minutes=(max(active_time_windows) + 1) * tvtw_indexer.time_bin_minutes)

    windows = generate_casa_windows(
        active_start_abs, active_end_abs, ccasa_window_length_min, window_stride_min
    )

    carry = 0.0
    capacity_per_window_fractional = hourly_rate * (ccasa_window_length_min / 60.0)
    
    all_events = sorted(casa_events, key=lambda x: x['t_entry_star'])

    for window_start, window_end in windows:
        all_events.sort(key=lambda x: x['t_entry_star'])
        
        entrants = [
            event for event in all_events if window_start <= event['t_entry_star'] < window_end
        ]
        
        if not entrants:
            continue

        carry += capacity_per_window_fractional
        capacity = int(carry)
        carry -= capacity
        
        assign_delays(entrants, capacity, window_end)

    delays = {}
    for event in all_events:
        delay_seconds = (event["t_entry_star"] - event["t_entry_orig"]).total_seconds()
        delays[event["flight_id"]] = max(0, delay_seconds / 60.0)

    for flight_id in identifier_list:
        if flight_id not in delays:
            delays[flight_id] = 0.0

    console.print(f"[bold green]C-CASA simulation complete. Calculated delays for {len(delays)} flights.[/bold green]")
    return delays
