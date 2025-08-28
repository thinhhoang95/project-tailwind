from __future__ import annotations

from typing import Dict, List, Tuple

import math
from datetime import datetime, timedelta
import logging

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


def _decode_tvtw(indexer: TVTWIndexer, tvtw_index: int) -> Tuple[str, int] | None:
    """
    Decode a global TVTW index into (traffic_volume_id, time_window_idx).
    Returns None if the index cannot be decoded.
    """
    try:
        decoded = indexer.get_tvtw_from_index(int(tvtw_index))
    except Exception:
        decoded = None
    return decoded


def assign_delays(
    flight_list: FlightList,
    identifier_list: List[str],
    reference_location: str,
    tvtw_indexer: TVTWIndexer,
    hourly_rate: int,
    active_time_windows: List[int],
) -> Dict[str, int]:
    """
    Assign FCFS delays to flights crossing a reference traffic volume within
    the specified active time windows.

    Continuous single-server model:
      - Service time per flight s = 3600 / hourly_rate seconds
      - Entrants ordered by scheduled crossing datetime (takeoff_time + entry_time_s),
        tie-broken deterministically by (scheduled_dt, flight_id)

    Args:
        flight_list: FlightList holding flight metadata and intervals
        identifier_list: Candidate flight IDs to consider
        reference_location: Traffic volume ID to regulate
        tvtw_indexer: TVTWIndexer for decoding TVTW indices
        hourly_rate: Service rate (flights per hour); must be > 0
        active_time_windows: List of time window indices (integers)

    Returns:
        Mapping flight_id -> delay_minutes (integer, ceil of seconds/60)
    """
    if hourly_rate is None or hourly_rate <= 0:
        logging.warning("assign_delays: hourly_rate <= 0; returning no delays")
        return {}

    if not active_time_windows:
        return {}

    active_set = set(int(w) for w in active_time_windows)

    entrants: List[Tuple[datetime, str]] = []

    # Build entrant list: earliest crossing of reference_location within active windows
    for flight_id in identifier_list:
        # Guard: skip unknown flights
        if flight_id not in flight_list.flight_metadata:
            continue

        meta = flight_list.flight_metadata[flight_id]
        takeoff_time: datetime = meta["takeoff_time"]

        earliest_dt: datetime | None = None

        for interval in meta.get("occupancy_intervals", []):
            decoded = _decode_tvtw(tvtw_indexer, interval["tvtw_index"])  # (tv_id, time_idx)
            if not decoded:
                continue
            tv_id, time_idx = decoded
            if tv_id != reference_location:
                continue
            if int(time_idx) not in active_set:
                continue

            entry_seconds = float(interval.get("entry_time_s", 0))
            scheduled_dt = takeoff_time + timedelta(seconds=entry_seconds)
            if earliest_dt is None or scheduled_dt < earliest_dt:
                earliest_dt = scheduled_dt

        if earliest_dt is not None:
            entrants.append((earliest_dt, flight_id))

    # Deterministic ordering
    entrants.sort(key=lambda x: (x[0], x[1]))

    if not entrants:
        return {}

    service_seconds = 3600.0 / float(hourly_rate)
    service_delta = timedelta(seconds=service_seconds)

    delays: Dict[str, int] = {}
    server_available_at: datetime | None = None

    for scheduled_dt, flight_id in entrants:
        if server_available_at is None or scheduled_dt >= server_available_at:
            # No wait
            delay_seconds = 0.0
            service_start = scheduled_dt
        else:
            # Wait until server available
            delay_seconds = (server_available_at - scheduled_dt).total_seconds()
            service_start = server_available_at

        # Ceil to minutes
        delay_minutes = int(math.ceil(delay_seconds / 60.0)) if delay_seconds > 0 else 0
        delays[flight_id] = delay_minutes

        # Advance server
        server_available_at = service_start + service_delta

    return delays


__all__ = ["assign_delays"]


