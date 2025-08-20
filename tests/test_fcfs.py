import sys
from pathlib import Path

project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

import pytest

from project_tailwind.optimize.fcfs.scheduler import assign_delays
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


class _MiniFlightList:
    """
    Minimal FlightList stub exposing flight_metadata with required fields:
      - takeoff_time: datetime
      - occupancy_intervals: list of {tvtw_index, entry_time_s, exit_time_s}
    """

    def __init__(self, flight_metadata):
        self.flight_metadata = flight_metadata
        # Provide attributes used by consumers in other places if needed
        self.time_bin_minutes = 15


def _make_indexer(tv_ids, time_bin_minutes=15) -> TVTWIndexer:
    indexer = TVTWIndexer(time_bin_minutes=time_bin_minutes)
    # Seed TV ids deterministically
    indexer._tv_id_to_idx = {tv: i for i, tv in enumerate(tv_ids)}
    indexer._idx_to_tv_id = {i: tv for tv, i in indexer._tv_id_to_idx.items()}
    indexer._populate_tvtw_mappings()
    return indexer


def test_fcfs_no_overload_zero_delays():
    tv_ids = ["TVA"]
    indexer = _make_indexer(tv_ids)

    t0 = datetime(2024, 1, 1, 8, 0, 0)
    # Two flights 10 minutes apart at same TV and same active window
    # Bin size 15 min; pick window 32 arbitrarily
    win = 32
    meta = {
        "F1": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win), "entry_time_s": 600, "exit_time_s": 900}
            ],
        },
        "F2": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win), "entry_time_s": 1200, "exit_time_s": 1500}
            ],
        },
    }
    fl = _MiniFlightList(meta)

    # Hourly rate 6 flights/hour => service time 600s = 10 minutes
    delays = assign_delays(fl, ["F1", "F2"], "TVA", indexer, hourly_rate=6, active_time_windows=[win])
    assert delays["F1"] == 0
    assert delays["F2"] == 0


def test_fcfs_overload_cascade_minutes_ceiling():
    tv_ids = ["TVA"]
    indexer = _make_indexer(tv_ids)

    t0 = datetime(2024, 1, 1, 8, 0, 0)
    win = 32

    # Three flights 1 minute apart, service time 1200s (20 min) => queue builds
    meta = {
        f"F{i}": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win), "entry_time_s": i * 60, "exit_time_s": i * 60 + 60}
            ],
        }
        for i in range(3)
    }
    fl = _MiniFlightList(meta)

    delays = assign_delays(fl, ["F0", "F1", "F2"], "TVA", indexer, hourly_rate=3, active_time_windows=[win])
    # service time = 1200s; arrivals at t=0,60,120; all wait behind previous
    # F0: 0
    # F1: waits 1140s => ceil(19.0) = 19 min
    # F2: waits 2280s => ceil(38.0) = 38 min
    assert delays["F0"] == 0
    assert delays["F1"] == 19
    assert delays["F2"] == 38


def test_fcfs_partial_active_windows_filtering():
    tv_ids = ["TVA", "TVB"]
    indexer = _make_indexer(tv_ids)

    t0 = datetime(2024, 1, 1, 9, 0, 0)
    win_a, win_b = 10, 11

    meta = {
        "A": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win_a), "entry_time_s": 0, "exit_time_s": 60}
            ],
        },
        "B": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win_b), "entry_time_s": 0, "exit_time_s": 60}
            ],
        },
        "C": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVB", win_a), "entry_time_s": 0, "exit_time_s": 60}
            ],
        },
    }
    fl = _MiniFlightList(meta)

    delays = assign_delays(fl, ["A", "B", "C"], "TVA", indexer, hourly_rate=60, active_time_windows=[win_a])
    # Only A matches window and location
    assert set(delays.keys()) == {"A"}
    assert delays["A"] == 0


def test_fcfs_multiple_entries_uses_earliest_in_active_windows():
    tv_ids = ["TVA"]
    indexer = _make_indexer(tv_ids)

    t0 = datetime(2024, 1, 1, 7, 0, 0)
    win1, win2 = 5, 6
    meta = {
        "F": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win2), "entry_time_s": 1800, "exit_time_s": 2000},
                {"tvtw_index": indexer.get_tvtw_index("TVA", win1), "entry_time_s": 600, "exit_time_s": 800},
            ],
        }
    }
    fl = _MiniFlightList(meta)

    delays = assign_delays(fl, ["F"], "TVA", indexer, hourly_rate=120, active_time_windows=[win1, win2])
    # Earliest active is entry_time_s=600
    assert delays["F"] == 0


def test_fcfs_zero_rate_returns_empty():
    tv_ids = ["TVA"]
    indexer = _make_indexer(tv_ids)

    t0 = datetime(2024, 1, 1, 7, 0, 0)
    meta = {
        "F": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", 1), "entry_time_s": 0, "exit_time_s": 10},
            ],
        }
    }
    fl = _MiniFlightList(meta)

    delays = assign_delays(fl, ["F"], "TVA", indexer, hourly_rate=0, active_time_windows=[1])
    assert delays == {}


