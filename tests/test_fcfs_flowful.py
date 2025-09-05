import sys
from pathlib import Path

project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

import pytest

from parrhesia.fcfs.flowful import _normalize_flight_spec, preprocess_flights_for_scheduler, assign_delays_flowful
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


def _make_indexer(bin_minutes: int = 15) -> TVTWIndexer:
    idx = TVTWIndexer(time_bin_minutes=bin_minutes)
    # Seed a couple of TVs for completeness; names are arbitrary for these tests
    idx._tv_id_to_idx = {"TVA": 0, "TVB": 1}
    idx._idx_to_tv_id = {0: "TVA", 1: "TVB"}
    idx._populate_tvtw_mappings()
    return idx


def test_normalize_flight_spec_from_takeoff_plus_entry_seconds():
    idx = _make_indexer(bin_minutes=15)
    base = datetime(2025, 1, 1, 9, 0, 0)  # 09:00 is bin 36 for 15-min bins
    spec = {
        "flight_id": "X",
        "takeoff_time": base,
        "entry_time_s": 120.0,  # 2 minutes into the bin
    }
    fid, rdt, rbin = _normalize_flight_spec(spec, idx)
    assert fid == "X"
    assert rdt == base + timedelta(seconds=120)
    assert rbin == idx.bin_of_datetime(rdt)


def test_preprocess_sorts_within_bin_by_requested_dt():
    idx = _make_indexer(bin_minutes=15)
    base = datetime(2025, 1, 1, 9, 0, 0)
    # Two flights in same bin (09:00-09:15), different within-bin seconds
    flights_by_flow = {
        "flow": [
            {"flight_id": "F2", "requested_dt": base + timedelta(minutes=7)},
            {"flight_id": "F1", "requested_dt": base + timedelta(minutes=2)},
        ]
    }
    out = preprocess_flights_for_scheduler(flights_by_flow, idx)
    norm = out["flow"]
    # Expect F1 (2 min) before F2 (7 min)
    assert [fid for fid, _, _ in norm] == ["F1", "F2"]


def test_assign_delays_flowful_minute_precision_two_in_same_bin():
    idx = _make_indexer(bin_minutes=15)
    base = datetime(2025, 1, 1, 9, 0, 0)
    # Two flights: 2 min and 7 min into bin 09:00-09:15
    flights_by_flow = {
        "flow": [
            {"flight_id": "A", "requested_dt": base + timedelta(minutes=2)},
            {"flight_id": "B", "requested_dt": base + timedelta(minutes=7)},
        ]
    }
    T = int(idx.num_time_bins)
    sched = [0] * (T + 1)
    b = idx.bin_of_datetime(base)
    sched[b] = 1
    sched[b + 1] = 1
    delays, realised = assign_delays_flowful(flights_by_flow, {"flow": sched}, idx)
    # First released in-bin, no delay
    assert delays["A"] == 0
    # Second released next bin: from 7 min to 15 min -> 8 minutes delay
    assert delays["B"] == 8

