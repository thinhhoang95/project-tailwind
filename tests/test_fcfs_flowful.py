import sys
from pathlib import Path

project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

import pytest

from parrhesia.fcfs.flowful import (
    _normalize_flight_spec,
    preprocess_flights_for_scheduler,
    assign_delays_flowful,
)
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


def test_spill_mode_overflow_bin_default_behavior():
    idx = _make_indexer(bin_minutes=30)
    T = int(idx.num_time_bins)
    base_bin = 6
    flights_by_flow = {
        "flow": [
            {"flight_id": f"F{i}", "requested_bin": base_bin}
            for i in range(5)
        ]
    }
    sched = [0] * (T + 1)
    sched[base_bin] = 1
    sched[T] = 4
    delays, realised = assign_delays_flowful(flights_by_flow, {"flow": sched}, idx)
    assert realised["F0"]["bin"] == base_bin
    for fid in ["F1", "F2", "F3", "F4"]:
        assert realised[fid]["bin"] == T


def test_spill_mode_dump_to_next_bin_assigns_single_bin():
    idx = _make_indexer(bin_minutes=15)
    T = int(idx.num_time_bins)
    base_bin = 10
    flights_by_flow = {
        "flow": [
            {"flight_id": f"F{i}", "requested_bin": base_bin}
            for i in range(4)
        ]
    }
    sched = [0] * (T + 1)
    sched[base_bin] = 1
    delays, realised = assign_delays_flowful(
        flights_by_flow,
        {"flow": sched},
        idx,
        spill_mode="dump_to_next_bin",
    )
    spill_bin = base_bin + 1
    assert realised["F0"]["bin"] == base_bin
    for fid in ["F1", "F2", "F3"]:
        assert realised[fid]["bin"] == spill_bin


def test_spill_mode_defined_rate_token_bucket_distribution():
    idx = _make_indexer(bin_minutes=30)
    T = int(idx.num_time_bins)
    base_bin = 8
    flights_by_flow = {
        "flow": [
            {"flight_id": f"F{i}", "requested_bin": base_bin}
            for i in range(6)
        ]
    }
    sched = [0] * (T + 1)
    sched[base_bin] = 1
    delays, realised = assign_delays_flowful(
        flights_by_flow,
        {"flow": sched},
        idx,
        spill_mode="defined_release_rate_for_spills",
        release_rate_for_spills=4.0,  # 4 flights/hour -> 2 per 30-min bin
    )
    expected_bins = {
        "F0": base_bin,
        "F1": base_bin + 1,
        "F2": base_bin + 1,
        "F3": base_bin + 2,
        "F4": base_bin + 2,
        "F5": base_bin + 3,
    }
    for fid, expected in expected_bins.items():
        assert realised[fid]["bin"] == expected


def test_defined_rate_requires_release_rate():
    idx = _make_indexer(bin_minutes=30)
    T = int(idx.num_time_bins)
    base_bin = 5
    flights_by_flow = {
        "flow": [
            {"flight_id": "F0", "requested_bin": base_bin},
            {"flight_id": "F1", "requested_bin": base_bin},
        ]
    }
    sched = [0] * (T + 1)
    sched[base_bin] = 0
    sched[T] = 2
    with pytest.raises(ValueError):
        assign_delays_flowful(
            flights_by_flow,
            {"flow": sched},
            idx,
            spill_mode="defined_release_rate_for_spills",
        )
