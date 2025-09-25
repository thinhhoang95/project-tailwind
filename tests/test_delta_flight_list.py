from datetime import datetime, timedelta

import numpy as np

from project_tailwind.optimize.eval.delta_flight_list import DeltaFlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


class _StubFlightList:
    def __init__(self, *, time_bin_minutes: int, indexer: TVTWIndexer, metadata):
        self.time_bin_minutes = int(time_bin_minutes)
        self._indexer = indexer
        self.tv_id_to_idx = dict(indexer.tv_id_to_idx)
        self.num_tvtws = len(self.tv_id_to_idx) * indexer.num_time_bins
        self.flight_metadata = metadata
        self.flight_ids = list(metadata.keys())
        self.num_flights = len(self.flight_ids)
        self._vectors: dict[str, np.ndarray] = {}

    @property
    def indexer(self) -> TVTWIndexer:
        return self._indexer

    def get_occupancy_vector(self, flight_id: str) -> np.ndarray:
        if flight_id in self._vectors:
            return self._vectors[flight_id]
        vec = np.zeros(self.num_tvtws, dtype=np.float32)
        meta = self.flight_metadata.get(flight_id, {}) or {}
        for interval in meta.get("occupancy_intervals", []) or []:
            try:
                idx = int(interval.get("tvtw_index"))
            except Exception:
                continue
            if 0 <= idx < self.num_tvtws:
                vec[idx] = 1.0
        self._vectors[flight_id] = vec
        return vec

    def shift_flight_occupancy(self, flight_id: str, delay_minutes: int) -> np.ndarray:
        shift_bins = delay_minutes // self.time_bin_minutes
        if delay_minutes % self.time_bin_minutes != 0:
            shift_bins += 1
        orig = self.get_occupancy_vector(flight_id)
        shifted = np.zeros_like(orig)
        if shift_bins > 0:
            if shift_bins < len(orig):
                shifted[shift_bins:] = orig[:-shift_bins]
        elif shift_bins < 0:
            back = abs(shift_bins)
            if back < len(orig):
                shifted[:-back] = orig[back:]
        else:
            shifted[:] = orig
        return shifted

    def get_total_occupancy_by_tvtw(self) -> np.ndarray:
        total = np.zeros(self.num_tvtws, dtype=np.float32)
        for fid in self.flight_ids:
            total += self.get_occupancy_vector(fid)
        return total


def _make_indexer(time_bin_minutes: int = 15) -> TVTWIndexer:
    indexer = TVTWIndexer(time_bin_minutes=time_bin_minutes)
    indexer._tv_id_to_idx = {"TVA": 0}
    indexer._idx_to_tv_id = {0: "TVA"}
    indexer._populate_tvtw_mappings()
    return indexer


def test_iter_hotspot_crossings_applies_delay_shift_and_filters_active_windows():
    indexer = _make_indexer()
    base_takeoff = datetime(2024, 1, 1, 8, 0, 0)
    original_bin = 10
    tvtw_index = indexer.get_tvtw_index("TVA", original_bin)
    metadata = {
        "F1": {
            "takeoff_time": base_takeoff,
            "occupancy_intervals": [
                {"tvtw_index": tvtw_index, "entry_time_s": 120},
            ],
        }
    }
    base = _StubFlightList(time_bin_minutes=15, indexer=indexer, metadata=metadata)
    delta = DeltaFlightList(base, {"F1": 20})

    events = list(delta.iter_hotspot_crossings(["TVA"]))
    assert len(events) == 1
    fid, tv_id, entry_dt, time_idx = events[0]
    assert fid == "F1"
    assert tv_id == "TVA"
    assert time_idx == original_bin + 2  # ceil(20 / 15) = 2 bin shift
    expected_entry = base_takeoff + timedelta(minutes=20) + timedelta(seconds=120)
    assert entry_dt == expected_entry

    allowed = list(delta.iter_hotspot_crossings(["TVA"], active_windows=[original_bin + 2]))
    blocked = list(delta.iter_hotspot_crossings(["TVA"], active_windows=[original_bin]))
    assert allowed == events
    assert blocked == []

    allowed_map = list(delta.iter_hotspot_crossings(["TVA"], active_windows={"TVA": [original_bin + 2]}))
    blocked_map = list(delta.iter_hotspot_crossings(["TVA"], active_windows={"TVA": [original_bin]}))
    assert allowed_map == events
    assert blocked_map == []
