import asyncio
import json
from pathlib import Path
from typing import Dict, List
import sys

# Ensure repo src/ is on the import path
_repo_src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_repo_src))

from parrhesia.flows.flow_pipeline import build_global_flows

# Provide a minimal geopandas stub to satisfy imports in server code during tests
import types as _types
if "geopandas" not in sys.modules:
    _gp = _types.ModuleType("geopandas")
    class _GeoDataFrame:  # minimal placeholder
        pass
    def _read_file(*args, **kwargs):
        # Return a minimal placeholder; any usage during tests is guarded/overridden
        return _GeoDataFrame()
    _gp.GeoDataFrame = _GeoDataFrame
    _gp.read_file = _read_file
    sys.modules["geopandas"] = _gp

from server_tailwind.airspace.airspace_api_wrapper import AirspaceAPIWrapper


class SimpleFlightList:
    def __init__(self, tv_ids: List[str], time_bin_minutes: int = 15, bins_per_tv: int = 96):
        self.time_bin_minutes = int(time_bin_minutes)
        self.tv_id_to_idx = {str(tv): i for i, tv in enumerate(tv_ids)}
        self.idx_to_tv_id = {i: str(tv) for tv, i in self.tv_id_to_idx.items()}
        self.num_time_bins_per_tv = int(bins_per_tv)
        self.flight_metadata: Dict[str, Dict] = {}

    @property
    def num_tvtws(self) -> int:
        return int(len(self.tv_id_to_idx) * int(self.num_time_bins_per_tv))

    def get_flight_tv_sequence_indices(self, fid: str):
        meta = self.flight_metadata.get(str(fid)) or {}
        intervals = meta.get("occupancy_intervals", []) or []
        if not intervals:
            return []
        try:
            order = sorted(range(len(intervals)), key=lambda i: float(intervals[i].get("entry_time_s", 0.0)))
        except Exception:
            order = list(range(len(intervals)))
        seq = []
        for i in order:
            tvtw_index = int(intervals[i].get("tvtw_index", 0))
            tv_idx = int(tvtw_index) // int(self.num_time_bins_per_tv)
            if not seq or seq[-1] != tv_idx:
                seq.append(tv_idx)
        import numpy as np
        return np.asarray(seq, dtype=int)

    def get_footprints_for_flights(self, flight_ids: List[str], hotspot_tv_index: int = None):  # type: ignore[override]
        import numpy as np
        fps = []
        for fid in flight_ids:
            seq = self.get_flight_tv_sequence_indices(fid).tolist()
            if hotspot_tv_index is not None:
                cut = None
                for i, v in enumerate(seq):
                    if int(v) == int(hotspot_tv_index):
                        cut = i
                        break
                if cut is not None:
                    seq = seq[: cut + 1]
            fps.append(np.unique(np.asarray(seq, dtype=int)))
        return fps


def _make_interval(tv_idx: int, bin_offset: int, bins_per_tv: int) -> Dict[str, int]:
    tvtw_index = tv_idx * bins_per_tv + bin_offset
    return {
        "tvtw_index": int(tvtw_index),
        "entry_time_s": int(bin_offset) * 60,
        "exit_time_s": int(bin_offset + 1) * 60,
    }


def _build_stub_flight_list() -> SimpleFlightList:
    # Synthetic 6 flights: 3 in each group
    # Group 1: TVA -> TVB -> TVC (trim at TVB)
    # Group 2: TVA -> TVD -> TVB (trim at TVB)
    bins_per_tv = 24 * 4  # 15-min bins
    fl = SimpleFlightList(["TVA", "TVB", "TVC", "TVD"], time_bin_minutes=15, bins_per_tv=bins_per_tv)

    def add_flight(fid: str, seq: List[int]):
        intervals = []
        for i, tv_idx in enumerate(seq):
            intervals.append(_make_interval(tv_idx, bin_offset=2 * i, bins_per_tv=bins_per_tv))
        fl.flight_metadata[fid] = {
            "takeoff_time": "2023-10-03T00:00:00",
            "origin": "AAA",
            "destination": "BBB",
            "distance": 1000,
            "occupancy_intervals": intervals,
        }

    # Group 1
    add_flight("A1", [0, 1, 2])
    add_flight("A2", [0, 1, 2])
    add_flight("A3", [0, 1, 2])
    # Group 2
    add_flight("B1", [0, 3, 1])
    add_flight("B2", [0, 3, 1])
    add_flight("B3", [0, 3, 1])
    return fl


def _make_wrapper_with_fl(fl) -> AirspaceAPIWrapper:
    # Build a wrapper and inject the synthetic FlightList and a minimal evaluator stub
    w = AirspaceAPIWrapper()
    w._flight_list = fl
    # Minimal evaluator stub with required attributes
    class _E:
        def __init__(self, fl_):
            self.time_bin_minutes = fl_.time_bin_minutes
            self.tv_id_to_idx = fl_.tv_id_to_idx
            self.hourly_capacity_by_tv = {}

    w._evaluator = _E(fl)
    return w


def test_flow_extraction_pipeline_matches_build_global_flows(tmp_path: Path):
    fl = _build_stub_flight_list()

    traffic_volume_id = "TVB"
    flight_ids = ["A1", "A2", "A3", "B1", "B2", "B3"]

    # Expected assignments via direct pipeline call
    expected = build_global_flows(
        flight_list=fl,
        union_flight_ids=flight_ids,
        hotspots=[traffic_volume_id],
        trim_policy="earliest_hotspot",
        leiden_params={"threshold": 0.8, "resolution": 1.0, "seed": 123},
        direction_opts={"mode": "coord_cosine"},
    )

    w = _make_wrapper_with_fl(fl)

    async def _run():
        return await w.get_flow_extraction(
            traffic_volume_id=traffic_volume_id,
            ref_time_str="08:00:00",
            threshold=0.8,
            resolution=1.0,
            flight_ids=",".join(flight_ids),
            seed=123,
        )

    res = asyncio.get_event_loop().run_until_complete(_run())
    got = res.get("communities", {})
    # Exact mapping match expected
    assert got == expected


def test_flow_extraction_legacy_returns_results(tmp_path: Path):
    fl = _build_stub_flight_list()

    w = _make_wrapper_with_fl(fl)
    traffic_volume_id = "TVB"
    flight_ids = ["A1", "A2", "A3", "B1", "B2", "B3"]

    async def _run():
        return await w.get_flow_extraction_legacy(
            traffic_volume_id=traffic_volume_id,
            ref_time_str="08:00:00",
            threshold=0.8,
            resolution=1.0,
            flight_ids=",".join(flight_ids),
            seed=123,
        )

    res = asyncio.get_event_loop().run_until_complete(_run())
    comm = res.get("communities", {})
    # Covers all flights
    assert set(comm.keys()) == set(flight_ids)
    # Expect two distinct labels corresponding to the two groups
    labels = set(comm.values())
    assert len(labels) == 2
