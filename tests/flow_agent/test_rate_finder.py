from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from parrhesia.flow_agent.rate_finder import RateFinder, RateFinderConfig
from parrhesia.flow_agent.state import PlanState
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator


class DummyFlightList:
    """Lightweight FlightList substitute for rate finder tests."""

    def __init__(self, *, indexer: TVTWIndexer, flight_metadata: Dict[str, Dict[str, object]]):
        self.indexer = indexer
        self.time_bin_minutes = indexer.time_bin_minutes
        self.tv_id_to_idx = indexer.tv_id_to_idx
        self.num_tvtws = len(self.tv_id_to_idx) * indexer.num_time_bins
        self.flight_metadata = flight_metadata
        self.flight_ids = list(flight_metadata.keys())
        self.num_flights = len(self.flight_ids)
        self.occupancy_matrix = self._build_sparse()

    def _build_sparse(self):
        rows = []
        cols = []
        data = []
        for row, fid in enumerate(self.flight_ids):
            for interval in self.flight_metadata[fid]["occupancy_intervals"]:
                cols.append(int(interval["tvtw_index"]))
                rows.append(row)
                data.append(1.0)
        shape = (self.num_flights, self.num_tvtws)
        return sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)

    def get_occupancy_vector(self, flight_id: str) -> np.ndarray:
        row = self.flight_ids.index(flight_id)
        return self.occupancy_matrix.getrow(row).toarray().ravel()

    def shift_flight_occupancy(self, flight_id: str, delay_minutes: int) -> np.ndarray:
        bins = int(math.ceil(max(delay_minutes, 0) / self.time_bin_minutes))
        vec = self.get_occupancy_vector(flight_id)
        if bins <= 0:
            return vec.copy()
        shifted = np.zeros_like(vec)
        if bins < len(vec):
            shifted[bins:] = vec[:-bins]
        return shifted

    def get_total_occupancy_by_tvtw(self) -> np.ndarray:
        return np.asarray(self.occupancy_matrix.sum(axis=0)).ravel()

    def iter_hotspot_crossings(self, hotspot_ids, active_windows=None):
        window_map = None
        if isinstance(active_windows, dict):
            window_map = {k: set(int(x) for x in v) for k, v in active_windows.items()}
        hotspots = set(str(h) for h in hotspot_ids)
        for fid, meta in self.flight_metadata.items():
            takeoff = meta["takeoff_time"]
            for interval in meta["occupancy_intervals"]:
                idx = int(interval["tvtw_index"])
                decoded = self.indexer.get_tvtw_from_index(idx)
                if not decoded:
                    continue
                tv_id, time_idx = decoded
                if tv_id not in hotspots:
                    continue
                if window_map is not None and int(time_idx) not in window_map.get(tv_id, set()):
                    continue
                entry_s = float(interval.get("entry_time_s", 0.0))
                entry_dt = takeoff + timedelta(seconds=entry_s)
                yield (fid, tv_id, entry_dt, int(time_idx))

    def copy(self):
        # NetworkEvaluator snapshots the original baseline via copy(); tests do not mutate.
        return self


@pytest.fixture(scope="module")
def rate_finder_env() -> Dict[str, object]:
    indexer = TVTWIndexer(time_bin_minutes=30)
    indexer._tv_id_to_idx = {"TV1": 0}
    indexer._idx_to_tv_id = {0: "TV1"}
    indexer._populate_tvtw_mappings()

    window_bins = (16, 18)  # covers 08:00-09:00
    tv_index_a = indexer.get_tvtw_index("TV1", window_bins[0])
    tv_index_b = indexer.get_tvtw_index("TV1", window_bins[0] + 1)

    t0 = datetime(2024, 1, 1, 8, 0, 0)
    flights = {
        "F1": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": int(tv_index_a), "entry_time_s": 0, "exit_time_s": 300}
            ],
        },
        "F2": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": int(tv_index_a), "entry_time_s": 120, "exit_time_s": 420}
            ],
        },
        "F3": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": int(tv_index_b), "entry_time_s": 60, "exit_time_s": 360}
            ],
        },
    }

    flight_list = DummyFlightList(indexer=indexer, flight_metadata=flights)

    gdf = gpd.GeoDataFrame(
        {
            "traffic_volume_id": ["TV1"],
            "capacity": [{"08:00-09:00": 1}],
            "geometry": [None],
        }
    )

    evaluator = NetworkEvaluator(gdf, flight_list)
    config = RateFinderConfig(rate_grid=(math.inf, 3.0, 2.0, 1.0), passes=2, max_eval_calls=20)
    rf = RateFinder(evaluator=evaluator, flight_list=flight_list, indexer=indexer, config=config)

    flows = {
        "A": ("F1", "F2"),
        "B": ("F3",),
    }
    plan_state = PlanState()
    return {
        "rate_finder": rf,
        "plan_state": plan_state,
        "control_volume": "TV1",
        "window_bins": window_bins,
        "flows": flows,
        "config": config,
    }


def test_rate_finder_per_flow_improves_or_matches_baseline(rate_finder_env):
    rf = rate_finder_env["rate_finder"]
    plan_state = rate_finder_env["plan_state"]
    rates, delta_j, info = rf.find_rates(
        plan_state=plan_state,
        control_volume_id=rate_finder_env["control_volume"],
        window_bins=rate_finder_env["window_bins"],
        flows=rate_finder_env["flows"],
        mode="per_flow",
    )
    assert set(rates.keys()) == set(rate_finder_env["flows"].keys())
    assert delta_j <= 1e-6  # no regression
    assert info["eval_calls"] <= rate_finder_env["config"].max_eval_calls
    assert info["timing_seconds"] < 0.5
    assert info["passes_ran"] >= 1
    assert info["aggregate_delays_size"] >= 0


def test_rate_finder_reuses_caches(rate_finder_env):
    rf = rate_finder_env["rate_finder"]
    plan_state = rate_finder_env["plan_state"]

    # First run to populate caches
    rf.find_rates(
        plan_state=plan_state,
        control_volume_id=rate_finder_env["control_volume"],
        window_bins=rate_finder_env["window_bins"],
        flows=rate_finder_env["flows"],
        mode="per_flow",
    )

    rates, delta_j, info = rf.find_rates(
        plan_state=plan_state,
        control_volume_id=rate_finder_env["control_volume"],
        window_bins=rate_finder_env["window_bins"],
        flows=rate_finder_env["flows"],
        mode="per_flow",
    )
    assert set(rates.keys()) == set(rate_finder_env["flows"].keys())
    assert info["cache_hits"] > 0
    assert info["eval_calls"] == 0
    assert math.isclose(delta_j, info["delta_j"])  # consistency check
