"""Tests for CountAPIWrapper ranking options."""

from pathlib import Path
import sys
import types
from types import MethodType, SimpleNamespace

import numpy as np

_repo_src = Path(__file__).resolve().parents[1] / "src"
if str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

if "networkx" not in sys.modules:
    sys.modules["networkx"] = types.ModuleType("networkx")

if "geopandas" not in sys.modules:
    _gpd = types.ModuleType("geopandas")

    class _GeoDataFrame:
        pass

    def _read_file(*_args, **_kwargs):
        return _GeoDataFrame()

    _gpd.GeoDataFrame = _GeoDataFrame
    _gpd.read_file = _read_file
    sys.modules["geopandas"] = _gpd

from server_tailwind.CountAPIWrapper import CountAPIWrapper


def _build_wrapper_with_mock_data() -> CountAPIWrapper:
    wrapper = CountAPIWrapper.__new__(CountAPIWrapper)

    wrapper._flight_list = SimpleNamespace(
        tv_id_to_idx={"TV_A": 0, "TV_B": 1, "TV_C": 2},
        flight_id_to_row={"F1": 0, "F2": 1, "F3": 2},
        num_time_bins_per_tv=3,
        time_bin_minutes=15,
    )

    wrapper._capacity_per_bin_matrix = None
    wrapper._hourly_capacity_by_tv = {}

    total_vector = np.array(
        [20, 20, 20, 5, 5, 5, 7, 7, 7],
        dtype=np.float32,
    )

    contrib_vectors = {
        0: np.array([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.float32),
        1: np.array([1, 1, 1, 2, 2, 2, 0, 0, 0], dtype=np.float32),
        2: np.array([18, 18, 18, 2, 2, 2, 7, 7, 7], dtype=np.float32),
    }

    wrapper._total_occupancy_vector = total_vector.copy()

    def fake_aggregate(self, rows_arr):
        if rows_arr is None:
            return self._total_occupancy_vector.copy()
        if rows_arr.size == 0:
            return np.zeros_like(self._total_occupancy_vector)
        acc = np.zeros_like(self._total_occupancy_vector)
        for idx in rows_arr:
            acc += contrib_vectors[int(idx)]
        return acc

    wrapper._aggregate_vector_for_rows = MethodType(fake_aggregate, wrapper)

    return wrapper


def _run_ranked_request(wrapper: CountAPIWrapper, rank_by: str):
    return wrapper._compute_flight_contrib_counts(
        traffic_volume_ids=None,
        from_time_str=None,
        to_time_str=None,
        flight_ids=["F1", "F2"],
        rank_by=rank_by,
        rolling_hour=False,
        top_k=3,
    )


def test_rank_by_flight_list_count_orders_by_contribution():
    wrapper = _build_wrapper_with_mock_data()
    result = _run_ranked_request(wrapper, "flight_list_count")

    assert result["metadata"]["ranked_tv_ids"] == ["TV_B", "TV_A", "TV_C"]
    assert result["metadata"]["rank_by"] == "flight_list_count"
    assert result["metadata"]["total_flights_considered"] == 2


def test_rank_by_flight_list_relative_orders_by_ratio():
    wrapper = _build_wrapper_with_mock_data()
    result = _run_ranked_request(wrapper, "flight_list_relative")

    assert result["metadata"]["ranked_tv_ids"] == ["TV_B", "TV_A", "TV_C"]
    assert result["metadata"]["rank_by"] == "flight_list_relative"
