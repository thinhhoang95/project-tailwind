import asyncio
from datetime import datetime
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if "geopandas" not in sys.modules:
    geopandas_stub = types.ModuleType("geopandas")
    geopandas_stub.read_file = lambda *args, **kwargs: None
    geopandas_stub.GeoDataFrame = object  # minimal placeholder for type hints
    sys.modules["geopandas"] = geopandas_stub

from server_tailwind.query.QueryAPIWrapper import QueryAPIWrapper


class DummyFlightList:
    def __init__(self):
        self.time_bin_minutes = 15
        self.num_time_bins_per_tv = 4
        self.tv_id_to_idx = {"TVA": 0, "TVB": 1}
        self.idx_to_tv_id = {0: "TVA", 1: "TVB"}
        self.flight_ids = ["F1", "F2", "F3"]
        self.num_flights = len(self.flight_ids)
        self.num_tvtws = len(self.tv_id_to_idx) * self.num_time_bins_per_tv

        def _interval(tvtw_index: int, entry: int, exit_: int) -> dict:
            return {
                "tvtw_index": tvtw_index,
                "entry_time_s": entry,
                "exit_time_s": exit_,
            }

        self.flight_metadata = {
            "F1": {
                "takeoff_time": datetime(2024, 1, 1, 6, 0, 0),
                "origin": "AAA",
                "destination": "BBB",
                "distance": 1000,
                "occupancy_intervals": [
                    _interval(0, 0, 600),
                    _interval(1, 600, 1200),
                    _interval(4, 1500, 2100),
                ],
            },
            "F2": {
                "takeoff_time": datetime(2024, 1, 1, 7, 0, 0),
                "origin": "CCC",
                "destination": "DDD",
                "distance": 800,
                "occupancy_intervals": [
                    _interval(5, 900, 1500),
                ],
            },
            "F3": {
                "takeoff_time": datetime(2024, 1, 1, 5, 30, 0),
                "origin": "AAA",
                "destination": "EEE",
                "distance": 900,
                "occupancy_intervals": [
                    _interval(0, 0, 300),
                    _interval(4, 600, 1200),
                    _interval(6, 1350, 1650),
                ],
            },
        }

        matrix = np.zeros((self.num_flights, self.num_tvtws), dtype=np.float32)
        for row_idx, flight_id in enumerate(self.flight_ids):
            for interval in self.flight_metadata[flight_id]["occupancy_intervals"]:
                matrix[row_idx, int(interval["tvtw_index"]) ] = 1.0
        self.occupancy_matrix = sparse.csr_matrix(matrix)

    def get_total_occupancy_by_tvtw(self) -> np.ndarray:
        return np.asarray(self.occupancy_matrix.sum(axis=0)).ravel()


class DummyResources:
    def __init__(self, flight_list: DummyFlightList):
        self.flight_list = flight_list
        self.indexer = None
        self.capacity_per_bin_matrix = np.full((2, flight_list.num_time_bins_per_tv), 10.0, dtype=np.float32)
        self.hourly_capacity_by_tv = {"TVA": {0: 10.0}, "TVB": {0: 10.0}}
        self.traffic_volumes_gdf = None


@pytest.fixture
def wrapper() -> QueryAPIWrapper:
    flight_list = DummyFlightList()
    resources = DummyResources(flight_list)
    return QueryAPIWrapper(resources=resources)


def test_cross_query(wrapper: QueryAPIWrapper):
    payload = {"query": {"type": "cross", "tv": "TVA"}}
    result = asyncio.run(wrapper.evaluate(payload))
    assert result["flight_ids"] == ["F1", "F3"]


def test_sequence_query(wrapper: QueryAPIWrapper):
    payload = {
        "query": {
            "type": "sequence",
            "steps": [
                {"type": "cross", "tv": "TVA"},
                {"type": "cross", "tv": "TVB"},
            ],
            "select": "flight_ids",
            "order_by": "takeoff_time",
        }
    }
    result = asyncio.run(wrapper.evaluate(payload))
    assert result["flight_ids"] == ["F3", "F1"]


def test_flight_scope(wrapper: QueryAPIWrapper):
    payload = {
        "query": {"type": "cross", "tv": "TVA"},
        "options": {"flight_ids": ["F1"]},
    }
    result = asyncio.run(wrapper.evaluate(payload))
    assert result["flight_ids"] == ["F1"]
    assert result["metadata"].get("scope_flights") == 1


def test_flight_scope_unknown_id(wrapper: QueryAPIWrapper):
    payload = {
        "query": {"type": "cross", "tv": "TVA"},
        "options": {"flight_ids": ["NOPE"]},
    }
    with pytest.raises(ValueError, match="Unknown flight id"):
        asyncio.run(wrapper.evaluate(payload))
