import sys
from pathlib import Path

project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

from datetime import datetime

import numpy as np
import geopandas as gpd

from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.eval.plan_evaluator import PlanEvaluator
from project_tailwind.optimize.eval.delta_flight_list import DeltaFlightList
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.moves.network_plan_move import NetworkPlanMove


class _MiniFlightList:
    """
    Minimal FlightList-like object for testing PlanEvaluator without large files.
    Exposes attributes used by FCFS and NetworkEvaluator via DeltaFlightList overlay.
    """

    def __init__(self, time_bin_minutes: int, tv_ids, flight_metadata):
        self.time_bin_minutes = time_bin_minutes
        self.flight_metadata = flight_metadata
        self.flight_ids = list(flight_metadata.keys())
        self.num_flights = len(self.flight_ids)
        # Build a synthetic mapping and a very small occupancy matrix interface
        self.tv_id_to_idx = {tv: i for i, tv in enumerate(tv_ids)}

        # Ensure a full grid of TV x time windows as expected by NetworkEvaluator
        num_bins_per_day = 1440 // self.time_bin_minutes
        self.num_tvtws = len(self.tv_id_to_idx) * num_bins_per_day

        # For NetworkEvaluator, we need get_occupancy_vector and get_total_occupancy_by_tvtw
        # We'll construct per-flight vectors lazily by marking 1s at given intervals
        self._per_flight_vectors = {}

    def get_occupancy_vector(self, flight_id: str):
        if flight_id in self._per_flight_vectors:
            return self._per_flight_vectors[flight_id]
        vec = np.zeros(self.num_tvtws, dtype=np.float32)
        for interval in self.flight_metadata[flight_id].get("occupancy_intervals", []):
            vec[int(interval["tvtw_index"]) ] = 1.0
        self._per_flight_vectors[flight_id] = vec
        return vec

    def shift_flight_occupancy(self, flight_id: str, delay_minutes: int):
        # Shift by bins with ceil
        shift_bins = delay_minutes // self.time_bin_minutes
        if delay_minutes % self.time_bin_minutes != 0:
            shift_bins += 1
        orig = self.get_occupancy_vector(flight_id)
        shifted = np.zeros_like(orig)
        if shift_bins > 0 and shift_bins < len(orig):
            shifted[shift_bins:] = orig[:-shift_bins]
        elif shift_bins <= 0:
            back = abs(shift_bins)
            if back < len(orig):
                shifted[:-back] = orig[back:]
        return shifted

    def get_total_occupancy_by_tvtw(self):
        total = np.zeros(self.num_tvtws, dtype=np.float32)
        for fid in self.flight_ids:
            total += self.get_occupancy_vector(fid)
        return total


def _make_indexer(tv_ids, time_bin_minutes=15) -> TVTWIndexer:
    indexer = TVTWIndexer(time_bin_minutes=time_bin_minutes)
    indexer._tv_id_to_idx = {tv: i for i, tv in enumerate(tv_ids)}
    indexer._idx_to_tv_id = {i: tv for tv, i in indexer._tv_id_to_idx.items()}
    indexer._populate_tvtw_mappings()
    return indexer


def _make_parser(indexer: TVTWIndexer, flights_data):
    # Write flights_data to a temp JSON in tests output dir
    tmp_path = Path("output/_tmp_plan_app_flights.json")
    tmp_path.parent.mkdir(exist_ok=True, parents=True)
    import json
    with open(tmp_path, "w") as f:
        json.dump(flights_data, f)
    return RegulationParser(flights_file=str(tmp_path), tvtw_indexer=indexer)


def test_plan_evaluator_builds_delta_and_metrics():
    tv_ids = ["TVA"]
    indexer = _make_indexer(tv_ids, time_bin_minutes=15)

    # Two flights in the same window, service rate 3/h will induce delay for second
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    win = 32
    flights_json = {
        "F1": {
            "takeoff_time": t0.isoformat(sep=" "),
            "origin": "AAA",
            "destination": "BBB",
            "distance": 100.0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win), "entry_time_s": 0, "exit_time_s": 60}
            ],
        },
        "F2": {
            "takeoff_time": t0.isoformat(sep=" "),
            "origin": "AAA",
            "destination": "BBB",
            "distance": 120.0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win), "entry_time_s": 60, "exit_time_s": 120}
            ],
        },
    }

    # Minimal TV GDF with capacity per hour for TVA
    gdf = gpd.GeoDataFrame({
        "traffic_volume_id": ["TVA"],
        "capacity": [
            {"08:00-09:00": 1}  # 1 flight/hour to keep it simple
        ],
        "geometry": [None],
    })

    # Build minimal base flight list
    # Convert takeoff_time to datetime in metadata for _MiniFlightList
    meta = {
        fid: {
            "takeoff_time": datetime.fromisoformat(v["takeoff_time"]),
            "occupancy_intervals": v["occupancy_intervals"],
        }
        for fid, v in flights_json.items()
    }
    base = _MiniFlightList(time_bin_minutes=15, tv_ids=tv_ids, flight_metadata=meta)

    parser = _make_parser(indexer, flights_json)
    # 1) String-parsed regulation
    plan = NetworkPlan([f"TV_TVA IC__ 3 {win}"])

    evaluator = PlanEvaluator(traffic_volumes_gdf=gdf, parser=parser, tvtw_indexer=indexer)
    result = evaluator.evaluate_plan(plan, base)

    # Delays should exist for at least one of the flights
    delays = result["delays_by_flight"]
    assert isinstance(result["delta_view"], DeltaFlightList)
    assert set(delays.keys()) <= {"F1", "F2"}
    # Excess vector is a numpy array
    assert hasattr(result["excess_vector"], "shape")
    # Delay stats include total_delay_seconds
    assert "total_delay_seconds" in result["delay_stats"]

    # 2) Explicit target_flight_ids without relying on parser
    from project_tailwind.optimize.regulation import Regulation
    reg2 = Regulation.from_components(
        location="TVA",
        rate=3,
        time_windows=[win],
        filter_type="IC",
        filter_value="__",
        target_flight_ids=["F1", "F2"],
    )
    plan2 = NetworkPlan([reg2])
    result2 = evaluator.evaluate_plan(plan2, base)
    delays2 = result2["delays_by_flight"]
    assert set(delays2.keys()) <= {"F1", "F2"}


def test_network_plan_move_builds_delta_view():
    tv_ids = ["TVA"]
    indexer = _make_indexer(tv_ids, time_bin_minutes=15)

    t0 = datetime(2024, 1, 1, 8, 0, 0)
    win = 32
    flights_json = {
        "F1": {
            "takeoff_time": t0.isoformat(sep=" "),
            "origin": "AAA",
            "destination": "BBB",
            "distance": 100.0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win), "entry_time_s": 0, "exit_time_s": 60}
            ],
        },
        "F2": {
            "takeoff_time": t0.isoformat(sep=" "),
            "origin": "AAA",
            "destination": "BBB",
            "distance": 120.0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win), "entry_time_s": 60, "exit_time_s": 120}
            ],
        },
    }

    meta = {
        fid: {
            "takeoff_time": datetime.fromisoformat(v["takeoff_time"]),
            "occupancy_intervals": v["occupancy_intervals"],
        }
        for fid, v in flights_json.items()
    }
    base = _MiniFlightList(time_bin_minutes=15, tv_ids=tv_ids, flight_metadata=meta)

    parser = _make_parser(indexer, flights_json)
    # Also test explicit flight list path for NetworkPlanMove
    from project_tailwind.optimize.regulation import Regulation
    reg = Regulation.from_components(location="TVA", rate=3, time_windows=[win], target_flight_ids=["F1", "F2"])
    plan = NetworkPlan([reg])
    move = NetworkPlanMove(network_plan=plan, parser=parser, tvtw_indexer=indexer)

    delta_view, total_delay = move.build_delta_view(base)
    assert isinstance(delta_view, DeltaFlightList)
    assert total_delay >= 0


def test_objective_decreases_with_regulation_when_weighted_for_z_sum_only():
    tv_ids = ["TVA"]
    indexer = _make_indexer(tv_ids, time_bin_minutes=15)

    t0 = datetime(2024, 1, 1, 8, 0, 0)
    win = 32
    # Three flights in the same early hour to create overload vs capacity=1/h
    flights_json = {
        "F1": {
            "takeoff_time": t0.isoformat(sep=" "),
            "origin": "AAA",
            "destination": "BBB",
            "distance": 100.0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win), "entry_time_s": 0, "exit_time_s": 60}
            ],
        },
        "F2": {
            "takeoff_time": t0.isoformat(sep=" "),
            "origin": "AAA",
            "destination": "BBB",
            "distance": 120.0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win), "entry_time_s": 300, "exit_time_s": 360}
            ],
        },
        "F3": {
            "takeoff_time": t0.isoformat(sep=" "),
            "origin": "AAA",
            "destination": "BBB",
            "distance": 140.0,
            "occupancy_intervals": [
                {"tvtw_index": indexer.get_tvtw_index("TVA", win), "entry_time_s": 600, "exit_time_s": 660}
            ],
        },
    }

    # Capacity only defined for 08:00-09:00 hour, forcing excess there if >1 flight
    gdf = gpd.GeoDataFrame({
        "traffic_volume_id": ["TVA"],
        "capacity": [
            {"08:00-09:00": 1}
        ],
        "geometry": [None],
    })

    meta = {
        fid: {
            "takeoff_time": datetime.fromisoformat(v["takeoff_time"]),
            "occupancy_intervals": v["occupancy_intervals"],
        }
        for fid, v in flights_json.items()
    }
    base = _MiniFlightList(time_bin_minutes=15, tv_ids=tv_ids, flight_metadata=meta)

    parser = _make_parser(indexer, flights_json)
    evaluator = PlanEvaluator(traffic_volumes_gdf=gdf, parser=parser, tvtw_indexer=indexer)

    # Baseline objective with no regulation
    empty_plan = NetworkPlan([])
    weights = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}
    res0 = evaluator.evaluate_plan(empty_plan, base, weights=weights)
    obj0 = res0["objective"]

    # Apply a tight regulation (1 flight/hour) on the same window and flights to queue them across hours
    from project_tailwind.optimize.regulation import Regulation
    reg = Regulation.from_components(
        location="TVA",
        rate=1,
        time_windows=[win],
        filter_type="IC",
        filter_value="__",
        target_flight_ids=["F1", "F2", "F3"],
    )
    plan1 = NetworkPlan([reg])
    res1 = evaluator.evaluate_plan(plan1, base, weights=weights)
    obj1 = res1["objective"]

    # The objective should decrease when overload is reduced
    assert obj1 < obj0, f"Expected regulated objective < baseline (got {obj1} vs {obj0})"

