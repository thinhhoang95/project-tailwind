import json
from pathlib import Path

import geopandas as gpd  # type: ignore
import numpy as np

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.features import FlightFeatures


def _write_mock_inputs(tmp_path: Path):
    # Minimal TV indexer for two TVs and 30-min bins (48 bins per day)
    tvtw_indexer = {
        "time_bin_minutes": 30,
        "tv_id_to_idx": {"TV_A": 0, "TV_B": 1},
    }

    # Build a tiny occupancy file with 3 flights
    # Flight F1: traverses TV_A hours 6 and 7 (4 bins total)
    # Flight F2: traverses TV_A hour 7, TV_B hour 7
    # Flight F3: traverses TV_B hour 10 only
    def tw(hour: int, bin_in_hour: int) -> int:
        return hour * 2 + bin_in_hour

    f1_intervals = [
        {"tvtw_index": tw(6, 0), "entry_time_s": 0, "exit_time_s": 0},
        {"tvtw_index": tw(6, 1), "entry_time_s": 0, "exit_time_s": 0},
        {"tvtw_index": tw(7, 0), "entry_time_s": 0, "exit_time_s": 0},
        {"tvtw_index": tw(7, 1), "entry_time_s": 0, "exit_time_s": 0},
    ]

    f2_intervals = [
        {"tvtw_index": tw(7, 0), "entry_time_s": 0, "exit_time_s": 0},
        {"tvtw_index": 48 + tw(7, 1), "entry_time_s": 0, "exit_time_s": 0},  # TV_B offset
    ]

    f3_intervals = [
        {"tvtw_index": 48 + tw(10, 0), "entry_time_s": 0, "exit_time_s": 0},
        {"tvtw_index": 48 + tw(10, 1), "entry_time_s": 0, "exit_time_s": 0},
        # Ensure matrix shape covers full day across both TVs (2 * 48 = 96 bins)
        {"tvtw_index": 48 + 47, "entry_time_s": 0, "exit_time_s": 0},
    ]

    flights = {
        "F1": {
            "takeoff_time": "2024-01-01 06:00:00",
            "origin": "AAA",
            "destination": "BBB",
            "distance": 100.0,
            "occupancy_intervals": f1_intervals,
        },
        "F2": {
            "takeoff_time": "2024-01-01 07:00:00",
            "origin": "AAA",
            "destination": "CCC",
            "distance": 120.0,
            "occupancy_intervals": f2_intervals,
        },
        "F3": {
            "takeoff_time": "2024-01-01 10:00:00",
            "origin": "DDD",
            "destination": "EEE",
            "distance": 90.0,
            "occupancy_intervals": f3_intervals,
        },
    }

    occ_path = tmp_path / "occ.json"
    idx_path = tmp_path / "idx.json"
    with occ_path.open("w", encoding="utf-8") as f:
        json.dump(flights, f)
    with idx_path.open("w", encoding="utf-8") as f:
        json.dump(tvtw_indexer, f)

    # Build a minimal GeoDataFrame with capacities
    # TV_A heavily constrained at hours 6 and 7 so they overload
    # TV_B constrained at hour 7 but hour 10 is ample
    data = [
        {
            "traffic_volume_id": "TV_A",
            "capacity": {"06:00-07:00": 1, "07:00-08:00": 1},
        },
        {
            "traffic_volume_id": "TV_B",
            "capacity": {"07:00-08:00": 1, "10:00-11:00": 10},
        },
    ]
    # geopandas requires a geometry column; we can supply empty geometries
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy([0, 1], [0, 1]))

    return occ_path, idx_path, gdf


def test_flight_features_basic(tmp_path):
    occ_path, idx_path, tvs_gdf = _write_mock_inputs(tmp_path)

    fl = FlightList(str(occ_path), str(idx_path))
    ev = NetworkEvaluator(tvs_gdf, fl)

    feats = FlightFeatures(fl, ev, overload_threshold=0.0)

    # Footprints
    assert feats.get_footprint("F1") == {"TV_A"}
    assert feats.get_footprint("F2") == {"TV_A", "TV_B"}
    assert feats.get_footprint("F3") == {"TV_B"}

    # Multiplicity: count of overloaded (tv, hour)
    # Overloads expected:
    #   TV_A @ 6 and 7 -> F1 contributes 2; F2 contributes 1 (hour 7)
    #   TV_B @ 7 might overload depending on F2; TV_B @ 10 should not overload
    m1 = feats.multiplicity("F1")
    m2 = feats.multiplicity("F2")
    m3 = feats.multiplicity("F3")
    assert m1 >= 1
    assert m2 >= 1
    assert m3 >= 0

    # Slack valley is non-negative
    assert feats.slack_valley("F1") >= 0.0
    assert feats.slack_valley("F2") >= 0.0
    assert feats.slack_valley("F3") >= 0.0

    # Ranking using F1 as seed should favor F1 or F2 due to similarity/multiplicity
    seed = feats.compute_seed_footprint(["F1"])  # {TV_A}
    ranked = feats.rank_candidates(seed, ["F1", "F2", "F3"], top_k=3)
    assert ranked[0]["flight_id"] in ("F1", "F2")
    # Scores are within [0, 1] when normalized
    for r in ranked:
        assert 0.0 <= r["score"] <= 1.0


