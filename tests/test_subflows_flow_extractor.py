import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.subflows.flow_extractor import (
    assign_communities_for_hotspot,
    compute_jaccard_similarity,
    run_leiden_from_similarity,
)


def _write_tvtw_indexer(tmpdir: Path) -> Path:
    tvtw = {
        "time_bin_minutes": 15,
        "tv_id_to_idx": {"TVA": 0, "TVB": 1, "TVC": 2, "TVD": 3},
    }
    p = tmpdir / "tvtw_indexer.json"
    p.write_text(json.dumps(tvtw))
    return p


def _make_interval(tv_idx: int, bin_offset: int, bins_per_tv: int) -> Dict[str, int]:
    tvtw_index = tv_idx * bins_per_tv + bin_offset
    return {
        "tvtw_index": int(tvtw_index),
        "entry_time_s": int(bin_offset) * 60,
        "exit_time_s": int(bin_offset + 1) * 60,
    }


def _write_occupancy(tmpdir: Path) -> Path:
    # Synthetic 6 flights: 3 in each group
    # Group 1: TVA -> TVB -> TVC (trim at TVB)
    # Group 2: TVA -> TVD -> TVB (trim at TVB)
    bins_per_tv = 24 * 4  # 15-min bins
    flights: Dict[str, Dict] = {}

    def add_flight(fid: str, seq: List[int]):
        intervals = []
        for i, tv_idx in enumerate(seq):
            intervals.append(_make_interval(tv_idx, bin_offset=2 * i, bins_per_tv=bins_per_tv))
        flights[fid] = {
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

    p = tmpdir / "so6_occupancy_matrix_with_times.json"
    p.write_text(json.dumps(flights))
    return p


def test_sequences_and_footprints(tmp_path: Path):
    tvtw = _write_tvtw_indexer(tmp_path)
    occ = _write_occupancy(tmp_path)
    fl = FlightList(str(occ), str(tvtw))

    tvb_index = fl.tv_id_to_idx["TVB"]

    # Validate sequence extraction and consecutive de-duplication
    seq_a1 = fl.get_flight_tv_sequence_indices("A1")
    seq_b1 = fl.get_flight_tv_sequence_indices("B1")
    assert np.array_equal(seq_a1, np.array([0, 1, 2]))
    assert np.array_equal(seq_b1, np.array([0, 3, 1]))

    # Footprints trimmed to first TVB occurrence
    fp_a1 = fl.get_flight_tv_footprint_indices("A1", hotspot_tv_index=tvb_index)
    fp_b1 = fl.get_flight_tv_footprint_indices("B1", hotspot_tv_index=tvb_index)
    assert np.array_equal(fp_a1, np.array([0, 1]))
    assert np.array_equal(fp_b1, np.array([0, 1, 3]))


def test_jaccard_and_leiden(tmp_path: Path):
    tvtw = _write_tvtw_indexer(tmp_path)
    occ = _write_occupancy(tmp_path)
    fl = FlightList(str(occ), str(tvtw))
    tvb_index = fl.tv_id_to_idx["TVB"]

    flight_ids = ["A1", "A2", "A3", "B1", "B2", "B3"]
    fps = fl.get_footprints_for_flights(flight_ids, hotspot_tv_index=tvb_index)

    S = compute_jaccard_similarity(fps)
    assert isinstance(S, np.ndarray) and S.shape == (6, 6)

    # Within-group should be higher than between-group
    # A1 vs A2 (same set)
    assert np.isclose(S[0, 1], 1.0)
    # A1 vs B1: {0,1} vs {0,1,3} -> 2/3
    assert np.isclose(S[0, 3], 2.0 / 3.0)
    assert S[0, 1] > S[0, 3]

    # Leiden with threshold that keeps only within-group edges
    membership = run_leiden_from_similarity(S, threshold=0.8, resolution=1.0, seed=42)
    # Expect two clusters
    assert len(membership) == 6
    group_a = {membership[0], membership[1], membership[2]}
    group_b = {membership[3], membership[4], membership[5]}
    assert len(group_a) == 1 and len(group_b) == 1 and group_a != group_b


def test_public_wrapper(tmp_path: Path):
    tvtw = _write_tvtw_indexer(tmp_path)
    occ = _write_occupancy(tmp_path)
    fl = FlightList(str(occ), str(tvtw))

    hotspot = {
        "traffic_volume_id": "TVB",
        "hour": 0,
        "flight_ids": ["A1", "A2", "A3", "B1", "B2", "B3"],
    }

    res = assign_communities_for_hotspot(
        fl, hotspot["flight_ids"], hotspot["traffic_volume_id"], threshold=0.8, resolution=1.0, seed=123
    )
    assert set(res.keys()) == set(hotspot["flight_ids"])  # mapping covers all
    # Two communities
    labels = list(res.values())
    assert len(set(labels)) == 2
    # Intra-group equal, inter-group different
    assert res["A1"] == res["A2"] == res["A3"]
    assert res["B1"] == res["B2"] == res["B3"]
    assert res["A1"] != res["B1"]

    # Edge-case: single-flight hotspot returns {fid: 0}
    single = assign_communities_for_hotspot(
        fl, ["A1"], "TVB"
    )
    assert single == {"A1": 0}


