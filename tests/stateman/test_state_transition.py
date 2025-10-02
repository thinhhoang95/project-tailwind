import json
import sys
from pathlib import Path

import pytest

if "geopandas" not in sys.modules:
    import types

    geopandas_stub = types.ModuleType("geopandas")
    geopandas_stub.GeoDataFrame = object  # type: ignore[attr-defined]
    geopandas_stub.GeoSeries = object  # type: ignore[attr-defined]
    sys.modules["geopandas"] = geopandas_stub

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.stateman.delay_assignment import DelayAssignmentTable
from project_tailwind.stateman.delta_view import DeltaOccupancyView
from project_tailwind.stateman.flight_list_with_delta import FlightListWithDelta
from project_tailwind.stateman.regulation_history import RegulationHistory


@pytest.fixture
def sample_flight_files(tmp_path: Path) -> tuple[Path, Path]:
    """
    Create two temporary JSON files (occupancy and tvtw indexer) and return their paths.
    
    Parameters:
        tmp_path (Path): Pytest temporary directory in which to create the files.
    
    Returns:
        tuple[Path, Path]: Paths to the created occupancy JSON and tvtw_indexer JSON, respectively.
    """
    indexer_path = tmp_path / "tvtw_indexer.json"
    occupancy_path = tmp_path / "occupancy.json"

    indexer_payload = {
        "time_bin_minutes": 15,
        "tv_id_to_idx": {"TV1": 0},
    }
    with indexer_path.open("w", encoding="utf-8") as handle:
        json.dump(indexer_payload, handle)

    occupancy_payload = {
        "F1": {
            "takeoff_time": "2024-01-01T00:00:00",
            "origin": "AAA",
            "destination": "BBB",
            "distance": 1000,
            "occupancy_intervals": [
                {"tvtw_index": 10, "entry_time_s": 0.0, "exit_time_s": 600.0},
                {"tvtw_index": 11, "entry_time_s": 600.0, "exit_time_s": 900.0},
            ],
        },
        "F2": {
            "takeoff_time": "2024-01-01T01:00:00",
            "origin": "CCC",
            "destination": "DDD",
            "distance": 1200,
            "occupancy_intervals": [
                {"tvtw_index": 95, "entry_time_s": 8000.0, "exit_time_s": 8300.0},
            ],
        },
    }
    with occupancy_path.open("w", encoding="utf-8") as handle:
        json.dump(occupancy_payload, handle)

    return occupancy_path, indexer_path


@pytest.fixture
def base_flight_list(sample_flight_files: tuple[Path, Path]) -> FlightList:
    """
    Create a FlightList initialized from a pair of occupancy and indexer JSON file paths.
    
    Parameters:
        sample_flight_files (tuple[Path, Path]): A tuple containing (occupancy_path, indexer_path).
    
    Returns:
        FlightList: An instance loaded from the provided occupancy and indexer files.
    """
    occupancy_path, indexer_path = sample_flight_files
    return FlightList(str(occupancy_path), str(indexer_path))


@pytest.fixture
def stateful_flight_list(sample_flight_files: tuple[Path, Path]) -> FlightListWithDelta:
    occupancy_path, indexer_path = sample_flight_files
    return FlightListWithDelta(str(occupancy_path), str(indexer_path))


def test_delay_assignment_table_merge_and_io(tmp_path: Path) -> None:
    table = DelayAssignmentTable.from_dict({"F1": 5, "F2": 0})
    other = DelayAssignmentTable.from_dict({"F1": 7, "F3": 3})

    merged_max = table.merge(other, policy="max")
    assert merged_max.to_dict()["F1"] == 7
    merged_sum = table.merge({"F1": 4, "F4": 2}, policy="sum")
    assert merged_sum.to_dict()["F1"] == 9
    assert merged_sum.to_dict()["F4"] == 2

    json_path = tmp_path / "delays.json"
    csv_path = tmp_path / "delays.csv"

    merged_sum.save_json(json_path)
    merged_sum.save_csv(csv_path)

    json_loaded = DelayAssignmentTable.load_json(json_path)
    csv_loaded = DelayAssignmentTable.load_csv(csv_path)

    assert dict(json_loaded.nonzero_items()) == {"F1": 9, "F4": 2}
    assert dict(csv_loaded.nonzero_items()) == {"F1": 9, "F4": 2}


def test_delta_view_shift_and_drop(base_flight_list: FlightList) -> None:
    delays = DelayAssignmentTable.from_dict({"F1": 15, "F2": 15})

    view = DeltaOccupancyView.from_delay_table(base_flight_list, delays, regulation_id="R1")

    dense_delta = view.as_dense_delta()
    assert dense_delta.shape[0] == base_flight_list.num_tvtws
    assert dense_delta[10] == -1
    assert dense_delta[11] == 0
    assert dense_delta[12] == 1
    assert dense_delta[95] == -1

    intervals_f1 = view.per_flight_new_intervals["F1"]
    assert [iv["tvtw_index"] for iv in intervals_f1] == [11, 12]
    assert intervals_f1[0]["entry_time_s"] == pytest.approx(900.0)
    assert intervals_f1[1]["entry_time_s"] == pytest.approx(1500.0)
    assert view.per_flight_new_intervals["F2"] == []

    stats = view.stats()
    assert stats["num_changed_flights"] == 2
    assert stats["total_delay_minutes"] == 30
    assert stats["nonzero_entries"] == 3
    assert set(view.changed_flights()) == {"F1", "F2"}


def test_flight_list_with_delta_step(stateful_flight_list: FlightListWithDelta) -> None:
    delays = DelayAssignmentTable.from_dict({"F1": 15, "F2": 15})
    view = DeltaOccupancyView.from_delay_table(stateful_flight_list, delays, regulation_id="R1")

    stateful_flight_list.step_by_delay(view)

    occupancy_f1 = stateful_flight_list.get_occupancy_vector("F1")
    assert occupancy_f1[11] == pytest.approx(1.0)
    assert occupancy_f1[12] == pytest.approx(1.0)
    assert occupancy_f1[10] == pytest.approx(0.0)

    occupancy_f2 = stateful_flight_list.get_occupancy_vector("F2")
    assert occupancy_f2[95] == pytest.approx(0.0)

    intervals_f1 = stateful_flight_list.flight_metadata["F1"]["occupancy_intervals"]
    assert [iv["tvtw_index"] for iv in intervals_f1] == [11, 12]

    aggregate = stateful_flight_list.get_delta_aggregate()
    assert aggregate[10] == -1
    assert aggregate[12] == 1
    assert aggregate[95] == -1

    assert stateful_flight_list.total_delay_assigned_min == 30
    assert stateful_flight_list.num_delayed_flights == 2
    assert stateful_flight_list.num_regulations == 1
    assert stateful_flight_list.applied_regulations == ["R1"]
    assert stateful_flight_list.delay_histogram[15] == 2


def test_regulation_history_roundtrip(base_flight_list: FlightList) -> None:
    delays = DelayAssignmentTable.from_dict({"F1": 15})
    view = DeltaOccupancyView.from_delay_table(base_flight_list, delays, regulation_id="R1")

    history = RegulationHistory()
    history.record("R1", view)
    assert "R1" in history
    assert history.get("R1") is view
    assert history.list_ids() == ["R1"]

    history.record("R1", view)
    assert len(history) == 1
