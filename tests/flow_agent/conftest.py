from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList

_MAX_SAMPLE_FLIGHTS = 120


def _discover_data_root() -> Path:
    """Locate the directory that holds the tailwind artifacts used for tests."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "data" / "tailwind",
        repo_root / "output",
    ]
    for root in candidates:
        if (root / "tvtw_indexer.json").exists() and (root / "so6_occupancy_matrix_with_times.json").exists():
            return root
    raise FileNotFoundError(
        "Could not locate tvtw_indexer.json and so6_occupancy_matrix_with_times.json; "
        "checked data/tailwind and output."
    )


@pytest.fixture(scope="session")
def tailwind_sample(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """Stream a small real-data sample for end-to-end flow agent testing."""
    data_root = _discover_data_root()
    indexer_path = data_root / "tvtw_indexer.json"
    occupancy_path = data_root / "so6_occupancy_matrix_with_times.json"

    ijson = pytest.importorskip("ijson", reason="ijson is required to stream the SO6 occupancy file")

    sample_dir = tmp_path_factory.mktemp("flow_agent_sample")
    sample_path = sample_dir / "sample_flights.json"

    sample: Dict[str, Dict[str, Any]] = {}
    with open(occupancy_path, "rb") as handle:
        for flight_id, payload in ijson.kvitems(handle, ""):
            if not isinstance(payload, dict):
                continue
            intervals = payload.get("occupancy_intervals") or []
            if not intervals:
                continue
            curated = []
            for iv in intervals:
                if "tvtw_index" not in iv:
                    continue
                try:
                    tvtw_index = int(iv["tvtw_index"])
                except Exception:
                    continue
                try:
                    entry_time = float(iv.get("entry_time_s", 0.0))
                except Exception:
                    entry_time = 0.0
                try:
                    exit_time = float(iv.get("exit_time_s", entry_time))
                except Exception:
                    exit_time = entry_time
                curated.append(
                    {
                        "tvtw_index": tvtw_index,
                        "entry_time_s": entry_time,
                        "exit_time_s": exit_time,
                    }
                )
            if not curated:
                continue
            sample[str(flight_id)] = {
                "takeoff_time": payload.get("takeoff_time"),
                "origin": payload.get("origin"),
                "destination": payload.get("destination"),
                "distance": payload.get("distance"),
                "occupancy_intervals": curated,
            }
            if len(sample) >= _MAX_SAMPLE_FLIGHTS:
                break

    if not sample:
        pytest.skip("SO6 occupancy sample did not yield any flights")

    with open(sample_path, "w", encoding="utf-8") as out:
        json.dump(sample, out)

    flight_list = FlightList(str(sample_path), str(indexer_path))
    indexer = TVTWIndexer.load(str(indexer_path))

    tv_counts: Counter[str] = Counter()
    tv_time_counts: Dict[str, Counter[int]] = defaultdict(Counter)
    for meta in flight_list.flight_metadata.values():
        for interval in meta.get("occupancy_intervals", []):
            tvtw_idx = int(interval["tvtw_index"])
            tv_id, time_idx = indexer.get_tvtw_from_index(tvtw_idx)
            tv_counts[tv_id] += 1
            tv_time_counts[tv_id][int(time_idx)] += 1

    hotspots: List[str] = [tv for tv, _ in tv_counts.most_common(3)]
    if not hotspots:
        pytest.skip("Sample dataset contained no hotspot crossings")

    active_windows: Dict[str, List[int]] = {}
    for tv in hotspots:
        bins = [b for b, _ in tv_time_counts[tv].most_common(3)]
        active_windows[tv] = bins if bins else list(tv_time_counts[tv].keys())[:1]

    return {
        "flight_list": flight_list,
        "indexer": indexer,
        "hotspots": hotspots,
        "active_windows": active_windows,
    }


def uniform_capacities(
    indexer: TVTWIndexer,
    tv_ids: List[str],
    hourly_cap: float = 120.0,
) -> Dict[str, np.ndarray]:
    """Build a simple per-TV uniform capacity vector for testing."""
    T = indexer.num_time_bins
    proto = np.full(T, hourly_cap, dtype=np.float32)
    return {tv: proto.copy() for tv in tv_ids}
