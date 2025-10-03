# Load data from resources.py
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from parrhesia.flow_agent35.regen.hotspot_segment_extractor import (
    extract_hotspot_segments_from_resources,
    segment_to_hotspot_payload,
)
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from server_tailwind.core.resources import AppResources, ResourcePaths, get_resources


def build_data(occupancy_path: Path, indexer_path: Path, caps_path: Path) -> Tuple[FlightList, TVTWIndexer]:
    """
    Loads core data structures using shared AppResources (delta-enabled).

    This function initializes the main objects required for network evaluation
    and regulation proposals using the shared resources flight list for
    consistency with the server. It preserves defensive occupancy padding.

    Args:
        occupancy_path: Path to the flight occupancy matrix JSON file.
        indexer_path: Path to the TVTW indexer JSON file.
        caps_path: Path to the traffic volume capacities GeoJSON file.

    Returns:
        A tuple containing the initialized FlightList and TVTWIndexer objects.
    """
    # Use AppResources so flows/regen share the same flight list/indexer as the server
    res = AppResources(
        ResourcePaths(
            occupancy_file_path=occupancy_path,
            tvtw_indexer_path=indexer_path,
            traffic_volumes_path=caps_path,
        )
    ).preload_all()

    # Ensure subsequent calls to get_resources() return this same instance
    try:
        from server_tailwind.core import resources as _res_mod  # type: ignore
        _res_mod._GLOBAL_RESOURCES = res  # type: ignore[attr-defined]
    except Exception:
        pass

    indexer = res.indexer
    flight_list = res.flight_list

    # Defensive occupancy width alignment (keep as before)
    expected_tvtws = len(indexer.tv_id_to_idx) * indexer.num_time_bins
    if getattr(flight_list, "num_tvtws", 0) < expected_tvtws:
        from scipy import sparse  # type: ignore
        pad_cols = expected_tvtws - int(flight_list.num_tvtws)
        pad_matrix = sparse.lil_matrix((int(flight_list.num_flights), pad_cols))
        flight_list._occupancy_matrix_lil = sparse.hstack(  # type: ignore[attr-defined]
            [flight_list._occupancy_matrix_lil, pad_matrix], format="lil"  # type: ignore[attr-defined]
        )
        flight_list.num_tvtws = expected_tvtws  # type: ignore[assignment]
        flight_list._temp_occupancy_buffer = np.zeros(expected_tvtws, dtype=np.float32)  # type: ignore[attr-defined]
        flight_list._lil_matrix_dirty = True  # type: ignore[attr-defined]
        flight_list._sync_occupancy_matrix()  # type: ignore[attr-defined]

    # Capacities from resources
    caps_gdf = res.traffic_volumes_gdf
    if getattr(caps_gdf, "empty", False):
        raise SystemExit("Traffic volume capacity file is empty; cannot proceed.")

    return flight_list, indexer


@pytest.mark.slow
def test_hse() -> None:
    """Compute hotspot segments and surface a ranked summary table."""

    occupancy_path = REPO_ROOT / "output" / "so6_occupancy_matrix_with_times.json"
    indexer_path = REPO_ROOT / "output" / "tvtw_indexer.json"
    caps_path = REPO_ROOT / "output" / "wxm_sm_ih_maxpool.geojson"

    missing = [path for path in (occupancy_path, indexer_path, caps_path) if not path.exists()]
    if missing:
        pytest.skip(f"required artifacts missing: {[str(p) for p in missing]}")

    # Load resources once so the extractor shares the same FlightList the server uses.
    build_data(occupancy_path, indexer_path, caps_path)

    resources = get_resources().preload_all()
    segments = extract_hotspot_segments_from_resources(resources=resources)

    # There should be a substantial number of hotspot segments for the dataset.
    assert len(segments) == 1707

    # The implementation sorts by descending max exceedance; confirm monotonic ordering.
    leading_max_excess = [seg["max_excess"] for seg in segments[:10]]
    assert leading_max_excess == sorted(leading_max_excess, reverse=True)

    expected_top5 = [
        ("LECBBAS", "11:15", "11:45", 34.0, 62.0),
        ("LFBZX15", "11:15", "11:45", 28.0, 51.0),
        ("LFBZX35", "11:15", "11:45", 28.0, 54.0),
        ("LECMSEI", "11:30", "11:45", 27.0, 39.0),
        ("LFBZNX35", "11:15", "11:45", 25.0, 49.0),
    ]
    observed_top5 = [
        (
            seg["traffic_volume_id"],
            seg["start_label"],
            seg["end_label"],
            seg["max_excess"],
            seg["sum_excess"],
        )
        for seg in segments[:5]
    ]
    assert observed_top5 == expected_top5

    # Regen expects inclusive bins translated to [start, end_exclusive].
    top_payload = segment_to_hotspot_payload(segments[0])
    assert top_payload == {
        "control_volume_id": "LECBBAS",
        "window_bins": [45, 48],
        "metadata": {},
        "mode": "inventory",
    }

    # Build a Rich table for interactive inspection if the dependency is available.
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        # Fallback text rendering ensures the table logic still executes.
        fallback_rows = [
            f"#{idx} {seg['traffic_volume_id']} max={seg['max_excess']}"
            for idx, seg in enumerate(segments[:5], start=1)
        ]
        assert fallback_rows[0].startswith("#1 LECBBAS")
    else:
        console = Console(record=True, width=100)
        table = Table(title="Top Hotspot Segments")
        table.add_column("Rank", justify="right")
        table.add_column("Traffic Volume", style="cyan")
        table.add_column("Start")
        table.add_column("End")
        table.add_column("Max Excess", justify="right")
        table.add_column("Sum Excess", justify="right")

        for idx, seg in enumerate(segments[:5], start=1):
            table.add_row(
                str(idx),
                seg["traffic_volume_id"],
                seg["start_label"],
                seg["end_label"],
                f"{seg['max_excess']:.1f}",
                f"{seg['sum_excess']:.1f}",
            )

        console.print(table)
        rendered = console.export_text(clear=False)
        assert "Top Hotspot Segments" in rendered
        assert "LECBBAS" in rendered
