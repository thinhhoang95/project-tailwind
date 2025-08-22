import sys
from pathlib import Path
import os
import argparse

# Ensure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from typing import Dict, List, Tuple

import geopandas as gpd  # type: ignore

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.features.flight_features import FlightFeatures


def load_traffic_volumes_gdf(tvs_geojson: str | None) -> gpd.GeoDataFrame:
    """
    Load the traffic volumes GeoDataFrame.
    Follows the same fallbacks as tests/tabu/regulation_use_example.py
    """
    candidates: List[Path] = []
    if tvs_geojson:
        candidates.append(Path(tvs_geojson))
    env = os.environ.get("TVS_GEOJSON") or os.environ.get("TRAFFIC_VOLUMES_GEOJSON")
    if env:
        candidates.append(Path(env))
    # Same path used in tests/test_hotspot_flow_retrieval.py
    candidates.append(Path("/Volumes/CrucialX/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"))
    # Windows example path from comments
    candidates.append(Path("D:/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"))

    for p in candidates:
        try:
            if p.exists():
                return gpd.read_file(str(p))
        except Exception:
            continue

    raise FileNotFoundError(
        "Traffic volumes GeoJSON not found. Provide via --tvs-geojson or TVS_GEOJSON env."
    )


def parse_time_bin_label(label: str, time_bin_minutes: int) -> int:
    """
    Convert a label like "06:00-06:15" to a time-window index (0..num_bins-1)
    based on the configured time_bin_minutes.
    """
    try:
        start, _end = label.strip().split("-")
        hh, mm = start.split(":")
        start_minutes = int(hh) * 60 + int(mm)
    except Exception:
        raise ValueError(f"Invalid time bin label: {label!r}. Expected HH:MM-HH:MM")

    if start_minutes % time_bin_minutes != 0:
        raise ValueError(
            f"Start minute {start_minutes} not aligned to {time_bin_minutes}-minute bins"
        )
    return start_minutes // time_bin_minutes


def build_overloaded_bin_flight_map(evaluator: NetworkEvaluator) -> Dict[int, List[str]]:
    """
    Build a mapping: overloaded_tvtw_index -> [flight_ids]
    Using evaluator.get_hotspot_flights(mode="bin").
    """
    results = evaluator.get_hotspot_flights(threshold=0.0, mode="bin")
    mapping: Dict[int, List[str]] = {}
    for item in results:
        tvtw_idx = int(item.get("tvtw_index"))
        mapping[tvtw_idx] = list(item.get("flight_ids", []))
    return mapping


def list_overloaded_bin_hotspots(
    evaluator: NetworkEvaluator, indexer: TVTWIndexer
) -> List[Tuple[int, str, int]]:
    """
    Build a list of overloaded hotspots at bin-level for interactive selection.
    Returns a list of tuples: (tvtw_index, label, flight_count)
    where label is like "TV_NAME at HH:MM-HH:MM".
    """
    results = evaluator.get_hotspot_flights(threshold=0.0, mode="bin")
    menu: List[Tuple[int, str, int]] = []
    for item in results:
        try:
            tvtw_idx = int(item.get("tvtw_index"))
        except Exception:
            continue
        tv_name, time_label = indexer.get_human_readable_tvtw(tvtw_idx) or ("?", "?")
        flight_count = len(item.get("flight_ids", []))
        menu.append((tvtw_idx, f"{tv_name} at {time_label}", flight_count))
    return menu


def prompt_hotspot_selection(options: List[Tuple[int, str, int]]) -> int | None:
    """
    Present a numeric menu and return the selected tvtw_index, or None if aborted.
    """
    if not options:
        print("X No overloaded hotspots found (threshold > 0.0).")
        return None

    print("\n2) Select a hotspot to inspect")
    for idx, (_tvtw, label, cnt) in enumerate(options, start=1):
        print(f"  [{idx}] {label} — {cnt} flights")
    print("  [q] Quit")

    while True:
        choice = input("Enter selection #: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return None
        try:
            sel = int(choice)
            if 1 <= sel <= len(options):
                return options[sel - 1][0]
        except Exception:
            pass
        print("Invalid selection. Please enter a valid number or 'q' to quit.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive verification: enter a TV ID and time bin; list flights that "
            "pass through multiple overloaded hotspots (including the selected one)."
        )
    )
    parser.add_argument(
        "--occupancy-json",
        default="output/so6_occupancy_matrix_with_times.json",
        help="Path to so6_occupancy_matrix_with_times.json",
    )
    parser.add_argument(
        "--tvtw-indexer",
        default="output/tvtw_indexer.json",
        help="Path to tvtw_indexer.json",
    )
    parser.add_argument(
        "--tvs-geojson",
        default=None,
        help="Path to traffic volumes GeoJSON (falls back to env or test defaults)",
    )
    args = parser.parse_args()

    # Validate required files
    required = [args.occupancy_json, args.tvtw_indexer]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        print(f"X Missing required files: {missing}")
        print("Please ensure these exist relative to project root.")
        return

    print("1) Loading inputs...")
    tvs_gdf = load_traffic_volumes_gdf(args.tvs_geojson)
    print(f"   OK traffic volumes: {len(tvs_gdf)}")

    flight_list = FlightList(
        occupancy_file_path=args.occupancy_json,
        tvtw_indexer_path=args.tvtw_indexer,
    )
    print(
        f"   OK flights: {flight_list.num_flights} with {flight_list.num_tvtws} TVTWs (bin={flight_list.time_bin_minutes}m)"
    )

    evaluator = NetworkEvaluator(tvs_gdf, flight_list)
    indexer = TVTWIndexer.load(args.tvtw_indexer)

    # Interactive menu for hotspot selection (bin-level)
    options = list_overloaded_bin_hotspots(evaluator, indexer)
    tvtw_index = prompt_hotspot_selection(options)
    if tvtw_index is None:
        return

    print("\n3) Finding overloaded hotspots and flights...")
    bin_to_flights = build_overloaded_bin_flight_map(evaluator)

    if tvtw_index not in bin_to_flights:
        print(
            "   Selected TVTW is not overloaded at threshold > 0.0. No flights to report for verification."
        )
        return

    seed_flights = set(bin_to_flights[tvtw_index])
    print(f"   Seed (selected hotspot) flights: {len(seed_flights)}")

    # Build reverse map flight -> list of overloaded tvtw indices it occupies
    flight_to_overloaded_bins: Dict[str, List[int]] = {fid: [] for fid in seed_flights}
    for ov_idx, fids in bin_to_flights.items():
        for fid in seed_flights:
            # Avoid O(n*m) string scans if no match
            if fid in fids:
                flight_to_overloaded_bins[fid].append(ov_idx)

    # Filter to flights that cross multiple hotspots (>= 2)
    qualifying: List[Tuple[str, List[int]]] = []
    for fid, idxs in flight_to_overloaded_bins.items():
        uniq = sorted(set(idxs))
        if len(uniq) >= 2:
            qualifying.append((fid, uniq))

    if not qualifying:
        print("   No flights from the selected hotspot pass through multiple overloaded hotspots.")
        return

    # Cross-check with FlightFeatures multiplicity (hour-level)
    ff = FlightFeatures(flight_list, evaluator, overload_threshold=0.0, limit_to_flight_ids=[fid for fid, _ in qualifying])
    bins_per_hour = 60 // flight_list.time_bin_minutes

    def hour_label(h: int) -> str:
        s = f"{h:02d}:00"
        e = f"{(h + 1) % 24:02d}:00"
        return f"{s}-{e}"

    print("\n=== Flights crossing multiple hotspots (including selected) ===")
    for fid, idxs in qualifying:
        # Derive hour-level unique pairs for comparison
        hour_pairs: List[Tuple[str, int]] = []
        for gi in idxs:
            tv_name, time_idx = indexer.get_tvtw_from_index(gi) or (None, None)
            if tv_name is None or time_idx is None:
                continue
            hour = int(time_idx) // bins_per_hour
            hour_pairs.append((str(tv_name), hour))
        unique_hour_pairs = sorted(set(hour_pairs))

        ff_mult = ff.multiplicity(fid)
        status = "OK" if ff_mult == len(unique_hour_pairs) else f"MISMATCH (ff={ff_mult}, ours={len(unique_hour_pairs)})"

        print(f"Flight {fid}  [FlightFeatures multiplicity={ff_mult} | {status}]")
        # Print bin-level hotspots
        for gi in idxs:
            tv_name, tw_label = indexer.get_human_readable_tvtw(gi) or ("?", "?")
            print(f"  - Hotspot {tv_name} at {tw_label}")
        # Print hour-level aggregation that drives multiplicity
        print("    Hour-level hotspots (for multiplicity):")
        for tv_name, h in unique_hour_pairs:
            print(f"      - {tv_name} at {hour_label(h)}")


if __name__ == "__main__":
    main()


