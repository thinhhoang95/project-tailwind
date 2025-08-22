import sys
from pathlib import Path
import os
import argparse
from typing import Dict, List, Tuple, Set

import numpy as np  # type: ignore
import geopandas as gpd  # type: ignore

# Ensure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.features.flight_features import FlightFeatures
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


def load_traffic_volumes_gdf(tvs_geojson: str | None) -> gpd.GeoDataFrame:
    """
    Load the traffic volumes GeoDataFrame.

    Follows the same fallbacks as tests/tabu/verifications/verify_hotspot_multiplicity.py
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


def compute_footprint_and_hours_from_occupancy(
    flight_list: FlightList, flight_id: str
) -> Tuple[Set[str], Dict[str, Set[int]]]:
    """
    Derive the set of traffic volumes visited and the set of hours per TV from the
    flight's occupancy vector.
    """
    occ_vec = flight_list.get_occupancy_vector(flight_id)
    nz_indices = np.nonzero(occ_vec > 0.0)[0] # non-zero indices in the occupancy vector

    bins_per_hour = 60 // int(flight_list.time_bin_minutes)
    num_tvtws = int(flight_list.num_tvtws)
    num_tvs = int(len(flight_list.tv_id_to_idx))
    num_time_bins_per_tv = int(num_tvtws // max(1, num_tvs))
    row_to_tv_id: Dict[int, str] = {idx: tv for tv, idx in flight_list.tv_id_to_idx.items()}

    tv_set: Set[str] = set()
    hours_by_tv: Dict[str, Set[int]] = {}

    for tvtw_idx in nz_indices.tolist():
        tv_row = int(tvtw_idx // max(1, num_time_bins_per_tv)) # row index of the traffic volume
        time_idx = int(tvtw_idx % max(1, num_time_bins_per_tv)) # hour index of the traffic volume
        tv_id = row_to_tv_id.get(tv_row)
        if tv_id is None:
            continue
        hour = int(time_idx // max(1, bins_per_hour))
        tv_set.add(tv_id)
        hours_by_tv.setdefault(tv_id, set()).add(hour)

    return tv_set, hours_by_tv


def manual_hourly_occupancy_for(
    total_occupancy_by_tvtw: np.ndarray,
    flight_list: FlightList,
    tv_id: str,
    hour: int,
) -> float:
    """
    Compute the network-wide hourly occupancy for a given traffic volume and hour by
    summing per-bin occupancies.
    """
    bins_per_hour = 60 // int(flight_list.time_bin_minutes)
    num_tvtws = int(flight_list.num_tvtws)
    num_time_bins_per_tv = int(num_tvtws // max(1, len(flight_list.tv_id_to_idx)))

    tv_row = flight_list.tv_id_to_idx.get(tv_id)
    if tv_row is None:
        return 0.0
    tv_start = int(tv_row * num_time_bins_per_tv)
    start_bin = int(hour * bins_per_hour)
    end_bin = min(start_bin + bins_per_hour, num_time_bins_per_tv)

    s = 0.0 # sum of occupancy for the traffic volume and hour
    for bin_offset in range(start_bin, end_bin):
        tvtw_idx = tv_start + bin_offset
        if 0 <= tvtw_idx < num_tvtws:
            s += float(total_occupancy_by_tvtw[tvtw_idx]) # add the occupancy for the traffic volume and hour
    return s # return the sum of occupancy for the traffic volume and hour


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verification: footprint and slack-valley metrics for flights within an overloaded hotspot."
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
    parser.add_argument(
        "--max-flights", type=int, default=10, help="Maximum number of flights to inspect from the selected hotspot"
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

    # Ensure evaluator caches are populated (hourly occupancy matrix)
    _ = evaluator.compute_excess_traffic_vector()

    # Interactive menu for hotspot selection (bin-level)
    options = list_overloaded_bin_hotspots(evaluator, indexer)
    tvtw_index = prompt_hotspot_selection(options)
    if tvtw_index is None:
        return

    print("\n3) Collect flights from selected hotspot and prepare verification...")
    bin_to_flights = build_overloaded_bin_flight_map(evaluator)
    if tvtw_index not in bin_to_flights:
        print(
            "   Selected TVTW is not overloaded at threshold > 0.0. No flights to verify."
        )
        return

    selected_flights: List[str] = bin_to_flights[tvtw_index][: max(1, int(args.max_flights))]
    print(f"   Selected {len(selected_flights)} flights for verification")

    ff = FlightFeatures(
        flight_list,
        evaluator,
        overload_threshold=0.0,
        limit_to_flight_ids=selected_flights,
    )

    hourly_matrix = getattr(evaluator, "last_hourly_occupancy_matrix", None)
    if hourly_matrix is None:
        print("X Hourly occupancy matrix not available; aborting slack verification.")
        return

    total_occupancy = flight_list.get_total_occupancy_by_tvtw()

    print("\n=== Verification per flight ===")
    for fid in selected_flights:
        print(f"\nFlight {fid}")

        # Footprint check
        occ_tv_set, hours_by_tv = compute_footprint_and_hours_from_occupancy(
            flight_list, fid
        )
        ff_tv_set = ff.get_footprint(fid)

        occ_tvs_sorted = sorted(occ_tv_set)
        ff_tvs_sorted = sorted(ff_tv_set)
        status_footprint = "OK" if set(occ_tvs_sorted) == set(ff_tvs_sorted) else "MISMATCH"
        print("  - Footprint from occupancy:", ", ".join(occ_tvs_sorted))
        print("  - Footprint from FlightFeatures:", ", ".join(ff_tvs_sorted))
        print(f"  => Footprint match: {status_footprint}")

        # Slack value check
        all_tv_p5_values: List[float] = []
        all_occ_ok = True
        for tv_id in sorted(occ_tv_set):
            hours = sorted(hours_by_tv.get(tv_id, set()))
            if not hours:
                continue
            row_idx = evaluator.tv_id_to_row_idx.get(tv_id)
            if row_idx is None:
                continue

            slacks_for_tv: List[float] = []
            for h in hours:
                manual_occ = manual_hourly_occupancy_for(total_occupancy, flight_list, tv_id, int(h))
                evaluator_occ = float(hourly_matrix[row_idx, int(h)])
                cap = float(evaluator.hourly_capacity_by_tv.get(tv_id, {}).get(int(h), 0.0))
                slack = cap - evaluator_occ
                
                print(f"    TV {tv_id}, Hour {h}: manual_occ={manual_occ:.3f}, evaluator_occ={evaluator_occ:.3f}, cap={cap:.3f}, slack={slack:.3f}")
                
                if abs(manual_occ - evaluator_occ) > 1e-6:
                    all_occ_ok = False
                slacks_for_tv.append(slack)

            if slacks_for_tv:
                try:
                    p5 = float(np.percentile(np.asarray(slacks_for_tv, dtype=np.float64), 5))
                except Exception:
                    p5 = float("nan")
                if not np.isnan(p5):
                    all_tv_p5_values.append(p5)

        manual_slack_min_p5 = (
            min(all_tv_p5_values) if all_tv_p5_values else 0.0
        )
        ff_vals = ff.get(fid)
        ff_slack_min_p5 = float(ff_vals.slack_min_p5)
        status_slack = (
            "OK" if abs(manual_slack_min_p5 - ff_slack_min_p5) <= 1e-6 else "MISMATCH"
        )

        print(f"  - Hourly occupancy match vs evaluator: {'OK' if all_occ_ok else 'MISMATCH'}")
        print(
            f"  - slack_min_p5 manual={manual_slack_min_p5:.6f} | features={ff_slack_min_p5:.6f} => {status_slack}"
        )


if __name__ == "__main__":
    main()


