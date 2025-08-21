import sys
from pathlib import Path
import argparse
import os
import time

# Ensure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import geopandas as gpd  # type: ignore

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.moves.network_plan_move import NetworkPlanMove
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.optimize.regulation import Regulation
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

# Optional pretty printing with rich
try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None


def load_traffic_volumes_gdf(tvs_geojson: str | None) -> gpd.GeoDataFrame:
    """
    Load the traffic volumes GeoDataFrame.
    If tvs_geojson is None, try environment variable or a repo-default path.
    """
    # Priority: CLI arg -> ENV -> repo default used in existing test
    candidates: list[Path] = []
    if tvs_geojson:
        candidates.append(Path(tvs_geojson))
    env = os.environ.get("TVS_GEOJSON") or os.environ.get("TRAFFIC_VOLUMES_GEOJSON")
    if env:
        candidates.append(Path(env))
    # Same path used in tests/test_hotspot_flow_retrieval.py
    candidates.append(Path("/Volumes/CrucialX/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"))
    # Windows example path from comment in the test file
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


def pick_hotspot_per_hour(evaluator: NetworkEvaluator) -> dict:
    """Pick the most overloaded (tv, hour) item from per-hour hotspots, or return first if none overloaded."""
    per_hour = evaluator.get_hotspot_flights(threshold=0.0, mode="hour")
    if not per_hour:
        return {}

    # Prefer overloaded ones sorted by (hourly_occupancy - hourly_capacity) desc
    overloaded = [x for x in per_hour if x.get("is_overloaded")]
    def deficit(item: dict) -> float:
        return float(item.get("hourly_occupancy", 0.0)) - float(item.get("hourly_capacity", 0.0))

    if overloaded:
        overloaded.sort(key=deficit, reverse=True)
        return overloaded[0]
    # Fallback: just take the one with highest hourly_occupancy
    per_hour.sort(key=lambda x: float(x.get("hourly_occupancy", 0.0)), reverse=True)
    return per_hour[0]


def hour_to_time_windows(hour: int, time_bin_minutes: int) -> list[int]:
    bins_per_hour = 60 // int(time_bin_minutes)
    start = int(hour) * bins_per_hour
    return list(range(start, start + bins_per_hour))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Demonstration: find a hotspot, propose a blanket regulation at that hotspot,\n"
            "set a rate, apply via NetworkPlanMove (DeltaFlightList), and re-evaluate."
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
        "--tv-id",
        default=None,
        help="Override hotspot traffic_volume_id (otherwise pick most overloaded)",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=None,
        help="Override hotspot hour (0-23). If tv-id provided, must also provide hour.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=None,
        help="Rate for regulation (flights/hour). Defaults to hotspot hourly_capacity.",
    )
    args = parser.parse_args()

    # Validate required files
    required = [args.occupancy_json, args.tvtw_indexer]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        print(f"X Missing required files: {missing}")
        print("Please ensure these exist relative to project root.")
        return

    # Load traffic volumes and flights
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

    # Build evaluator and find hotspots
    print("\n2) Finding hotspots and retrieving flights (per TV hour)...")
    evaluator = NetworkEvaluator(tvs_gdf, flight_list)
    # Pre-regulation metrics
    time_start = time.time()
    pre_excess = evaluator.compute_excess_traffic_vector()
    pre_delay_stats = evaluator.compute_delay_stats()
    time_end = time.time()
    print(f"Computation completed in {time_end - time_start} seconds")
    # Technically it could also be done through calling compute_horizon_metrics(horizon_time_windows=0)
    # but we will write everything out here for now for clarity
    pre_z_sum = float(pre_excess.sum()) if pre_excess.size > 0 else 0.0
    pre_z_max = float(pre_excess.max()) if pre_excess.size > 0 else 0.0
    pre_metrics = {
        "stage": "Pre",
        "z_sum": pre_z_sum,
        "z_max": pre_z_max,
        "total_delay_min": pre_delay_stats.get("total_delay_seconds", 0.0) / 60.0,
        "mean_delay_min": pre_delay_stats.get("mean_delay_seconds", 0.0) / 60.0,
        "max_delay_min": pre_delay_stats.get("max_delay_seconds", 0.0) / 60.0,
        "delayed_flights": pre_delay_stats.get("delayed_flights_count", 0),
    }
    per_hour = evaluator.get_hotspot_flights(threshold=0.0, mode="hour")
    print(f"   Found {len(per_hour)} (tv, hour) hotspot entries")

    # Print hotspots table using rich if available
    if RICH_AVAILABLE and console:
        table = Table(title="Hotspot Analysis (Per TV-Hour)")
        table.add_column("TV ID", style="cyan", no_wrap=True)
        table.add_column("Hour", justify="center", style="magenta")
        table.add_column("Hourly Occ", justify="right", style="yellow")
        table.add_column("Hourly Cap", justify="right", style="green")
        table.add_column("Excess", justify="right", style="red")
        table.add_column("Util %", justify="right")
        table.add_column("Overloaded", justify="center")
        
        # Sort hotspots by excess (descending) for better readability
        sorted_hotspots = sorted(per_hour, key=lambda x: x.get("excess_traffic", 0), reverse=True)
        
        for item in sorted_hotspots[:20]:  # Show top 20 hotspots
            tv_id = item.get("traffic_volume_id", "N/A")
            hour = item.get("hour", "N/A")
            occupancy = item.get("hourly_occupancy", 0)
            capacity = item.get("hourly_capacity", 0)
            excess = item.get("excess_traffic", 0)
            utilization = (occupancy / capacity * 100) if capacity > 0 else 0
            is_overloaded = item.get("is_overloaded", False)
            
            table.add_row(
                str(tv_id),
                str(hour),
                f"{occupancy:.0f}",
                str(capacity) if capacity > 0 else "N/A",
                f"{excess:.1f}" if excess > 0 else "0.0",
                f"{utilization:.1f}%" if capacity > 0 else "N/A",
                "✓" if is_overloaded else "✗"
            )
        
        console.print(table)
    else:
        # Fallback to simple text table
        print("   Top hotspots (TV-Hour):")
        sorted_hotspots = sorted(per_hour, key=lambda x: x.get("excess_traffic", 0), reverse=True)
        for i, item in enumerate(sorted_hotspots[:10], 1):
            tv_id = item.get("traffic_volume_id", "N/A")
            hour = item.get("hour", "N/A")
            occupancy = item.get("hourly_occupancy", 0)
            capacity = item.get("hourly_capacity", 0)
            excess = item.get("excess_traffic", 0)
            utilization = (occupancy / capacity * 100) if capacity > 0 else 0
            is_overloaded = "YES" if item.get("is_overloaded", False) else "NO"
            print(f"   {i:2d}. TV={tv_id} H={hour} Occ={occupancy:.0f} Cap={capacity} Excess={excess:.1f} Util={utilization:.1f}% Over={is_overloaded}")

    if not per_hour:
        print("   No hotspots detected (excess == 0 across all). Exiting example.")
        return

    if args.tv_id is not None and args.hour is not None:
        selected = None
        for item in per_hour:
            if item.get("traffic_volume_id") == args.tv_id and int(item.get("hour", -1)) == int(args.hour):
                selected = item
                break
        if selected is None:
            print("   Provided --tv-id/--hour not found among hotspots. Exiting.")
            return
    else:
        selected = pick_hotspot_per_hour(evaluator)

    tv_id = selected["traffic_volume_id"]
    hour = int(selected["hour"])
    hourly_capacity = int(selected.get("hourly_capacity", -1))
    hourly_occupancy = float(selected.get("hourly_occupancy", -1))
    is_overloaded = bool(selected.get("is_overloaded", False))
    print(
        f"   Selected hotspot: tv={tv_id}, hour={hour}, occupancy={hourly_occupancy:.0f}, capacity={hourly_capacity}, overloaded={is_overloaded}"
    )

    # 3) Build a blanket regulation for that (tv, hour) — IC__ means no filtering condition
    bins_per_hour = 60 // flight_list.time_bin_minutes
    time_windows = hour_to_time_windows(hour, flight_list.time_bin_minutes)
    rate = int(args.rate) if args.rate is not None else int(hourly_capacity if hourly_capacity > 0 else max(1, int(hourly_occupancy)))

    regulation = Regulation.from_components(
        location=tv_id,
        rate=rate,
        time_windows=time_windows,
        # filter_type defaults to IC; filter_value defaults to '__' (wildcard both sides)
    )
    plan = NetworkPlan([regulation])

    # Prepare parser and indexer (parser reads flight JSON; we reuse the same occupancy JSON with intervals)
    tvtw_indexer = TVTWIndexer.load(args.tvtw_indexer)
    parser_obj = RegulationParser(flights_file=args.occupancy_json, tvtw_indexer=tvtw_indexer)

    # 4) Apply network plan move to compute delays and build DeltaFlightList view
    print("\n3) Applying NetworkPlanMove (FCFS), building DeltaFlightList view...")
    move = NetworkPlanMove(plan, parser_obj, tvtw_indexer)
    delta_view, total_delay_min = move.build_delta_view(flight_list)
    print(f"   OK assigned total delay: {total_delay_min} minutes across all {len(delta_view.flight_ids)} flights (view)") # this is the total number of flights in the dataset, not the number of flights that acutally got delayed.

    # 5) Re-run network evaluation on the delta view
    print("\n4) Re-evaluating network with delays applied (delta view)...")
    post_eval = NetworkEvaluator(tvs_gdf, delta_view)
    post_per_hour = post_eval.get_hotspot_flights(threshold=0.0, mode="hour")
    post_delay_stats = post_eval.compute_delay_stats()
    post_excess = post_eval.compute_excess_traffic_vector()
    post_z_sum = float(post_excess.sum()) if post_excess.size > 0 else 0.0
    post_z_max = float(post_excess.max()) if post_excess.size > 0 else 0.0
    post_metrics = {
        "stage": "Post",
        "z_sum": post_z_sum,
        "z_max": post_z_max,
        "total_delay_min": post_delay_stats.get("total_delay_seconds", 0.0) / 60.0,
        "mean_delay_min": post_delay_stats.get("mean_delay_seconds", 0.0) / 60.0,
        "max_delay_min": post_delay_stats.get("max_delay_seconds", 0.0) / 60.0,
        "delayed_flights": post_delay_stats.get("delayed_flights_count", 0),
    }

    # Find the same tv/hour in the post state
    post_selected = None
    for item in post_per_hour:
        if item.get("traffic_volume_id") == tv_id and int(item.get("hour", -1)) == hour:
            post_selected = item
            break

    if post_selected is None:
        print("   Post-regulation hotspot entry not found for the same tv/hour (possible if no longer overloaded).")
    else:
        post_hourly_occupancy = float(post_selected.get("hourly_occupancy", -1))
        post_hourly_capacity = int(post_selected.get("hourly_capacity", -1))
        post_overloaded = bool(post_selected.get("is_overloaded", False))
        print(
            f"   Post: tv={tv_id}, hour={hour}, hourly_occupancy={post_hourly_occupancy:.0f}, hourly_capacity={post_hourly_capacity}, overloaded={post_overloaded}"
        )

    print("\n=== Summary ===")
    print(f"Hotspot: tv={tv_id}, hour={hour}, windows={time_windows}, bins_per_hour={bins_per_hour}")
    print(f"Regulation: rate={rate} flights/hour (blanket, no filtering)")
    print(f"Assigned total delay: {total_delay_min} minutes")

    # Pretty metrics table (pre vs post)
    if RICH_AVAILABLE:
        table = Table(title="Network Evaluator Metrics")
        table.add_column("Stage", justify="left")
        table.add_column("z_sum", justify="right")
        table.add_column("z_max", justify="right")
        table.add_column("Total delay (min)", justify="right")
        table.add_column("Mean delay (min)", justify="right")
        table.add_column("Max delay (min)", justify="right")
        table.add_column("Delayed flights", justify="right")

        for m in (pre_metrics, post_metrics):
            table.add_row(
                m["stage"],
                f"{m['z_sum']:.1f}",
                f"{m['z_max']:.1f}",
                f"{m['total_delay_min']:.1f}",
                f"{m['mean_delay_min']:.2f}",
                f"{m['max_delay_min']:.2f}",
                str(m["delayed_flights"]),
            )
        console.print(table)
    else:
        print("\nPre vs Post metrics:")
        for m in (pre_metrics, post_metrics):
            print(
                f"  {m['stage']}: z_sum={m['z_sum']:.1f}, z_max={m['z_max']:.1f}, "
                f"total_delay_min={m['total_delay_min']:.1f}, mean_delay_min={m['mean_delay_min']:.2f}, "
                f"max_delay_min={m['max_delay_min']:.2f}, delayed_flights={m['delayed_flights']}"
            )


if __name__ == "__main__":
    main()


