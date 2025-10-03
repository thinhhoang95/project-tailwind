from __future__ import annotations

"""
Tiny end-to-end pipeline to smoke-test the SA optimizer on real data.

This script:
  1) Loads the TVTW indexer and streams a small sample of flights from the
     SO6 occupancy JSON using ijson.
  2) Chooses a few hotspot TVs and active windows from the sample.
  3) Builds flows globally (earliest-hotspot trimming) and per-flow requested
     bins at a controlled volume via earliest-median policy.
  4) Builds per-TV capacities from the GeoJSON.
  5) Runs 1-2 iterations of simulated annealing and prints objective parts.

If you hit missing dependencies, activate the right environment:
  conda activate silverdrizzle
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

from src.parrhesia.indexing.tvtw_indexer import TVTWIndexer
from src.parrhesia.optim.flight_list import FlightList, _parse_naive_utc
from src.parrhesia.flows.flow_pipeline import collect_hotspot_flights, build_global_flows
from src.parrhesia.optim.capacity import build_bin_capacities
from src.parrhesia.optim.objective import ObjectiveWeights, score
from src.parrhesia.optim.sa_optimizer import SAParams, prepare_flow_scheduling_inputs, run_sa


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    root = _project_root()
    indexer_path = root / "data" / "tailwind" / "tvtw_indexer.json"
    occ_path = root / "data" / "tailwind" / "so6_occupancy_matrix_with_times.json"
    caps_path = root / "data" / "cirrus" / "wxm_sm_ih_maxpool.geojson"

    print("Loading TVTW indexer...")
    idx = TVTWIndexer.load(str(indexer_path))

    # Stream a small sample to decide hotspots/windows and build a small FlightList
    print("Sampling flights via ijson (up to 120 flights)...")
    try:
        import ijson  # type: ignore
    except Exception:
        print("ijson not available; please `conda activate silverdrizzle` if missing.")
        import sys
        sys.exit(1)

    sample_flights: Dict[str, dict] = {}
    tv_counts: Counter[str] = Counter()
    tv_time_counts: dict[str, Counter[int]] = defaultdict(Counter)
    with open(occ_path, "rb") as f:
        for flight_id, obj in ijson.kvitems(f, ""):
            if not isinstance(obj, dict):
                continue
            takeoff_raw = obj.get("takeoff_time")
            intervals_in = obj.get("occupancy_intervals", []) or []
            if not takeoff_raw or not intervals_in:
                continue
            curated_intervals = []
            for it in intervals_in:
                try:
                    tvtw_idx = int(it.get("tvtw_index"))
                except Exception:
                    continue
                decoded = idx.get_tvtw_from_index(tvtw_idx)
                if not decoded:
                    continue
                tv_id, time_idx = decoded
                tv_counts[tv_id] += 1
                tv_time_counts[tv_id][int(time_idx)] += 1
                entry_s = it.get("entry_time_s", 0.0)
                exit_s = it.get("exit_time_s", entry_s)
                try:
                    entry_s = float(entry_s)
                except Exception:
                    entry_s = 0.0
                try:
                    exit_s = float(exit_s)
                except Exception:
                    exit_s = entry_s
                curated_intervals.append({
                    "tvtw_index": tvtw_idx,
                    "entry_time_s": entry_s,
                    "exit_time_s": exit_s,
                })
            if not curated_intervals:
                continue
            sample_flights[str(flight_id)] = {
                "takeoff_time": _parse_naive_utc(str(takeoff_raw)),
                "origin": obj.get("origin"),
                "destination": obj.get("destination"),
                "distance": obj.get("distance"),
                "occupancy_intervals": curated_intervals,
            }
            if len(sample_flights) >= 120:
                break

    if not sample_flights:
        print("No flights sampled; aborting.")
        return 1

    # Pick top-2 hotspots and their top-2 windows
    hotspot_ids = [tv for tv, _ in tv_counts.most_common(2)]
    active_windows: dict[str, list[int]] = {}
    for tv in hotspot_ids:
        bins = [b for b, _ in tv_time_counts[tv].most_common(2)]
        active_windows[tv] = bins if bins else list(tv_time_counts[tv].keys())[:1]

    print(f"Hotspots: {hotspot_ids}; windows: {active_windows}")

    # Build small FlightList
    fl = FlightList(idx)
    fl.flight_metadata = sample_flights

    # Build union and flows
    union_ids, _meta = collect_hotspot_flights(fl, hotspot_ids, active_windows=active_windows)
    flow_map = build_global_flows(
        fl,
        union_ids,
        hotspots=hotspot_ids,
        trim_policy="earliest_hotspot",
        leiden_params={"threshold": 0.1, "resolution": 1.0, "seed": 0},
    )
    flights_by_flow, ctrl_by_flow = prepare_flow_scheduling_inputs(
        flight_list=fl,
        flow_map=flow_map,
        hotspot_ids=hotspot_ids,
    )

    # Build capacities
    caps = build_bin_capacities(str(caps_path), idx)

    # Attention cells: directly the chosen hotspots + their top windows
    target_cells = [(tv, b) for tv, bins in active_windows.items() for b in bins]
    ripple_cells = []  # keep empty for a tiny smoke test

    # Compute baseline (n=d) objective
    w = ObjectiveWeights()
    T = idx.num_time_bins
    n0 = {}
    for f, specs in flights_by_flow.items():
        arr = [0] * (T + 1)
        for sp in specs:
            rb = int(sp.get("requested_bin", 0))
            if 0 <= rb <= T:
                arr[rb] += 1
        n0[f] = arr
    J0, comps0, _ = score(
        n0,
        flights_by_flow=flights_by_flow,
        indexer=idx,
        capacities_by_tv=caps,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        flight_list=fl,
        weights=w,
        spill_mode="dump_to_next_bin",
    )

    # Run 150 iterations to validate improvement
    iters = 150
    print(f"Running SA for {iters} iterations...")
    params = SAParams(iterations=iters, warmup_moves=30, seed=0)
    n_best, J, comps, arts = run_sa(
        flights_by_flow=flights_by_flow,
        flight_list=fl,
        indexer=idx,
        capacities_by_tv=caps,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        weights=w,
        spill_mode="dump_to_next_bin",
    )

    improvement = float(J0 - J)
    pct = (improvement / J0 * 100.0) if J0 != 0 else 0.0
    print(json.dumps({
        "hotspots": hotspot_ids,
        "objective_baseline": J0,
        "objective_best": J,
        "improvement": improvement,
        "improvement_pct": round(pct, 2),
        "components": comps,
        "num_flows": len(n_best),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
