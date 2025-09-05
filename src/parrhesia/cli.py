from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .indexing.tvtw_indexer import TVTWIndexer
from .optim.flight_list import FlightList
from .flows.flow_pipeline import collect_hotspot_flights, build_global_flows
from .optim.capacity import build_bin_capacities
from .optim.objective import ObjectiveWeights
from .optim.sa_optimizer import SAParams, prepare_flow_scheduling_inputs, run_sa


def _parse_cells(csv: str) -> List[Tuple[str, int]]:
    """Parse cells like 'TV1:18,TV2:19' into [(TV1,18),(TV2,19)]."""
    out: List[Tuple[str, int]] = []
    if not csv:
        return out
    for tok in csv.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            continue
        tv, b = tok.split(":", 1)
        try:
            out.append((tv, int(b)))
        except Exception:
            continue
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Parrhesia SA optimizer")
    ap.add_argument("--indexer", required=True, help="Path to tvtw_indexer.json")
    ap.add_argument("--occupancy", required=True, help="Path to SO6 occupancy JSON")
    ap.add_argument("--capacities", required=True, help="Path to GeoJSON with capacities")
    ap.add_argument("--hotspots", required=True, help="Comma-separated TV ids")
    ap.add_argument("--windows", default="", help="Comma-separated bin indices (global)")
    ap.add_argument("--iterations", type=int, default=200, help="SA iterations")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--targets", default="", help="CSV cells like 'TV:bin,...'")
    ap.add_argument("--ripples", default="", help="CSV cells like 'TV:bin,...'")
    args = ap.parse_args(argv)

    idx = TVTWIndexer.load(str(args.indexer))
    fl = FlightList.from_json(str(args.occupancy), idx)
    caps = build_bin_capacities(str(args.capacities), idx)

    hotspot_ids = [s.strip() for s in args.hotspots.split(",") if s.strip()]
    windows = [int(x) for x in args.windows.split(",") if x.strip()] if args.windows else None

    union_ids, _meta = collect_hotspot_flights(fl, hotspot_ids, active_windows=windows)
    flow_map = build_global_flows(
        fl, union_ids, hotspots=hotspot_ids, trim_policy="earliest_hotspot",
        leiden_params={"threshold": 0.1, "resolution": 1.0, "seed": args.seed},
    )
    flights_by_flow, ctrl_by_flow = prepare_flow_scheduling_inputs(
        flight_list=fl, flow_map=flow_map, hotspot_ids=hotspot_ids
    )

    target_cells = _parse_cells(args.targets)
    ripple_cells = _parse_cells(args.ripples)

    w = ObjectiveWeights()
    params = SAParams(iterations=args.iterations, seed=args.seed)
    n_best, J, comps, _arts = run_sa(
        flights_by_flow=flights_by_flow,
        flight_list=fl,
        indexer=idx,
        capacities_by_tv=caps,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        weights=w,
    )

    print(json.dumps({
        "objective": J,
        "components": comps,
        "flows": {str(int(k)): int(sum(v)) for k, v in n_best.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

