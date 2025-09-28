from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np


# Ensure 'src' is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Core data structures
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator

# Hotspot inventory (flows + window discovery)
from parrhesia.flow_agent.hotspot_discovery import (
    HotspotInventory,
    HotspotDiscoveryConfig,
)


# Time Profiling helpers ===
from contextlib import contextmanager
from collections import Counter, defaultdict, deque
import time, atexit

_stats = defaultdict(lambda: [0, 0.0])  # name -> [calls, total_seconds]
@contextmanager
def timed(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _stats[name][0] += 1
        _stats[name][1] += dt

@atexit.register
def _report_timings():
    if not _stats:
        return
    total = sum(sec for _, sec in _stats.values())
    width = max(len(k) for k in _stats)
    print("\n=== Timing summary (wall time) ===")
    for name, (calls, sec) in sorted(_stats.items(), key=lambda kv: kv[1][1], reverse=True):
        avg = sec / calls
        share = (sec / total) if total else 0
        print(f"{name:<{width}}  total {sec*1000:8.1f} ms  avg {avg*1000:7.1f} ms  calls {calls:5d}  share {share:5.1%}")
# End profiling helpers ===


def _pick_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_artifacts() -> Tuple[Path, Path, Path]:
    """
    Resolve and validate artifact paths (occupancy, indexer, capacities).
    """
    project_root = REPO_ROOT

    occupancy_candidates = [
        project_root / "output" / "so6_occupancy_matrix_with_times.json",
        project_root / "data" / "tailwind" / "so6_occupancy_matrix_with_times.json",
    ]
    indexer_candidates = [
        project_root / "output" / "tvtw_indexer.json",
        project_root / "data" / "tailwind" / "tvtw_indexer.json",
    ]
    caps_candidates = [
        Path("/mnt/d/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"),
        Path("D:/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"),
        Path("/Volumes/CrucialX/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"),
        project_root / "data" / "cirrus" / "wxm_sm_ih_maxpool.geojson",
        project_root / "output" / "wxm_sm_ih_maxpool.geojson",
    ]

    occupancy_path = _pick_existing_path(occupancy_candidates)
    indexer_path = _pick_existing_path(indexer_candidates)
    caps_path = _pick_existing_path(caps_candidates)

    if not occupancy_path or not indexer_path or not caps_path:
        raise SystemExit(
            "Required artifacts not found. Ensure occupancy, indexer, and capacity GeoJSON exist."
        )

    return occupancy_path, indexer_path, caps_path


def build_data(occupancy_path: Path, indexer_path: Path, caps_path: Path) -> Tuple[FlightList, TVTWIndexer, NetworkEvaluator]:
    # Load indexer and occupancy
    indexer = TVTWIndexer.load(str(indexer_path))

    # Touch occupancy JSON to validate read
    with open(occupancy_path, "r", encoding="utf-8") as fh:
        _ = json.load(fh)

    flight_list = FlightList(str(occupancy_path), str(indexer_path))

    # Align occupancy matrix width (defensive, mirroring examples/flow_agent/run_agent.py)
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

    # Capacities
    caps_gdf = gpd.read_file(str(caps_path))
    if caps_gdf.empty:
        raise SystemExit("Traffic volume capacity file is empty; cannot proceed.")

    evaluator = NetworkEvaluator(caps_gdf, flight_list)
    return flight_list, indexer, evaluator


def pick_top_hotspot(inventory: HotspotInventory, *, threshold: float = 0.0) -> Optional[Dict[str, Any]]:
    cfg = HotspotDiscoveryConfig(
        threshold=float(threshold),
        top_hotspots=16,
        top_flows=12,
        max_flights_per_flow=64,
        min_flights_per_flow=5,
        direction_opts={"mode": "none"},
    )
    descs = inventory.build_from_segments(
        threshold=cfg.threshold,
        top_hotspots=cfg.top_hotspots,
        top_flows=cfg.top_flows,
        min_flights_per_flow=cfg.min_flights_per_flow,
        max_flights_per_flow=cfg.max_flights_per_flow,
        leiden_params={"threshold": 0.64, "resolution": 1.0, "seed": 0},
        direction_opts=cfg.direction_opts,
    )
    if not descs:
        return None

    # Already ranked by exceedance (via evaluator.get_hotspot_segments), pick first
    d0 = descs[0]
    payload = {
        "control_volume_id": d0.control_volume_id,
        "window_bins": list(d0.window_bins),
        "candidate_flow_ids": list(d0.candidate_flow_ids),
        "metadata": dict(d0.metadata),
        "hotspot_prior": float(d0.hotspot_prior),
        "mode": d0.mode,
    }
    return payload


def propose_one_regulation(
    *,
    hotspot_payload: Dict[str, Any],
    evaluator: NetworkEvaluator,
    indexer: TVTWIndexer,
) -> Dict[str, Any]:
    pass # TODO: implement this


def main() -> None:
    print("[regen] Resolving artifacts ...")
    occ_path, idx_path, caps_path = load_artifacts()
    print(f"[regen] occupancy: {occ_path}")
    print(f"[regen] indexer:   {idx_path}")
    print(f"[regen] capacities:{caps_path}")

    print("[regen] Loading data ...")
    flight_list, indexer, evaluator = build_data(occ_path, idx_path, caps_path)

    print("[regen] Building hotspot inventory ...")
    inventory = HotspotInventory(evaluator=evaluator, flight_list=flight_list, indexer=indexer)

    top = pick_top_hotspot(inventory, threshold=0.0)
    if not top:
        print("[regen] No hotspots detected above threshold.")
        return

    ctrl = top.get("control_volume_id")
    win = top.get("window_bins")
    print(f"[regen] Top hotspot: TV={ctrl} window={win}")

    print("[regen] Proposing one regulation (per-flow) ...")
    proposal = propose_one_regulation(hotspot_payload=top, evaluator=evaluator, indexer=indexer)

    # Print concise summary
    flow = proposal["flows"][0]
    diag = proposal["diagnostics"]
    print(
        "[regen] Proposal → TV=\n"
        f"  {proposal['control_volume_id']} bins={proposal['window_bins'][0]}-{proposal['window_bins'][1]}\n"
        f"  flow={flow['flow_id']} r0={flow['baseline_rate_per_hour']:.1f}/h → R={flow['allowed_rate_per_hour']:.1f}/h (Δ={flow['assigned_cut_per_hour']:.0f}/h)\n"
        f"  entrants={flow['entrants_in_window']:.0f} flights={flow['num_flights']} E_target={diag['E_target']:.1f} (D_peak={diag['D_peak']:.1f}, D_sum={diag['D_sum']:.1f})"
    )

    # Optional: write to disk next to artifacts
    out_dir = REPO_ROOT / "agent_runs" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regen_proposal.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(proposal, fh, indent=2)
    print(f"[regen] Proposal saved: {out_path}")


if __name__ == "__main__":
    main()


