from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pytest

# Ensure 'src' is importable when running tests directly
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from parrhesia.flow_agent import (
    MCTSAgent,
    MCTSConfig,
    RateFinderConfig,
    SearchLogger,
    HotspotDiscoveryConfig,
)
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator


@pytest.mark.slow
def test_mcts_agent_real_data_smoke(tmp_path: Path):
    # Locate required artifacts
    project_root = Path(__file__).resolve().parents[2]
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
    ]

    def _pick_path(cands: List[Path]) -> Path | None:
        for p in cands:
            if p.exists():
                return p
        return None

    occupancy_path = _pick_path(occupancy_candidates)
    indexer_path = _pick_path(indexer_candidates)
    caps_path = _pick_path(caps_candidates)

    if not occupancy_path or not indexer_path or not caps_path:
        pytest.skip("Artifacts for agent smoke test are unavailable")

    # Load indexer and flight list
    indexer = TVTWIndexer.load(str(indexer_path))
    with open(occupancy_path, "r", encoding="utf-8") as handle:
        _ = json.load(handle)

    flight_list = FlightList(str(occupancy_path), str(indexer_path))

    # Align matrix width in case the artifact was built with fewer TVs than indexer knows
    expected_tvtws = len(indexer.tv_id_to_idx) * indexer.num_time_bins
    if flight_list.num_tvtws < expected_tvtws:
        from scipy import sparse

        pad_cols = expected_tvtws - flight_list.num_tvtws
        pad_matrix = sparse.lil_matrix((flight_list.num_flights, pad_cols))
        flight_list._occupancy_matrix_lil = sparse.hstack(
            [flight_list._occupancy_matrix_lil, pad_matrix], format="lil"
        )
        flight_list.num_tvtws = expected_tvtws
        flight_list._temp_occupancy_buffer = np.zeros(expected_tvtws, dtype=np.float32)
        flight_list._lil_matrix_dirty = True
        flight_list._sync_occupancy_matrix()

    # Capacities
    caps_gdf = gpd.read_file(str(caps_path))
    if caps_gdf.empty:
        pytest.skip("Traffic volume capacity file did not contain any records")

    evaluator = NetworkEvaluator(caps_gdf, flight_list)

    # Logger to tmp path
    log_dir = tmp_path / "runs"
    print(f"[test] Log dir: {log_dir}")
    logger = SearchLogger.to_timestamped(str(log_dir))

    # Configure agent budgets small to keep runtime reasonable
    # Limit to a single regulation to shorten runtime and match the request
    mcts_cfg = MCTSConfig(max_sims=12, commit_depth=1, commit_eval_limit=6, seed=0)
    rf_cfg = RateFinderConfig(use_adaptive_grid=True, max_eval_calls=128)
    disc_cfg = HotspotDiscoveryConfig(
        threshold=0.0,
        top_hotspots=5,
        top_flows=3,
        max_flights_per_flow=15,
        leiden_params={"threshold": 0.3, "resolution": 1.0, "seed": 0},
        direction_opts={"mode": "none"},
    )

    agent = MCTSAgent(
        evaluator=evaluator,
        flight_list=flight_list,
        indexer=indexer,
        mcts_cfg=mcts_cfg,
        rate_finder_cfg=rf_cfg,
        discovery_cfg=disc_cfg,
        logger=logger,
        max_regulations=1,
    )

    print("[test] Starting agent.run() ...")
    state, info = agent.run()
    logger.close()

    # Assertions
    print(f"[test] Commits: {info.commits}")
    assert info.commits >= 1, "agent should commit at least one regulation"
    assert state.plan, "plan should contain committed regulations"

    # At least one commit should have negative delta_j (improvement)
    deltas: List[float] = []
    for i, reg in enumerate(state.plan, 1):
        di = (reg.diagnostics or {}).get("rate_finder", {})
        if "delta_j" in di:
            v = float(di["delta_j"]) 
            deltas.append(v)
            print(f"[test] Commit {i}: delta_j={v}")
    assert deltas and min(deltas) < 0.0, "expected at least one improving commit"

    # Final global objective sanity
    summary = info.summary or {}
    print(f"[test] Final objective: {summary.get('objective')}")
    assert np.isfinite(float(summary.get("objective", np.inf))), "final objective must be finite"

    # Log file exists and contains at least run_start and run_end entries
    assert info.log_path and os.path.exists(info.log_path), "expected search log file to be created"
    print(f"[test] Log path: {info.log_path}")
    with open(info.log_path, "r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh.readlines() if l.strip()]
    assert lines, "log file should not be empty"
    assert any('"type":"run_start"' in ln for ln in lines), "missing run_start event"
    assert any('"type":"run_end"' in ln for ln in lines), "missing run_end event"
    print(f"[test] Wrote {len(lines)} log lines")
