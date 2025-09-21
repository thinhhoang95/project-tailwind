from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pytest
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
from rich import box
from rich.live import Live
from rich.console import Group

# Time Profiling helpers ===
from contextlib import contextmanager
from collections import defaultdict
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



console = Console()


def _build_params_table(
    *,
    mcts_cfg: "MCTSConfig",
    rf_cfg: "RateFinderConfig",
    disc_cfg: "HotspotDiscoveryConfig",
    indexer: "TVTWIndexer",
    flight_list: "FlightList",
    occupancy_path: Path,
    indexer_path: Path,
    caps_path: Path,
) -> Table:
    tbl = Table(title="MCTS Agent Parameters", box=box.SIMPLE)
    tbl.add_column("Component", style="cyan", no_wrap=True)
    tbl.add_column("Parameter", style="magenta", no_wrap=True)
    tbl.add_column("Value", style="white")

    def add_cfg_rows(name: str, cfg: Any) -> None:
        for k, v in getattr(cfg, "__dict__", {}).items():
            tbl.add_row(name, str(k), str(v))

    # Data artifacts
    tbl.add_row("Artifacts", "occupancy", str(occupancy_path))
    tbl.add_row("Artifacts", "indexer", str(indexer_path))
    tbl.add_row("Artifacts", "capacities", str(caps_path))
    # Dataset summary
    tbl.add_row("Dataset", "flights", str(getattr(flight_list, "num_flights", "?")))
    tbl.add_row("Dataset", "tvs", str(len(getattr(indexer, "tv_id_to_idx", {}))))
    tbl.add_row("Dataset", "time_bins", str(getattr(indexer, "num_time_bins", "?")))
    # Configs
    add_cfg_rows("MCTS", mcts_cfg)
    add_cfg_rows("RateFinder", rf_cfg)
    add_cfg_rows("Discovery", disc_cfg)
    return tbl


def initiate_agent(tmp_path: Path) -> Optional[tuple]:
    # Locate required artifacts
    project_root = Path(__file__).resolve().parents[2]
    console.print(f"[runner] Project root: {project_root}")
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
        project_root / "output" / "wxm_sm_ih_maxpool.geojson"
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
        if os.environ.get("PYTEST_CURRENT_TEST"):
            pytest.skip("Artifacts for agent smoke test are unavailable")
        console.print("[yellow]Required artifacts not found; aborting run.[/yellow]")
        return None
        

    # Load indexer and flight list
    with timed("load_tvtw_indexer"):
        indexer = TVTWIndexer.load(str(indexer_path))
    with timed("load_occupancy_json"):
        with open(occupancy_path, "r", encoding="utf-8") as handle:
            _ = json.load(handle)

    with timed("init_flight_list"):
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
    with timed("load_caps_geojson"):
        caps_gdf = gpd.read_file(str(caps_path))
    if caps_gdf.empty:
        pytest.skip("Traffic volume capacity file did not contain any records")

    with timed("init_network_evaluator"):
        evaluator = NetworkEvaluator(caps_gdf, flight_list)

    # Logger to tmp path
    log_dir = tmp_path / "runs"
    logger, loggerpath = SearchLogger.to_timestamped(str(log_dir))
    debug_logger, debug_logger_path = SearchLogger.to_timestamped(str(log_dir), prefix="debug")
    console.print(f"[runner] Log path: {loggerpath}")
    console.print(f"[runner] Debug log path: {debug_logger_path}")

    # Configure agent budgets small to keep runtime reasonable
    # Limit to a single regulation to shorten runtime and match the request
    mcts_cfg = MCTSConfig(max_sims=128, commit_depth=16, commit_eval_limit=16, seed=69420, debug_prints=True)
    rf_cfg = RateFinderConfig(use_adaptive_grid=True, max_eval_calls=4)
    disc_cfg = HotspotDiscoveryConfig(
        threshold=0.0,
        top_hotspots=32,
        top_flows=6,
        max_flights_per_flow=64,
        leiden_params={"threshold": 0.64, "resolution": 1.0, "seed": 0},
        direction_opts={"mode": "none"},
    )

    # Prepare Rich progress bar and callback
    prog = Progress(
        TextColumn("[bold blue]MCTS[/bold blue]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} sims"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn(" • nodes {task.fields[nodes]} • best ΔJ {task.fields[best]} • last ΔJ {task.fields[last]} • evals {task.fields[evals]}")
    )

    task_id = prog.add_task(
        "search",
        total=int(mcts_cfg.max_sims),
        nodes=0,
        best="—",
        last="—",
        evals=0,
    )

    last_payload: Dict[str, Any] = {}
    # Track total simulations attempted across all internal MCTS runs
    sim_counter: Dict[str, Any] = {"total": 0, "last_sd": 0, "init": False}
    live_holder: Dict[str, Any] = {"live": None}

    def _sig_to_label(sig: Any) -> str:
        try:
            if not isinstance(sig, (list, tuple)) or not sig:
                return str(sig)
            kind = sig[0]
            if kind == "add":
                return f"add({sig[1]})"
            if kind == "rem":
                return f"rem({sig[1]})"
            if kind == "cont":
                return "cont"
            if kind == "back":
                return "back"
            if kind == "commit":
                return f"commit({max(0, len(sig)-1)})"
            if kind == "new_reg":
                return "new_reg"
            if kind == "hotspot":
                return f"hotspot({sig[1]},{sig[2]}-{sig[3]})"
            if kind == "stop":
                return "stop"
            return str(sig)
        except Exception:
            return str(sig)

    def _build_root_table(payload: Dict[str, Any]) -> Table:
        tbl = Table(title="Root: Top Visits", box=box.SIMPLE_HEAVY)
        tbl.add_column("Action", style="cyan")
        tbl.add_column("N", justify="right")
        tbl.add_column("Q", justify="right")
        tbl.add_column("P", justify="right")
        rows = payload.get("root_top", []) or []
        for item in rows:
            try:
                sig, n, q, p = item
            except Exception:
                continue
            tbl.add_row(_sig_to_label(sig), str(int(n)), f"{float(q):.3f}", f"{float(p):.3f}")
        if not rows:
            tbl.add_row("—", "0", "0.000", "0.000")
        return tbl

    def _build_actions_table(payload: Dict[str, Any]) -> Table:
        tbl = Table(title="Action Counts", box=box.SIMPLE_HEAVY)
        tbl.add_column("Action", style="magenta")
        tbl.add_column("Count", justify="right")
        ac = payload.get("action_counts", {}) or {}
        for k, v in sorted(ac.items(), key=lambda kv: (-kv[1], kv[0])):
            tbl.add_row(str(k), str(int(v)))
        if not ac:
            tbl.add_row("—", "0")
        return tbl

    def _compose_live() -> Group:
        return Group(
            prog,
            _build_root_table(last_payload),
            _build_actions_table(last_payload),
        )

    def _on_progress(payload: Dict[str, Any]) -> None:
        sims_done = int(payload.get("sims_done", 0))
        sims_total = int(payload.get("sims_total", 0)) or int(mcts_cfg.max_sims)
        nodes = int(payload.get("nodes", 0))
        best = payload.get("best_delta_j")
        last = payload.get("last_delta_j")
        evals = int(payload.get("commit_evals", 0))
        best_s = f"{best:.3f}" if isinstance(best, (int, float)) else "—"
        last_s = f"{last:.3f}" if isinstance(last, (int, float)) else "—"
        # Accumulate simulations across runs. If sims_done resets (new run),
        # only count the forward progress and ignore negative deltas.
        if not sim_counter["init"]:
            sim_counter["init"] = True
            sim_counter["total"] += max(0, sims_done)
            sim_counter["last_sd"] = sims_done
        else:
            delta = sims_done - int(sim_counter["last_sd"])
            if delta >= 0:
                sim_counter["total"] += delta
            else:
                # New run likely started (counter reset); add current as progress from 0
                sim_counter["total"] += max(0, sims_done)
            sim_counter["last_sd"] = sims_done
        prog.update(
            task_id,
            completed=sims_done,
            total=max(sims_total, sims_done),
            nodes=nodes,
            best=best_s,
            last=last_s,
            evals=evals,
        )
        last_payload.clear(); last_payload.update(payload)
        live = live_holder.get("live")
        if live is not None:
            live.update(_compose_live())

    agent = MCTSAgent(
        evaluator=evaluator,
        flight_list=flight_list,
        indexer=indexer,
        mcts_cfg=mcts_cfg,
        rate_finder_cfg=rf_cfg,
        discovery_cfg=disc_cfg,
        logger=logger,
        debug_logger=debug_logger,
        max_regulations=24,
        timer=timed,
        progress_cb=_on_progress,
    )

    # Show parameter table once before starting
    console.print(_build_params_table(
        mcts_cfg=mcts_cfg,
        rf_cfg=rf_cfg,
        disc_cfg=disc_cfg,
        indexer=indexer,
        flight_list=flight_list,
        occupancy_path=occupancy_path,
        indexer_path=indexer_path,
        caps_path=caps_path,
    ))

    console.print("[runner] Starting agent.run() ...")
    with Live(_compose_live(), refresh_per_second=8, console=console) as _live:
        live_holder["live"] = _live
        with timed("agent.run"):
            state, info = agent.run()
        live_holder["live"] = None
    logger.close()
    debug_logger.close()

    # Final summary and quick checks (non-fatal)
    console.print(f"[runner] Commits: {info.commits}")
    # Also report total simulations tried across all MCTS runs
    try:
        console.print(f"[runner] Total simulations tried: {int(sim_counter['total'])}")
    except Exception:
        pass
    if info.summary:
        console.print(f"[runner] Final objective: {info.summary.get('objective')}")
    if info.log_path:
        console.print(f"[runner] Log path: {info.log_path}")
    if info.debug_log_path:
        console.print(f"[runner] Debug log path: {info.debug_log_path}")
    return state, info

if __name__ == '__main__':
    from rich.panel import Panel
    
    console.print(Panel.fit(
        "[bold cyan]AIR TRAFFIC FLOW MANAGEMENT REGULATION AGENT[/bold cyan]\n\n"
        "[dim]Created by:[/dim] [green]Thinh Hoang[/green] ([blue]thinh.hoangdinh@enac.fr[/blue])\n"
        "[dim]Latest Revisions:[/dim] [yellow]21/09/2025[/yellow]",
        title="[bold magenta]MCTS Flow Agent[/bold magenta]",
        border_style="bright_blue",
        padding=(1, 2)
    ))

    run_dir = Path('agent_runs')
    run_dir.mkdir(parents=True, exist_ok=True)
    initiate_agent(run_dir)
