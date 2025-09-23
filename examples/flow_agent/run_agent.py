from __future__ import annotations

import json
import math
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
from collections import defaultdict, deque
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
    validate_plan_file,
    validate_plan_and_run_file,
)
from parrhesia.flow_agent.plan_export import save_plan_to_file
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
    early_stop_no_improvement: bool,
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
    # Agent flags
    tbl.add_row("Agent", "early_stop_no_improvement", str(early_stop_no_improvement))
    return tbl


def print_last_debug_lines(log_path: Path | str, num_lines: int = 200) -> None:
    """Print the last N lines of the given debug log file."""
    try:
        p = Path(log_path)
        if not p.exists():
            console.print(f"[yellow]Debug log not found:[/yellow] {p}")
            return
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            tail_lines = deque(fh, maxlen=int(num_lines))
        console.print(f"[bold]Last {int(num_lines)} debug log lines[/bold] — {p}")
        if tail_lines:
            console.print("".join(tail_lines))
        else:
            console.print("[dim](debug log is empty)[/dim]")
    except Exception as exc:
        console.print(f"[red]Failed to read debug log:[/red] {exc}")


# Moved the 15-minute delay validation into the shared panel under plan validation.


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
    cold_logger, cold_logger_path = SearchLogger.to_timestamped(str(log_dir), prefix="cold")
    # Debug logger intentionally disabled for this run (kept in codebase)
    debug_logger = None
    debug_logger_path = None
    console.print(f"[runner] Log path: {loggerpath}")
    console.print(f"[runner] Cold Feet log path: {cold_logger_path}")

    # Additional commit-attempts logger (separate JSONL file)
    commit_logger, commit_logger_path = SearchLogger.to_timestamped(str(log_dir), prefix="commit_attempts")
    console.print(f"[runner] Commit attempts log path: {commit_logger_path}")

    # Instrument CheapTransition inside the agent to record every CommitRegulation step
    try:
        import parrhesia.flow_agent.agent as _agent_mod
        from parrhesia.flow_agent.actions import CommitRegulation as _CommitRegulation
        from parrhesia.flow_agent.transition import CheapTransition as _BaseCheapTransition

        class _LoggingCheapTransition(_BaseCheapTransition):
            def step(self, state, action):  # type: ignore[override]
                if isinstance(action, _CommitRegulation):
                    try:
                        ctx = getattr(state, "hotspot_context", None)
                        if ctx is not None:
                            t0 = int(getattr(ctx, "window_bins", (0, 0))[0])
                            t1 = int(getattr(ctx, "window_bins", (0, 0))[1])
                            payload = {
                                "control_volume_id": str(getattr(ctx, "control_volume_id", "")),
                                "window_bins": [t0, t1],
                                "mode": str(getattr(ctx, "mode", "per_flow")),
                                "flow_ids": [str(fid) for fid in getattr(ctx, "selected_flow_ids", [])],
                                "committed_rates": getattr(action, "committed_rates", None),
                                "diagnostics": dict(getattr(action, "diagnostics", {})),
                            }
                            commit_logger.event("commit_attempt", payload)
                    except Exception:
                        pass
                return super().step(state, action)

        _agent_mod.CheapTransition = _LoggingCheapTransition
    except Exception as _exc:
        console.print(f"[yellow]Failed to instrument commit attempts:[/yellow] {_exc}")

    # Configure agent budgets small to keep runtime reasonable
    # Limit to a single regulation to shorten runtime and match the request
    mcts_cfg = MCTSConfig(
        max_sims=7680,
        commit_depth=64,
        commit_eval_limit=152,
        max_actions=1e32,
        seed=69420,
        debug_prints=False,
    )
    # Force full scorer for consistency investigation (can be toggled via env)
    os.environ.setdefault("RATE_FINDER_FAST_SCORER", "0")
    # Add warning panel about fast scorer being disabled
    from rich.panel import Panel
    
    
    rf_cfg = RateFinderConfig(use_adaptive_grid=True, max_eval_calls=192, fast_scorer_enabled=False) 

    warning_panel = Panel(
        "[yellow]⚠️  FAST_SCORER is disabled as it gives faulty results.[/yellow]\n"
        "[dim]This is the correct behavior. Investigation into FAST_SCORER is required.[/dim]",
        title="[bold red]WARNING[/bold red]",
        border_style="red",
        expand=False
    )
    console.print(warning_panel)
    console.print()


    disc_cfg = HotspotDiscoveryConfig(
        threshold=0.0,
        top_hotspots=64,
        top_flows=12,
        max_flights_per_flow=64,
        leiden_params={"threshold": 0.64, "resolution": 1.0, "seed": 0},
        direction_opts={"mode": "none"},
    )

    # Agent-level flags
    early_stop_no_improvement = False

    # Prepare Rich progress bar and callback
    prog = Progress(
        TextColumn("[bold blue]MCTS[/bold blue]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} sims"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn(" • nodes {task.fields[nodes]} • best ΔJ {task.fields[best]} • last ΔJ {task.fields[last]} • evals {task.fields[evals]} • acts {task.fields[actions]}/{task.fields[action_budget]}")
    )

    initial_budget = getattr(mcts_cfg, "max_actions", None)
    if isinstance(initial_budget, (int, float)) and initial_budget not in (None, 0):
        initial_budget_field = str(int(initial_budget))
    else:
        initial_budget_field = "∞"

    task_id = prog.add_task(
        "search",
        total=int(mcts_cfg.max_sims),
        nodes=0,
        best="—",
        last="—",
        evals=0,
        actions=0,
        action_budget=initial_budget_field,
    )

    last_payload: Dict[str, Any] = {}
    # Track total simulations attempted across all internal MCTS runs
    sim_counter: Dict[str, Any] = {"total": 0, "last_sd": 0, "init": False}
    live_holder: Dict[str, Any] = {"live": None}
    # Track outer-loop status (commits/limits/stop monitoring)
    outer_state: Dict[str, Any] = {
        "outer_commits": 0,
        "outer_commit_depth": int(mcts_cfg.commit_depth),
        "outer_max_regulation_count": None,
        "outer_limit": int(mcts_cfg.commit_depth),
        "outer_early_stop_no_improvement": bool(early_stop_no_improvement),
        "outer_stop_reason": None,
        "outer_stop_info": None,
        "outer_last_delta_j": None,
        "outer_run_index": 1,
    }

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
        tbl = Table(title="Root: Action Summary", box=box.SIMPLE_HEAVY)
        tbl.add_column("Action", style="cyan")
        tbl.add_column("Stage", style="yellow")
        tbl.add_column("Plan State", style="white")
        tbl.add_column("N", justify="right")
        tbl.add_column("avg Q", justify="right")
        tbl.add_column("avg P", justify="right")
        tbl.add_column("avg Reward", justify="right")
        aggregated = payload.get("all_action_stats") or {}
        rows = []
        for stage, stage_data in aggregated.items():
            for label, stats in stage_data.items():
                rows.append(
                    (
                        label,
                        str(stage),
                        str(stats.get("plan_state", "—")),
                        int(stats.get("N", 0)),
                        float(stats.get("avg_q", 0.0)),
                        float(stats.get("avg_prior", 0.0)),
                        float(stats.get("avg_reward", 0.0)),
                    )
                )
        rows.sort(key=lambda row: (-row[3], row[0]))
        if rows:
            for label, stage, plan_state, n, avg_q, avg_prior, avg_reward in rows:
                tbl.add_row(
                    label,
                    stage,
                    plan_state,
                    str(n),
                    f"{avg_q:.3f}",
                    f"{avg_prior:.3f}",
                    f"{avg_reward:.3f}",
                )
        else:
            plan_state = str(payload.get("root_plan_state", "—"))
            tbl.add_row("—", "?", plan_state, "0", "0.000", "0.000", "0.000")
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

    def _build_delta_table(payload: Dict[str, Any]) -> Table:
        tbl = Table(title="ΔJ", box=box.SIMPLE_HEAVY)
        tbl.add_column("Metric", style="green")
        tbl.add_column("Value", justify="right")
        best = payload.get("best_delta_j")
        last = payload.get("last_delta_j")
        best_s = f"{float(best):.3f}" if isinstance(best, (int, float)) else "—"
        last_s = f"{float(last):.3f}" if isinstance(last, (int, float)) else "—"
        tbl.add_row("Best ΔJ", best_s)
        tbl.add_row("Latest ΔJ", last_s)
        return tbl

    def _build_outer_table(state: Dict[str, Any]) -> Table:
        tbl = Table(title="Outer Loop", box=box.SIMPLE_HEAVY)
        tbl.add_column("Metric", style="yellow")
        tbl.add_column("Value", justify="right")
        commits = state.get("outer_commits", 0) or 0
        limit = state.get("outer_limit")
        limit_s = (str(int(limit)) if isinstance(limit, (int, float)) else "?")
        tbl.add_row("Commits", f"{int(commits)}/{limit_s}")
        cd = state.get("outer_commit_depth")
        tbl.add_row("Commit depth", (str(int(cd)) if isinstance(cd, (int, float)) else "?"))
        mr = state.get("outer_max_regulation_count")
        tbl.add_row("Max regulations", ("∞" if mr in (None, 0) else str(int(mr))))
        eno = state.get("outer_early_stop_no_improvement")
        tbl.add_row("Early stop if ΔJ≥0", ("true" if bool(eno) else "false"))
        lcdj = state.get("outer_last_delta_j")
        tbl.add_row("Last committed ΔJ", (f"{float(lcdj):.3f}" if isinstance(lcdj, (int, float)) else "—"))
        sr = state.get("outer_stop_reason")
        tbl.add_row("Stop reason", (str(sr) if sr else "—"))
        si = state.get("outer_stop_info") or {}
        if isinstance(si, dict) and si:
            # show a compact one-line view
            keys = list(si.keys())[:4]
            compact = {k: si[k] for k in keys}
            tbl.add_row("Stop info", str(compact))
        else:
            tbl.add_row("Stop info", "—")
        return tbl

    def _compose_live() -> Group:
        return Group(
            prog,
            # _build_outer_table(outer_state),  # Temporarily hidden per request
            _build_delta_table(last_payload),
            _build_root_table(last_payload),
            _build_actions_table(last_payload),
        )

    def _on_progress(payload: Dict[str, Any]) -> None:
        # Update inner-loop progress only when relevant keys are present
        if "sims_done" in payload:
            sims_done = int(payload.get("sims_done", 0))
            sims_total = int(payload.get("sims_total", 0)) or int(mcts_cfg.max_sims)
            nodes = int(payload.get("nodes", 0))
            best = payload.get("best_delta_j")
            last = payload.get("last_delta_j")
            evals = int(payload.get("commit_evals", 0))
            actions_raw = payload.get("actions_done", 0)
            actions_done = int(actions_raw) if isinstance(actions_raw, (int, float)) else 0
            budget_raw = payload.get("max_actions")
            cfg_budget = getattr(mcts_cfg, "max_actions", None)
            if isinstance(budget_raw, (int, float)) and budget_raw not in (0, None):
                action_budget_s = str(int(budget_raw))
            elif isinstance(cfg_budget, (int, float)) and cfg_budget not in (0, None):
                action_budget_s = str(int(cfg_budget))
            else:
                action_budget_s = "∞"
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
                actions=actions_done,
                action_budget=action_budget_s,
            )
            # Merge new inner-loop fields while preserving any outer-loop fields
            last_payload.update(payload)

        # Update outer-loop tracking if payload carries any outer_* fields
        if any(k.startswith("outer_") for k in payload.keys()):
            try:
                outer_state.update({k: v for k, v in payload.items() if k.startswith("outer_")})
            except Exception:
                pass
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
        cold_logger=cold_logger,
        max_regulation_count=1,
        timer=timed,
        progress_cb=_on_progress,
        early_stop_no_improvement=early_stop_no_improvement,
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
        early_stop_no_improvement=early_stop_no_improvement,
    ))

    console.print("[runner] Starting agent.run() ...")
    with Live(_compose_live(), refresh_per_second=8, console=console) as _live:
        live_holder["live"] = _live
        with timed("agent.run"):
            state, info = agent.run()
        live_holder["live"] = None
    logger.close()
    try:
        if cold_logger is not None:
            cold_logger.close()
    except Exception:
        pass
    try:
        if commit_logger is not None:
            commit_logger.close()
    except Exception:
        pass

    # Final summary and quick checks (non-fatal)
    console.print(f"[runner] Commits: {info.commits}")
    try:
        reason = getattr(info, "stop_reason", None)
        if reason:
            console.print(f"[runner] Stop reason: {reason}")
            si = getattr(info, "stop_info", None) or {}
            if isinstance(si, dict) and si:
                # Keep it concise on one line
                compact = {k: si[k] for k in list(si.keys())[:6]}
                console.print(f"[runner] Stop info: {compact}")
    except Exception:
        pass
    # Also report total simulations tried across all MCTS runs
    try:
        console.print(f"[runner] Total simulations tried: {int(sim_counter['total'])}")
    except Exception:
        pass
    if info.summary:
        final_obj = info.summary.get("objective")
        total_delta = info.total_delta_j
        if isinstance(final_obj, (int, float)):
            base_obj = final_obj - total_delta
            console.print(f"[runner] Final objective: {base_obj:,.3f} → {final_obj:,.3f} (ΔJ: {total_delta:,.3f})")
        else:
            console.print(f"[runner] Final objective: {final_obj}")
    if info.log_path:
        console.print(f"[runner] Log path: {info.log_path}")
    # Debug log is disabled for this run
    # Repeat the Action Counts table again for downstream quality-control parsing
    try:
        console.print(_build_actions_table(last_payload))
    except Exception:
        pass

    # Persist best-found regulation plan (including per-flow rates) next to logs
    try:
        out_path = save_plan_to_file(state, info, indexer)
        console.print(f"[runner] Plan exported: {out_path}")
        try:
            # Unified validation panel: structural checks + delay granularity from run log
            run_log_path = getattr(info, "log_path", None) or loggerpath
            if run_log_path:
                ok, _ = validate_plan_and_run_file(out_path, run_log_path)
            else:
                # Fallback to plan-only validation if run log missing
                ok = bool(validate_plan_file(out_path))
            if not ok:
                # Preserve prior behavior: fail run in tests or CLI
                msg = "Validation failed (see panel above)."
                if os.environ.get("PYTEST_CURRENT_TEST"):
                    raise AssertionError(msg)
                else:
                    raise SystemExit(2)
        except (SystemExit, AssertionError):
            raise
        except Exception as exc:
            console.print(f"[yellow]Failed to validate plan:[/yellow] {exc}")
    except Exception as exc:
        console.print(f"[yellow]Failed to export plan:[/yellow] {exc}")

    # Report Cold Feet stats: per-run max commits and overall max
    try:
        summary_dict = info.summary if isinstance(info.summary, dict) else {}
        mc_list = summary_dict.get("max_commits_per_inner")
        mc_overall = summary_dict.get("max_commits_overall")
        cc_list = summary_dict.get("commit_calls_per_inner")
        cc_total = summary_dict.get("commit_calls_total")

        # Commit evaluations summary (less confusing, both per-run and total)
        if isinstance(cc_list, list) and cc_list:
            total_val = int(cc_total) if isinstance(cc_total, (int, float)) else int(sum(int(x) for x in cc_list))
            console.print(f"[runner] Commit evaluations (per inner run): {cc_list} • total: {total_val}")
        else:
            console.print("[runner] Commit evaluations (per inner run): (none recorded)")

        # Max commits achieved within a single simulation path (per inner run)
        if isinstance(mc_list, list) and mc_list:
            overall_val = int(mc_overall) if isinstance(mc_overall, (int, float)) else int(max(mc_list))
            console.print(f"[runner] Max commits in any single path (per inner run): {mc_list} • overall: {overall_val}")
        else:
            console.print("[runner] Max commits in any single path (per inner run): (none recorded)")
    except Exception:
        pass
    # Delay granularity validation is now included in the unified panel above.
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
