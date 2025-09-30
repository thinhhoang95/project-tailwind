from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping, Sequence, Tuple

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


class GlobalStats:
    """Session-wide aggregator for monitoring and telemetry."""

    def __init__(self, *, logger: Optional[SearchLogger] = None) -> None:
        self.outer_runs_started: int = 0
        self.outer_runs_completed: int = 0
        self._outer_last_index: int = 0
        self.started_at_ts: float = time.time()
        self.last_snapshot_ts: float = self.started_at_ts

        self.outer_max_run_param: Optional[int] = None
        self.outer_max_time: Optional[float] = None
        self.outer_max_evals: Optional[int] = None
        self.outer_max_expansions: Optional[int] = None
        self.outer_max_actions: Optional[int] = None

        self.last_stop_reason: Optional[str] = None
        self.last_stop_info: Optional[Dict[str, Any]] = None

        self.best_objective_value: Optional[float] = None
        self.best_delta_j: Optional[float] = None
        self.best_outer_iter: Optional[int] = None
        self.best_plan_id: Optional[str] = None
        self.best_timestamp: Optional[float] = None
        self.best_plan_summary: Optional[str] = None
        self.best_N_flights_flows: Optional[int] = None
        self.best_N_entrants_flows: Optional[int] = None

        self.action_counts: Counter[str] = Counter()
        self.inner_stats: Dict[str, Any] = {}

        self._logger = logger
        self._current_outer_index: Optional[int] = None
        self._current_outer_started_at: Optional[float] = None
        self._best_updated_this_outer: bool = False
        self._outer_runs_since_best: int = 0

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            iv = int(value)
            return iv
        except Exception:
            return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            fv = float(value)
            if math.isfinite(fv):
                return fv
            return None
        except Exception:
            return None

    @staticmethod
    def _normalize_flow_map(flow_map: Mapping[str, Sequence[Any]] | None) -> Dict[str, List[str]]:
        if not isinstance(flow_map, Mapping):
            return {}
        out: Dict[str, List[str]] = {}
        for key, flights in flow_map.items():
            try:
                key_s = str(key)
            except Exception:
                key_s = repr(key)
            items: List[str] = []
            if isinstance(flights, Sequence):
                for item in flights:
                    try:
                        items.append(str(item))
                    except Exception:
                        continue
            out[key_s] = items
        return out

    @staticmethod
    def _normalize_entrants(entrants: Mapping[str, Any] | None) -> Dict[str, int]:
        if not isinstance(entrants, Mapping):
            return {}
        out: Dict[str, int] = {}
        for key, value in entrants.items():
            try:
                key_s = str(key)
            except Exception:
                key_s = repr(key)
            try:
                out[key_s] = int(value)
            except Exception:
                out[key_s] = 0
        return out

    @staticmethod
    def _compute_flow_metrics(
        flow_to_flights: Mapping[str, Sequence[str]] | None,
        entrants_by_flow: Mapping[str, int] | None,
    ) -> Tuple[Optional[int], Optional[int]]:
        flights_total: Optional[int] = None
        if flow_to_flights:
            unique: set[str] = set()
            for flights in flow_to_flights.values():
                for fid in flights:
                    unique.add(str(fid))
            flights_total = len(unique)

        entrants_total: Optional[int] = None
        if entrants_by_flow:
            total = 0
            for count in entrants_by_flow.values():
                try:
                    total += int(count)
                except Exception:
                    continue
            entrants_total = total
        return flights_total, entrants_total

    @staticmethod
    def _summarize_plan(meta: Mapping[str, Any], flow_to_flights: Mapping[str, Sequence[str]]) -> str:
        ctrl = meta.get("control_volume_id")
        t0, t1 = meta.get("window_bins", [None, None])
        mode = meta.get("mode", "?")
        flow_ids = meta.get("flow_ids")
        if not isinstance(flow_ids, Sequence) or isinstance(flow_ids, (str, bytes)):
            flow_ids = list(flow_to_flights.keys())
        flow_count = len(flow_ids) if flow_ids else 0
        rates = meta.get("committed_rates") or meta.get("rates")
        rate_summary = ""
        if isinstance(rates, Mapping):
            try:
                vals = [float(v) for v in rates.values() if float(v) > 0]
                if vals:
                    rate_summary = f" rates≈[{min(vals):.0f},{max(vals):.0f}]"
            except Exception:
                rate_summary = ""
        elif isinstance(rates, (int, float)):
            try:
                if float(rates) > 0:
                    rate_summary = f" rate≈{float(rates):.0f}"
            except Exception:
                rate_summary = ""
        return (
            f"TV {ctrl} bins {t0}-{t1} [{mode}] flows={flow_count}{rate_summary}"
            if ctrl is not None
            else f"[{mode}] flows={flow_count}{rate_summary}"
        )

    def _top_actions(self, limit: int = 5) -> List[Dict[str, Any]]:
        total = sum(self.action_counts.values())
        if total <= 0:
            return []
        top_items = self.action_counts.most_common(limit)
        out: List[Dict[str, Any]] = []
        for action, count in top_items:
            percent = (count / total) * 100.0 if total else 0.0
            out.append({"action": action, "count": count, "percent": percent})
        return out

    def _emit_snapshot(self, event: str, extra: Optional[Mapping[str, Any]] = None) -> None:
        if self._logger is None:
            return
        payload: Dict[str, Any] = {
            "event": event,
            "timestamp": time.time(),
            "outer_runs_started": self.outer_runs_started,
            "outer_runs_completed": self.outer_runs_completed,
            "outer_max_run_param": self.outer_max_run_param,
            "outer_max_time": self.outer_max_time,
            "outer_max_actions": self.outer_max_actions,
            "outer_max_evals": self.outer_max_evals,
            "outer_max_expansions": self.outer_max_expansions,
            "last_stop_reason": self.last_stop_reason,
            "best_objective_value": self.best_objective_value,
            "best_delta_j": self.best_delta_j,
            "best_outer_iter": self.best_outer_iter,
            "best_plan_id": self.best_plan_id,
            "best_plan_summary": self.best_plan_summary,
            "best_N_flights_flows": self.best_N_flights_flows,
            "best_N_entrants_flows": self.best_N_entrants_flows,
            # Clarify scope of the "best" metrics captured here
            "best_scope": "local_candidate",
            "best_scope_desc": "Best candidate at a single TV/window; objective and ΔJ are local, not system-wide",
            "actions_top": self._top_actions(),
            "actions_total": sum(self.action_counts.values()),
        }
        if extra:
            for key, value in extra.items():
                if key not in payload:
                    payload[key] = value
        event_name = "global_best_update" if event == "best_update" else "global_stats_snapshot"
        self._logger.event(event_name, payload)

    def on_outer_start(self, outer_index: int, limits: Optional[Mapping[str, Any]] = None) -> None:
        self.outer_runs_started = max(self.outer_runs_started, int(max(outer_index, 0)))
        self._current_outer_index = int(max(outer_index, 0))
        self._current_outer_started_at = time.time()
        self.last_snapshot_ts = self._current_outer_started_at
        self._best_updated_this_outer = False

        if limits:
            maybe_runs = self._coerce_int(limits.get("outer_max_run_param"))
            if maybe_runs and maybe_runs > 0:
                self.outer_max_run_param = maybe_runs
            maybe_time = self._coerce_float(limits.get("outer_max_time"))
            if maybe_time and maybe_time > 0:
                self.outer_max_time = maybe_time
            maybe_actions = self._coerce_int(limits.get("outer_max_actions"))
            if maybe_actions and maybe_actions > 0:
                self.outer_max_actions = maybe_actions
            maybe_evals = self._coerce_int(limits.get("outer_max_evals"))
            if maybe_evals and maybe_evals > 0:
                self.outer_max_evals = maybe_evals
            maybe_exp = self._coerce_int(limits.get("outer_max_expansions"))
            if maybe_exp and maybe_exp > 0:
                self.outer_max_expansions = maybe_exp

    def on_outer_end(self, outer_index: int, meta: Optional[Mapping[str, Any]] = None) -> None:
        self.outer_runs_completed += 1
        self._outer_last_index = int(max(outer_index, 0))
        self.last_snapshot_ts = time.time()
        if not self._best_updated_this_outer and self.outer_runs_completed > 0:
            self._outer_runs_since_best += 1
        if meta:
            stop_reason = meta.get("stop_reason")
            if isinstance(stop_reason, str):
                self.last_stop_reason = stop_reason
            stop_info = meta.get("stop_info")
            if isinstance(stop_info, Mapping):
                self.last_stop_info = dict(stop_info)
            total_delta = meta.get("total_delta_j")
            if total_delta is not None:
                try:
                    meta_total_delta = float(total_delta)
                except Exception:
                    meta_total_delta = None
            else:
                meta_total_delta = None
            extra = {
                "outer_index": int(max(outer_index, 0)),
                "commits": meta.get("commits"),
                "total_delta_j": meta_total_delta,
                "stop_reason": self.last_stop_reason,
            }
        else:
            extra = {"outer_index": int(max(outer_index, 0))}
        self._emit_snapshot("outer_end", extra)
        self._current_outer_index = None
        self._current_outer_started_at = None

    def on_candidate_scored(
        self,
        candidate_id: str,
        objective: float,
        delta_j: float,
        meta: Mapping[str, Any],
    ) -> None:
        flow_map = self._normalize_flow_map(meta.get("flow_to_flights"))
        entrants_map = self._normalize_entrants(meta.get("entrants_by_flow"))
        self._maybe_update_best(
            source="candidate",
            identifier=candidate_id,
            objective=self._coerce_float(objective),
            delta_j=self._coerce_float(delta_j),
            flow_to_flights=flow_map,
            entrants_by_flow=entrants_map,
            meta=meta,
        )

    def on_action(self, action_type: str, count: int = 1) -> None:
        if count <= 0:
            return
        self.action_counts[action_type] += int(count)

    def on_plan_committed(
        self,
        plan: Sequence[Any],
        delta_j: float,
        flow_to_flights: Mapping[str, Sequence[str]] | None,
        entrants_by_flow: Mapping[str, Any] | None,
        meta: Mapping[str, Any],
    ) -> None:
        normalized_flows = self._normalize_flow_map(flow_to_flights)
        entrants_map = self._normalize_entrants(entrants_by_flow)
        extended_meta = dict(meta)
        extended_meta.setdefault("flow_ids", [str(idx) for idx in normalized_flows.keys()])
        extended_meta.setdefault("committed_rates", meta.get("committed_rates") or meta.get("rates"))
        self._maybe_update_best(
            source="commit",
            identifier=str(meta.get("plan_id") or len(plan)),
            objective=self._coerce_float(meta.get("objective")),
            delta_j=self._coerce_float(delta_j),
            flow_to_flights=normalized_flows,
            entrants_by_flow=entrants_map,
            meta=extended_meta,
        )

    def on_limit_hit(self, kind: str, meta: Optional[Mapping[str, Any]] = None) -> None:
        self.last_stop_reason = str(kind)
        if isinstance(meta, Mapping):
            self.last_stop_info = dict(meta)
        else:
            self.last_stop_info = None

    def update_inner_stats(self, payload: Mapping[str, Any]) -> None:
        try:
            self.inner_stats = dict(payload)
        except Exception:
            self.inner_stats = {}

    def snapshot_for_display(self) -> Dict[str, Any]:
        now = time.time()
        elapsed = now - self.started_at_ts
        time_since_outer = None
        if self._current_outer_started_at is not None:
            time_since_outer = now - self._current_outer_started_at
        time_since_best = None
        if self.best_timestamp is not None:
            time_since_best = max(0.0, now - self.best_timestamp)
        outer_pct = None
        if self.outer_max_run_param:
            outer_pct = (self.outer_runs_completed / self.outer_max_run_param) * 100.0
        time_pct = None
        if self.outer_max_time and self.outer_max_time > 0:
            time_pct = min(100.0, (elapsed / self.outer_max_time) * 100.0)
        progress = {
            "outer_runs_completed": self.outer_runs_completed,
            "outer_runs_started": self.outer_runs_started,
            "outer_last_index": self._outer_last_index,
            "outer_max_run_param": self.outer_max_run_param,
            "outer_progress_pct": outer_pct,
            "time_elapsed": elapsed,
            "outer_max_time": self.outer_max_time,
            "time_progress_pct": time_pct,
            "outer_max_actions": self.outer_max_actions,
            "outer_max_evals": self.outer_max_evals,
            "outer_max_expansions": self.outer_max_expansions,
            "current_outer_index": self._current_outer_index,
            "time_since_outer_start": time_since_outer,
            "last_stop_reason": self.last_stop_reason,
            "last_stop_info": self.last_stop_info,
            "outer_runs_since_best": self._outer_runs_since_best,
        }
        best = {
            "objective": self.best_objective_value,
            "delta_j": self.best_delta_j,
            "outer_iter": self.best_outer_iter,
            "plan_id": self.best_plan_id,
            "timestamp": self.best_timestamp,
            "time_since_best": time_since_best,
            "plan_summary": self.best_plan_summary,
            "n_flights_flows": self.best_N_flights_flows,
            "n_entrants_flows": self.best_N_entrants_flows,
        }
        actions = {
            "top": self._top_actions(),
            "total": sum(self.action_counts.values()),
        }
        return {
            "progress": progress,
            "best": best,
            "actions": actions,
            "inner": dict(self.inner_stats),
        }

    def _maybe_update_best(
        self,
        *,
        source: str,
        identifier: str,
        objective: Optional[float],
        delta_j: Optional[float],
        flow_to_flights: Mapping[str, Sequence[str]],
        entrants_by_flow: Mapping[str, int],
        meta: Mapping[str, Any],
    ) -> None:
        if objective is None or (isinstance(objective, (int, float)) and not math.isfinite(float(objective))):
            # Emit a diagnostic trace to the run log before raising to surface the root cause
            if self._logger is not None:
                try:
                    outer_idx_hint = None
                    plan_id_hint = None
                    meta_keys = None
                    try:
                        outer_idx_hint = meta.get("outer_index")
                    except Exception:
                        outer_idx_hint = None
                    try:
                        plan_id_hint = meta.get("plan_id")
                    except Exception:
                        plan_id_hint = None
                    try:
                        meta_keys = list(meta.keys())
                    except Exception:
                        meta_keys = None
                    self._logger.event(
                        "invalid_best_objective",
                        {
                            "source": str(source),
                            "identifier": str(identifier),
                            "objective": objective,
                            "delta_j": delta_j,
                            "outer_index_hint": outer_idx_hint,
                            "plan_id_hint": plan_id_hint,
                            "meta_keys": meta_keys,
                        },
                    )
                except Exception:
                    pass
            raise ValueError(f"Invalid objective for best-plan update (source={source}, id={identifier}): {objective}")
        if self.best_objective_value is not None and objective >= self.best_objective_value:
            return

        flights_total, entrants_total = self._compute_flow_metrics(flow_to_flights, entrants_by_flow)
        summary = self._summarize_plan(meta, flow_to_flights)
        now = time.time()
        outer_idx = meta.get("outer_index")
        if not isinstance(outer_idx, int):
            outer_idx = self._current_outer_index

        self.best_objective_value = objective
        self.best_delta_j = delta_j
        self.best_outer_iter = outer_idx
        self.best_plan_id = identifier
        self.best_timestamp = now
        self.best_plan_summary = summary
        self.best_N_flights_flows = flights_total
        self.best_N_entrants_flows = entrants_total
        self._best_updated_this_outer = True
        self._outer_runs_since_best = 0

        self._emit_snapshot(
            "best_update",
            {
                "source": source,
                "identifier": identifier,
                "objective": objective,
                "delta_j": delta_j,
                "outer_index": outer_idx,
            },
        )

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
    outer_max_runs: Optional[int],
    outer_max_time_s: Optional[float],
    outer_max_actions: Optional[int],
    diversify_ban_selected: bool,
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
    tbl.add_row(
        "Agent",
        "outer_max_runs",
        ("∞" if outer_max_runs in (None, 0) else str(int(outer_max_runs))),
    )
    tbl.add_row(
        "Agent",
        "outer_max_time_s",
        ("∞" if outer_max_time_s in (None, 0.0) else f"{float(outer_max_time_s):.1f}"),
    )
    tbl.add_row(
        "Agent",
        "outer_max_actions",
        ("∞" if outer_max_actions in (None, 0) else str(int(outer_max_actions))),
    )
    tbl.add_row("Agent", "diversify_ban_selected", str(bool(diversify_ban_selected)))
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

    global_stats = GlobalStats(logger=logger)

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
    mcts_cfg = MCTSConfig(
        max_sims=8192,
        commit_depth=512,
        commit_eval_limit=512,
        max_actions=None,
        seed=69420,
        debug_prints=False,
        flow_dirichlet_epsilon = 0.4,
        flow_dirichlet_alpha = 0.5
    )
    if mcts_cfg.max_actions in (None, 0):
        action_budget_msg = "∞"
    else:
        action_budget_msg = str(int(mcts_cfg.max_actions))
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
        min_flights_per_flow=5,
        leiden_params={"threshold": 0.64, "resolution": 1.0, "seed": 0},
        direction_opts={"mode": "none"},
    )

    # Agent-level flags
    early_stop_no_improvement = False
    outer_max_runs_env = os.environ.get("FLOW_AGENT_OUTER_MAX_RUNS")
    outer_max_time_env = os.environ.get("FLOW_AGENT_OUTER_MAX_TIME_S")
    outer_max_actions_env = os.environ.get("FLOW_AGENT_OUTER_MAX_ACTIONS")
    diversify_env = os.environ.get("FLOW_AGENT_DIVERSIFY_BAN_SELECTED", "0")

    def _parse_int(value: Optional[str]) -> Optional[int]:
        if not value:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _parse_float(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        try:
            return float(value)
        except Exception:
            return None

    outer_max_runs_param = _parse_int(outer_max_runs_env)
    if outer_max_runs_param is None:
        outer_max_runs_param = 1_000_000 # unlimited
    elif outer_max_runs_param <= 0:
        outer_max_runs_param = None

    outer_max_time_param = _parse_float(outer_max_time_env)
    if outer_max_time_param is None:
        outer_max_time_param = 300 # 5 minutes
    elif outer_max_time_param <= 0:
        outer_max_time_param = None

    outer_max_actions_param = _parse_int(outer_max_actions_env)
    if outer_max_actions_param is not None and outer_max_actions_param <= 0:
        outer_max_actions_param = None

    diversify_flag = bool(str(diversify_env).strip() not in ("", "0", "false", "False"))

    # Prepare Rich progress bar and callback
    prog = Progress(
        TextColumn("[bold blue]MCTS[/bold blue]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} sims"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn(
            " • nodes {task.fields[nodes]} • best ΔJ {task.fields[best]} • last ΔJ {task.fields[last]} "
            "• evals {task.fields[evals]} • acts {task.fields[actions]}/{task.fields[action_budget]}"
        ),
        auto_refresh=False,
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

    def _render_global_progress(progress: Dict[str, Any], inner_payload: Dict[str, Any], actions_total: int) -> Table:
        tbl = Table(title="Global Progress", box=box.SIMPLE_HEAVY)
        tbl.add_column("Metric", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="white")

        completed = int(progress.get("outer_runs_completed", 0))
        max_runs = progress.get("outer_max_run_param")
        pct = progress.get("outer_progress_pct")
        max_runs_s = "∞" if not max_runs else str(int(max_runs))
        pct_s = f"{float(pct):.1f}%" if isinstance(pct, (int, float)) and math.isfinite(float(pct)) else "—"
        tbl.add_row("Outer runs", f"{completed} / {max_runs_s} ({pct_s})")

        elapsed = progress.get("time_elapsed")
        elapsed_s = f"{float(elapsed):.1f}s" if isinstance(elapsed, (int, float)) else "—"
        max_time = progress.get("outer_max_time")
        if isinstance(max_time, (int, float)) and max_time > 0:
            time_pct = progress.get("time_progress_pct")
            pct_time_s = f"{float(time_pct):.1f}%" if isinstance(time_pct, (int, float)) and math.isfinite(float(time_pct)) else "—"
            tbl.add_row("Elapsed time", f"{elapsed_s} / {float(max_time):.1f}s ({pct_time_s})")
        else:
            tbl.add_row("Elapsed time", elapsed_s)

        sims_done = inner_payload.get("sims_done")
        sims_s = str(int(sims_done)) if isinstance(sims_done, (int, float)) else "—"
        max_exp = progress.get("outer_max_expansions")
        if isinstance(max_exp, (int, float)) and max_exp > 0:
            tbl.add_row("Search expansions", f"{sims_s} / {int(max_exp)}")
        else:
            tbl.add_row("Search expansions", sims_s)

        commit_evals = inner_payload.get("commit_evals")
        evals_s = str(int(commit_evals)) if isinstance(commit_evals, (int, float)) else "—"
        max_evals = progress.get("outer_max_evals")
        if isinstance(max_evals, (int, float)) and max_evals > 0:
            tbl.add_row("Plan evals", f"{evals_s} / {int(max_evals)}")
        else:
            tbl.add_row("Plan evals", evals_s)

        actions_val = actions_total
        max_actions = progress.get("outer_max_actions")
        if isinstance(max_actions, (int, float)) and max_actions > 0 and actions_total >= 0:
            pct_actions = (actions_total / float(max_actions)) * 100.0
            tbl.add_row("Actions", f"{actions_total} / {int(max_actions)} ({pct_actions:.1f}%)")
        else:
            tbl.add_row("Actions", str(actions_total))

        current_outer = progress.get("current_outer_index")
        if current_outer:
            tbl.add_row("Current outer", str(current_outer))
        time_since_outer = progress.get("time_since_outer_start")
        if isinstance(time_since_outer, (int, float)) and time_since_outer >= 0:
            tbl.add_row("Outer runtime", f"{time_since_outer:.1f}s")
        stalls = progress.get("outer_runs_since_best")
        if isinstance(stalls, int) and stalls >= 0:
            tbl.add_row("Runs since best", str(stalls))
        stop_reason = progress.get("last_stop_reason")
        if stop_reason:
            tbl.add_row("Last stop", str(stop_reason))
        stop_info = progress.get("last_stop_info")
        if isinstance(stop_info, dict) and stop_info:
            compact = {k: stop_info[k] for k in list(stop_info.keys())[:3]}
            tbl.add_row("Stop info", str(compact))
        return tbl

    def _render_global_best(best: Dict[str, Any]) -> Table:
        tbl = Table(title="Best Candidate (Local)", box=box.SIMPLE_HEAVY)
        tbl.add_column("Metric", style="green", no_wrap=True)
        tbl.add_column("Value", style="white")

        objective = best.get("objective")
        delta_j = best.get("delta_j")
        outer_iter = best.get("outer_iter")
        plan_id = best.get("plan_id")
        plan_summary = best.get("plan_summary")
        n_flights = best.get("n_flights_flows")
        n_entrants = best.get("n_entrants_flows")
        time_since_best = best.get("time_since_best")

        obj_s = f"{float(objective):.3f}" if isinstance(objective, (int, float)) else "—"
        delta_s = f"{float(delta_j):.3f}" if isinstance(delta_j, (int, float)) else "—"
        outer_s = str(int(outer_iter)) if isinstance(outer_iter, (int, float)) else "—"
        plan_id_s = str(plan_id) if plan_id is not None else "—"
        summary_s = str(plan_summary) if plan_summary else "—"
        flights_s = str(int(n_flights)) if isinstance(n_flights, (int, float)) else "—"
        entrants_s = str(int(n_entrants)) if isinstance(n_entrants, (int, float)) else "—"
        since_best_s = f"{float(time_since_best):.1f}s" if isinstance(time_since_best, (int, float)) else "—"

        tbl.add_row("Candidate objective (local)", obj_s)
        tbl.add_row("Best ΔJ (local)", delta_s)
        tbl.add_row("Outer iter", outer_s)
        tbl.add_row("Plan id", plan_id_s)
        tbl.add_row("Summary", summary_s)
        tbl.add_row("Flights (unique)", flights_s)
        tbl.add_row("Entrants", entrants_s)
        tbl.add_row("Since best", since_best_s)
        return tbl

    def _render_global_actions(actions: Dict[str, Any]) -> Table:
        tbl = Table(title="Global Actions (Top 5)", box=box.SIMPLE_HEAVY)
        tbl.add_column("Action", style="magenta")
        tbl.add_column("Count", justify="right")
        tbl.add_column("%", justify="right")

        top = actions.get("top") or []
        for entry in top:
            name = str(entry.get("action", ""))
            count = entry.get("count")
            percent = entry.get("percent")
            count_s = str(int(count)) if isinstance(count, (int, float)) else "—"
            percent_s = f"{float(percent):.1f}%" if isinstance(percent, (int, float)) else "—"
            tbl.add_row(name, count_s, percent_s)
        if not top:
            tbl.add_row("—", "0", "0.0%")

        total = actions.get("total")
        if isinstance(total, (int, float)):
            tbl.caption = f"Total actions: {int(total)}"
        return tbl

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
        tbl = Table(title="ΔJ (local)", box=box.SIMPLE_HEAVY)
        tbl.add_column("Metric", style="green")
        tbl.add_column("Value", justify="right")
        best = payload.get("best_delta_j")
        last = payload.get("last_delta_j")
        best_s = f"{float(best):.3f}" if isinstance(best, (int, float)) else "—"
        last_s = f"{float(last):.3f}" if isinstance(last, (int, float)) else "—"
        tbl.add_row("Best ΔJ", best_s)
        tbl.add_row("Latest ΔJ", last_s)
        return tbl

    # Controls for verbose Rich tables; keep data builders available but default hidden.
    SHOW_GLOBAL_ACTIONS_PANEL = False
    SHOW_ACTION_COUNTS_TABLE = False
    SHOW_ROOT_ACTION_SUMMARY = False

    def _compose_live() -> Group:
        snapshot = global_stats.snapshot_for_display()
        actions_data = snapshot.get("actions", {}) or {}
        actions_total = actions_data.get("total", 0)
        progress_tbl = _render_global_progress(snapshot.get("progress", {}), last_payload, actions_total)
        # best_tbl = _render_global_best(snapshot.get("best", {}))

        renderables = [
            progress_tbl,
            # best_tbl,
        ]
        if SHOW_GLOBAL_ACTIONS_PANEL:
            renderables.append(_render_global_actions(actions_data))

        renderables.append(prog)
        renderables.append(_build_delta_table(last_payload))

        if SHOW_ROOT_ACTION_SUMMARY:
            renderables.append(_build_root_table(last_payload))
        if SHOW_ACTION_COUNTS_TABLE:
            renderables.append(_build_actions_table(last_payload))

        return Group(*renderables)

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
                refresh=False,
            )
            # Merge new inner-loop fields while preserving any outer-loop fields
            last_payload.update(payload)
            try:
                global_stats.update_inner_stats(last_payload)
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
        # Allow up to three committed regulations before forcing commits to move the outer loop forward
        max_inner_loop_commits_and_evals=32, # the variable name could be misleading. It is the max number of commits before moving the outer loop forward
        timer=timed,
        progress_cb=_on_progress,
        early_stop_no_improvement=early_stop_no_improvement,
        outer_max_runs=outer_max_runs_param,
        outer_max_time_s=outer_max_time_param,
        outer_max_actions=outer_max_actions_param,
        diversify_ban_selected=diversify_flag,
        global_stats=global_stats,
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
        outer_max_runs=outer_max_runs_param,
        outer_max_time_s=outer_max_time_param,
        outer_max_actions=outer_max_actions_param,
        diversify_ban_selected=diversify_flag,
    ))

    console.print("[runner] Starting agent.run() ...")
    with Live(_compose_live(), refresh_per_second=8, console=console, vertical_overflow="visible") as _live:
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
    console.print(f"[runner] Number of Regulations: {info.commits}") # basically the final commit equals to the number of regulations because that's how regulation is added
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
            console.print(f"[runner] Final plan objective (plan domain): {base_obj:,.3f} → {final_obj:,.3f} (ΣΔJ_local: {total_delta:,.3f})")
            # Attempt to read system-wide objective from the last run_end payload captured in last_payload/global_stats
            try:
                # Prefer live last_payload if present
                sys_obj = last_payload.get("system_objective") if isinstance(last_payload, dict) else None
                if sys_obj is None and isinstance(info.summary, dict):
                    # Fallback to summary if run_end merged it (not typical)
                    sys_obj = info.summary.get("system_objective")
                if isinstance(sys_obj, (int, float)):
                    console.print(f"[runner] System objective (all TVs): {float(sys_obj):,.3f}")
            except Exception:
                pass
        else:
            console.print(f"[runner] Final plan objective (plan domain): {final_obj}")
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
        "[bold cyan]AIR TRAFFIC FLOW MANAGEMENT REGULATION AGENT 3.5[/bold cyan]\n\n"
        "[dim]Created by:[/dim] [green]Thinh Hoang[/green] ([blue]thinh.hoangdinh@enac.fr[/blue])\n"
        "[dim]Latest Revisions:[/dim] [yellow]26/09/2025[/yellow]",
        title="[bold magenta]MCTS Flow Agent[/bold magenta]",
        border_style="bright_blue",
        padding=(1, 2)
    ))

    run_dir = Path('agent_runs')
    run_dir.mkdir(parents=True, exist_ok=True)
    initiate_agent(run_dir)
