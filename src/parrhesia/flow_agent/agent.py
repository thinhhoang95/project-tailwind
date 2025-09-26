from __future__ import annotations

import copy
import multiprocessing
import queue
import time
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Callable, ContextManager, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

from .actions import Action, NewRegulation
from .logging import SearchLogger
from .mcts import MCTS, MCTSConfig
from .rate_finder import RateFinder, RateFinderConfig
from .state import PlanState
from .transition import CheapTransition
from .hotspot_discovery import (
    HotspotDiscoveryConfig,
    HotspotInventory,
)
from .monitoring import GlobalStatsHook

from parrhesia.optim.objective import ObjectiveWeights, build_score_context
from parrhesia.flow_agent.safespill_objective import score_with_context

_ENSEMBLE_RATE_FINDER: Optional[RateFinder] = None
_ENSEMBLE_SEED_OFFSET = 10_000


def _set_ensemble_rate_finder(rate_finder: RateFinder) -> None:
    global _ENSEMBLE_RATE_FINDER
    _ENSEMBLE_RATE_FINDER = rate_finder


def _root_parallel_mcts_worker(payload: Tuple[PlanState, Dict[str, Any], Optional[int], int, Optional[Any]]) -> Dict[str, Any]:
    state, cfg_dict, run_index, worker_index, progress_handle = payload
    rate_finder = _ENSEMBLE_RATE_FINDER
    if rate_finder is None:
        raise RuntimeError("Ensemble worker started without RateFinder context")

    cfg = MCTSConfig(**cfg_dict)
    cfg.root_parallel_workers = 1

    progress_queue = progress_handle

    def _emit_progress(progress_payload: Dict[str, Any]) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put_nowait(("progress", int(worker_index), dict(progress_payload)))
        except queue.Full:
            pass
        except Exception:
            pass

    mcts = MCTS(
        transition=CheapTransition(),
        rate_finder=rate_finder,
        config=cfg,
        timer=None,
        progress_cb=_emit_progress,
        debug_logger=None,
        cold_logger=None,
    )

    try:
        commit_action = mcts.run(state, run_index=run_index)
    except Exception as exc:
        if progress_queue is not None:
            try:
                progress_queue.put_nowait(("error", int(worker_index), {"message": repr(exc)}))
            except Exception:
                pass
        raise

    last_stats = dict(mcts.last_run_stats or {})

    root_key = state.canonical_key()
    root_node = mcts.nodes.get(root_key)
    root_visits = int(root_node.N) if root_node is not None else 0
    root_children = len(root_node.children) if root_node is not None else 0
    root_top: List[Tuple] = []
    if root_node is not None:
        tmp: List[Tuple] = []
        for sig, est in root_node.edges.items():
            p = float(root_node.P.get(sig, 0.0))
            tmp.append((sig, int(est.N), float(est.Q), p))
        tmp.sort(key=lambda x: (-x[1], -x[2]))
        root_top = tmp

    diagnostics = getattr(commit_action, "diagnostics", {}) or {}
    rf_diag = diagnostics.get("rate_finder", {}) if isinstance(diagnostics, Mapping) else {}
    try:
        delta_j = float(rf_diag.get("delta_j"))
    except Exception:
        delta_j = float("inf")

    best_delta = (delta_j if np.isfinite(delta_j) else None)
    progress_payload = {
        "sims_done": root_visits,
        "sims_total": int(cfg.max_sims),
        "elapsed_s": float(last_stats.get("elapsed_s", 0.0)),
        "eta_s": 0.0,
        "nodes": len(mcts.nodes),
        "root_visits": root_visits,
        "root_children": root_children,
        "root_top": root_top,
        "root_plan_state": MCTS._state_brief(state),
        "all_action_stats": mcts.get_action_stats(),
        "commit_evals": int(last_stats.get("commit_calls", 0)),
        "best_delta_j": best_delta,
        "last_delta_j": best_delta,
        "last_return": None,
        "action_counts": dict(mcts.action_counts),
        "actions_done": int(last_stats.get("actions_done", 0)),
        "max_actions": last_stats.get("max_actions"),
    }

    if progress_queue is not None:
        try:
            progress_queue.put_nowait(("complete", int(worker_index), dict(progress_payload)))
        except queue.Full:
            try:
                progress_queue.put(("complete", int(worker_index), dict(progress_payload)), timeout=0.5)
            except Exception:
                pass
        except Exception:
            pass

    return {
        "commit_action": commit_action,
        "action_counts": dict(mcts.action_counts),
        "action_stats": mcts.get_action_stats(),
        "last_run_stats": last_stats,
        "root_visits": root_visits,
        "root_children": root_children,
        "root_top": root_top,
        "root_plan_state": MCTS._state_brief(state),
        "nodes": len(mcts.nodes),
        "delta_j": float(delta_j),
        "progress": progress_payload,
        "worker_id": int(worker_index),
    }


@dataclass
class _EnsembleRunResult:
    commit_action: Action
    best_worker_index: int
    best_action_counts: Dict[str, int]
    combined_action_counts: Dict[str, int]
    last_run_stats: Dict[str, Any]
    best_delta_j: float
    progress_payload: Dict[str, Any]
    worker_summaries: List[Dict[str, Any]]


@dataclass
class RunInfo:
    commits: int
    total_delta_j: float
    log_path: Optional[str]
    debug_log_path: Optional[str]
    summary: Dict[str, Any]
    action_counts: Dict[str, int]
    stop_reason: Optional[str] = None
    stop_info: Dict[str, Any] = None  # type: ignore[assignment]


class MCTSAgent:
    """End-to-end agent that discovers hotspots, runs MCTS, and computes a final objective.

    Typical usage:
        agent = MCTSAgent(...)
        final_state, info = agent.run()
    """

    def __init__(
        self,
        *,
        evaluator: NetworkEvaluator,
        flight_list: FlightList,
        indexer: TVTWIndexer,
        mcts_cfg: Optional[MCTSConfig] = None,
        rate_finder_cfg: Optional[RateFinderConfig] = None,
        discovery_cfg: Optional[HotspotDiscoveryConfig] = None,
        logger: Optional[SearchLogger] = None,
        debug_logger: Optional[SearchLogger] = None,
        cold_logger: Optional[SearchLogger] = None,
        max_regulations: Optional[int] = None,
        max_inner_loop_commits_and_evals: Optional[int] = None,
        timer: Optional[Callable[[str], ContextManager[Any]]] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        global_stats: Optional[GlobalStatsHook] = None,
        early_stop_no_improvement: bool = False,
        outer_max_runs: Optional[int] = 1,
        outer_max_time_s: Optional[float] = None,
        outer_max_actions: Optional[int] = None,
        diversify_ban_selected: bool = False,
    ) -> None:
        self.evaluator = evaluator
        self.flight_list = flight_list
        self.indexer = indexer
        self.mcts_cfg = mcts_cfg or MCTSConfig()
        self.discovery_cfg = discovery_cfg or HotspotDiscoveryConfig()
        self.logger = logger
        self.debug_logger = debug_logger
        self.cold_logger = cold_logger
        if max_inner_loop_commits_and_evals is None:
            max_inner_loop_commits_and_evals = max_regulations
        if max_inner_loop_commits_and_evals is None:
            sanitized_inner_commit_limit: Optional[int] = None
        else:
            sanitized_inner_commit_limit = int(max(0, max_inner_loop_commits_and_evals))
        self.max_inner_loop_commits_and_evals = sanitized_inner_commit_limit
        # Preserve legacy attribute name for downstream callers that may still reference it.
        self.max_regulations = sanitized_inner_commit_limit
        self._timer_factory = timer
        self._progress_cb = progress_cb
        self._global_stats = global_stats
        self.early_stop_no_improvement = bool(early_stop_no_improvement)

        if outer_max_runs in (None, 0):
            self.outer_max_runs = None
        else:
            try:
                runs_val = int(outer_max_runs)
            except Exception:
                runs_val = 1
            self.outer_max_runs = max(1, runs_val)
        if outer_max_time_s is None or outer_max_time_s <= 0:
            self.outer_max_time_s = None
        else:
            try:
                self.outer_max_time_s = float(outer_max_time_s)
            except Exception:
                self.outer_max_time_s = None
        if outer_max_actions in (None, 0):
            self.outer_max_actions = None
        else:
            try:
                actions_val = int(outer_max_actions)
            except Exception:
                actions_val = 0
            self.outer_max_actions = actions_val if actions_val > 0 else None
        self.diversify_ban_selected = bool(diversify_ban_selected)

        self.rate_finder = RateFinder(
            evaluator=evaluator,
            flight_list=flight_list,
            indexer=indexer,
            config=rate_finder_cfg or RateFinderConfig(use_adaptive_grid=True),
            timer=timer,
            monitoring=global_stats,
        )
        self.inventory = HotspotInventory(evaluator=evaluator, flight_list=flight_list, indexer=indexer)

    def _timed(self, name: str) -> ContextManager[Any]:
        if self._timer_factory is None:
            return nullcontext()
        return self._timer_factory(name)

    def _ban_regulation_spec(
        self,
        state: PlanState,
        *,
        control_volume_id: Optional[str],
        window_bins: Tuple[int, int],
        flow_ids: Sequence[str],
        mode: str,
    ) -> None:
        if not control_volume_id or not flow_ids:
            return
        entry = {
            "control_volume_id": str(control_volume_id),
            "window_bins": [int(window_bins[0]), int(window_bins[1])],
            "flow_ids": [str(fid) for fid in sorted(flow_ids)],
            "mode": "per_flow" if mode == "per_flow" else "blanket",
        }
        banned_key = "banned_regulations"
        banned = list(state.metadata.get(banned_key) or [])
        if entry not in banned:
            banned.append(entry)
            state.metadata[banned_key] = banned

    def _remove_hotspot_candidate(
        self,
        state: PlanState,
        *,
        control_volume_id: Optional[str],
        window_bins: Tuple[int, int],
    ) -> None:
        if not control_volume_id:
            return
        cands = list(state.metadata.get("hotspot_candidates") or [])
        target = (str(control_volume_id), int(window_bins[0]), int(window_bins[1]))
        filtered: List[Dict[str, Any]] = []
        removed = False
        for item in cands:
            try:
                tv = str(item.get("control_volume_id"))
                wb = item.get("window_bins") or []
                t0, t1 = int(wb[0]), int(wb[1])
            except Exception:
                filtered.append(item)
                continue
            if (tv, t0, t1) == target:
                removed = True
                continue
            filtered.append(item)
        if removed:
            state.metadata["hotspot_candidates"] = filtered


    def _run_root_parallel(
        self,
        state: PlanState,
        *,
        run_index: int,
        worker_count: int,
    ) -> Optional[_EnsembleRunResult]:
        if worker_count <= 1:
            return None
        try:
            mp_ctx = multiprocessing.get_context("fork")
        except (ValueError, RuntimeError):
            mp_ctx = None
        if mp_ctx is None:
            return None

        _set_ensemble_rate_finder(self.rate_finder)
        cfg_base = asdict(self.mcts_cfg)

        manager = mp_ctx.Manager()
        try:
            progress_queue = manager.Queue()
            futures = []
            results: List[Dict[str, Any]] = []
            worker_progress: Dict[int, Dict[str, Any]] = {}
            worker_status: Dict[int, str] = {}
            worker_errors: Dict[int, str] = {}
            last_progress_payload: Optional[Dict[str, Any]] = None

            def _safe_int(value: Any, default: int = 0) -> int:
                try:
                    return int(value)
                except Exception:
                    return default

            def _safe_float(value: Any) -> Optional[float]:
                try:
                    f = float(value)
                except Exception:
                    return None
                if np.isfinite(f):
                    return f
                return None

            def _compose_worker_progress() -> Optional[Dict[str, Any]]:
                if not worker_progress:
                    return None
                _lead_id, lead_payload = max(
                    worker_progress.items(),
                    key=lambda kv: _safe_int((kv[1] or {}).get("sims_done"), 0),
                )
                aggregated = dict(lead_payload or {})
                max_sims_done = max(
                    _safe_int((data or {}).get("sims_done"), 0) for data in worker_progress.values()
                )
                aggregated["sims_done"] = max_sims_done
                sims_total_candidates = [
                    _safe_int((data or {}).get("sims_total"), int(self.mcts_cfg.max_sims))
                    for data in worker_progress.values()
                ]
                if sims_total_candidates:
                    aggregated["sims_total"] = max(sims_total_candidates)
                else:
                    aggregated["sims_total"] = int(self.mcts_cfg.max_sims)

                combined_counts: Dict[str, int] = {}
                total_commit_evals = 0
                total_actions_done = 0
                best_delta: Optional[float] = None
                best_worker_id: Optional[int] = None
                workers_summary: List[Dict[str, Any]] = []

                for wid, data in worker_progress.items():
                    payload = data or {}
                    counts = payload.get("action_counts") or {}
                    for label, count in counts.items():
                        try:
                            combined_counts[str(label)] = combined_counts.get(str(label), 0) + int(count)
                        except Exception:
                            continue
                    commit_evals = _safe_int(payload.get("commit_evals"), 0)
                    actions_done = _safe_int(payload.get("actions_done"), 0)
                    total_commit_evals += commit_evals
                    total_actions_done += actions_done
                    best_val = _safe_float(payload.get("best_delta_j"))
                    if best_val is not None and (best_delta is None or best_val < best_delta):
                        best_delta = best_val
                        best_worker_id = wid
                    worker_entry: Dict[str, Any] = {
                        "worker_id": int(wid),
                        "sims_done": _safe_int(payload.get("sims_done"), 0),
                        "best_delta_j": best_val,
                        "last_delta_j": _safe_float(payload.get("last_delta_j")),
                        "commit_evals": commit_evals,
                        "actions_done": actions_done,
                        "status": worker_status.get(wid, "running"),
                    }
                    if worker_entry["status"] == "error":
                        worker_entry["error"] = worker_errors.get(wid)
                    workers_summary.append(worker_entry)

                aggregated["action_counts"] = combined_counts
                aggregated["commit_evals"] = total_commit_evals
                aggregated["actions_done"] = total_actions_done
                aggregated["ensemble_workers"] = sorted(
                    workers_summary, key=lambda item: item["worker_id"]
                )
                aggregated["ensemble_worker_count"] = len(worker_progress)
                aggregated["ensemble_progress_source"] = "ensemble"
                if best_worker_id is not None:
                    aggregated["ensemble_best_worker"] = int(best_worker_id)
                if best_delta is not None:
                    aggregated["best_delta_j"] = best_delta
                    aggregated["last_delta_j"] = best_delta
                aggregated.setdefault("sims_total", int(self.mcts_cfg.max_sims))
                if aggregated.get("max_actions") is None:
                    aggregated["max_actions"] = cfg_base.get("max_actions")
                return aggregated

            def _drain_progress_queue(block: bool = False) -> bool:
                nonlocal last_progress_payload
                if progress_queue is None:
                    return False
                processed = False
                updated = False
                while True:
                    try:
                        if block:
                            item = progress_queue.get(timeout=0.1)
                        else:
                            item = progress_queue.get_nowait()
                    except (queue.Empty, EOFError):
                        break
                    processed = True
                    if not isinstance(item, tuple) or len(item) < 3:
                        continue
                    kind, worker_id, payload = item
                    try:
                        wid = int(worker_id)
                    except Exception:
                        continue
                    if kind == "error":
                        worker_status[wid] = "error"
                        if isinstance(payload, Mapping):
                            worker_errors[wid] = str(payload.get("message"))
                        if wid not in worker_progress:
                            worker_progress[wid] = {}
                        updated = True
                        continue
                    if kind in {"progress", "complete"}:
                        if isinstance(payload, Mapping):
                            worker_progress[wid] = dict(payload)
                        if kind == "complete":
                            worker_status[wid] = "done"
                        elif worker_status.get(wid) != "done":
                            worker_status[wid] = "running"
                        updated = True
                if updated:
                    aggregate = _compose_worker_progress()
                    if aggregate is not None:
                        last_progress_payload = aggregate
                        if self._progress_cb is not None:
                            try:
                                self._progress_cb(aggregate)
                            except Exception:
                                pass
                return processed

            with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_ctx) as executor:
                for idx in range(worker_count):
                    cfg_payload = dict(cfg_base)
                    cfg_payload["root_parallel_workers"] = 1
                    base_seed = int(self.mcts_cfg.seed)
                    cfg_payload["seed"] = base_seed + _ENSEMBLE_SEED_OFFSET * (idx + 1)
                    max_actions_val = cfg_payload.get("max_actions")
                    if max_actions_val in (None, 0):
                        cfg_payload["max_actions"] = None
                    else:
                        try:
                            cfg_payload["max_actions"] = int(max_actions_val)
                        except Exception:
                            cfg_payload["max_actions"] = None
                    state_payload = copy.deepcopy(state)
                    futures.append(
                        executor.submit(
                            _root_parallel_mcts_worker,
                            (state_payload, cfg_payload, run_index, idx, progress_queue),
                        )
                    )

                pending = set(futures)
                while pending:
                    done, not_done = wait(pending, timeout=0.1)
                    _drain_progress_queue(block=False)
                    for fut in done:
                        results.append(fut.result())
                    pending = not_done
                while _drain_progress_queue(block=False):
                    pass

            try:
                progress_queue.close()
                progress_queue.join_thread()
            except Exception:
                pass
        finally:
            try:
                manager.shutdown()
            except Exception:
                pass

        if not results:
            return None

        aggregate_payload = _compose_worker_progress()
        if aggregate_payload is not None:
            last_progress_payload = aggregate_payload

        results.sort(key=lambda entry: int(entry.get("worker_id", 0)))
        combined_counts: Dict[str, int] = {}
        worker_summaries: List[Dict[str, Any]] = []
        fallback_workers: List[Dict[str, Any]] = []
        best_entry: Optional[Dict[str, Any]] = None
        best_delta = float("inf")
        total_commit_evals = 0
        total_actions_done = 0

        for res in results:
            counts = {str(k): int(v) for k, v in (res.get("action_counts") or {}).items()}
            for label, count in counts.items():
                combined_counts[label] = combined_counts.get(label, 0) + count
            stats = res.get("last_run_stats") or {}
            delta = float(res.get("delta_j", float("inf")))
            worker_id = int(res.get("worker_id", 0))
            commit_calls = int(stats.get("commit_calls", 0))
            actions_done = int(stats.get("actions_done", 0))
            total_commit_evals += commit_calls
            total_actions_done += actions_done
            worker_summaries.append(
                {
                    "worker_id": worker_id,
                    "delta_j": float(delta),
                    "commit_calls": commit_calls,
                    "actions_done": actions_done,
                    "root_visits": int(res.get("root_visits", 0)),
                }
            )
            fallback_workers.append(
                {
                    "worker_id": worker_id,
                    "sims_done": int(res.get("root_visits", 0)),
                    "best_delta_j": float(delta) if np.isfinite(delta) else None,
                    "last_delta_j": float(delta) if np.isfinite(delta) else None,
                    "commit_evals": commit_calls,
                    "actions_done": actions_done,
                    "status": worker_status.get(worker_id, "done"),
                }
            )
            if delta < best_delta:
                best_delta = delta
                best_entry = res

        if best_entry is None:
            raise RuntimeError("Ensemble MCTS produced no valid result")

        progress_payload = dict(last_progress_payload or {})
        if not progress_payload:
            progress_payload = dict(best_entry.get("progress") or {})
        progress_payload["action_counts"] = combined_counts
        progress_payload["commit_evals"] = total_commit_evals
        progress_payload["actions_done"] = total_actions_done
        if np.isfinite(best_delta):
            progress_payload["best_delta_j"] = float(best_delta)
            progress_payload["last_delta_j"] = float(best_delta)
        else:
            if progress_payload.get("best_delta_j") is None:
                progress_payload["best_delta_j"] = None
            if progress_payload.get("last_delta_j") is None:
                progress_payload["last_delta_j"] = None
        progress_payload.setdefault("sims_total", int(self.mcts_cfg.max_sims))
        if not progress_payload.get("ensemble_workers"):
            progress_payload["ensemble_workers"] = fallback_workers
            progress_payload["ensemble_worker_count"] = len(fallback_workers)
            if fallback_workers:
                best_worker = min(
                    fallback_workers,
                    key=lambda item: item["best_delta_j"] if item["best_delta_j"] is not None else float("inf"),
                )
                progress_payload["ensemble_best_worker"] = int(best_worker["worker_id"])
        else:
            progress_payload.setdefault(
                "ensemble_worker_count", len(progress_payload.get("ensemble_workers") or [])
            )

        progress_payload.setdefault("ensemble_progress_source", "results")

        if self._progress_cb is not None and progress_payload:
            try:
                self._progress_cb(progress_payload)
            except Exception:
                pass

        if self.debug_logger is not None and worker_summaries:
            try:
                self.debug_logger.event(
                    "root_parallel_summary",
                    {
                        "workers": worker_summaries,
                        "best_worker_index": int(best_entry.get("worker_id", 0)),
                        "best_delta_j": float(best_delta),
                    },
                )
            except Exception:
                pass

        return _EnsembleRunResult(
            commit_action=best_entry["commit_action"],
            best_worker_index=int(best_entry.get("worker_id", 0)),
            best_action_counts={k: int(v) for k, v in (best_entry.get("action_counts") or {}).items()},
            combined_action_counts=combined_counts,
            last_run_stats=dict(best_entry.get("last_run_stats") or {}),
            best_delta_j=float(best_delta),
            progress_payload=progress_payload,
            worker_summaries=worker_summaries,
        )
    def run(self) -> Tuple[PlanState, RunInfo]:
        # Build hotspot inventory and seed plan state
        cfg = self.discovery_cfg
        with self._timed("agent.hotspot_inventory"):
            descs = self.inventory.build_from_segments(
                threshold=float(cfg.threshold),
                top_hotspots=int(cfg.top_hotspots),
                top_flows=int(cfg.top_flows),
                min_flights_per_flow=int(cfg.min_flights_per_flow),
                max_flights_per_flow=int(cfg.max_flights_per_flow),
                leiden_params=cfg.leiden_params,
                direction_opts=cfg.direction_opts,
            )
            candidates = self.inventory.to_candidate_payloads(descs)

        # Print out all extracted flows (not logged to file) for manual inspection
        try:
            # print("[agent] Hotspot flows extracted:")
            for idx, cand in enumerate(candidates, 1):
                tv = cand.get("control_volume_id")
                wb = cand.get("window_bins") or []
                try:
                    t0, t1 = (int(wb[0]), int(wb[1]))
                except Exception:
                    t0, t1 = 0, 0
                meta = cand.get("metadata") or {}
                flow_to_flights = meta.get("flow_to_flights", {}) or {}
                flows_sorted = sorted(flow_to_flights.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
                # print(f"  [{idx}] TV {tv} window [{t0},{t1}) • flows: {len(flows_sorted)}")
                for fid, flights in flows_sorted:
                    try:
                        fl_ids = [str(x) for x in (flights or [])]
                    except Exception:
                        fl_ids = []
                    # print(f"     - flow {fid}: {len(fl_ids)} flights: {fl_ids}")
        except Exception:
            pass

        base_state = PlanState()
        base_state.metadata["hotspot_candidates"] = copy.deepcopy(candidates)

        initial_summary: Dict[str, Any] = {}
        system_initial_objective: Optional[float] = None
        try:
            initial_summary = self._compute_final_objective(base_state)
        except Exception:
            initial_summary = {}
        try:
            _sys0 = self._compute_system_objective(base_state)
            system_initial_objective = float(_sys0.get("system_objective")) if isinstance(_sys0.get("system_objective"), (int, float)) else None
        except Exception:
            system_initial_objective = None

        transition = CheapTransition()
        mcts = MCTS(
            transition=transition,
            rate_finder=self.rate_finder,
            config=self.mcts_cfg,
            timer=self._timer_factory,
            progress_cb=self._progress_cb,
            debug_logger=self.debug_logger,
            cold_logger=self.cold_logger,
        )

        if self.logger is not None:
            payload: Dict[str, Any] = {
                "num_candidates": len(candidates),
                "mcts_cfg": self.mcts_cfg.__dict__,
            }
            if initial_summary:
                payload.update(
                    {
                        "objective": initial_summary.get("objective"),
                        "components": dict(initial_summary.get("components", {})),
                        "artifacts": initial_summary.get("artifacts"),
                        "num_flows": initial_summary.get("num_flows"),
                        "spill_T": initial_summary.get("spill_T"),
                        "in_window_releases": initial_summary.get("in_window_releases"),
                        # Clarify scope for the baseline objective at run_start
                        "objective_scope": "system_candidates_union",
                        "objective_scope_desc": "Baseline over the union of hotspot candidate TVs/windows",
                    }
                )
                if system_initial_objective is not None:
                    payload.update({
                        "system_objective": float(system_initial_objective),
                        "system_components": None,
                        "system_objective_scope": "all_tvs",
                    })
            self.logger.event("run_start", payload)

        root_parallel_workers_raw = getattr(self.mcts_cfg, "root_parallel_workers", 1)
        try:
            root_parallel_workers = int(root_parallel_workers_raw)
        except Exception:
            root_parallel_workers = 1
        if root_parallel_workers <= 0:
            root_parallel_workers = 1

        commit_depth_limit = int(max(0, self.mcts_cfg.commit_depth))
        configured_inner_commit_limit = (
            int(self.max_inner_loop_commits_and_evals)
            if self.max_inner_loop_commits_and_evals is not None
            else None
        )
        outer_loop_budget = (
            int(max(0, configured_inner_commit_limit)) if configured_inner_commit_limit is not None else commit_depth_limit
        )

        master_start = time.perf_counter()
        master_runs_executed = 0
        master_actions_total = 0
        best_state: Optional[PlanState] = None
        best_summary: Dict[str, Any] = {}
        best_action_counts: Dict[str, int] = {}
        best_stop_reason: Optional[str] = None
        best_stop_info: Dict[str, Any] = {}
        best_total_delta_j = 0.0
        best_commits = 0
        best_objective = float("inf")
        best_run_index = 0

        diversify_banned: List[Dict[str, Any]] = []

        if self.debug_logger is not None:
            try:
                self.debug_logger.event(
                    "outer_master_start",
                    {
                        "num_candidates": len(candidates),
                        "commit_depth": int(self.mcts_cfg.commit_depth),
                        "max_inner_loop_commits_and_evals": (
                            int(self.max_inner_loop_commits_and_evals)
                            if self.max_inner_loop_commits_and_evals is not None
                            else None
                        ),
                        "max_sims": int(self.mcts_cfg.max_sims),
                        "max_time_s": float(self.mcts_cfg.max_time_s),
                        "commit_eval_limit": int(self.mcts_cfg.commit_eval_limit),
                        "max_actions": (
                            int(self.mcts_cfg.max_actions)
                            if getattr(self.mcts_cfg, "max_actions", None) not in (None, 0)
                            else None
                        ),
                        "outer_max_runs": (
                            int(self.outer_max_runs)
                            if self.outer_max_runs is not None
                            else None
                        ),
                        "outer_max_time_s": (
                            float(self.outer_max_time_s)
                            if self.outer_max_time_s is not None
                            else None
                        ),
                        "outer_max_actions": (
                            int(self.outer_max_actions)
                            if self.outer_max_actions is not None
                            else None
                        ),
                    },
                )
            except Exception:
                pass

        def _emit_master_progress(payload: Dict[str, Any], *, master_run_index: int, partial_actions: Optional[int] = None) -> None:
            if self._progress_cb is None:
                return
            merged = dict(payload)
            merged.setdefault("outer_master_run_index", int(master_run_index + 1))
            merged.setdefault("outer_master_runs_completed", int(master_runs_executed))
            merged.setdefault(
                "outer_master_max_runs",
                (int(self.outer_max_runs) if self.outer_max_runs is not None else None),
            )
            merged.setdefault(
                "outer_master_max_time_s",
                (float(self.outer_max_time_s) if self.outer_max_time_s is not None else None),
            )
            merged.setdefault(
                "outer_master_max_actions",
                (int(self.outer_max_actions) if self.outer_max_actions is not None else None),
            )
            total_actions_value = master_actions_total
            if partial_actions is not None:
                total_actions_value = partial_actions
            merged.setdefault("outer_master_total_actions", int(total_actions_value))
            merged.setdefault("outer_master_elapsed_s", float(time.perf_counter() - master_start))
            try:
                self._progress_cb(merged)
            except Exception:
                pass

        base_candidates = copy.deepcopy(candidates)

        while True:
            now = time.perf_counter()
            if self.outer_max_runs is not None and master_runs_executed >= self.outer_max_runs:
                if self._global_stats is not None:
                    try:
                        self._global_stats.on_limit_hit(
                            "outer_max_run_param",
                            {
                                "outer_runs_completed": int(master_runs_executed),
                                "outer_max_run_param": int(self.outer_max_runs),
                            },
                        )
                    except Exception:
                        pass
                break
            if self.outer_max_time_s is not None and (now - master_start) >= float(self.outer_max_time_s):
                if self._global_stats is not None:
                    try:
                        self._global_stats.on_limit_hit(
                            "outer_max_time",
                            {
                                "elapsed_s": float(now - master_start),
                                "outer_max_time": float(self.outer_max_time_s),
                            },
                        )
                    except Exception:
                        pass
                break
            if self.outer_max_actions is not None and master_actions_total >= int(self.outer_max_actions):
                if self._global_stats is not None:
                    try:
                        self._global_stats.on_limit_hit(
                            "outer_max_actions",
                            {
                                "actions_total": int(master_actions_total),
                                "outer_max_actions": int(self.outer_max_actions),
                            },
                        )
                    except Exception:
                        pass
                break

            state = PlanState()
            state.metadata["hotspot_candidates"] = copy.deepcopy(base_candidates)
            if diversify_banned:
                state.metadata["banned_regulations"] = copy.deepcopy(diversify_banned)
            elif self.diversify_ban_selected:
                state.metadata.setdefault("banned_regulations", [])

            master_run_index = master_runs_executed
            if self._global_stats is not None:
                try:
                    self._global_stats.on_outer_start(
                        int(master_run_index + 1),
                        {
                            "outer_max_run_param": (int(self.outer_max_runs) if self.outer_max_runs is not None else None),
                            "outer_max_time": (float(self.outer_max_time_s) if self.outer_max_time_s is not None else None),
                            "outer_max_actions": (int(self.outer_max_actions) if self.outer_max_actions is not None else None),
                            "outer_max_evals": int(self.mcts_cfg.commit_eval_limit),
                            "outer_max_expansions": int(self.mcts_cfg.max_sims),
                            "outer_loop_budget": int(outer_loop_budget),
                            "commit_depth_limit": int(commit_depth_limit),
                        },
                    )
                except Exception:
                    pass
            try:
                seed_base = int(getattr(self.mcts_cfg, "seed", 0))
            except Exception:
                seed_base = 0
            try:
                mcts.rng = np.random.default_rng(seed_base + int(master_run_index))
            except Exception:
                pass

            commits = 0
            total_delta_j = 0.0
            aggregated_counts: Dict[str, int] = {}
            outer_stop_reason: Optional[str] = None
            outer_stop_info: Dict[str, Any] = {}
            run_idx = 1
            max_commits_per_inner: List[int] = []
            commit_calls_per_inner: List[int] = []
            run_actions_total = 0

            if self.debug_logger is not None:
                try:
                    self.debug_logger.event(
                        "outer_run_start",
                        {
                            "outer_master_run_index": int(master_run_index + 1),
                            "commit_depth": int(self.mcts_cfg.commit_depth),
                            "max_inner_loop_commits_and_evals": (
                                int(self.max_inner_loop_commits_and_evals)
                                if self.max_inner_loop_commits_and_evals is not None
                                else None
                            ),
                            "max_sims": int(self.mcts_cfg.max_sims),
                            "max_time_s": float(self.mcts_cfg.max_time_s),
                            "commit_eval_limit": int(self.mcts_cfg.commit_eval_limit),
                            "max_actions": (
                                int(self.mcts_cfg.max_actions)
                                if getattr(self.mcts_cfg, "max_actions", None) not in (None, 0)
                                else None
                            ),
                        },
                    )
                except Exception:
                    pass

            _emit_master_progress(
                {
                    "outer_commits": int(commits),
                    "outer_commit_depth": int(commit_depth_limit),
                    "outer_max_inner_loop_commits_and_evals": (
                        int(self.max_inner_loop_commits_and_evals)
                        if self.max_inner_loop_commits_and_evals is not None
                        else None
                    ),
                    "outer_limit": int(outer_loop_budget),
                    "outer_early_stop_no_improvement": bool(self.early_stop_no_improvement),
                    "outer_run_index": int(run_idx),
                },
                master_run_index=master_run_index,
                partial_actions=master_actions_total,
            )

            for _ in range(max(1, int(outer_loop_budget))):
                state.stage = "idle"
                action_counts_this_run: Dict[str, int] = {}
                combined_action_counts: Dict[str, int] = {}
                run_stats_this_run: Dict[str, Any] = {}
                run_result: Optional[_EnsembleRunResult] = None
                try:
                    with self._timed("agent.mcts.run"):
                        if root_parallel_workers > 1:
                            run_result = self._run_root_parallel(
                                state,
                                run_index=(master_run_index * max(1, int(outer_loop_budget))) + run_idx,
                                worker_count=root_parallel_workers,
                            )
                        if run_result is None:
                            commit_action = mcts.run(
                                state,
                                run_index=(master_run_index * max(1, int(outer_loop_budget))) + run_idx,
                            )
                            action_counts_this_run = dict(mcts.action_counts)
                            combined_action_counts = dict(mcts.action_counts)
                            run_stats_this_run = dict(mcts.last_run_stats or {})
                        else:
                            commit_action = run_result.commit_action
                            action_counts_this_run = dict(run_result.best_action_counts)
                            combined_action_counts = dict(run_result.combined_action_counts)
                            run_stats_this_run = dict(run_result.last_run_stats)
                except Exception as exc:
                    message = str(exc)
                    for k, v in combined_action_counts.items():
                        aggregated_counts[k] = aggregated_counts.get(k, 0) + int(v)
                        if self._global_stats is not None:
                            try:
                                self._global_stats.on_action(str(k), int(v))
                            except Exception:
                                pass
                    if isinstance(exc, RuntimeError) and "MCTS did not evaluate any commit" in message:
                        if self.logger is not None:
                            self.logger.event(
                                "mcts_no_commit",
                                {"error": message, "outer_run": master_run_index + 1},
                            )
                        if self.debug_logger is not None:
                            try:
                                self.debug_logger.event(
                                    "mcts_no_commit",
                                    {
                                        "error": message,
                                        "exc_type": type(exc).__name__,
                                        "outer_master_run_index": int(master_run_index + 1),
                                    },
                                )
                            except Exception:
                                pass
                        outer_stop_reason = "no_commit_available"
                        outer_stop_info = {
                            "status": "no_commit_evaluated",
                            "exc_type": type(exc).__name__,
                        }
                        _emit_master_progress(
                            {
                                "outer_stop_reason": "no_commit_available",
                                "outer_stop_info": outer_stop_info,
                                "outer_commits": int(commits),
                                "outer_commit_depth": int(commit_depth_limit),
                                "outer_max_inner_loop_commits_and_evals": (
                                    int(self.max_inner_loop_commits_and_evals)
                                    if self.max_inner_loop_commits_and_evals is not None
                                    else None
                                ),
                                "outer_limit": int(outer_loop_budget),
                                "outer_run_index": int(run_idx),
                            },
                            master_run_index=master_run_index,
                            partial_actions=master_actions_total + run_actions_total,
                        )
                        if self._global_stats is not None:
                            try:
                                self._global_stats.on_limit_hit(
                                    "no_commit_available",
                                    {
                                        "outer_index": int(master_run_index + 1),
                                        "message": message,
                                    },
                                )
                            except Exception:
                                pass
                        break
                    if self.logger is not None:
                        self.logger.event("mcts_error", {"error": message})
                    if self.debug_logger is not None:
                        try:
                            self.debug_logger.event(
                                "mcts_error",
                                {
                                    "error": message,
                                    "exc_type": type(exc).__name__,
                                    "outer_master_run_index": int(master_run_index + 1),
                                },
                            )
                        except Exception:
                            pass
                    outer_stop_reason = "mcts_error"
                    outer_stop_info = {"error": message, "exc_type": type(exc).__name__}
                    _emit_master_progress(
                        {
                            "outer_stop_reason": "mcts_error",
                            "outer_stop_info": outer_stop_info,
                            "outer_commits": int(commits),
                            "outer_commit_depth": int(commit_depth_limit),
                            "outer_max_inner_loop_commits_and_evals": (
                                int(self.max_inner_loop_commits_and_evals)
                                if self.max_inner_loop_commits_and_evals is not None
                                else None
                            ),
                            "outer_limit": int(outer_loop_budget),
                            "outer_run_index": int(run_idx),
                        },
                        master_run_index=master_run_index,
                        partial_actions=master_actions_total + run_actions_total,
                    )
                    if self._global_stats is not None:
                        try:
                            self._global_stats.on_limit_hit(
                                "mcts_error",
                                {
                                    "outer_index": int(master_run_index + 1),
                                    "message": message,
                                },
                            )
                        except Exception:
                            pass
                    break

                for k, v in combined_action_counts.items():
                    aggregated_counts[k] = aggregated_counts.get(k, 0) + int(v)
                    if self._global_stats is not None:
                        try:
                            self._global_stats.on_action(str(k), int(v))
                        except Exception:
                            pass

                run_actions_total += int(run_stats_this_run.get("actions_done", 0))

                try:
                    stats = dict(run_stats_this_run)
                    if isinstance(stats, dict):
                        mx = int(stats.get("max_commits_path", 0))
                        max_commits_per_inner.append(mx)
                        try:
                            commit_calls_per_inner.append(int(stats.get("commit_calls", 0)))
                        except Exception:
                            commit_calls_per_inner.append(0)
                        if self.cold_logger is not None:
                            try:
                                self.cold_logger.event(
                                    "cold_inner_run_summary",
                                    {
                                        "run_index": int(stats.get("run_index") or run_idx),
                                        "commit_calls": int(stats.get("commit_calls", 0)),
                                        "stop_reason": stats.get("stop_reason"),
                                        "stop_info": stats.get("stop_info"),
                                        "elapsed_s": float(stats.get("elapsed_s", 0.0)),
                                        "actions_done": int(stats.get("actions_done", 0)),
                                        "max_actions": stats.get("max_actions"),
                                        "max_commits_path": mx,
                                    },
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

                info = (commit_action.diagnostics or {}).get("rate_finder", {})
                delta_j = float(info.get("delta_j", 0.0))

                diag = (commit_action.diagnostics or {}).get("rate_finder", {})
                ctrl = str(diag.get("control_volume_id")) if diag.get("control_volume_id") is not None else None
                win_list = diag.get("window_bins") or []
                try:
                    wb = (int(win_list[0]), int(win_list[1]))
                except Exception:
                    wb = (0, 1)
                mode = str(diag.get("mode", "per_flow"))
                try:
                    orig_diag_payload: Dict[str, Any] = dict(commit_action.diagnostics or {})
                    rf_info_for_bounds: Dict[str, Any] = dict(orig_diag_payload.get("rate_finder", {}) or {})
                    grid_vals = rf_info_for_bounds.get("rate_grid", []) or []
                    finite_rates: List[float] = []
                    for val in grid_vals:
                        try:
                            fv = float(val)
                        except Exception:
                            continue
                        if np.isinf(fv) or np.isnan(fv):
                            continue
                        if fv <= 0.0:
                            continue
                        finite_rates.append(fv)
                    rate_menu_lower = int(round(min(finite_rates))) if finite_rates else None
                    rate_menu_upper = int(round(max(finite_rates))) if finite_rates else None
                    rf_info_for_bounds["rate_menu_lower"] = rate_menu_lower
                    rf_info_for_bounds["rate_menu_upper"] = rate_menu_upper
                    diag_enriched_payload: Dict[str, Any] = dict(orig_diag_payload)
                    diag_enriched_payload["rate_finder"] = rf_info_for_bounds
                except Exception:
                    diag_enriched_payload = dict(commit_action.diagnostics or {})
                if isinstance(commit_action.committed_rates, dict):
                    flow_ids = tuple(sorted(str(k) for k in commit_action.committed_rates.keys()))
                else:
                    entrants = diag.get("entrants_by_flow", {}) or {}
                    flow_ids = tuple(sorted(str(k) for k in entrants.keys()))

                from .state import RegulationSpec  # local import to avoid cycles at module import time

                if ctrl is None:
                    break

                rates = commit_action.committed_rates
                valid = False
                rates_to_store: Optional[object] = None
                if isinstance(rates, dict):
                    cleaned: Dict[str, int] = {}
                    for k, v in (rates or {}).items():
                        try:
                            iv = int(v)
                        except Exception:
                            iv = 0
                        if iv > 0:
                            cleaned[str(k)] = iv
                    if cleaned and len(flow_ids) > 0:
                        valid = True
                        rates_to_store = cleaned
                else:
                    valid = isinstance(rates, tuple)
                    rates_to_store = tuple(int(x) for x in (rates or ()))
                    if not rates_to_store:
                        valid = False

                if not valid:
                    continue

                regulation = RegulationSpec(
                    control_volume_id=ctrl,
                    window_bins=wb,
                    flow_ids=flow_ids,
                    mode="per_flow" if mode == "per_flow" else "blanket",
                    committed_rates=rates_to_store,
                    diagnostics=diag_enriched_payload,
                )

                if any(
                    existing.to_canonical_dict() == regulation.to_canonical_dict()
                    for existing in state.plan
                ):
                    self._ban_regulation_spec(
                        state,
                        control_volume_id=ctrl,
                        window_bins=wb,
                        flow_ids=flow_ids,
                        mode=mode,
                    )
                    self._remove_hotspot_candidate(
                        state,
                        control_volume_id=ctrl,
                        window_bins=wb,
                    )
                    if self.debug_logger is not None:
                        try:
                            self.debug_logger.event(
                                "outer_skip_duplicate_regulation",
                                {
                                    "control_volume_id": ctrl,
                                    "window_bins": [int(wb[0]), int(wb[1])],
                                    "mode": mode,
                                    "flow_ids": list(flow_ids),
                                    "committed_rates": rates_to_store,
                                    "outer_master_run_index": int(master_run_index + 1),
                                },
                            )
                        except Exception:
                            pass
                    continue

                state.plan.append(regulation)
                self._ban_regulation_spec(
                    state,
                    control_volume_id=ctrl,
                    window_bins=wb,
                    flow_ids=flow_ids,
                    mode=mode,
                )
                self._remove_hotspot_candidate(
                    state,
                    control_volume_id=ctrl,
                    window_bins=wb,
                )
                total_delta_j += delta_j
                commits += 1

                _emit_master_progress(
                    {
                        "outer_commits": int(commits),
                        "outer_commit_depth": int(commit_depth_limit),
                        "outer_max_inner_loop_commits_and_evals": (
                            int(self.max_inner_loop_commits_and_evals)
                            if self.max_inner_loop_commits_and_evals is not None
                            else None
                        ),
                        "outer_limit": int(outer_loop_budget),
                        "outer_last_delta_j": float(delta_j),
                        "outer_early_stop_no_improvement": bool(self.early_stop_no_improvement),
                        "outer_run_index": int(run_idx),
                    },
                    master_run_index=master_run_index,
                    partial_actions=master_actions_total + run_actions_total,
                )

                diag_payload: Dict[str, Any] = {}
                rf_info: Dict[str, Any] = {}
                fin_comp: Dict[str, Any] = {}
                fin_obj: Optional[float] = None
                rate_menu_lower: Optional[int] = None
                rate_menu_upper: Optional[int] = None
                cand_nonzero_delays: Optional[int] = None
                cand_max_delay: Optional[int] = None
                cand_spill: Optional[int] = None
                n_flights_flows: Optional[int] = None
                n_flights_flows_max: Optional[int] = None
                max_flow_id: Optional[str] = None
                n_entrants_flows: Optional[int] = None
                n_entrants_flows_max: Optional[int] = None
                entrants_max_flow_id: Optional[str] = None
                flow_to_flights_map: Dict[str, List[str]] = {}
                entrants_counts_map: Dict[str, int] = {}

                if self.logger is not None:
                    try:
                        diag_payload = dict(getattr(regulation, "diagnostics", {}) or {})
                    except Exception:
                        diag_payload = {}
                    rf_info = diag_payload.get("rate_finder", {}) if isinstance(diag_payload, dict) else {}
                    fin_comp = rf_info.get("final_components", {}) if isinstance(rf_info, dict) else {}
                    fin_obj = rf_info.get("final_objective", None)
                    rate_menu_lower = rf_info.get("rate_menu_lower") if isinstance(rf_info, dict) else None
                    rate_menu_upper = rf_info.get("rate_menu_upper") if isinstance(rf_info, dict) else None
                    try:
                        fin_delays = rf_info.get("final_delays_min", {}) if isinstance(rf_info, dict) else {}
                        if isinstance(fin_delays, dict) and fin_delays:
                            vals = [int(v) for v in fin_delays.values()]
                            cand_nonzero_delays = int(sum(1 for v in vals if v > 0))
                            cand_max_delay = int(max(vals)) if vals else 0
                    except Exception:
                        cand_nonzero_delays = None
                        cand_max_delay = None
                    try:
                        arts = rf_info.get("final_artifacts", {}) if isinstance(rf_info, dict) else {}
                        nmap = arts.get("n", {}) if isinstance(arts, dict) else {}
                        if nmap:
                            T_int = int(self.indexer.num_time_bins)
                            total = 0
                            for v in nmap.values():
                                arr = np.asarray(v)
                                if arr.size > T_int:
                                    total += int(arr[T_int])
                            cand_spill = int(total)
                    except Exception:
                        cand_spill = None
                    if cand_spill is None:
                        try:
                            v = rf_info.get("final_spill_T")
                            if isinstance(v, (int, float)):
                                cand_spill = int(v)
                        except Exception:
                            pass
                    try:
                        desc = self.inventory.get(str(ctrl), (int(wb[0]), int(wb[1])))
                        meta = getattr(desc, "metadata", {}) if desc is not None else {}
                        flow_to_flights = meta.get("flow_to_flights", {}) if isinstance(meta, dict) else {}
                        if isinstance(flow_to_flights, dict):
                            for fid, flights in flow_to_flights.items():
                                flow_to_flights_map[str(fid)] = [str(x) for x in (flights or [])]
                        if flow_ids:
                            unique_total_set: set[str] = set()
                            unique_max = -1
                            unique_max_fid: Optional[str] = None
                            for fid in flow_ids:
                                fl_list = flow_to_flights_map.get(str(fid), [])
                                uniq = set(fl_list)
                                unique_total_set.update(uniq)
                                size = len(uniq)
                                if size > unique_max:
                                    unique_max = size
                                    unique_max_fid = str(fid)
                            n_flights_flows = int(len(unique_total_set))
                            if unique_max >= 0:
                                n_flights_flows_max = int(unique_max)
                                max_flow_id = unique_max_fid
                    except Exception:
                        pass
                    try:
                        ent_map = rf_info.get("entrants_by_flow", {}) if isinstance(rf_info, dict) else {}
                        if isinstance(ent_map, dict):
                            for fid, count in ent_map.items():
                                try:
                                    entrants_counts_map[str(fid)] = int(count)
                                except Exception:
                                    continue
                        if flow_ids and entrants_counts_map:
                            entrants_total = 0
                            entrants_max = -1
                            entrants_max_fid = None
                            for fid in flow_ids:
                                count = int(entrants_counts_map.get(str(fid), 0))
                                entrants_total += count
                                if count > entrants_max:
                                    entrants_max = count
                                    entrants_max_fid = str(fid)
                            n_entrants_flows = int(entrants_total)
                            if entrants_max >= 0:
                                n_entrants_flows_max = int(entrants_max)
                                entrants_max_flow_id = entrants_max_fid
                    except Exception:
                        pass
                    self.logger.event(
                        "after_commit",
                        {
                            "reg": state.plan[-1].to_canonical_dict(),
                            "delta_j": delta_j,
                            "commits": commits,
                            "candidate_objective": fin_obj,
                            "candidate_components": fin_comp,
                            "candidate_rate_menu_lower": rate_menu_lower,
                            "candidate_rate_menu_upper": rate_menu_upper,
                            "candidate_spill_T": cand_spill,
                            "candidate_nonzero_delay_count": cand_nonzero_delays,
                            "candidate_max_delay_min": cand_max_delay,
                            "N_flights_flows": n_flights_flows,
                            "N_flights_flows_max": n_flights_flows_max,
                            "max_flow_id": max_flow_id,
                            "N_entrants_flows": n_entrants_flows,
                            "N_entrants_flows_max": n_entrants_flows_max,
                            "entrants_max_flow_id": entrants_max_flow_id,
                            "outer_master_run_index": int(master_run_index + 1),
                        },
                    )
                if self._global_stats is not None:
                    plan_id = None
                    if isinstance(diag_payload.get("plan_id"), str):
                        plan_id = str(diag_payload.get("plan_id"))
                    if not plan_id:
                        plan_id = f"outer{master_run_index + 1}-commit{commits}"
                    self._global_stats.on_plan_committed(
                        plan=tuple(state.plan),
                        delta_j=float(delta_j),
                        flow_to_flights=flow_to_flights_map if flow_to_flights_map else None,
                        entrants_by_flow=entrants_counts_map if entrants_counts_map else None,
                        meta={
                            "control_volume_id": ctrl,
                            "window_bins": [int(wb[0]), int(wb[1])],
                            "mode": mode,
                            "objective": fin_obj,
                            "components": fin_comp,
                            "outer_index": int(master_run_index + 1),
                            "plan_id": plan_id,
                            "commits": int(commits),
                            "committed_rates": rates_to_store,
                            "rate_menu_lower": rate_menu_lower,
                            "rate_menu_upper": rate_menu_upper,
                            "delta_j": float(delta_j),
                            "flow_ids": [str(fid) for fid in flow_ids],
                            "timestamp": time.time(),
                            "cand_nonzero_delays": cand_nonzero_delays,
                            "cand_max_delay": cand_max_delay,
                            "cand_spill": cand_spill,
                        },
                    )
                if self.debug_logger is not None:
                    try:
                        self.debug_logger.event(
                            "outer_after_commit",
                            {
                                "delta_j": float(delta_j),
                                "commits": int(commits),
                                "reg": state.plan[-1].to_canonical_dict(),
                                "outer_master_run_index": int(master_run_index + 1),
                            },
                        )
                    except Exception:
                        pass

                if self.early_stop_no_improvement and delta_j >= 0.0:
                    if self.debug_logger is not None:
                        try:
                            self.debug_logger.event(
                                "outer_stop_no_improvement",
                                {
                                    "delta_j": float(delta_j),
                                    "commits": int(commits),
                                    "limit": int(outer_loop_budget),
                                    "outer_master_run_index": int(master_run_index + 1),
                                },
                            )
                        except Exception:
                            pass
                    outer_stop_reason = "no_improvement"
                    outer_stop_info = {
                        "delta_j": float(delta_j),
                        "commits": int(commits),
                        "limit": int(outer_loop_budget),
                    }
                    _emit_master_progress(
                        {
                            "outer_stop_reason": "no_improvement",
                            "outer_stop_info": outer_stop_info,
                            "outer_commits": int(commits),
                            "outer_commit_depth": int(commit_depth_limit),
                            "outer_max_inner_loop_commits_and_evals": (
                                int(self.max_inner_loop_commits_and_evals)
                                if self.max_inner_loop_commits_and_evals is not None
                                else None
                            ),
                            "outer_limit": int(outer_loop_budget),
                            "outer_last_delta_j": float(delta_j),
                            "outer_early_stop_no_improvement": bool(self.early_stop_no_improvement),
                            "outer_run_index": int(run_idx),
                        },
                        master_run_index=master_run_index,
                        partial_actions=master_actions_total + run_actions_total,
                    )
                    if self._global_stats is not None:
                        try:
                            self._global_stats.on_limit_hit(
                                "no_improvement",
                                {
                                    "outer_index": int(master_run_index + 1),
                                    "delta_j": float(delta_j),
                                    "commits": int(commits),
                                },
                            )
                        except Exception:
                            pass
                    break

                run_idx += 1

            if self.max_inner_loop_commits_and_evals is not None and commits >= self.max_inner_loop_commits_and_evals:
                if self.logger is not None:
                    self.logger.event(
                        "inner_commit_limit_reached",
                        {"limit": int(self.max_inner_loop_commits_and_evals), "outer_run": master_run_index + 1},
                    )
                if self.debug_logger is not None:
                    try:
                        self.debug_logger.event(
                            "outer_stop_inner_commit_limit",
                            {
                                "limit": int(self.max_inner_loop_commits_and_evals),
                                "commits": int(commits),
                                "outer_master_run_index": int(master_run_index + 1),
                            },
                        )
                    except Exception:
                        pass
                if self._global_stats is not None:
                    try:
                        self._global_stats.on_limit_hit(
                            "inner_commit_limit_reached",
                            {
                                "outer_index": int(master_run_index + 1),
                                "limit": int(self.max_inner_loop_commits_and_evals),
                                "commits": int(commits),
                            },
                        )
                    except Exception:
                        pass

            with self._timed("agent.final_objective"):
                final_summary = self._compute_final_objective(state)
            if aggregated_counts:
                final_summary = {**final_summary, "action_counts": aggregated_counts}

            final_stop_reason = outer_stop_reason
            if final_stop_reason is None:
                if configured_inner_commit_limit is not None and commits >= configured_inner_commit_limit:
                    final_stop_reason = "inner_commit_limit_reached"
                elif configured_inner_commit_limit is None and commits >= commit_depth_limit:
                    final_stop_reason = "commit_depth_limit_reached"
                    if self._global_stats is not None:
                        try:
                            self._global_stats.on_limit_hit(
                                "commit_depth_limit_reached",
                                {
                                    "outer_index": int(master_run_index + 1),
                                    "limit": int(commit_depth_limit),
                                    "commits": int(commits),
                                },
                            )
                        except Exception:
                            pass
                else:
                    final_stop_reason = "completed"
            final_stop_info = outer_stop_info or {}

            _emit_master_progress(
                {
                    "outer_stop_reason": final_stop_reason,
                    "outer_stop_info": final_stop_info,
                    "outer_commits": int(commits),
                    "outer_commit_depth": int(commit_depth_limit),
                    "outer_max_inner_loop_commits_and_evals": (
                        int(self.max_inner_loop_commits_and_evals)
                        if self.max_inner_loop_commits_and_evals is not None
                        else None
                    ),
                    "outer_limit": int(outer_loop_budget),
                },
                master_run_index=master_run_index,
                partial_actions=master_actions_total + run_actions_total,
            )

            if self._global_stats is not None:
                try:
                    self._global_stats.on_outer_end(
                        int(master_run_index + 1),
                        {
                            "stop_reason": final_stop_reason,
                            "stop_info": final_stop_info,
                            "total_delta_j": float(total_delta_j),
                            "commits": int(commits),
                            "objective": final_summary.get("objective"),
                            "action_counts": dict(aggregated_counts),
                        },
                    )
                except Exception:
                    pass

            if self.debug_logger is not None:
                try:
                    payload = {
                        "stop_reason": final_stop_reason,
                        "stop_info": (final_stop_info or None),
                        "commits": int(commits),
                        "limit": int(outer_loop_budget),
                        "commit_depth": int(commit_depth_limit),
                        "max_inner_loop_commits_and_evals": (
                            int(self.max_inner_loop_commits_and_evals)
                            if self.max_inner_loop_commits_and_evals is not None
                            else None
                        ),
                        "total_delta_j": float(total_delta_j),
                        "objective": float(final_summary.get("objective", 0.0)),
                        "num_flows": int(final_summary.get("num_flows", 0)),
                        "action_counts": aggregated_counts,
                        "outer_master_run_index": int(master_run_index + 1),
                    }
                    self.debug_logger.event("outer_run_end", payload)
                except Exception:
                    pass

            if max_commits_per_inner:
                try:
                    final_summary["max_commits_per_inner"] = list(max_commits_per_inner)
                    final_summary["max_commits_overall"] = int(max(max_commits_per_inner))
                except Exception:
                    pass
            if commit_calls_per_inner:
                try:
                    final_summary["commit_calls_per_inner"] = list(commit_calls_per_inner)
                    final_summary["commit_calls_total"] = int(sum(commit_calls_per_inner))
                except Exception:
                    pass

            objective_value = float("inf")
            obj_candidate = final_summary.get("objective")
            if isinstance(obj_candidate, (int, float)) and np.isfinite(obj_candidate):
                objective_value = float(obj_candidate)

            if best_state is None or objective_value < best_objective:
                best_objective = objective_value
                best_state = state.copy()
                best_summary = dict(final_summary)
                best_action_counts = dict(aggregated_counts)
                best_stop_reason = final_stop_reason
                best_stop_info = dict(final_stop_info)
                best_total_delta_j = float(total_delta_j)
                best_commits = commits
                best_run_index = int(master_run_index + 1)

            if self.diversify_ban_selected:
                banned_next = list(diversify_banned)
                for entry in state.metadata.get("banned_regulations", []) or []:
                    if entry not in banned_next:
                        banned_next.append(entry)
                diversify_banned = banned_next

            master_runs_executed += 1
            master_actions_total += int(run_actions_total)

            if self.outer_max_actions is not None and master_actions_total >= int(self.outer_max_actions):
                break

        if best_state is None:
            best_state = base_state
            best_summary = dict(initial_summary)
            best_action_counts = {}
            best_stop_reason = "no_runs"
            best_stop_info = {"status": "master_budget_exhausted"}
            best_total_delta_j = 0.0
            best_commits = 0
            best_run_index = 0

        elapsed_master = float(time.perf_counter() - master_start)

        best_summary.setdefault("outer_master_runs_executed", int(master_runs_executed))
        best_summary.setdefault("outer_master_total_actions", int(master_actions_total))
        best_summary.setdefault("outer_master_elapsed_s", elapsed_master)
        best_summary.setdefault("outer_master_best_run_index", int(best_run_index))
        best_summary.setdefault(
            "outer_master_max_runs",
            (int(self.outer_max_runs) if self.outer_max_runs is not None else None),
        )
        best_summary.setdefault(
            "outer_master_max_time_s",
            (float(self.outer_max_time_s) if self.outer_max_time_s is not None else None),
        )
        best_summary.setdefault(
            "outer_master_max_actions",
            (int(self.outer_max_actions) if self.outer_max_actions is not None else None),
        )
        best_summary["action_counts"] = dict(best_action_counts)

        _emit_master_progress(
            {
                "outer_stop_reason": best_stop_reason,
                "outer_stop_info": best_stop_info,
                "outer_commits": int(best_commits),
                "outer_commit_depth": int(commit_depth_limit),
                "outer_max_inner_loop_commits_and_evals": (
                    int(self.max_inner_loop_commits_and_evals)
                    if self.max_inner_loop_commits_and_evals is not None
                    else None
                ),
                "outer_limit": int(outer_loop_budget),
            },
            master_run_index=max(best_run_index - 1, 0),
            partial_actions=master_actions_total,
        )

        if self.logger is not None:
            # Clarify the scope of the final objective reported at run_end
            payload: Dict[str, Any] = {
                "commits": best_commits,
                "objective_scope": "plan_domain",
                "objective_scope_desc": "Final plan objective over TVs/windows with committed regulations only",
                **best_summary,
            }
            try:
                # Compute system-wide objective for the final plan
                sysf = self._compute_system_objective(best_state if best_state is not None else state)
                sys_final_obj = sysf.get("system_objective")
                if isinstance(sys_final_obj, (int, float)):
                    # Reflect system-wide objectives in both the log payload and the returned summary
                    best_summary["system_objective"] = float(sys_final_obj)
                    if system_initial_objective is not None:
                        best_summary["system_initial_objective"] = float(system_initial_objective)
                    payload.update({
                        "system_objective": float(sys_final_obj),
                        "system_components": sysf.get("system_components"),
                        "system_objective_scope": "all_tvs",
                        "system_initial_objective": (float(system_initial_objective) if system_initial_objective is not None else None),
                    })
            except Exception:
                pass
            self.logger.event("run_end", payload)

        if self.debug_logger is not None:
            try:
                self.debug_logger.event(
                    "outer_master_end",
                    {
                        "runs_executed": int(master_runs_executed),
                        "total_actions": int(master_actions_total),
                        "elapsed_s": elapsed_master,
                        "best_run_index": int(best_run_index),
                        "best_objective": (
                            float(best_summary.get("objective"))
                            if isinstance(best_summary.get("objective"), (int, float))
                            else None
                        ),
                    },
                )
            except Exception:
                pass

        info = RunInfo(
            commits=best_commits,
            total_delta_j=float(best_total_delta_j),
            log_path=(self.logger.path if self.logger else None),
            debug_log_path=(self.debug_logger.path if self.debug_logger else None),
            summary=best_summary,
            action_counts=best_action_counts,
            stop_reason=best_stop_reason,
            stop_info=best_stop_info,
        )
        return best_state, info

    # ------------------------------------------------------------------
    def _compute_final_objective(self, state: PlanState) -> Dict[str, Any]:
        # Build global flights_by_flow by stitching descriptors per (tv, window)
        flights_by_flow: Dict[str, List[Dict[str, Any]]] = {}
        capacities_by_tv: Dict[str, np.ndarray] = {}
        T = int(self.indexer.num_time_bins)

        def _ensure_caps(tv: str) -> None:
            if tv not in capacities_by_tv:
                caps = self.rate_finder._build_capacities_for_tv(tv)
                capacities_by_tv.update(caps)

        # Helper: entrants → flight specs with requested_bin for deterministic scheduling
        def _specs_for(comm_tv: str, window: Tuple[int, int], flow_to_flights: Mapping[str, Sequence[str]], flow_key_prefix: str) -> Dict[str, List[Dict[str, Any]]]:
            t0, t1 = int(window[0]), int(window[1])
            active = list(range(t0, max(t0 + 1, t1)))
            win_len = max(1, t1 - t0)
            reverse: Dict[str, str] = {}
            for f, ids in flow_to_flights.items():
                for fid in ids:
                    reverse[str(fid)] = str(f)
            # build raw entries
            out: Dict[str, List[Dict[str, Any]]] = {}
            iter_fn = getattr(self.flight_list, "iter_hotspot_crossings", None)
            if callable(iter_fn):
                for fid, tv, entry_dt, t in iter_fn([comm_tv], active_windows={comm_tv: active}):  # type: ignore[misc]
                    flow = reverse.get(str(fid))
                    if flow is None:
                        continue
                    rbin = int(t)
                    # clamp and remap to [0, T]
                    rbin = max(0, min(T, rbin))
                    key = f"{flow_key_prefix}:{flow}"
                    spec = {"flight_id": str(fid), "requested_bin": rbin}
                    if isinstance(entry_dt, datetime):
                        spec["requested_dt"] = entry_dt
                    else:
                        meta = getattr(self.flight_list, "flight_metadata", {}).get(str(fid), {}) or {}
                        takeoff = meta.get("takeoff_time")
                        intervals = meta.get("occupancy_intervals") or []
                        for iv in intervals:
                            try:
                                tvtw_idx = int(iv.get("tvtw_index"))
                            except Exception:
                                continue
                            decoded = self.indexer.get_tvtw_from_index(tvtw_idx)
                            if not decoded:
                                continue
                            tv_decoded, bin_idx = decoded
                            if str(tv_decoded) != str(comm_tv) or int(bin_idx) != rbin:
                                continue
                            entry_s = iv.get("entry_time_s", 0)
                            try:
                                entry_s = float(entry_s)
                            except Exception:
                                entry_s = 0.0
                            if isinstance(takeoff, datetime):
                                spec["requested_dt"] = takeoff + timedelta(seconds=entry_s)
                            break
                    out.setdefault(key, []).append(spec)
            # Ensure keys exist even if no entrants (edge-case)
            for f in flow_to_flights.keys():
                key = f"{flow_key_prefix}:{f}"
                out.setdefault(key, [])
            return out

        if state.plan:
            # Aggregate per regulation
            for reg in state.plan:
                tv = str(reg.control_volume_id)
                t0, t1 = int(reg.window_bins[0]), int(reg.window_bins[1])
                desc = self.inventory.get(tv, (t0, t1))
                if desc is None:
                    # Try fallback to candidates in state metadata
                    desc = None
                    for c in state.metadata.get("hotspot_candidates", []) or []:
                        if str(c.get("control_volume_id")) == tv:
                            wb = c.get("window_bins") or []
                            if int(wb[0]) == t0 and int(wb[1]) == t1:
                                desc = type("_Tmp", (), {"metadata": c.get("metadata", {}), "mode": c.get("mode", "per_flow")})
                                break
                meta = getattr(desc, "metadata", {}) if desc is not None else {}
                flow_to_flights: Mapping[str, Sequence[str]] = meta.get("flow_to_flights", {}) or {}

                _ensure_caps(tv)

                # Build unique key prefix per regulation to avoid id collisions across hotspots/windows
                flow_key_prefix = f"{tv}:{t0}-{t1}"
                if isinstance(reg.committed_rates, dict):
                    # Per-flow
                    specs_map = _specs_for(tv, (t0, t1), {f: flow_to_flights.get(f, []) for f in reg.flow_ids}, flow_key_prefix)
                    for k, v in specs_map.items():
                        flights_by_flow[k] = v
                else:
                    # Blanket: union all listed flows under a synthetic id
                    union_map: Dict[str, Sequence[str]] = {"__blanket__": [fid for f in reg.flow_ids for fid in flow_to_flights.get(f, [])]}
                    synthetic = f"{flow_key_prefix}:__blanket__"
                    specs_map = _specs_for(tv, (t0, t1), union_map, flow_key_prefix)
                    flights_by_flow[synthetic] = specs_map.get(f"{flow_key_prefix}:__blanket__", [])
        else:
            # Baseline (no committed regulations): derive flows from hotspot candidates
            for c in state.metadata.get("hotspot_candidates", []) or []:
                try:
                    tv = str(c.get("control_volume_id"))
                    wb = c.get("window_bins") or []
                    t0, t1 = int(wb[0]), int(wb[1])
                except Exception:
                    continue
                meta = c.get("metadata") or {}
                flow_to_flights = meta.get("flow_to_flights", {}) or {}
                if not isinstance(flow_to_flights, Mapping):
                    continue
                _ensure_caps(tv)
                flow_key_prefix = f"{tv}:{t0}-{t1}"
                specs_map = _specs_for(tv, (t0, t1), flow_to_flights, flow_key_prefix)
                for k, v in specs_map.items():
                    flights_by_flow[k] = v

        # Global context and schedule assembly
        weights = ObjectiveWeights()  # default weights
        context = build_score_context(
            flights_by_flow,
            indexer=self.indexer,
            capacities_by_tv=capacities_by_tv,
            target_cells=None,
            ripple_cells=None,
            flight_list=self.flight_list,
            weights=weights,
        )

        # Compute schedules n_f(t)
        if state.plan:
            # Use committed rates per regulation
            n_f_t: Dict[str, List[int]] = {k: [int(x) for x in np.zeros(self.indexer.num_time_bins + 1, dtype=int)] for k in flights_by_flow.keys()}
            bin_minutes = max(1, int(getattr(self.indexer, "time_bin_minutes", 60)))

            for reg in state.plan:
                tv = str(reg.control_volume_id)
                t0, t1 = int(reg.window_bins[0]), int(reg.window_bins[1])
                flow_key_prefix = f"{tv}:{t0}-{t1}"
                active = sorted(set(range(t0, max(t0 + 1, t1))))
                if isinstance(reg.committed_rates, dict):
                    for f, rate in reg.committed_rates.items():
                        key = f"{flow_key_prefix}:{f}"
                        d = context.d_by_flow.get(key)
                        if d is None:
                            continue
                        schedule = list(d.tolist())
                        rate_val = float(rate)
                        quota = 0 if not (rate_val > 0 and not np.isinf(rate_val)) else int(round(rate_val * bin_minutes / 60.0))
                        ready = 0
                        released = 0
                        for t in active:
                            t = int(t)
                            if t < 0 or t >= self.indexer.num_time_bins:
                                continue
                            ready += int(d[t])
                            available = max(0, ready - released)
                            take = min(quota, available) if quota > 0 else 0
                            schedule[t] = int(take)
                            released += int(take)
                        total_non_overflow = int(sum(d[: self.indexer.num_time_bins]))
                        scheduled_non_overflow = int(sum(schedule[: self.indexer.num_time_bins]))
                        base_overflow = int(d[self.indexer.num_time_bins]) if len(d) > self.indexer.num_time_bins else 0
                        schedule[self.indexer.num_time_bins] = int(base_overflow + max(0, total_non_overflow - scheduled_non_overflow))
                        n_f_t[key] = schedule
                else:
                    # Blanket
                    key = f"{flow_key_prefix}:__blanket__"
                    d = context.d_by_flow.get(key)
                    if d is None:
                        continue
                    schedule = list(d.tolist())
                    rate_val = float(reg.committed_rates) if reg.committed_rates is not None else float("inf")
                    quota = 0 if not (rate_val > 0 and not np.isinf(rate_val)) else int(round(rate_val * bin_minutes / 60.0))
                    ready = 0
                    released = 0
                    for t in active:
                        t = int(t)
                        if t < 0 or t >= self.indexer.num_time_bins:
                            continue
                        ready += int(d[t])
                        available = max(0, ready - released)
                        take = min(quota, available) if quota > 0 else 0
                        schedule[t] = int(take)
                        released += int(take)
                    total_non_overflow = int(sum(d[: self.indexer.num_time_bins]))
                    scheduled_non_overflow = int(sum(schedule[: self.indexer.num_time_bins]))
                    base_overflow = int(d[self.indexer.num_time_bins]) if len(d) > self.indexer.num_time_bins else 0
                    schedule[self.indexer.num_time_bins] = int(base_overflow + max(0, total_non_overflow - scheduled_non_overflow))
                    n_f_t[key] = schedule
        else:
            # Baseline schedule: release at requested bins (no regulation)
            n_f_t = {}
            for k, d in context.d_by_flow.items():
                try:
                    n_f_t[str(k)] = list(d.tolist())
                except Exception:
                    arr = np.asarray(d)
                    n_f_t[str(k)] = [int(x) for x in arr.tolist()]

        objective, components, artifacts = score_with_context(n_f_t, flights_by_flow=flights_by_flow, capacities_by_tv=capacities_by_tv, flight_list=self.flight_list, context=context)

        # Compute final plan spill and in-window release counts
        try:
            T_int = int(self.indexer.num_time_bins)
            spill_T_val = int(sum(int(np.asarray(v)[T_int]) for v in artifacts.get("n", {}).values()))
        except Exception:
            spill_T_val = None  # type: ignore[assignment]
        try:
            T_int = int(self.indexer.num_time_bins)
            inwin_total_val = int(sum(int(np.asarray(v)[:T_int].sum()) for v in artifacts.get("n", {}).values()))
        except Exception:
            inwin_total_val = None  # type: ignore[assignment]

        return {
            "objective": float(objective),
            "components": components,
            "artifacts": artifacts,
            "num_flows": len(flights_by_flow),
            "spill_T": spill_T_val,
            "in_window_releases": inwin_total_val,
        }

    # ------------------------------------------------------------------
    def _compute_system_objective(self, state: PlanState) -> Dict[str, Any]:
        """
        Compute objective over ALL TVs in the indexer (global, not scoped).

        Scheduling is applied only to flows that belong to the current plan's
        regulated TVs/windows (if any). Unregulated TVs/windows contribute via
        baseline occupancy (zero-delay) captured by the context.
        """
        flights_by_flow: Dict[str, List[Dict[str, Any]]] = {}
        capacities_by_tv: Dict[str, np.ndarray] = {}
        T = int(self.indexer.num_time_bins)

        def _ensure_caps(tv: str) -> None:
            if tv not in capacities_by_tv:
                caps = self.rate_finder._build_capacities_for_tv(tv)
                capacities_by_tv.update(caps)

        # Include capacities for all TVs
        for tv in getattr(self.indexer, "tv_id_to_idx", {}).keys():
            _ensure_caps(str(tv))

        # Build schedules only for plan domain (if any)
        if state.plan:
            # For each regulation, reconstruct flow_to_flights from inventory or candidates
            for reg in state.plan:
                tv = str(reg.control_volume_id)
                t0, t1 = int(reg.window_bins[0]), int(reg.window_bins[1])
                desc = self.inventory.get(tv, (t0, t1))
                if desc is None:
                    # Fallback to candidates stored in state metadata
                    for c in state.metadata.get("hotspot_candidates", []) or []:
                        if str(c.get("control_volume_id")) == tv:
                            wb = c.get("window_bins") or []
                            if int(wb[0]) == t0 and int(wb[1]) == t1:
                                desc = type("_Tmp", (), {"metadata": c.get("metadata", {}), "mode": c.get("mode", "per_flow")})
                                break
                meta = getattr(desc, "metadata", {}) if desc is not None else {}
                flow_to_flights: Mapping[str, Sequence[str]] = meta.get("flow_to_flights", {}) or {}

                prefix = f"{tv}:{t0}-{t1}"
                if isinstance(reg.committed_rates, dict):
                    specs_map = {}
                    # Only include flows listed in the regulation
                    for f in reg.flow_ids:
                        specs_map.update(self._specs_for_system(tv, (t0, t1), {f: flow_to_flights.get(f, [])}, prefix))
                    for k, v in specs_map.items():
                        flights_by_flow[k] = v
                else:
                    # Blanket: union all listed flows under a synthetic id
                    union_map: Dict[str, Sequence[str]] = {"__blanket__": [fid for f in reg.flow_ids for fid in flow_to_flights.get(f, [])]}
                    specs_map = self._specs_for_system(tv, (t0, t1), union_map, prefix)
                    flights_by_flow[f"{prefix}:__blanket__"] = specs_map.get(f"{prefix}:__blanket__", [])
        # Else: no plan schedules; flights_by_flow remains empty (baseline-only global)

        # Build global context (tvs_of_interest = all TVs via capacities_by_tv)
        context = build_score_context(
            flights_by_flow,
            indexer=self.indexer,
            capacities_by_tv=capacities_by_tv,
            target_cells=None,
            ripple_cells=None,
            flight_list=self.flight_list,
            weights=ObjectiveWeights(),
        )

        # Assemble schedules
        if state.plan and flights_by_flow:
            n_f_t: Dict[str, List[int]] = {k: [int(x) for x in np.zeros(self.indexer.num_time_bins + 1, dtype=int)] for k in flights_by_flow.keys()}
            bin_minutes = max(1, int(getattr(self.indexer, "time_bin_minutes", 60)))
            for reg in state.plan:
                tv = str(reg.control_volume_id)
                t0, t1 = int(reg.window_bins[0]), int(reg.window_bins[1])
                prefix = f"{tv}:{t0}-{t1}"
                active = sorted(set(range(t0, max(t0 + 1, t1))))
                if isinstance(reg.committed_rates, dict):
                    for f, rate in reg.committed_rates.items():
                        key = f"{prefix}:{f}"
                        d = context.d_by_flow.get(key)
                        if d is None:
                            continue
                        schedule = list(d.tolist())
                        rate_val = float(rate)
                        quota = 0 if not (rate_val > 0 and not np.isinf(rate_val)) else int(round(rate_val * bin_minutes / 60.0))
                        ready = 0
                        released = 0
                        for t in active:
                            t = int(t)
                            if t < 0 or t >= self.indexer.num_time_bins:
                                continue
                            ready += int(d[t])
                            available = max(0, ready - released)
                            take = min(quota, available) if quota > 0 else 0
                            schedule[t] = int(take)
                            released += int(take)
                        total_non_overflow = int(sum(d[: self.indexer.num_time_bins]))
                        scheduled_non_overflow = int(sum(schedule[: self.indexer.num_time_bins]))
                        base_overflow = int(d[self.indexer.num_time_bins]) if len(d) > self.indexer.num_time_bins else 0
                        schedule[self.indexer.num_time_bins] = int(base_overflow + max(0, total_non_overflow - scheduled_non_overflow))
                        n_f_t[key] = schedule
                else:
                    key = f"{prefix}:__blanket__"
                    d = context.d_by_flow.get(key)
                    if d is None:
                        continue
                    schedule = list(d.tolist())
                    rate_val = float(reg.committed_rates) if reg.committed_rates is not None else float("inf")
                    quota = 0 if not (rate_val > 0 and not np.isinf(rate_val)) else int(round(rate_val * bin_minutes / 60.0))
                    ready = 0
                    released = 0
                    for t in active:
                        t = int(t)
                        if t < 0 or t >= self.indexer.num_time_bins:
                            continue
                        ready += int(d[t])
                        available = max(0, ready - released)
                        take = min(quota, available) if quota > 0 else 0
                        schedule[t] = int(take)
                        released += int(take)
                    total_non_overflow = int(sum(d[: self.indexer.num_time_bins]))
                    scheduled_non_overflow = int(sum(schedule[: self.indexer.num_time_bins]))
                    base_overflow = int(d[self.indexer.num_time_bins]) if len(d) > self.indexer.num_time_bins else 0
                    schedule[self.indexer.num_time_bins] = int(base_overflow + max(0, total_non_overflow - scheduled_non_overflow))
                    n_f_t[key] = schedule
        else:
            n_f_t = {}

        objective, components, artifacts = score_with_context(
            n_f_t,
            flights_by_flow=flights_by_flow,
            capacities_by_tv=capacities_by_tv,
            flight_list=self.flight_list,
            context=context,
        )

        return {
            "system_objective": float(objective),
            "system_components": components,
        }

    # Helper for system objective: reuse entrants -> specs mapping
    def _specs_for_system(self, comm_tv: str, window: Tuple[int, int], flow_to_flights: Mapping[str, Sequence[str]], flow_key_prefix: str) -> Dict[str, List[Dict[str, Any]]]:
        T = int(self.indexer.num_time_bins)
        t0, t1 = int(window[0]), int(window[1])
        active = list(range(t0, max(t0 + 1, t1)))
        reverse: Dict[str, str] = {}
        for f, ids in flow_to_flights.items():
            for fid in ids:
                reverse[str(fid)] = str(f)
        out: Dict[str, List[Dict[str, Any]]] = {}
        iter_fn = getattr(self.flight_list, "iter_hotspot_crossings", None)
        if callable(iter_fn):
            for fid, tv, entry_dt, t in iter_fn([comm_tv], active_windows={comm_tv: active}):  # type: ignore[misc]
                flow = reverse.get(str(fid))
                if flow is None:
                    continue
                rbin = max(0, min(T, int(t)))
                key = f"{flow_key_prefix}:{flow}"
                spec = {"flight_id": str(fid), "requested_bin": rbin}
                if isinstance(entry_dt, datetime):
                    spec["requested_dt"] = entry_dt
                out.setdefault(key, []).append(spec)
        for f in flow_to_flights.keys():
            key = f"{flow_key_prefix}:{f}"
            out.setdefault(key, [])
        return out


__all__ = ["MCTSAgent", "RunInfo"]
