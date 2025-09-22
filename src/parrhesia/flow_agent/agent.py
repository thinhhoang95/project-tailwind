from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

from .actions import NewRegulation
from .logging import SearchLogger
from .mcts import MCTS, MCTSConfig
from .rate_finder import RateFinder, RateFinderConfig
from .state import PlanState
from .transition import CheapTransition
from .hotspot_discovery import (
    HotspotDiscoveryConfig,
    HotspotInventory,
)
from parrhesia.optim.objective import ObjectiveWeights, build_score_context, score_with_context


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
        timer: Optional[Callable[[str], ContextManager[Any]]] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.evaluator = evaluator
        self.flight_list = flight_list
        self.indexer = indexer
        self.mcts_cfg = mcts_cfg or MCTSConfig()
        self.discovery_cfg = discovery_cfg or HotspotDiscoveryConfig()
        self.logger = logger
        self.debug_logger = debug_logger
        self.cold_logger = cold_logger
        self.max_regulations = None if max_regulations is None else int(max(0, max_regulations))
        self._timer_factory = timer
        self._progress_cb = progress_cb

        self.rate_finder = RateFinder(
            evaluator=evaluator,
            flight_list=flight_list,
            indexer=indexer,
            config=rate_finder_cfg or RateFinderConfig(use_adaptive_grid=True),
            timer=timer,
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

    # ------------------------------------------------------------------
    def run(self) -> Tuple[PlanState, RunInfo]:
        # Build hotspot inventory and seed plan state
        cfg = self.discovery_cfg
        with self._timed("agent.hotspot_inventory"):
            descs = self.inventory.build_from_segments(
                threshold=float(cfg.threshold),
                top_hotspots=int(cfg.top_hotspots),
                top_flows=int(cfg.top_flows),
                max_flights_per_flow=int(cfg.max_flights_per_flow),
                leiden_params=cfg.leiden_params,
                direction_opts=cfg.direction_opts,
            )
            candidates = self.inventory.to_candidate_payloads(descs)

        state = PlanState()
        state.metadata["hotspot_candidates"] = candidates

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
            self.logger.event("run_start", {"num_candidates": len(candidates), "mcts_cfg": self.mcts_cfg.__dict__})
        if self.debug_logger is not None:
            try:
                self.debug_logger.event(
                    "outer_run_start",
                    {
                        "num_candidates": len(candidates),
                        "commit_depth": int(self.mcts_cfg.commit_depth),
                        "max_regulations": (int(self.max_regulations) if self.max_regulations is not None else None),
                        "max_sims": int(self.mcts_cfg.max_sims),
                        "max_time_s": float(self.mcts_cfg.max_time_s),
                        "commit_eval_limit": int(self.mcts_cfg.commit_eval_limit),
                        "max_actions": (int(self.mcts_cfg.max_actions) if getattr(self.mcts_cfg, "max_actions", None) not in (None, 0) else None),
                    },
                )
            except Exception:
                pass

        commits = 0
        total_delta_j = 0.0
        # Loop: allow multiple commits until STOP logic or inventory exhaustion
        limit = int(self.mcts_cfg.commit_depth)
        if self.max_regulations is not None:
            limit = min(limit, int(self.max_regulations))
        aggregated_counts: Dict[str, int] = {}
        outer_stop_reason: Optional[str] = None
        outer_stop_info: Dict[str, Any] = {}
        run_idx = 1
        max_commits_per_inner: List[int] = []
        commit_calls_per_inner: List[int] = []
        for _ in range(max(1, limit)):
            # Ensure we are in idle to start a new regulation
            state.stage = "idle"
            try:
                with self._timed("agent.mcts.run"):
                    commit_action = mcts.run(state, run_index=run_idx)
            except Exception as exc:
                message = str(exc)
                for k, v in mcts.action_counts.items():
                    aggregated_counts[k] = aggregated_counts.get(k, 0) + int(v)
                if isinstance(exc, RuntimeError) and "MCTS did not evaluate any commit" in message:
                    if self.logger is not None:
                        self.logger.event("mcts_no_commit", {"error": message})
                    if self.debug_logger is not None:
                        try:
                            self.debug_logger.event(
                                "mcts_no_commit",
                                {"error": message, "exc_type": type(exc).__name__},
                            )
                        except Exception:
                            pass
                    outer_stop_reason = "no_commit_available"
                    outer_stop_info = {"status": "no_commit_evaluated", "exc_type": type(exc).__name__}
                    break
                if self.logger is not None:
                    self.logger.event("mcts_error", {"error": message})
                if self.debug_logger is not None:
                    try:
                        self.debug_logger.event("mcts_error", {"error": message, "exc_type": type(exc).__name__})
                    except Exception:
                        pass
                outer_stop_reason = "mcts_error"
                outer_stop_info = {"error": message, "exc_type": type(exc).__name__}
                break

            for k, v in mcts.action_counts.items():
                aggregated_counts[k] = aggregated_counts.get(k, 0) + int(v)

            # Cold Feet: capture per-run stats
            try:
                stats = getattr(mcts, "last_run_stats", {}) or {}
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

            # Extract improvement for bookkeeping
            info = (commit_action.diagnostics or {}).get("rate_finder", {})
            delta_j = float(info.get("delta_j", 0.0))

            # Materialize regulation and append to plan without relying on stage guards
            # Extract details from diagnostics
            diag = (commit_action.diagnostics or {}).get("rate_finder", {})
            ctrl = str(diag.get("control_volume_id")) if diag.get("control_volume_id") is not None else None
            win_list = diag.get("window_bins") or []
            try:
                wb = (int(win_list[0]), int(win_list[1]))
            except Exception:
                wb = (0, 1)
            mode = str(diag.get("mode", "per_flow"))
            # Flow ids for the regulation
            flow_ids: Tuple[str, ...]
            if isinstance(commit_action.committed_rates, dict):
                flow_ids = tuple(sorted(str(k) for k in commit_action.committed_rates.keys()))
            else:
                entrants = diag.get("entrants_by_flow", {}) or {}
                flow_ids = tuple(sorted(str(k) for k in entrants.keys()))

            from .state import RegulationSpec  # local import to avoid cycles at module import time

            if ctrl is None:
                # As a fallback, skip appending if control volume is unknown
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
                try:
                    iv = int(round(float(rates))) if rates is not None else 0
                except Exception:
                    iv = 0
                if iv > 0 and len(flow_ids) > 0:
                    valid = True
                    rates_to_store = iv

            if not valid:
                self._ban_regulation_spec(
                    state,
                    control_volume_id=ctrl,
                    window_bins=wb,
                    flow_ids=flow_ids,
                    mode=mode,
                )
                if self.debug_logger is not None:
                    try:
                        self.debug_logger.event(
                            "outer_skip_empty_regulation",
                            {
                                "control_volume_id": ctrl,
                                "window_bins": [int(wb[0]), int(wb[1])],
                                "mode": mode,
                                "flow_ids": list(flow_ids),
                                "committed_rates": rates,
                                "reason": "no_effective_rates_or_no_flows",
                            },
                        )
                    except Exception:
                        pass
                continue

            regulation = RegulationSpec(
                control_volume_id=ctrl,
                window_bins=wb,
                flow_ids=flow_ids,
                mode="per_flow" if mode == "per_flow" else "blanket",
                committed_rates=rates_to_store,
                diagnostics=dict(commit_action.diagnostics or {}),
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

            if self.logger is not None:
                self.logger.event("after_commit", {"reg": state.plan[-1].to_canonical_dict(), "delta_j": delta_j, "commits": commits})
            if self.debug_logger is not None:
                try:
                    self.debug_logger.event(
                        "outer_after_commit",
                        {
                            "delta_j": float(delta_j),
                            "commits": int(commits),
                            "reg": state.plan[-1].to_canonical_dict(),
                        },
                    )
                except Exception:
                    pass

            # Optional early stop if no improvement
            if delta_j >= 0.0:
                if self.debug_logger is not None:
                    try:
                        self.debug_logger.event(
                            "outer_stop_no_improvement",
                            {"delta_j": float(delta_j), "commits": int(commits), "limit": int(limit)},
                        )
                    except Exception:
                        pass
                outer_stop_reason = "no_improvement"
                outer_stop_info = {"delta_j": float(delta_j), "commits": int(commits), "limit": int(limit)}
                break

            run_idx += 1

        if self.max_regulations is not None and commits >= self.max_regulations and self.logger is not None:
            self.logger.event("regulation_limit_reached", {"limit": int(self.max_regulations)})
        if self.max_regulations is not None and commits >= self.max_regulations and self.debug_logger is not None:
            try:
                self.debug_logger.event("outer_stop_regulation_limit", {"limit": int(self.max_regulations), "commits": int(commits)})
            except Exception:
                pass

        # Final global objective across all committed regulations
        with self._timed("agent.final_objective"):
            final_summary = self._compute_final_objective(state)
        if aggregated_counts:
            final_summary = {**final_summary, "action_counts": aggregated_counts}
        if self.logger is not None:
            self.logger.event("run_end", {"commits": commits, **final_summary})

        # Determine final stop reason and info irrespective of debug logging
        final_stop_reason = outer_stop_reason
        if final_stop_reason is None:
            if self.max_regulations is not None and commits >= self.max_regulations:
                final_stop_reason = "regulation_limit_reached"
            elif commits >= limit:
                final_stop_reason = "commit_depth_limit_reached"
            else:
                final_stop_reason = "completed"
        final_stop_info = outer_stop_info or {}

        # Emit a consolidated outer-run termination record to the debug log
        if self.debug_logger is not None:
            try:
                payload = {
                    "stop_reason": final_stop_reason,
                    "stop_info": (final_stop_info or None),
                    "commits": int(commits),
                    "limit": int(limit),
                    "commit_depth": int(self.mcts_cfg.commit_depth),
                    "max_regulations": (int(self.max_regulations) if self.max_regulations is not None else None),
                    "total_delta_j": float(total_delta_j),
                    "objective": float(final_summary.get("objective", 0.0)),
                    "num_flows": int(final_summary.get("num_flows", 0)),
                    "action_counts": aggregated_counts,
                }
                self.debug_logger.event("outer_run_end", payload)
            except Exception:
                pass

        # Attach cold feet aggregate stats to final summary
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

        info = RunInfo(
            commits=commits,
            total_delta_j=float(total_delta_j),
            log_path=(self.logger.path if self.logger else None),
            debug_log_path=(self.debug_logger.path if self.debug_logger else None),
            summary=final_summary,
            action_counts=aggregated_counts,
            stop_reason=final_stop_reason,
            stop_info=final_stop_info,
        )
        return state, info

    # ------------------------------------------------------------------
    def _compute_final_objective(self, state: PlanState) -> Dict[str, Any]:
        if not state.plan:
            return {"objective": 0.0, "components": {}, "artifacts": {}, "num_flows": 0}

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
                for fid, tv, _dt, t in iter_fn([comm_tv], active_windows={comm_tv: active}):  # type: ignore[misc]
                    flow = reverse.get(str(fid))
                    if flow is None:
                        continue
                    rbin = int(t)
                    # clamp and remap to [0, T]
                    rbin = max(0, min(T, rbin))
                    key = f"{flow_key_prefix}:{flow}"
                    out.setdefault(key, []).append({"flight_id": str(fid), "requested_bin": rbin})
            # Ensure keys exist even if no entrants (edge-case)
            for f in flow_to_flights.keys():
                key = f"{flow_key_prefix}:{f}"
                out.setdefault(key, [])
            return out

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

        # Compute schedules n_f(t) using committed rates per regulation
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

        objective, components, artifacts = score_with_context(n_f_t, flights_by_flow=flights_by_flow, capacities_by_tv=capacities_by_tv, flight_list=self.flight_list, context=context)

        return {
            "objective": float(objective),
            "components": components,
            "artifacts": artifacts,
            "num_flows": len(flights_by_flow),
        }


__all__ = ["MCTSAgent", "RunInfo"]
