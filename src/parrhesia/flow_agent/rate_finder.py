from __future__ import annotations

import math
import time
from collections import Counter, OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

from parrhesia.optim.objective import (
    ObjectiveWeights,
    ScoreContext,
    build_score_context,
)
from .safespill_objective import (
    score_with_context,
    score_with_context_precomputed_occ,
)
from .state import PlanState


@dataclass
class RateFinderConfig:
    """Tunable parameters for `RateFinder`."""

    rate_grid: Tuple[float, ...] = (
        math.inf,
        60.0,
        48.0,
        36.0,
        24.0,
        18.0,
        12.0,
        6.0,
    )
    passes: int = 2
    epsilon: float = 1e-3
    max_eval_calls: int = 256
    cache_size: int = 256
    objective_weights: Optional[Dict[str, float]] = None
    use_adaptive_grid: bool = False
    max_adaptive_rate: int = 120
    max_adaptive_candidates: int = 8
    # Fast scorer and verbosity controls
    fast_scorer_enabled: bool = True
    verbose: bool = False


@dataclass
class _CandidateResult:
    delta_j: float
    objective: float
    components: Dict[str, float]
    artifacts: Dict[str, Any]


@dataclass
class _BaselineResult:
    objective: float
    components: Dict[str, float]
    artifacts: Dict[str, Any]


class RateFinder:
    """Deterministic coordinate descent over discrete hourly rate grids."""

    def __init__(
        self,
        *,
        evaluator: NetworkEvaluator,
        flight_list: FlightList,
        indexer: TVTWIndexer,
        config: Optional[RateFinderConfig] = None,
        timer: Optional[Callable[[str], ContextManager[Any]]] = None,
    ) -> None:
        self.evaluator = evaluator
        self._base_flight_list = flight_list
        self._indexer = indexer
        self.config = config or RateFinderConfig()
        self._score_context_cache: Dict[Tuple, ScoreContext] = {}
        self._baseline_cache: Dict[Tuple, _BaselineResult] = {}
        self._candidate_cache: "OrderedDict[Tuple, _CandidateResult]" = OrderedDict()
        self._entrants_cache: "OrderedDict[Tuple, Dict[str, Tuple[Tuple[Any, Any, int], ...]]]" = OrderedDict()
        self._rate_grid_cache: "OrderedDict[Tuple, Tuple[float, ...]]" = OrderedDict()
        self._base_occ_cache: "OrderedDict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._timer_factory = timer

    def _timed(self, name: str) -> ContextManager[Any]:
        if self._timer_factory is None:
            return nullcontext()
        return self._timer_factory(name)

    # ------------------------------------------------------------------
    def find_rates(
        self,
        *,
        plan_state: PlanState,
        control_volume_id: str,
        window_bins: Tuple[int, int],
        flows: Mapping[str, Sequence[str]],
        mode: str = "per_flow",
    ) -> Tuple[Union[int, Dict[str, float]], float, Dict[str, object]]:
        if mode not in {"per_flow", "blanket"}:
            raise ValueError("mode must be 'per_flow' or 'blanket'")

        flow_map = {str(fid): tuple(str(f) for f in flights) for fid, flights in flows.items()}
        flow_ids = sorted(flow_map.keys())
        if not flow_ids:
            return ({} if mode == "per_flow" else math.inf, 0.0, {"reason": "no_flows"})

        window_start = int(window_bins[0])
        window_end = int(window_bins[1])
        if window_end <= window_start:
            window_end = window_start + 1
        active_windows = list(range(window_start, window_end))
        plan_key = plan_state.canonical_key()

        with self._timed("rate_finder.compute_entrants"):
            entrants = self._compute_entrants(control_volume_id, active_windows, flow_map)

        with self._timed("rate_finder.resolve_rate_grid"):
            rate_grid = self._resolve_rate_grid(
                control_volume_id=control_volume_id,
                window_bins=window_bins,
                flow_map=flow_map,
                mode=mode,
                active_windows=active_windows,
                entrants=entrants,
            )

        if self.config.verbose:
            if mode == "per_flow":
                for flow_id in flow_ids:
                    print(f"[RateFinder] candidate rates for flow {flow_id}: {tuple(rate_grid)}")
            else:
                print(f"[RateFinder] candidate rates (blanket mode): {tuple(rate_grid)}")

        flow_order = sorted(flow_ids, key=lambda f: (-len(entrants.get(f, [])), f))

        context_flow_ids = flow_ids if mode == "per_flow" else ["__blanket__"]
        context_key = self._context_key(
            plan_key=str(plan_key),
            control_volume_id=str(control_volume_id),
            window_bins=tuple(int(b) for b in window_bins),
            flow_ids=context_flow_ids,
        )

        with self._timed("rate_finder.build_flights_by_flow"):
            flights_by_flow = self._build_flights_by_flow(
                control_volume_id=control_volume_id,
                active_windows=active_windows,
                flow_map=flow_map,
                entrants=entrants,
                mode=mode,
            )
        with self._timed("rate_finder.build_capacities"):
            capacities_by_tv = self._build_capacities_for_tv(control_volume_id)

        target_cells = {(str(control_volume_id), int(t)) for t in active_windows}
        weights_cfg = self.config.objective_weights or {}
        weights = ObjectiveWeights(**weights_cfg) if weights_cfg else ObjectiveWeights()

        with self._timed("rate_finder.ensure_context_and_baseline"):
            context, baseline = self._ensure_context_and_baseline(
                context_key=context_key,
                flights_by_flow=flights_by_flow,
                capacities_by_tv=capacities_by_tv,
                target_cells=target_cells,
                weights=weights,
                tv_filter=[control_volume_id],
                plan_key=str(plan_key),
                control_volume_id=str(control_volume_id),
            )
        baseline_obj = baseline.objective
        baseline_components = baseline.components
        baseline_artifacts = baseline.artifacts

        eval_calls = 0
        cache_hits = 0
        best_delta = 0.0
        best_objective = baseline_obj
        pass_improvements: list[float] = []
        stopped_early = False

        start_ts = time.perf_counter()

        bin_minutes = max(1, int(getattr(self._indexer, "time_bin_minutes", 60)))
        T = int(self._indexer.num_time_bins)
        active_bins_sorted = sorted({int(b) for b in active_windows if 0 <= int(b) < T})

        # Adaptive pass control
        passes_to_use = 1 if bool(self.config.use_adaptive_grid) else int(self.config.passes)

        # Evaluation call budgeting
        rate_grid_len = len(rate_grid)
        flow_count = len(context_flow_ids)
        eval_call_limit = min(int(self.config.max_eval_calls), int(flow_count) * (int(rate_grid_len) + 3))

        if mode == "per_flow":
            best_rates: Dict[str, float] = {fid: math.inf for fid in context_flow_ids}
            history_out: Dict[str, Dict[str, float]] = {fid: {} for fid in context_flow_ids}

            for _ in range(int(passes_to_use)):
                prev_delta = best_delta
                for flow_id in flow_order:
                    if stopped_early:
                        break
                    for rate in rate_grid:
                        rate_val = float(rate)
                        rates_tuple = self._as_rate_tuple(best_rates, context_flow_ids, override=(flow_id, rate_val))
                        signature = self._candidate_signature(
                            plan_key,
                            control_volume_id,
                            window_bins,
                            context_flow_ids,
                            mode,
                            rates_tuple,
                        )
                        cached_result = self._candidate_cache_lookup(signature)
                        if cached_result is not None:
                            result = cached_result
                            cache_hits += 1
                        else:
                            candidate_rates = dict(best_rates)
                            candidate_rates[flow_id] = rate_val
                            with self._timed("rate_finder.build_schedule"):
                                schedule = self._build_schedule_from_rates(
                                    rates_map=candidate_rates,
                                    context=context,
                                    active_bins=active_bins_sorted,
                                    bin_minutes=bin_minutes,
                                )
                            with self._timed("rate_finder.evaluate_candidate"):
                                result = self._evaluate_candidate(
                                    signature=signature,
                                    schedule=schedule,
                                    flights_by_flow=flights_by_flow,
                                    capacities_by_tv=capacities_by_tv,
                                    flight_list=self._base_flight_list,
                                    context=context,
                                    baseline_obj=baseline_obj,
                                )
                            eval_calls += 1
                            if eval_calls >= eval_call_limit:
                                stopped_early = True
                        history_out.setdefault(flow_id, {})[str(rate_val)] = result.delta_j
                        if result.delta_j < best_delta:
                            best_delta = result.delta_j
                            best_objective = result.objective
                            best_rates[flow_id] = rate_val
                    if stopped_early:
                        break
                pass_improvements.append(prev_delta - best_delta)
                tol = self.config.epsilon * max(1.0, abs(baseline_obj))
                if pass_improvements[-1] <= tol or stopped_early:
                    break
            tie_tol = max(1e-6, abs(best_delta) * 1e-6)
            for flow_id in context_flow_ids:
                current = float(best_rates.get(flow_id, math.inf))
                if math.isfinite(current) and current > 0:
                    continue
                history = (history_out.get(flow_id) or {})
                candidate_rate = None
                candidate_delta = None
                for rate_key, delta_val in history.items():
                    try:
                        rate_candidate = float(rate_key)
                        delta_candidate = float(delta_val)
                    except Exception:
                        continue
                    if not math.isfinite(rate_candidate) or rate_candidate <= 0:
                        continue
                    if delta_candidate > best_delta + tie_tol:
                        continue
                    if (
                        candidate_delta is None
                        or delta_candidate < candidate_delta - tie_tol
                        or (
                            abs(delta_candidate - candidate_delta) <= tie_tol
                            and (candidate_rate is None or rate_candidate < candidate_rate)
                        )
                    ):
                        candidate_delta = delta_candidate
                        candidate_rate = rate_candidate
                if candidate_rate is not None:
                    best_rates[flow_id] = float(candidate_rate)
            rates_out: Union[int, Dict[str, float]] = {fid: float(best_rates[fid]) for fid in context_flow_ids}
            final_rates_map = {fid: float(best_rates[fid]) for fid in context_flow_ids}
        else:
            history_out = {"__blanket__": {}}
            best_rate = math.inf
            for _ in range(int(passes_to_use)):
                prev_delta = best_delta
                for rate in rate_grid:
                    rate_val = float(rate)
                    rates_tuple = (rate_val,)
                    signature = self._candidate_signature(
                        plan_key,
                        control_volume_id,
                        window_bins,
                        context_flow_ids,
                        mode,
                        rates_tuple,
                    )
                    cached_result = self._candidate_cache_lookup(signature)
                    if cached_result is not None:
                        result = cached_result
                        cache_hits += 1
                    else:
                        candidate_rates = {context_flow_ids[0]: rate_val}
                        with self._timed("rate_finder.build_schedule"):
                            schedule = self._build_schedule_from_rates(
                                rates_map=candidate_rates,
                                context=context,
                                active_bins=active_bins_sorted,
                                bin_minutes=bin_minutes,
                            )
                        with self._timed("rate_finder.evaluate_candidate"):
                            result = self._evaluate_candidate(
                                signature=signature,
                                schedule=schedule,
                                flights_by_flow=flights_by_flow,
                                capacities_by_tv=capacities_by_tv,
                                flight_list=self._base_flight_list,
                                context=context,
                                baseline_obj=baseline_obj,
                            )
                        eval_calls += 1
                        if eval_calls >= eval_call_limit:
                            stopped_early = True
                    history_out.setdefault("__blanket__", {})[str(rate_val)] = result.delta_j
                    if result.delta_j < best_delta:
                        best_delta = result.delta_j
                        best_objective = result.objective
                        best_rate = rate_val
                pass_improvements.append(prev_delta - best_delta)
                tol = self.config.epsilon * max(1.0, abs(baseline_obj))
                if pass_improvements[-1] <= tol or stopped_early:
                    break
            tie_tol = max(1e-6, abs(best_delta) * 1e-6)
            if not (math.isfinite(best_rate) and best_rate > 0):
                candidate_rate = None
                candidate_delta = None
                history = history_out.get("__blanket__") or {}
                for rate_key, delta_val in history.items():
                    try:
                        rate_candidate = float(rate_key)
                        delta_candidate = float(delta_val)
                    except Exception:
                        continue
                    if not math.isfinite(rate_candidate) or rate_candidate <= 0:
                        continue
                    if delta_candidate > best_delta + tie_tol:
                        continue
                    if (
                        candidate_delta is None
                        or delta_candidate < candidate_delta - tie_tol
                        or (
                            abs(delta_candidate - candidate_delta) <= tie_tol
                            and (candidate_rate is None or rate_candidate < candidate_rate)
                        )
                    ):
                        candidate_delta = delta_candidate
                        candidate_rate = rate_candidate
                if candidate_rate is not None:
                    best_rate = float(candidate_rate)
            rates_out = float(best_rate)
            final_rates_map = {context_flow_ids[0]: float(best_rate)}

        final_rates_tuple = self._as_rate_tuple(final_rates_map, context_flow_ids)
        final_signature = self._candidate_signature(
            plan_key,
            control_volume_id,
            window_bins,
            context_flow_ids,
            mode,
            final_rates_tuple,
        )
        final_result = self._candidate_cache_lookup(final_signature)
        if final_result is not None:
            cache_hits += 1
        else:
            with self._timed("rate_finder.build_schedule"):
                final_schedule = self._build_schedule_from_rates(
                    rates_map=final_rates_map,
                    context=context,
                    active_bins=active_bins_sorted,
                    bin_minutes=bin_minutes,
                )
            with self._timed("rate_finder.evaluate_candidate"):
                final_result = self._evaluate_candidate(
                    signature=final_signature,
                    schedule=final_schedule,
                    flights_by_flow=flights_by_flow,
                    capacities_by_tv=capacities_by_tv,
                    flight_list=self._base_flight_list,
                    context=context,
                    baseline_obj=baseline_obj,
                )
            eval_calls += 1

        best_delta = final_result.delta_j
        best_objective = final_result.objective
        final_components = final_result.components
        final_artifacts = final_result.artifacts

        elapsed = time.perf_counter() - start_ts
        # Derive simple spill metrics from final artifacts
        try:
            T_final = int(self._indexer.num_time_bins)
            nmap_final = final_artifacts.get("n", {}) if isinstance(final_artifacts, dict) else {}
            final_spill_T = int(sum(int(np.asarray(v)[T_final]) for v in nmap_final.values())) if nmap_final else 0
        except Exception:
            final_spill_T = None
        try:
            T_final = int(self._indexer.num_time_bins)
            nmap_final = final_artifacts.get("n", {}) if isinstance(final_artifacts, dict) else {}
            final_inwin_total = int(sum(int(np.asarray(v)[:T_final].sum()) for v in nmap_final.values())) if nmap_final else 0
        except Exception:
            final_inwin_total = None

        # Ensure delay maps are defined before using them in diagnostics
        baseline_delays = (
            dict(baseline_artifacts.get("delays_min", {}))
            if isinstance(baseline_artifacts, dict)
            else {}
        )
        final_delays = (
            dict(final_artifacts.get("delays_min", {}))
            if isinstance(final_artifacts, dict)
            else {}
        )

        diagnostics = {
            "mode": mode,
            "control_volume_id": control_volume_id,
            "window_bins": [int(window_bins[0]), int(window_bins[1])],
            "passes_ran": len(pass_improvements),
            "pass_improvements": pass_improvements,
            "eval_calls": eval_calls,
            "cache_hits": cache_hits,
            "baseline_objective": baseline_obj,
            "baseline_components": baseline_components,
            "final_objective": best_objective,
            "final_components": final_components,
            "delta_j": best_delta,
            "rate_grid": list(rate_grid),
            "entrants_by_flow": {fid: len(entrants.get(fid, [])) for fid in flow_ids},
            "per_flow_history": history_out,
            "timing_seconds": elapsed,
            "stopped_early": stopped_early,
            "fast_scorer_enabled": bool(getattr(self.config, "fast_scorer_enabled", True)),
            "final_spill_T": final_spill_T,
            "final_in_window_releases": final_inwin_total,
            # Small delay summaries
            "final_nonzero_delay_count": int(sum(1 for v in final_delays.values() if int(v) > 0)) if isinstance(final_delays, dict) else None,
            "final_max_delay_min": int(max((int(v) for v in final_delays.values()), default=0)) if isinstance(final_delays, dict) else None,
        }
        diagnostics["baseline_delays_size"] = len(baseline_delays)
        diagnostics["final_delays_min"] = final_delays
        diagnostics["aggregate_delays_size"] = len(final_delays)
        diagnostics["baseline_delays_min"] = baseline_delays

        return rates_out, best_delta, diagnostics

    def _context_key(
        self,
        *,
        plan_key: str,
        control_volume_id: str,
        window_bins: Tuple[int, int],
        flow_ids: Sequence[str],
    ) -> Tuple:
        return (
            str(plan_key),
            str(control_volume_id),
            tuple(int(b) for b in window_bins),
            tuple(str(fid) for fid in flow_ids),
        )

    def _ensure_context_and_baseline(
        self,
        *,
        context_key: Tuple,
        flights_by_flow: Mapping[str, Sequence[Any]],
        capacities_by_tv: Mapping[str, np.ndarray],
        target_cells: Iterable[Tuple[str, int]],
        weights: ObjectiveWeights,
        tv_filter: Sequence[str],
        plan_key: Optional[str] = None,
        control_volume_id: Optional[str] = None,
    ) -> Tuple[ScoreContext, _BaselineResult]:
        context = self._score_context_cache.get(context_key)
        target_set = set(target_cells)
        target_arg = target_set if target_set else None

        if context is None:
            with self._timed("rate_finder.build_score_context"):
                context = build_score_context(
                    flights_by_flow,
                    indexer=self._indexer,
                    capacities_by_tv=capacities_by_tv,
                    target_cells=target_arg,
                    ripple_cells=None,
                    flight_list=self._base_flight_list,
                    weights=weights,
                    tv_filter=tv_filter,
                )
                # Inject/reuse base occupancy arrays across flow subsets when possible
                try:
                    T = int(self._indexer.num_time_bins)
                    tv = str(control_volume_id or (tv_filter[0] if tv_filter else next(iter(capacities_by_tv.keys()))))
                    pk = str(plan_key) if plan_key is not None else str(context_key[0])
                    occ_key = (pk, tv)
                    cached_bases = self._base_occ_cache.get(occ_key)
                    if cached_bases is not None:
                        base_all, base_zero = cached_bases
                        if base_all.size == T and base_zero.size == T:
                            context.base_occ_all_by_tv[tv] = np.asarray(base_all, dtype=np.int64)
                            context.base_occ_sched_zero_by_tv[tv] = np.asarray(base_zero, dtype=np.int64)
                        # refresh LRU position
                        self._base_occ_cache.move_to_end(occ_key)
                    else:
                        # Store for reuse if available
                        ba = np.asarray(context.base_occ_all_by_tv.get(tv, np.zeros(T, dtype=np.int64)), dtype=np.int64)
                        bz = np.asarray(context.base_occ_sched_zero_by_tv.get(tv, np.zeros(T, dtype=np.int64)), dtype=np.int64)
                        self._base_occ_cache[occ_key] = (ba, bz)
                        self._base_occ_cache.move_to_end(occ_key)
                        self._trim_lru_cache(self._base_occ_cache)
                except Exception:
                    pass
            self._score_context_cache[context_key] = context

        baseline = self._baseline_cache.get(context_key)
        if baseline is None:
            with self._timed("rate_finder.score_with_context.baseline"):
                objective, components, artifacts = score_with_context(
                    context.d_by_flow,
                    flights_by_flow=flights_by_flow,
                    capacities_by_tv=capacities_by_tv,
                    flight_list=self._base_flight_list,
                    context=context,
                )
            baseline = _BaselineResult(
                objective=float(objective),
                components=components,
                artifacts=artifacts,
            )
            self._baseline_cache[context_key] = baseline

        return context, baseline

    def _build_flights_by_flow(
        self,
        *,
        control_volume_id: str,
        active_windows: Sequence[int],
        flow_map: Mapping[str, Sequence[str]],
        entrants: Mapping[str, Sequence[Tuple[str, object, int]]],
        mode: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        flights_by_flow: Dict[str, List[Dict[str, Any]]] = {}
        if mode == "per_flow":
            for flow_id in sorted(flow_map.keys()):
                entries = sorted(entrants.get(flow_id, []), key=lambda item: int(item[2]))
                specs: List[Dict[str, Any]] = []
                for fid, entry_dt, time_idx in entries:
                    spec: Dict[str, Any] = {
                        "flight_id": str(fid),
                        "requested_bin": int(time_idx),
                    }
                    if isinstance(entry_dt, datetime):
                        spec["requested_dt"] = entry_dt
                    specs.append(spec)
                flights_by_flow[flow_id] = specs
        else:
            synthetic_id = "__blanket__"
            union_entries: List[Tuple[str, object, int]] = []
            for flow_id in flow_map.keys():
                union_entries.extend(entrants.get(flow_id, []))
            union_entries.sort(key=lambda item: int(item[2]))
            specs: List[Dict[str, Any]] = []
            for fid, entry_dt, time_idx in union_entries:
                spec: Dict[str, Any] = {
                    "flight_id": str(fid),
                    "requested_bin": int(time_idx),
                }
                if isinstance(entry_dt, datetime):
                    spec["requested_dt"] = entry_dt
                specs.append(spec)
            flights_by_flow[synthetic_id] = specs
        return flights_by_flow

    def _build_capacities_for_tv(self, control_volume_id: str) -> Dict[str, np.ndarray]:
        T = int(self._indexer.num_time_bins)
        capacities = np.zeros(T, dtype=np.int64)
        bin_minutes = max(1, int(getattr(self._indexer, "time_bin_minutes", 60)))
        bins_per_hour = max(1, int(round(60.0 / float(bin_minutes))))
        hourly_map = self.evaluator.hourly_capacity_by_tv.get(str(control_volume_id)) or {}

        for hour_key, value in hourly_map.items():
            try:
                hour_idx = int(hour_key)
                cap_val = int(round(float(value)))
            except Exception:
                continue
            if cap_val < 0:
                cap_val = 0
            start_bin = hour_idx * bins_per_hour
            end_bin = min(start_bin + bins_per_hour, T)
            if start_bin >= T:
                continue
            capacities[start_bin:end_bin] = cap_val

        return {str(control_volume_id): capacities}

    def _build_schedule_from_rates(
        self,
        *,
        rates_map: Mapping[str, float],
        context: ScoreContext,
        active_bins: Sequence[int],
        bin_minutes: int,
    ) -> Dict[str, Sequence[int]]:
        T = int(context.indexer.num_time_bins)
        active_set = {int(b) for b in active_bins if 0 <= int(b) < T}
        schedules: Dict[str, List[int]] = {}

        for flow_id, demand_arr in context.d_by_flow.items():
            demand_vec = np.asarray(demand_arr, dtype=np.int64)
            demand_list = demand_vec.tolist()
            if len(demand_list) < T + 1:
                demand_list.extend([0] * ((T + 1) - len(demand_list)))
            schedule = [int(x) for x in demand_list[: T + 1]]

            rate = float(rates_map.get(flow_id, math.inf))
            if not active_set or math.isinf(rate) or rate <= 0:
                schedules[flow_id] = schedule
                continue

            quota = max(0, int(round(rate * bin_minutes / 60.0)))
            ready = 0
            released = 0
            for t in sorted(active_set):
                if t < 0 or t >= T:
                    continue
                ready += demand_list[t]
                available = max(0, ready - released)
                take = min(quota, available) if quota > 0 else 0
                schedule[t] = take
                released += take

            total_non_overflow = sum(demand_list[:T])
            overflow_base = demand_list[T] if len(demand_list) > T else 0
            scheduled_non_overflow = sum(schedule[:T])
            schedule[T] = overflow_base + max(0, total_non_overflow - scheduled_non_overflow)
            schedules[flow_id] = schedule

        return schedules

    def _evaluate_candidate(
        self,
        *,
        signature: Tuple,
        schedule: Mapping[str, Sequence[int]],
        flights_by_flow: Mapping[str, Sequence[Any]],
        capacities_by_tv: Mapping[str, np.ndarray],
        flight_list: FlightList,
        context: ScoreContext,
        baseline_obj: float,
    ) -> _CandidateResult:
        # Fast scorer toggle via config and env var (RATE_FINDER_FAST_SCORER=0 disables)
        use_fast = bool(getattr(self.config, "fast_scorer_enabled", True))
        try:
            import os
            env_flag = os.getenv("RATE_FINDER_FAST_SCORER")
            if env_flag is not None and str(env_flag).strip().lower() in {"0", "false", "no"}:
                use_fast = False
        except Exception:
            pass

        if use_fast:
            # Build occ_by_tv for the single control TV using base caches + schedule sum
            T = int(self._indexer.num_time_bins)
            sched_sum = np.zeros(T, dtype=np.int64)
            for f, arr in schedule.items():
                a = np.asarray(arr, dtype=np.int64)
                if a.size > 0:
                    take = a[:T] if a.size >= T else np.pad(a, (0, T - a.size), mode="constant")[:T]
                    sched_sum += take.astype(np.int64, copy=False)
            # Assume single TV in capacities/context
            tv_id = next(iter(capacities_by_tv.keys())) if capacities_by_tv else next(iter(context.tvs_of_interest))
            base_all = np.asarray(context.base_occ_all_by_tv.get(str(tv_id), np.zeros(T, dtype=np.int64)), dtype=np.int64)
            base_zero = np.asarray(context.base_occ_sched_zero_by_tv.get(str(tv_id), np.zeros(T, dtype=np.int64)), dtype=np.int64)
            occ_tv = base_all - base_zero + sched_sum
            occ_by_tv = {str(tv_id): occ_tv.astype(np.int64, copy=False)}

            # Lightweight debug values: scheduled in-window vs overflow totals
            overflow_total = None
            inwin_total = None
            try:
                overflow_total = int(sum(int(np.asarray(v)[-1]) for v in schedule.values() if hasattr(v, "__len__")))
            except Exception:
                overflow_total = None
            try:
                inwin_total = int(sum(int(np.asarray(v)[:T].sum()) for v in schedule.values() if hasattr(v, "__len__")))
            except Exception:
                inwin_total = None
            with self._timed("rate_finder.score_with_context.candidate"):
                objective, components, artifacts = score_with_context_precomputed_occ(
                    schedule,
                    flights_by_flow=flights_by_flow,
                    capacities_by_tv=capacities_by_tv,
                    flight_list=flight_list,
                    context=context,
                    occ_by_tv=occ_by_tv,
                )
            # Attach fast scorer footprint into artifacts so callers can inspect
            try:
                enriched = dict(artifacts)
                enriched["fast_scorer_used"] = True
                enriched["fast_in_window_total"] = inwin_total
                enriched["fast_overflow_total"] = overflow_total
                artifacts = enriched
            except Exception:
                pass
        else:
            with self._timed("rate_finder.score_with_context.candidate"):
                objective, components, artifacts = score_with_context(
                    schedule,
                    flights_by_flow=flights_by_flow,
                    capacities_by_tv=capacities_by_tv,
                    flight_list=flight_list,
                    context=context,
                )
        delta_j = float(objective) - float(baseline_obj)
        result = _CandidateResult(
            delta_j=delta_j,
            objective=float(objective),
            components=components,
            artifacts=artifacts,
        )
        self._candidate_cache_store(signature, result)
        return result

    def _compute_entrants(
        self,
        control_volume_id: str,
        active_windows: Sequence[int],
        flow_map: Mapping[str, Sequence[str]],
    ) -> Dict[str, list]:
        # Restrict to selected flights only
        allowed_fids: Tuple[str, ...] = tuple(sorted({str(fid) for flights in flow_map.values() for fid in flights}))
        key = self._entrants_cache_key(control_volume_id, active_windows, allowed_fids)
        cached = self._entrants_cache_lookup(key)
        if cached is None:
            entries_by_flight: Dict[str, list[Tuple[Any, Any, int]]] = {}
            # Decode intervals arithmetically for allowed flights only
            fm = getattr(self._base_flight_list, "flight_metadata", {}) or {}
            indexer = self._indexer
            T = int(getattr(indexer, "num_time_bins", 0))
            tv_to_idx = getattr(indexer, "tv_id_to_idx", {}) or {}
            target_row = int(tv_to_idx.get(str(control_volume_id), -1))
            aw_set = {int(b) for b in active_windows}
            for fid in allowed_fids:
                meta = fm.get(str(fid)) or {}
                takeoff = meta.get("takeoff_time")
                if not isinstance(takeoff, datetime):
                    # If we don't have a datetime, we still collect the time_idx without entry_dt
                    takeoff = None
                for iv in (meta.get("occupancy_intervals") or []):
                    try:
                        tvtw_idx = int(iv.get("tvtw_index"))
                    except Exception:
                        continue
                    if T <= 0:
                        continue
                    row = tvtw_idx // T
                    bin_idx = int(tvtw_idx - row * T)
                    if row != target_row or bin_idx not in aw_set:
                        continue
                    entry_s = iv.get("entry_time_s", 0)
                    try:
                        entry_s = float(entry_s)
                    except Exception:
                        entry_s = 0.0
                    entry_dt = (takeoff + timedelta(seconds=entry_s)) if isinstance(takeoff, datetime) else None
                    fid_key = str(fid)
                    bucket = entries_by_flight.setdefault(fid_key, [])
                    bucket.append((fid, entry_dt, int(bin_idx)))
            cached = {
                fid_key: tuple(values)
                for fid_key, values in entries_by_flight.items()
                if values
            }
            self._entrants_cache_store(key, cached)

        entrants: Dict[str, list] = {fid: [] for fid in flow_map}
        for flow_id, flights in flow_map.items():
            records = entrants.setdefault(flow_id, [])
            for fid in flights:
                fid_key = str(fid)
                entries = cached.get(fid_key) if cached else None
                if not entries:
                    continue
                records.extend(entries)
        return entrants

    def _resolve_rate_grid(
        self,
        *,
        control_volume_id: str,
        window_bins: Tuple[int, int],
        flow_map: Mapping[str, Sequence[str]],
        mode: str,
        active_windows: Sequence[int],
        entrants: Optional[Mapping[str, Sequence[Tuple[Any, Any, int]]]] = None,
    ) -> Tuple[float, ...]:
        base_grid: Tuple[float, ...] = tuple(self.config.rate_grid) if self.config.rate_grid else tuple()
        if not self.config.use_adaptive_grid:
            return base_grid or (math.inf,)

        cache_key = self._rate_grid_cache_key(
            control_volume_id=control_volume_id,
            window_bins=window_bins,
            flow_map=flow_map,
            mode=mode,
            active_windows=active_windows,
            base_grid=base_grid,
        )
        cached = self._rate_grid_cache_lookup(cache_key)
        if cached is not None:
            return cached

        adaptive = self._build_adaptive_rate_grid(
            control_volume_id=control_volume_id,
            window_bins=window_bins,
            flow_map=flow_map,
            active_windows=active_windows,
            mode=mode,
            entrants=entrants,
        )

        # Merge explicit grid values with adaptive suggestions for reproducibility.
        combined: list[float] = []
        seen: set[float] = set()

        def _append(value: float) -> None:
            if math.isnan(value):
                return
            if value in seen:
                return
            seen.add(value)
            combined.append(value)

        for item in adaptive:
            _append(float(item))
        for item in base_grid:
            _append(float(item))

        if not combined:
            return (math.inf,)

        # Ensure Infinity is first while preserving the remainder order.
        finite = [v for v in combined if not math.isinf(v)]
        finite_sorted = sorted(set(finite), reverse=True)
        result = [math.inf]
        result.extend(finite_sorted)
        resolved = tuple(result)
        self._rate_grid_cache_store(cache_key, resolved)
        return resolved

    def _build_adaptive_rate_grid(
        self,
        *,
        control_volume_id: str,
        window_bins: Tuple[int, int],
        flow_map: Mapping[str, Sequence[str]],
        active_windows: Sequence[int],
        mode: str,
        entrants: Optional[Mapping[str, Sequence[Tuple[Any, Any, int]]]] = None,
    ) -> Tuple[float, ...]:
        _timer_cm = self._timed("rate_finder.build_adaptive_rate_grid")
        _timer_exit = _timer_cm.__exit__
        _timer_cm.__enter__()
        try:
            if not flow_map:
                return (math.inf,)

            window_start = int(window_bins[0])
            window_end = int(window_bins[1])
            if window_end <= window_start:
                window_end = window_start + 1
            active_bins = list(range(window_start, window_end))
            if not active_bins:
                active_bins = [window_start]

            bin_minutes = max(1, int(getattr(self._indexer, "time_bin_minutes", 60)))
            window_length_bins = max(1, window_end - window_start)
            window_hours = max(window_length_bins * bin_minutes / 60.0, 1e-3)
            bins_per_hour = max(1, int(round(60.0 / float(bin_minutes))))

            entrants_map = entrants
            if entrants_map is None:
                entrants_map = self._compute_entrants(control_volume_id, active_bins, flow_map)

            flow_stats: Dict[str, Dict[str, float]] = {}
            total_counts: Counter[int] = Counter()
            total_entrants = 0

            for flow_id, flights in flow_map.items():
                entries = entrants_map.get(flow_id, []) if entrants_map else []
                bin_counter: Counter[int] = Counter()
                for _, _, time_idx in entries:
                    idx = int(time_idx)
                    if idx < window_start or idx >= window_end:
                        continue
                    bin_counter[idx] += 1
                total_counts.update(bin_counter)
                num_entrants = sum(bin_counter.values())
                total_entrants += num_entrants
                peak = self._max_rolling_count(bin_counter, active_bins, bins_per_hour)
                clear_rate = math.ceil(num_entrants / window_hours) if num_entrants else 0
                flow_stats[str(flow_id)] = {
                    "entrants": float(num_entrants),
                    "peak": float(peak),
                    "clear_rate": float(clear_rate),
                }

            if total_entrants <= 0:
                return (math.inf,)

            for flow_id, stats in flow_stats.items():
                stats["share"] = stats["entrants"] / float(total_entrants)

            total_peak = self._max_rolling_count(total_counts, active_bins, bins_per_hour)
            total_clear = math.ceil(total_entrants / window_hours)

            cap_map = self.evaluator.hourly_capacity_by_tv.get(str(control_volume_id)) or {}
            hour_indices = {int(bin_idx // bins_per_hour) for bin_idx in active_bins}
            caps: list[float] = []
            for hour in sorted(hour_indices):
                value = cap_map.get(hour)
                if isinstance(value, (int, float)) and value >= 0:
                    caps.append(float(value))
            cap_stat: float = 0.0
            if caps:
                try:
                    cap_stat = float(median(caps))
                except Exception:
                    cap_stat = float(caps[0])

            max_rate = max(1, int(self.config.max_adaptive_rate))
            candidate_ints: set[int] = set()

            def _push_rate(value: float) -> None:
                if value is None or math.isnan(value) or math.isinf(value):
                    return
                if value <= 0:
                    return
                normalized = max(1, min(max_rate, int(round(value))))
                candidate_ints.add(normalized)

            # Always consider global anchors.
            _push_rate(total_clear)
            _push_rate(total_peak)
            if cap_stat > 0:
                _push_rate(cap_stat)

            # Flow-aware anchors.
            multipliers = (0.5, 0.67, 0.8, 1.0, 1.25, 1.5)
            for flow_id, stats in flow_stats.items():
                entrants_count = stats["entrants"]
                if entrants_count <= 0:
                    continue
                _push_rate(stats["clear_rate"])
                _push_rate(stats["peak"])
                share = stats.get("share", 0.0)
                if cap_stat > 0 and share > 0:
                    base = cap_stat * share
                    for mult in multipliers:
                        _push_rate(base * mult)
                peak_base = stats["peak"]
                if peak_base > 0:
                    for mult in (0.67, 1.0, 1.5):
                        _push_rate(peak_base * mult)

            # Global multipliers around capacity and peak.
            for base in (cap_stat, float(total_peak), float(total_clear)):
                if base <= 0:
                    continue
                for mult in multipliers:
                    _push_rate(base * mult)

            if not candidate_ints:
                candidate_ints.update({1, 2, 3})

            max_candidates = max(1, int(self.config.max_adaptive_candidates))
            finite_limit = max(1, max_candidates - 1)

            ordered = sorted(candidate_ints, reverse=True)
            if len(ordered) > finite_limit:
                # Prefer to keep extremes and a balanced middle.
                selected: list[int] = []
                selected.append(ordered[0])
                if ordered[-1] not in selected:
                    selected.append(ordered[-1])

                remaining_slots = finite_limit - len(selected)
                if remaining_slots > 0:
                    step = max(1, len(ordered) // remaining_slots)
                    idx = step // 2
                    while len(selected) < finite_limit and idx < len(ordered):
                        candidate = ordered[idx]
                        if candidate not in selected:
                            selected.append(candidate)
                        idx += step
                ordered = sorted(set(selected), reverse=True)

            rates_out: list[float] = [math.inf]
            for value in ordered[:finite_limit]:
                if value not in rates_out:
                    rates_out.append(float(value))
            return tuple(rates_out)
        finally:
            _timer_exit(None, None, None)

    @staticmethod
    def _max_rolling_count(counter: Counter[int], bins: Sequence[int], window: int) -> int:
        if not counter:
            return 0
        if window <= 1:
            return max(counter.get(b, 0) for b in bins) if bins else 0
        if not bins:
            keys = list(counter.keys())
            bins = keys
        start = min(bins)
        end = max(bins) + window
        length = max(0, end - start)
        if length <= 0:
            return 0
        series = [0] * length
        for idx, count in counter.items():
            offset = int(idx) - start
            if 0 <= offset < length:
                series[offset] += int(count)
        window = min(window, len(series))
        if window <= 0:
            return 0
        current = sum(series[:window])
        best = current
        for i in range(1, len(series) - window + 1):
            current = current - series[i - 1] + series[i + window - 1]
            if current > best:
                best = current
        return best

    def _entrants_cache_key(
        self, control_volume_id: str, active_windows: Sequence[int], allowed_fids: Sequence[str]
    ) -> Tuple[str, Tuple[int, ...], Tuple[str, ...]]:
        windows_key = tuple(sorted(int(b) for b in active_windows))
        allowed_key = tuple(sorted(str(x) for x in allowed_fids))
        return (str(control_volume_id), windows_key, allowed_key)

    def _entrants_cache_lookup(
        self, key: Tuple[str, Tuple[int, ...], Tuple[str, ...]]
    ) -> Optional[Dict[str, Tuple[Tuple[Any, Any, int], ...]]]:
        cached = self._entrants_cache.get(key)
        if cached is not None:
            self._entrants_cache.move_to_end(key)
        return cached

    def _entrants_cache_store(
        self,
        key: Tuple[str, Tuple[int, ...], Tuple[str, ...]],
        value: Dict[str, Tuple[Tuple[Any, Any, int], ...]],
    ) -> None:
        self._entrants_cache[key] = value
        self._entrants_cache.move_to_end(key)
        self._trim_lru_cache(self._entrants_cache)

    def _rate_grid_cache_key(
        self,
        *,
        control_volume_id: str,
        window_bins: Tuple[int, int],
        flow_map: Mapping[str, Sequence[str]],
        mode: str,
        active_windows: Sequence[int],
        base_grid: Tuple[float, ...],
    ) -> Tuple:
        windows_key = tuple(sorted(int(b) for b in active_windows))
        flow_signature = self._flow_map_signature(flow_map)
        return (
            str(control_volume_id),
            tuple(int(b) for b in window_bins),
            windows_key,
            flow_signature,
            str(mode),
            tuple(float(x) for x in base_grid),
            int(self.config.max_adaptive_rate),
            int(self.config.max_adaptive_candidates),
        )

    def _rate_grid_cache_lookup(self, key: Tuple) -> Optional[Tuple[float, ...]]:
        cached = self._rate_grid_cache.get(key)
        if cached is not None:
            self._rate_grid_cache.move_to_end(key)
        return cached

    def _rate_grid_cache_store(self, key: Tuple, value: Tuple[float, ...]) -> None:
        self._rate_grid_cache[key] = value
        self._rate_grid_cache.move_to_end(key)
        self._trim_lru_cache(self._rate_grid_cache)

    def _flow_map_signature(
        self, flow_map: Mapping[str, Sequence[str]]
    ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        normalized: list[Tuple[str, Tuple[str, ...]]] = []
        for flow_id in sorted(flow_map.keys()):
            flights = flow_map[flow_id]
            normalized.append(
                (
                    str(flow_id),
                    tuple(sorted(str(f) for f in flights)),
                )
            )
        return tuple(normalized)

    def _trim_lru_cache(self, cache: "OrderedDict[Tuple, Any]") -> None:
        max_size = max(1, int(self.config.cache_size))
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _candidate_signature(
        self,
        plan_key: str,
        control_volume_id: str,
        window_bins: Tuple[int, int],
        flow_ids: Sequence[str],
        mode: str,
        rates_tuple: Sequence[float],
    ) -> Tuple:
        return (
            plan_key,
            str(control_volume_id),
            tuple(int(b) for b in window_bins),
            tuple(flow_ids),
            str(mode),
            tuple(float(x) for x in rates_tuple),
        )

    def _candidate_cache_lookup(self, signature: Tuple) -> Optional[_CandidateResult]:
        result = self._candidate_cache.get(signature)
        if result is not None:
            self._candidate_cache.move_to_end(signature)
        return result

    def _candidate_cache_store(self, signature: Tuple, result: _CandidateResult) -> None:
        self._candidate_cache[signature] = result
        self._candidate_cache.move_to_end(signature)
        while len(self._candidate_cache) > int(self.config.cache_size):
            self._candidate_cache.popitem(last=False)

    def _as_rate_tuple(
        self,
        rates: Mapping[str, float],
        flow_ids: Sequence[str],
        *,
        override: Optional[Tuple[str, float]] = None,
    ) -> Tuple[float, ...]:
        override_id, override_val = override if override else (None, None)
        out = []
        for fid in flow_ids:
            if override_id is not None and fid == override_id:
                out.append(float(override_val))
            else:
                out.append(float(rates.get(fid, math.inf)))
        return tuple(out)



__all__ = ["RateFinder", "RateFinderConfig"]
