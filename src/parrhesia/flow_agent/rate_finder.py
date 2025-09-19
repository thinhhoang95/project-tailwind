from __future__ import annotations

import math
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

from parrhesia.optim.objective import (
    ObjectiveWeights,
    ScoreContext,
    build_score_context,
    score_with_context,
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
    max_adaptive_candidates: int = 12


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
    ) -> None:
        self.evaluator = evaluator
        self._base_flight_list = flight_list
        self._indexer = indexer
        self.config = config or RateFinderConfig()
        self._score_context_cache: Dict[Tuple, ScoreContext] = {}
        self._baseline_cache: Dict[Tuple, _BaselineResult] = {}
        self._candidate_cache: "OrderedDict[Tuple, _CandidateResult]" = OrderedDict()

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

        rate_grid = self._resolve_rate_grid(
            control_volume_id=control_volume_id,
            window_bins=window_bins,
            flow_map=flow_map,
            mode=mode,
            active_windows=active_windows,
        )

        if mode == "per_flow":
            for flow_id in flow_ids:
                print(f"[RateFinder] candidate rates for flow {flow_id}: {tuple(rate_grid)}")
        else:
            print(f"[RateFinder] candidate rates (blanket mode): {tuple(rate_grid)}")

        entrants = self._compute_entrants(control_volume_id, active_windows, flow_map)
        flow_order = sorted(flow_ids, key=lambda f: (-len(entrants.get(f, [])), f))

        context_flow_ids = flow_ids if mode == "per_flow" else ["__blanket__"]
        context_key = self._context_key(
            plan_key=str(plan_key),
            control_volume_id=str(control_volume_id),
            window_bins=tuple(int(b) for b in window_bins),
            flow_ids=context_flow_ids,
        )

        flights_by_flow = self._build_flights_by_flow(
            control_volume_id=control_volume_id,
            active_windows=active_windows,
            flow_map=flow_map,
            entrants=entrants,
            mode=mode,
        )
        capacities_by_tv = self._build_capacities_for_tv(control_volume_id)

        target_cells = {(str(control_volume_id), int(t)) for t in active_windows}
        weights_cfg = self.config.objective_weights or {}
        weights = ObjectiveWeights(**weights_cfg) if weights_cfg else ObjectiveWeights()

        context, baseline = self._ensure_context_and_baseline(
            context_key=context_key,
            flights_by_flow=flights_by_flow,
            capacities_by_tv=capacities_by_tv,
            target_cells=target_cells,
            weights=weights,
            tv_filter=[control_volume_id],
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

        if mode == "per_flow":
            best_rates: Dict[str, float] = {fid: math.inf for fid in context_flow_ids}
            history_out: Dict[str, Dict[str, float]] = {fid: {} for fid in context_flow_ids}

            for _ in range(int(self.config.passes)):
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
                            schedule = self._build_schedule_from_rates(
                                rates_map=candidate_rates,
                                context=context,
                                active_bins=active_bins_sorted,
                                bin_minutes=bin_minutes,
                            )
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
                            if eval_calls >= self.config.max_eval_calls:
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
            rates_out: Union[int, Dict[str, float]] = {fid: float(best_rates[fid]) for fid in context_flow_ids}
            final_rates_map = {fid: float(best_rates[fid]) for fid in context_flow_ids}
        else:
            history_out = {"__blanket__": {}}
            best_rate = math.inf
            for _ in range(int(self.config.passes)):
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
                        schedule = self._build_schedule_from_rates(
                            rates_map=candidate_rates,
                            context=context,
                            active_bins=active_bins_sorted,
                            bin_minutes=bin_minutes,
                        )
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
                        if eval_calls >= self.config.max_eval_calls:
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
            final_schedule = self._build_schedule_from_rates(
                rates_map=final_rates_map,
                context=context,
                active_bins=active_bins_sorted,
                bin_minutes=bin_minutes,
            )
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
        }
        baseline_delays = dict(baseline_artifacts.get("delays_min", {}))
        final_delays = dict(final_artifacts.get("delays_min", {}))
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
    ) -> Tuple[ScoreContext, _BaselineResult]:
        context = self._score_context_cache.get(context_key)
        target_set = set(target_cells)
        target_arg = target_set if target_set else None

        if context is None:
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
            self._score_context_cache[context_key] = context

        baseline = self._baseline_cache.get(context_key)
        if baseline is None:
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
        entrants: Dict[str, list] = {fid: [] for fid in flow_map}
        reverse: Dict[str, str] = {}
        for flow_id, flights in flow_map.items():
            for fid in flights:
                reverse[str(fid)] = flow_id
        iter_fn = getattr(self._base_flight_list, "iter_hotspot_crossings", None)
        if callable(iter_fn):
            for fid, tv_id, entry_dt, time_idx in iter_fn(
                [control_volume_id], active_windows={control_volume_id: active_windows}
            ):
                flow = reverse.get(str(fid))
                if flow is not None:
                    entrants.setdefault(flow, []).append((fid, entry_dt, int(time_idx)))
        return entrants

    def _resolve_rate_grid(
        self,
        *,
        control_volume_id: str,
        window_bins: Tuple[int, int],
        flow_map: Mapping[str, Sequence[str]],
        mode: str,
        active_windows: Sequence[int],
    ) -> Tuple[float, ...]:
        base_grid: Tuple[float, ...] = tuple(self.config.rate_grid) if self.config.rate_grid else tuple()
        if not self.config.use_adaptive_grid:
            return base_grid or (math.inf,)

        adaptive = self._build_adaptive_rate_grid(
            control_volume_id=control_volume_id,
            window_bins=window_bins,
            flow_map=flow_map,
            active_windows=active_windows,
            mode=mode,
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
        return tuple(result)

    def _build_adaptive_rate_grid(
        self,
        *,
        control_volume_id: str,
        window_bins: Tuple[int, int],
        flow_map: Mapping[str, Sequence[str]],
        active_windows: Sequence[int],
        mode: str,
    ) -> Tuple[float, ...]:
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

        entrants = self._compute_entrants(control_volume_id, active_bins, flow_map)

        flow_stats: Dict[str, Dict[str, float]] = {}
        total_counts: Counter[int] = Counter()
        total_entrants = 0

        for flow_id, flights in flow_map.items():
            entries = entrants.get(flow_id, [])
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
