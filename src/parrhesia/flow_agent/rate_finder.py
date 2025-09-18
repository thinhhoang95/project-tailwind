from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from project_tailwind.optimize.eval.delta_flight_list import DeltaFlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

from parrhesia.fcfs.scheduler import assign_delays
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


@dataclass
class _CandidateResult:
    delta_j: float
    objective: float
    components: Dict[str, float]


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
        self._baseline_cache: Dict[str, Tuple[float, Dict[str, float]]] = {}
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

        active_windows = list(range(int(window_bins[0]), int(window_bins[1])))
        plan_key = plan_state.canonical_key()
        baseline_obj, baseline_components = self._ensure_baseline(plan_key, num_regs=len(plan_state.plan))

        entrants = self._compute_entrants(control_volume_id, active_windows, flow_map)
        flow_order = sorted(flow_ids, key=lambda f: (-len(entrants.get(f, [])), f))

        eval_calls = 0
        cache_hits = 0
        best_delta = 0.0
        best_objective = baseline_obj
        pass_improvements: list[float] = []
        elapsed = 0.0
        stopped_early = False

        start_ts = time.perf_counter()

        if mode == "per_flow":
            best_rates: Dict[str, float] = {fid: math.inf for fid in flow_ids}
            per_flow_delays: Dict[str, Dict[str, int]] = {fid: {} for fid in flow_ids}
            history: Dict[str, Dict[str, float]] = {fid: {} for fid in flow_ids}

            for _ in range(int(self.config.passes)):
                prev_delta = best_delta
                for flow_id in flow_order:
                    if stopped_early:
                        break
                    current_best_delta = best_delta
                    current_best_rate = best_rates[flow_id]
                    current_best_delays = per_flow_delays.get(flow_id, {})

                    for rate in self.config.rate_grid:
                        rate_val = float(rate)
                        rates_tuple = self._as_rate_tuple(best_rates, flow_ids, override=(flow_id, rate_val))
                        signature = self._candidate_signature(
                            plan_key,
                            control_volume_id,
                            window_bins,
                            flow_ids,
                            mode,
                            rates_tuple,
                        )
                        aggregate, candidate_delays = self._build_candidate_per_flow(
                            control_volume_id=control_volume_id,
                            active_windows=active_windows,
                            flow_id=flow_id,
                            rate=rate_val,
                            flow_map=flow_map,
                            current_per_flow=per_flow_delays,
                        )
                        result, cached = self._evaluate_candidate(
                            signature,
                            aggregate,
                            num_regs=len(plan_state.plan) + 1,
                            baseline_obj=baseline_obj,
                        )
                        if cached:
                            cache_hits += 1
                        else:
                            eval_calls += 1
                            if eval_calls >= self.config.max_eval_calls:
                                stopped_early = True
                        history[flow_id][str(rate_val)] = result.delta_j

                        if result.delta_j < current_best_delta:
                            current_best_delta = result.delta_j
                            current_best_rate = rate_val
                            current_best_delays = candidate_delays
                            best_objective = baseline_obj + result.delta_j
                            per_flow_delays[flow_id] = candidate_delays
                            best_rates[flow_id] = rate_val
                            best_delta = result.delta_j
                    # ensure selected flow snapshot sticks even without improvement
                    per_flow_delays[flow_id] = current_best_delays

                pass_improvements.append(prev_delta - best_delta)
                tol = self.config.epsilon * max(1.0, abs(baseline_obj))
                if pass_improvements[-1] <= tol or stopped_early:
                    break

            rates_out: Union[int, Dict[str, float]] = {fid: best_rates[fid] for fid in flow_ids}
            history_out = history
            aggregate_final = self._aggregate_delays(per_flow_delays)
        else:
            union_flights: Tuple[str, ...] = tuple(sorted({f for flights in flow_map.values() for f in flights}))
            best_rate = math.inf
            aggregate_final: Dict[str, int] = {}
            history_out = {"__blanket__": {}}

            for _ in range(int(self.config.passes)):
                prev_delta = best_delta
                for rate in self.config.rate_grid:
                    rate_val = float(rate)
                    signature = self._candidate_signature(
                        plan_key,
                        control_volume_id,
                        window_bins,
                        flow_ids,
                        mode,
                        (rate_val,),
                    )
                    aggregate = self._build_candidate_blanket(
                        control_volume_id=control_volume_id,
                        active_windows=active_windows,
                        union_flights=union_flights,
                        rate=rate_val,
                    )
                    result, cached = self._evaluate_candidate(
                        signature,
                        aggregate,
                        num_regs=len(plan_state.plan) + 1,
                        baseline_obj=baseline_obj,
                    )
                    if cached:
                        cache_hits += 1
                    else:
                        eval_calls += 1
                        if eval_calls >= self.config.max_eval_calls:
                            stopped_early = True
                    history_out.setdefault("__blanket__", {})[str(rate_val)] = result.delta_j
                    if result.delta_j < best_delta:
                        best_delta = result.delta_j
                        best_rate = rate_val
                        aggregate_final = aggregate
                        best_objective = baseline_obj + result.delta_j
                pass_improvements.append(prev_delta - best_delta)
                tol = self.config.epsilon * max(1.0, abs(baseline_obj))
                if pass_improvements[-1] <= tol or stopped_early:
                    break

            rates_out = float(best_rate)

        elapsed = time.perf_counter() - start_ts
        diagnostics = {
            "mode": mode,
            "control_volume_id": control_volume_id,
            "window_bins": list(window_bins),
            "passes_ran": len(pass_improvements),
            "pass_improvements": pass_improvements,
            "eval_calls": eval_calls,
            "cache_hits": cache_hits,
            "baseline_objective": baseline_obj,
            "baseline_components": baseline_components,
            "final_objective": best_objective,
            "delta_j": best_delta,
            "rate_grid": list(self.config.rate_grid),
            "entrants_by_flow": {fid: len(entrants.get(fid, [])) for fid in flow_ids},
            "per_flow_history": history_out,
            "timing_seconds": elapsed,
            "stopped_early": stopped_early,
        }
        diagnostics["aggregate_delays_size"] = len(aggregate_final)
        return rates_out, best_delta, diagnostics

    # ------------------------------------------------------------------
    def _evaluate_candidate(
        self,
        signature: Tuple,
        delays_by_flight: Mapping[str, int],
        *,
        num_regs: int,
        baseline_obj: float,
    ) -> Tuple[_CandidateResult, bool]:
        cached = self._candidate_cache_lookup(signature)
        if cached is not None:
            return cached, True
        delta_view = DeltaFlightList(self._base_flight_list, dict(delays_by_flight))
        try:
            self.evaluator.update_flight_list(delta_view)
            excess = self.evaluator.compute_excess_traffic_vector()
            delay_stats = self.evaluator.compute_delay_stats()
            objective, components = self._compute_objective(
                excess_vector=excess,
                delay_stats=delay_stats,
                num_regs=num_regs,
            )
        finally:
            self.evaluator.update_flight_list(self._base_flight_list)
        delta_j = float(objective) - float(baseline_obj)
        result = _CandidateResult(delta_j=delta_j, objective=float(objective), components=components)
        self._candidate_cache_store(signature, result)
        return result, False

    def _ensure_baseline(self, plan_key: str, *, num_regs: int) -> Tuple[float, Dict[str, float]]:
        cached = self._baseline_cache.get(plan_key)
        if cached is not None:
            return cached
        try:
            self.evaluator.update_flight_list(self._base_flight_list)
            excess = self.evaluator.compute_excess_traffic_vector()
            delay_stats = self.evaluator.compute_delay_stats()
            objective, components = self._compute_objective(
                excess_vector=excess,
                delay_stats=delay_stats,
                num_regs=num_regs,
            )
        finally:
            self.evaluator.update_flight_list(self._base_flight_list)
        payload = (float(objective), components)
        self._baseline_cache[plan_key] = payload
        return payload

    # ------------------------------------------------------------------
    def _build_candidate_per_flow(
        self,
        *,
        control_volume_id: str,
        active_windows: Sequence[int],
        flow_id: str,
        rate: float,
        flow_map: Mapping[str, Sequence[str]],
        current_per_flow: Mapping[str, Mapping[str, int]],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        if math.isinf(rate) or rate <= 0:
            candidate = {}
        else:
            candidate = assign_delays(
                flight_list=self._base_flight_list,
                identifier_list=[str(fid) for fid in flow_map.get(flow_id, ())],
                reference_location=control_volume_id,
                tvtw_indexer=self._indexer,
                hourly_rate=max(1, int(round(rate))),
                active_time_windows=[int(w) for w in active_windows],
            )
        merged: Dict[str, Dict[str, int]] = {k: dict(v) for k, v in current_per_flow.items()}
        merged[str(flow_id)] = dict(candidate)
        aggregate = self._aggregate_delays(merged)
        return aggregate, dict(candidate)

    def _build_candidate_blanket(
        self,
        *,
        control_volume_id: str,
        active_windows: Sequence[int],
        union_flights: Sequence[str],
        rate: float,
    ) -> Dict[str, int]:
        if math.isinf(rate) or rate <= 0:
            return {}
        return assign_delays(
            flight_list=self._base_flight_list,
            identifier_list=[str(fid) for fid in union_flights],
            reference_location=control_volume_id,
            tvtw_indexer=self._indexer,
            hourly_rate=max(1, int(round(rate))),
            active_time_windows=[int(w) for w in active_windows],
        )

    def _aggregate_delays(self, per_flow: Mapping[str, Mapping[str, int]]) -> Dict[str, int]:
        union: Dict[str, int] = {}
        for delays in per_flow.values():
            for fid, value in delays.items():
                if value <= 0:
                    continue
                prev = union.get(fid, 0)
                if value > prev:
                    union[fid] = int(value)
        return union

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

    def _compute_objective(
        self,
        *,
        excess_vector: np.ndarray,
        delay_stats: Mapping[str, float],
        num_regs: int,
    ) -> Tuple[float, Dict[str, float]]:
        weights = {
            "alpha": 1.0,
            "beta": 2.0,
            "gamma": 0.1,
            "delta": 25.0,
        }
        if self.config.objective_weights:
            weights.update(self.config.objective_weights)
        ev = np.asarray(excess_vector, dtype=float)
        z_sum = float(np.sum(ev)) if ev.size else 0.0
        z_max = float(np.max(ev)) if ev.size else 0.0
        delay_min = float(delay_stats.get("total_delay_seconds", 0.0)) / 60.0
        objective = (
            weights["alpha"] * z_sum
            + weights["beta"] * z_max
            + weights["gamma"] * delay_min
            + weights["delta"] * float(num_regs)
        )
        components = {
            "z_sum": z_sum,
            "z_max": z_max,
            "delay_min": delay_min,
            "num_regs": float(num_regs),
            "alpha": float(weights["alpha"]),
            "beta": float(weights["beta"]),
            "gamma": float(weights["gamma"]),
            "delta": float(weights["delta"]),
        }
        return float(objective), components


__all__ = ["RateFinder", "RateFinderConfig"]
