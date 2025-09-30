"""Prediction utilities leveraging the safe-spill objective."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from parrhesia.flow_agent.safespill_objective import score_with_context
from parrhesia.optim.objective import ObjectiveWeights, ScoreContext, build_score_context

from .rates import distribute_hourly_rate_to_bins
from .types import RateCut, Window


def build_local_context(
    *,
    indexer,
    flight_list,
    capacities_by_tv: Mapping[str, np.ndarray],
    target_cells: Sequence[Tuple[str, int]],
    flights_by_flow: Mapping[int, Sequence[Mapping[str, Any]]],
    weights: Optional[ObjectiveWeights] = None,
    tv_filter: Optional[Iterable[str]] = None,
    ripple_cells: Optional[Iterable[Tuple[str, int]]] = None,
) -> ScoreContext:
    """Build a localized :class:`ScoreContext` for evaluation."""

    return build_score_context(
        flights_by_flow,
        indexer=indexer,
        capacities_by_tv=capacities_by_tv,
        target_cells=list(target_cells),
        ripple_cells=list(ripple_cells) if ripple_cells is not None else None,
        flight_list=flight_list,
        weights=weights,
        tv_filter=tv_filter,
    )


def baseline_schedule_from_context(context: ScoreContext) -> Dict[Any, np.ndarray]:
    """Extract baseline demand histograms from the context."""

    out: Dict[Any, np.ndarray] = {}
    for flow_id, arr in context.d_by_flow.items():
        out[flow_id] = np.asarray(arr, dtype=np.int64).copy()
    return out


def apply_regulation_to_schedule(
    baseline: Mapping[Any, np.ndarray],
    demand_by_flow: Mapping[int, np.ndarray],
    *,
    rates: Sequence[RateCut],
    indexer,
    window: Window,
) -> Dict[Any, np.ndarray]:
    """Apply rate limits to the baseline schedule inside the window."""

    regulated: Dict[Any, np.ndarray] = {fid: np.asarray(arr, dtype=np.int64).copy() for fid, arr in baseline.items()}
    bph = int(indexer.rolling_window_size())
    start = int(window.start_bin)
    end = int(window.end_bin)
    for rate in rates:
        flow_id = rate.flow_id
        baseline_arr = regulated.get(flow_id)
        if baseline_arr is None:
            continue
        demand_arr = np.asarray(demand_by_flow.get(flow_id, baseline_arr), dtype=np.int64)
        allowances = distribute_hourly_rate_to_bins(
            rate.allowed_rate_R,
            bins_per_hour=bph,
            start_bin=start,
            end_bin=end,
        )
        if allowances.size == 0:
            continue
        max_bin = baseline_arr.shape[0] - 1
        for offset, bin_idx in enumerate(range(start, end + 1)):
            if bin_idx >= max_bin:
                break
            allowed = allowances[offset] if offset < allowances.size else allowances[-1]
            demand_val = demand_arr[bin_idx] if bin_idx < demand_arr.shape[0] else 0
            baseline_arr[bin_idx] = min(int(demand_val), int(allowed))
        regulated[flow_id] = baseline_arr
    return regulated


def score_pair(
    baseline: Mapping[Any, np.ndarray],
    regulated: Mapping[Any, np.ndarray],
    *,
    flights_by_flow: Mapping[int, Sequence[Mapping[str, Any]]],
    capacities_by_tv: Mapping[str, np.ndarray],
    flight_list,
    context: ScoreContext,
) -> Tuple[
    float,
    float,
    Mapping[str, np.ndarray],
    Mapping[str, np.ndarray],
    Dict[str, float],
    Dict[str, float],
]:
    """Score before and after schedules with the safe-spill objective."""

    score_before, components_before, artifacts_before = score_with_context(
        baseline,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=capacities_by_tv,
        flight_list=flight_list,
        context=context,
        audit_exceedances=True,
    )
    score_after, components_after, artifacts_after = score_with_context(
        regulated,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=capacities_by_tv,
        flight_list=flight_list,
        context=context,
        audit_exceedances=True,
    )
    occ_before = artifacts_before.get("occupancy", {})
    occ_after = artifacts_after.get("occupancy", {})
    return (
        float(score_before),
        float(score_after),
        occ_before,
        occ_after,
        dict(components_before),
        dict(components_after),
    )


def compute_delta_deficit_per_hour(
    occ_before: Mapping[str, np.ndarray],
    occ_after: Mapping[str, np.ndarray],
    *,
    capacities_by_tv: Mapping[str, np.ndarray],
    target_cells: Sequence[Tuple[str, int]],
    indexer,
    window: Window,
) -> float:
    """Compute deficit reduction averaged over the window."""

    unique_tvs = sorted({str(tv) for tv, _ in target_cells})
    start = int(window.start_bin)
    end = int(window.end_bin)
    if end < start:
        return 0.0
    bins_per_hour = int(indexer.rolling_window_size())
    total_delta = 0.0
    count = 0
    for tv in unique_tvs:
        occ_before_vec = np.asarray(occ_before.get(tv, np.zeros(int(indexer.num_time_bins), dtype=np.float64)))
        occ_after_vec = np.asarray(occ_after.get(tv, np.zeros(int(indexer.num_time_bins), dtype=np.float64)))
        cap_vec = np.asarray(capacities_by_tv.get(tv, np.zeros(int(indexer.num_time_bins), dtype=np.float64)))
        T = min(len(occ_before_vec), len(cap_vec))
        for t in range(start, min(end, T - 1) + 1):
            t_end = min(T, t + bins_per_hour)
            occ_before_sum = float(occ_before_vec[t:t_end].sum())
            occ_after_sum = float(occ_after_vec[t:t_end].sum())
            # Map the rolling-bin slice onto its parent hour and use a single
            # representative hourly capacity so we mirror the simulated annealing
            # objective, which subtracts one hourly capacity per window.
            hour_idx = int(t) // bins_per_hour
            h_start = hour_idx * bins_per_hour
            h_end = min(h_start + bins_per_hour, T)
            slice_caps = cap_vec[h_start:h_end]
            if slice_caps.size:
                hourly_cap = float(np.median(slice_caps))
            else:
                hourly_cap = float(cap_vec[t]) if t < len(cap_vec) else 0.0
            ex_before = max(0.0, occ_before_sum - hourly_cap)
            ex_after = max(0.0, occ_after_sum - hourly_cap)
            total_delta += ex_before - ex_after
            count += 1
    if count == 0:
        return 0.0
    return float(total_delta) / float(count)
