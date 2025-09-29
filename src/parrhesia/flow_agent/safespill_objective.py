from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from parrhesia.fcfs.flowful_safespill import assign_delays_flowful_preparsed
from parrhesia.optim.occupancy import compute_occupancy
from parrhesia.optim.objective import (
    ScoreContext,
    _to_len_T_plus_1_array,
    _compute_J_cap,
    _compute_J_reg_and_J_tv,
    _compute_J_share,
    _compute_J_spill,
)


def score_with_context(
    n_f_t: Mapping[Any, Union[Sequence[int], Mapping[int, int]]],
    *,
    flights_by_flow: Mapping[Any, Sequence[Any]],
    capacities_by_tv: Mapping[str, np.ndarray],
    flight_list: Optional[object],
    context: ScoreContext,
    audit_exceedances: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    indexer: TVTWIndexer = context.indexer
    T = int(indexer.num_time_bins)

    # Normalize n
    n_by_flow: Dict[Any, np.ndarray] = {f: _to_len_T_plus_1_array(arr, T) for f, arr in n_f_t.items()}

    # Delays using preprocessed sorted flights with safe spill
    delays_min, realised_start = assign_delays_flowful_preparsed(
        context.flights_sorted_by_flow,
        n_by_flow,
        indexer,
        spill_mode="dump_to_next_bin",
    )

    # Occupancy only for TVs of interest
    if flight_list is not None and context.base_occ_all_by_tv:
        try:
            import inspect  # type: ignore
            supports_filter = "flight_filter" in inspect.signature(compute_occupancy).parameters
        except Exception:
            supports_filter = False
        if supports_filter:
            occ_sched = compute_occupancy(
                flight_list,
                delays_min,
                indexer,
                tv_filter=context.tvs_of_interest,
                flight_filter=context.sched_fids,
            )
        else:
            fm = getattr(flight_list, "flight_metadata", {}) or {}
            sub_meta = {fid: fm.get(fid) for fid in context.sched_fids if fid in fm}
            _Sub = type("_SubFlights", (), {})
            sub_flight_list = _Sub()
            setattr(sub_flight_list, "flight_metadata", sub_meta)
            delays_sched = {fid: delays_min.get(fid, 0) for fid in context.sched_fids}
            occ_sched = compute_occupancy(
                sub_flight_list,
                delays_sched,
                indexer,
                tv_filter=context.tvs_of_interest,
            )
        occ_by_tv: Dict[str, np.ndarray] = {}
        T_int = int(indexer.num_time_bins)
        zeros = np.zeros(T_int, dtype=np.int64)
        for tv in context.tvs_of_interest:
            tvs = str(tv)
            base_all = context.base_occ_all_by_tv.get(tvs, zeros)
            base_sched_zero = context.base_occ_sched_zero_by_tv.get(tvs, zeros)
            sched_cur = occ_sched.get(tvs, zeros)
            occ_by_tv[tvs] = (
                base_all.astype(np.int64)
                + sched_cur.astype(np.int64)
                - base_sched_zero.astype(np.int64)
            )
    else:
        occ_by_tv = compute_occupancy(
            flight_list if flight_list is not None else type("_Dummy", (), {"flight_metadata": {}})(),
            delays_min,
            indexer,
            tv_filter=context.tvs_of_interest,
        )

    # J_cap (vectorized) using cached alpha
    K = int(indexer.rolling_window_size())
    J_cap = _compute_J_cap(
        occ_by_tv,
        capacities_by_tv,
        context.alpha_by_tv,
        K,
        audit_exceedances=audit_exceedances,
        indexer=indexer,
        target_cells=context.target_cells,
        ripple_cells=context.ripple_cells,
    )

    # J_delay, J_reg, J_tv
    total_delay_min = sum(int(v) for v in delays_min.values())
    J_delay = float(context.weights.lambda_delay) * float(total_delay_min)
    J_reg, J_tv = _compute_J_reg_and_J_tv(
        n_by_flow,
        context.d_by_flow,
        context.beta_gamma_by_flow,
        context.weights.beta_ctx,
        context.weights.gamma_ctx,
    )

    # Optional terms
    J_share = _compute_J_share(n_by_flow, context.d_by_flow, context.weights.theta_share) if context.weights.theta_share > 0 else 0.0
    J_spill = _compute_J_spill(n_by_flow, context.weights.eta_spill) if context.weights.eta_spill > 0 else 0.0

    J_total = 100.0 * J_cap + J_delay + J_reg + J_tv + J_share + J_spill

    components: Dict[str, float] = {
        "J_cap": float(J_cap),
        "J_delay": float(J_delay),
        "J_reg": float(J_reg),
        "J_tv": float(J_tv),
    }
    if context.weights.theta_share > 0:
        components["J_share"] = float(J_share)
    if context.weights.eta_spill > 0:
        components["J_spill"] = float(J_spill)

    artifacts: Dict[str, Any] = {
        "delays_min": delays_min,
        "realised_start": realised_start,
        "occupancy": occ_by_tv,
        "demand": context.d_by_flow,
        "n": n_by_flow,
        "beta_gamma": context.beta_gamma_by_flow,
        "alpha": context.alpha_by_tv,
    }

    return float(J_total), components, artifacts


def score_with_context_precomputed_occ(
    n_f_t: Mapping[Any, Union[Sequence[int], Mapping[int, int]]],
    *,
    flights_by_flow: Mapping[Any, Sequence[Any]],
    capacities_by_tv: Mapping[str, np.ndarray],
    flight_list: Optional[object],
    context: ScoreContext,
    occ_by_tv: Mapping[str, np.ndarray],
    audit_exceedances: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    indexer: TVTWIndexer = context.indexer
    T = int(indexer.num_time_bins)

    # Normalize n
    n_by_flow: Dict[Any, np.ndarray] = {f: _to_len_T_plus_1_array(arr, T) for f, arr in n_f_t.items()}

    # Delays using preprocessed sorted flights with safe spill
    delays_min, realised_start = assign_delays_flowful_preparsed(
        context.flights_sorted_by_flow,
        n_by_flow,
        indexer,
        spill_mode="dump_to_next_bin",
    )

    # Use provided occupancy for J_cap
    K = int(indexer.rolling_window_size())
    J_cap = _compute_J_cap(
        occ_by_tv,
        capacities_by_tv,
        context.alpha_by_tv,
        K,
        audit_exceedances=audit_exceedances,
        indexer=indexer,
        target_cells=context.target_cells,
        ripple_cells=context.ripple_cells,
    )

    # J_delay, J_reg, J_tv
    total_delay_min = sum(int(v) for v in delays_min.values())
    J_delay = float(context.weights.lambda_delay) * float(total_delay_min)
    J_reg, J_tv = _compute_J_reg_and_J_tv(
        n_by_flow,
        context.d_by_flow,
        context.beta_gamma_by_flow,
        context.weights.beta_ctx,
        context.weights.gamma_ctx,
    )

    # Optional terms
    J_share = _compute_J_share(n_by_flow, context.d_by_flow, context.weights.theta_share) if context.weights.theta_share > 0 else 0.0
    J_spill = _compute_J_spill(n_by_flow, context.weights.eta_spill) if context.weights.eta_spill > 0 else 0.0

    J_total = J_cap + J_delay + J_reg + J_tv + J_share + J_spill

    components: Dict[str, float] = {
        "J_cap": float(J_cap),
        "J_delay": float(J_delay),
        "J_reg": float(J_reg),
        "J_tv": float(J_tv),
    }
    if context.weights.theta_share > 0:
        components["J_share"] = float(J_share)
    if context.weights.eta_spill > 0:
        components["J_spill"] = float(J_spill)

    artifacts: Dict[str, Any] = {
        "delays_min": delays_min,
        "realised_start": realised_start,
        "occupancy": dict(occ_by_tv),
        "demand": context.d_by_flow,
        "n": n_by_flow,
        "beta_gamma": context.beta_gamma_by_flow,
        "alpha": context.alpha_by_tv,
    }

    return float(J_total), components, artifacts


__all__ = [
    "score_with_context",
    "score_with_context_precomputed_occ",
]

