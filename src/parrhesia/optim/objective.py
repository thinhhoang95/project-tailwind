"""
Objective evaluation for flowful scheduling across multiple hotspots.

This module computes the objective components described in the planning docs:

  - J_cap: weighted rolling-hour capacity exceedance across (v, t) cells
  - J_delay: sum of pushback delays (in minutes)
  - J_reg: per-flow rate deviation from baseline demand d_f(t)
  - J_tv: per-flow total variation (temporal smoothness)
  - J_share (optional): deviation of per-bin flow shares from demand shares
  - J_spill (optional): penalty on overflow releases n_f(T)

A top-level function `score(...)` orchestrates
  n_f_t  ->  FIFO scheduling  ->  occupancy/exceedance  ->  J-components.

Inputs are designed to align with the rest of the framework:
  - flights_by_flow: mapping flow_id -> sequence of flight specs
  - n_f_t: mapping flow_id -> sequence[int] (length T+1) or mapping[int,int]
  - indexer: TVTWIndexer with time and TVTW helpers
  - capacities_by_tv: tv_id -> array[T] of rolling-hour capacities per bin
  - target_cells, ripple_cells: sets of (tv_id, bin) pairs indicating the
    operator attention regions; remaining cells use context weights
  - flight_list: to compute occupancy after delays and to derive offsets for
    class mapping (f, t) via median offsets toward TVs in target/ripple sets

The design favors clarity and correctness. Incremental update facilities can
be built atop the helpers here (e.g., caching demands, class maps, and
occupancy slices), but are not required for baseline correctness.

Examples
--------
Minimal end-to-end evaluation with two flights, one TV, and n=d baseline:

>>> import numpy as np
>>> import datetime as dt
>>> from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
>>> from parrhesia.optim.objective import score, ObjectiveWeights
>>>
>>> # 30-minute bins; define a single TV "V1"
>>> idx = TVTWIndexer(time_bin_minutes=30)
>>> idx._tv_id_to_idx = {"V1": 0}; idx._idx_to_tv_id = {0: "V1"}; idx._populate_tvtw_mappings()
>>> T = idx.num_time_bins
>>>
>>> # Two flights in one flow requesting bins 18 and 20 (09:00 and 10:00)
>>> base = dt.datetime(2025, 1, 1)
>>> flights_by_flow = {
...   "F": [
...     {"flight_id": "A1", "requested_dt": base.replace(hour=9, minute=0)},
...     {"flight_id": "A2", "requested_dt": base.replace(hour=10, minute=0)},
...   ]
... }
>>> schedule = [0] * (T + 1); schedule[18] = 1; schedule[20] = 1
>>> n_f_t = {"F": schedule}
>>>
>>> # Minimal flight_list with one occupancy interval per flight on V1
>>> class _Flights: ...
>>> fl = _Flights()
>>> fl.flight_metadata = {
...   "A1": {"occupancy_intervals": [{"tvtw_index": idx.get_tvtw_index("V1", 18), "entry_time_s": 0.0, "exit_time_s": 60.0}]},
...   "A2": {"occupancy_intervals": [{"tvtw_index": idx.get_tvtw_index("V1", 20), "entry_time_s": 0.0, "exit_time_s": 60.0}]},
... }
>>>
>>> # High capacity -> no exceedance
>>> capacities_by_tv = {"V1": np.full(T, 10, dtype=int)}
>>> w = ObjectiveWeights(gamma_gt=0.0, gamma_rip=0.0, gamma_ctx=0.0)
>>> J, comps, arts = score(
...   n_f_t,
...   flights_by_flow=flights_by_flow,
...   indexer=idx,
...   capacities_by_tv=capacities_by_tv,
...   target_cells=None, ripple_cells=None,
...   flight_list=fl, weights=w,
... )
>>> round(comps["J_cap"], 6), comps["J_delay"], comps["J_reg"], comps["J_tv"]
(0.0, 0.0, 0.0, 0.0)
>>> int(arts["occupancy"]["V1"][18]), int(arts["occupancy"]["V1"][20])
(1, 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
from datetime import datetime

import numpy as np

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from ..fcfs.flowful import assign_delays_flowful, _normalize_flight_spec, preprocess_flights_for_scheduler, assign_delays_flowful_preparsed
from .occupancy import compute_occupancy
from .capacity import rolling_hour_sum
# --------------------------------- Public API --------------------------------
# Optional timing logs (set True for debug)
DEBUG_TIMING = False


@dataclass
class ScoreContext:
    indexer: TVTWIndexer
    weights: ObjectiveWeights
    target_cells: Optional[Iterable[Cell]]
    ripple_cells: Optional[Iterable[Cell]]
    tvs_of_interest: Iterable[str]
    alpha_by_tv: Dict[str, np.ndarray]
    d_by_flow: Dict[Any, np.ndarray]
    fids_by_flow: Dict[Any, List[str]]
    bins_by_flow: Dict[Any, List[int]]
    flights_sorted_by_flow: Dict[Any, List[Tuple[str, Optional[object], int]]]
    attn_tv_ids: List[str]
    beta_gamma_by_flow: Dict[Any, Tuple[np.ndarray, np.ndarray]]


def build_score_context(
    flights_by_flow: Mapping[Any, Sequence[Any]],
    *,
    indexer: TVTWIndexer,
    capacities_by_tv: Mapping[str, np.ndarray],
    target_cells: Optional[Iterable[Cell]] = None,
    ripple_cells: Optional[Iterable[Cell]] = None,
    flight_list: Optional[object] = None,
    weights: Optional[ObjectiveWeights] = None,
    tv_filter: Optional[Iterable[str]] = None,
) -> ScoreContext:
    weights = weights or ObjectiveWeights()
    T = int(indexer.num_time_bins)

    # Static: demand and pre-parsed flights
    d_by_flow, fids_by_flow, bins_by_flow = _compute_demands(flights_by_flow, indexer)
    flights_sorted_by_flow = preprocess_flights_for_scheduler(flights_by_flow, indexer)

    # TVs of interest
    if tv_filter is not None:
        tvs_of_interest = set(str(tv) for tv in tv_filter)
    else:
        tvs_of_interest = set(capacities_by_tv.keys())
        for s in (target_cells or []):
            tvs_of_interest.add(str(s[0]))
        for s in (ripple_cells or []):
            tvs_of_interest.add(str(s[0]))
        if not tvs_of_interest:
            tvs_of_interest = set(indexer.tv_id_to_idx.keys())

    # Alpha weights
    alpha_by_tv = _build_alpha_weights(indexer, target_cells, ripple_cells, weights, restrict_to_tvs=tvs_of_interest)

    # Classification: beta/gamma per flow
    attn_tv_ids = sorted({str(tv) for (tv, _t) in (target_cells or [])} | {str(tv) for (tv, _t) in (ripple_cells or [])})
    beta_gamma_by_flow: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
    if attn_tv_ids:
        if flight_list is not None:
            median_offsets = _compute_median_offsets(
                flights_by_flow, fids_by_flow, bins_by_flow, flight_list, indexer, attn_tv_ids
            )
        else:
            median_offsets = {f: {tv: 0.0 for tv in attn_tv_ids} for f in flights_by_flow.keys()}

        tgt_mask, rip_mask, _ = _build_cell_masks_2d(attn_tv_ids, T, target_cells, ripple_cells, int(weights.class_tolerance_w))
        for f in flights_by_flow.keys():
            offs_map = median_offsets.get(f, {})
            offs = np.array([int(round(float(offs_map.get(tv, 0.0)))) for tv in attn_tv_ids], dtype=np.int32)
            is_gt, is_rip, _ = _classify_flow_bins_from_masks(offs, tgt_mask, rip_mask)
            beta = np.where(is_gt, weights.beta_gt, np.where(is_rip, weights.beta_rip, weights.beta_ctx)).astype(np.float32)
            gamma = np.where(is_gt, weights.gamma_gt, np.where(is_rip, weights.gamma_rip, weights.gamma_ctx)).astype(np.float32)
            beta_gamma_by_flow[f] = (beta, gamma)
    else:
        for f in flights_by_flow.keys():
            beta_gamma_by_flow[f] = (
                np.full(T, float(weights.beta_ctx), dtype=np.float32),
                np.full(T, float(weights.gamma_ctx), dtype=np.float32),
            )

    return ScoreContext(
        indexer=indexer,
        weights=weights,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        tvs_of_interest=list(tvs_of_interest),
        alpha_by_tv=alpha_by_tv,
        d_by_flow=d_by_flow,
        fids_by_flow=fids_by_flow,
        bins_by_flow=bins_by_flow,
        flights_sorted_by_flow=flights_sorted_by_flow,
        attn_tv_ids=attn_tv_ids,
        beta_gamma_by_flow=beta_gamma_by_flow,
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
    weights = context.weights
    indexer = context.indexer
    T = int(indexer.num_time_bins)

    # Normalize n
    n_by_flow: Dict[Any, np.ndarray] = {f: _to_len_T_plus_1_array(arr, T) for f, arr in n_f_t.items()}

    # Delays using preprocessed sorted flights
    import time
    if DEBUG_TIMING:
        time_start = time.time()
    delays_min, realised_start = assign_delays_flowful_preparsed(context.flights_sorted_by_flow, n_by_flow, indexer)
    if DEBUG_TIMING:
        time_end = time.time(); print(f"assign_delays_flowful(pre) time: {time_end - time_start} seconds")

    # Occupancy only for TVs of interest
    occ_by_tv = compute_occupancy(
        flight_list if flight_list is not None else type("_Dummy", (), {"flight_metadata": {}})(),
        delays_min,
        indexer,
        tv_filter=context.tvs_of_interest,
    )

    # J_cap (vectorized) using cached alpha
    if DEBUG_TIMING:
        time_start = time.time()
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
    if DEBUG_TIMING:
        time_end = time.time(); print(f"J_cap time: {time_end - time_start} seconds")

    # J_delay, J_reg, J_tv
    if DEBUG_TIMING:
        time_start = time.time()
    total_delay_min = sum(int(v) for v in delays_min.values())
    J_delay = float(weights.lambda_delay) * float(total_delay_min)
    J_reg, J_tv = _compute_J_reg_and_J_tv(n_by_flow, context.d_by_flow, context.beta_gamma_by_flow, weights.beta_ctx, weights.gamma_ctx)
    if DEBUG_TIMING:
        time_end = time.time(); print(f"J_reg, J_delay and J_tv time: {time_end - time_start} seconds")

    # Optional terms
    if DEBUG_TIMING:
        time_start = time.time()
    J_share = _compute_J_share(n_by_flow, context.d_by_flow, weights.theta_share) if weights.theta_share > 0 else 0.0
    if DEBUG_TIMING:
        time_end = time.time(); print(f"J_share time: {time_end - time_start} seconds")
    J_spill = _compute_J_spill(n_by_flow, weights.eta_spill) if weights.eta_spill > 0 else 0.0

    J_total = J_cap + J_delay + J_reg + J_tv + J_share + J_spill

    components: Dict[str, float] = {
        "J_cap": float(J_cap),
        "J_delay": float(J_delay),
        "J_reg": float(J_reg),
        "J_tv": float(J_tv),
    }
    if weights.theta_share > 0:
        components["J_share"] = float(J_share)
    if weights.eta_spill > 0:
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


# --------------------------------- Public API --------------------------------


# ------------------------------- Data classes -------------------------------


@dataclass
class ObjectiveWeights:
    """
    Weights for objective components and per-class penalties.

    Defaults follow docs/recipes/flowful_sa.md §5.10.

    Examples
    --------
    Use defaults but enable a small spill penalty and fairness term:

    >>> from parrhesia.optim.objective import ObjectiveWeights
    >>> w = ObjectiveWeights(theta_share=0.05, eta_spill=0.1)
    >>> w.alpha_gt, w.beta_ctx, w.gamma_rip
    (10.0, 1.0, 0.25)
    """

    # Capacity exceedance
    alpha_gt: float = 10.0
    alpha_rip: float = 3.0
    alpha_ctx: float = 0.5

    # Regularisation (rate deviation)
    beta_gt: float = 0.1
    beta_rip: float = 0.5
    beta_ctx: float = 1.0

    # Smoothness (total variation)
    gamma_gt: float = 0.1
    gamma_rip: float = 0.25
    gamma_ctx: float = 0.5

    # Delay weight
    lambda_delay: float = 0.1

    # Optional fairness and spill
    theta_share: float = 0.0
    eta_spill: float = 0.0

    # Tolerance (in bins) when mapping (f, t) via median offsets to cells
    class_tolerance_w: int = 1


# ------------------------------- Helper types -------------------------------


Cell = Tuple[str, int]  # (tv_id, bin)


# ------------------------------- Core helpers -------------------------------


def _to_len_T_plus_1_array(n_t: Union[Sequence[int], Mapping[int, int]], T: int) -> np.ndarray:
    """Normalize a per-bin schedule into an int64 numpy array of length T+1."""
    out = np.zeros(T + 1, dtype=np.int64)
    if isinstance(n_t, Mapping):
        for k, v in n_t.items():
            try:
                kk = int(k)
                if 0 <= kk <= T:
                    out[kk] = int(v)
            except Exception:
                continue
        return out
    # Sequence path
    try:
        arr = np.asarray(list(n_t), dtype=np.int64)
    except Exception:
        return out
    if arr.size >= T + 1:
        return arr[: T + 1].astype(np.int64, copy=False)
    out[: arr.size] = arr
    return out


def _extract_requested_bins_for_flow(
    specs: Sequence[Any], indexer: TVTWIndexer
) -> Tuple[List[str], List[int]]:
    """
    Extract (flight_id, requested_bin) pairs for a single flow.

    Reuses `_normalize_flight_spec` from the scheduler for consistent parsing
    of heterogeneous flight spec shapes.
    """
    fids: List[str] = []
    bins: List[int] = []
    for sp in specs or []:
        fid, r_dt, r_bin = _normalize_flight_spec(sp, indexer)
        fids.append(fid)
        bins.append(int(r_bin))
    return fids, bins


def _compute_demands(
    flights_by_flow: Mapping[Any, Sequence[Any]],
    indexer: TVTWIndexer,
) -> Tuple[Dict[Any, np.ndarray], Dict[Any, List[str]], Dict[Any, List[int]]]:
    """
    Compute baseline per-flow demand histograms d_f(t) from requested bins.

    Returns
    -------
    demand_by_flow : Dict[flow, np.ndarray[int64]]
        Arrays of shape (T+1,) where index T is the overflow bin (typically 0
        unless explicitly provided via 'requested_bin' == T).
    fids_by_flow : Dict[flow, List[str]]
        Flight ids per flow in the same order as the parsed bins.
    bins_by_flow : Dict[flow, List[int]]
        Requested bins per flight (0..T). Overflow bin T is accepted.
    """
    T = int(indexer.num_time_bins)
    d: Dict[Any, np.ndarray] = {}
    fids: Dict[Any, List[str]] = {}
    bins_map: Dict[Any, List[int]] = {}

    for f, specs in flights_by_flow.items():
        ids, rbins = _extract_requested_bins_for_flow(specs, indexer)
        # Fast histogram using np.bincount with clipping and explicit overflow bin at T
        rb = np.asarray(rbins, dtype=np.int64)
        if rb.size:
            rb = np.clip(rb, 0, T)
            arr = np.bincount(rb, minlength=T + 1)
        else:
            arr = np.zeros(T + 1, dtype=np.int64)
        d[f] = arr
        fids[f] = ids
        bins_map[f] = rb.tolist() if rb.size else []
    return d, fids, bins_map


def _earliest_bins_by_tv_for_flight(
    flight_meta: Mapping[str, Any], indexer: TVTWIndexer, tv_filter: Optional[Iterable[str]] = None
) -> Dict[str, int]:
    """
    From a flight's occupancy intervals, derive the earliest bin index for
    each TV encountered. Unknown/malformed intervals are ignored.
    """
    earliest: Dict[str, int] = {}
    whitelist = set(str(x) for x in (tv_filter or []))
    for iv in flight_meta.get("occupancy_intervals", []) or []:
        try:
            tvtw_idx = int(iv.get("tvtw_index"))
        except Exception:
            continue
        decoded = indexer.get_tvtw_from_index(tvtw_idx)
        if not decoded:
            continue
        tv_id, tbin = decoded
        if whitelist and str(tv_id) not in whitelist:
            continue
        tbin = int(tbin)
        cur = earliest.get(tv_id)
        if cur is None or tbin < cur:
            earliest[tv_id] = tbin
    return earliest


def _compute_median_offsets(
    flights_by_flow: Mapping[Any, Sequence[Any]],
    fids_by_flow: Mapping[Any, Sequence[str]],
    bins_by_flow: Mapping[Any, Sequence[int]],
    flight_list,  # object with `flight_metadata`
    indexer: TVTWIndexer,
    tv_ids_of_interest: Iterable[str],
) -> Dict[Any, Dict[str, float]]:
    """
    For each flow and each TV in `tv_ids_of_interest`, compute the median
    offset Δ̄_{f→v*} = median_i [ τ_i^{v*} - τ_i ] where τ_i is the requested
    bin at the controlled volume (from flights_by_flow) and τ_i^{v*} is the
    earliest baseline bin at TV v* from the flight's footprints.
    """
    result: Dict[Any, Dict[str, float]] = {}
    # Build per-flight earliest-bin maps once
    earliest_by_flight: Dict[str, Dict[str, int]] = {}
    tv_filter = list(tv_ids_of_interest)
    for fid, meta in getattr(flight_list, "flight_metadata", {}).items():
        earliest_by_flight[str(fid)] = _earliest_bins_by_tv_for_flight(meta, indexer, tv_filter)

    for flow_id in flights_by_flow.keys():
        ids = [str(x) for x in (fids_by_flow.get(flow_id) or [])]
        rbins = [int(x) for x in (bins_by_flow.get(flow_id) or [])]
        per_tv: Dict[str, float] = {}
        for tv_id in tv_ids_of_interest:
            offsets: List[int] = []
            for fid, rbin in zip(ids, rbins):
                eb = earliest_by_flight.get(fid, {})
                if tv_id in eb:
                    offsets.append(int(eb[tv_id]) - int(rbin))
            if offsets:
                per_tv[str(tv_id)] = float(np.median(np.asarray(offsets, dtype=np.float64)))
            else:
                per_tv[str(tv_id)] = 0.0
        result[flow_id] = per_tv
    return result


def _build_alpha_weights(
    indexer: TVTWIndexer,
    target_cells: Optional[Iterable[Cell]],
    ripple_cells: Optional[Iterable[Cell]],
    weights: ObjectiveWeights,
    restrict_to_tvs: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Build per-TV per-bin alpha weights based on target/ripple cells using a
    vectorized layout internally, then convert back to dict-of-arrays.
    """
    T = int(indexer.num_time_bins)
    if restrict_to_tvs is None:
        tv_ids: List[str] = [str(x) for x in indexer.tv_id_to_idx.keys()]
    else:
        tv_ids = [str(x) for x in restrict_to_tvs]

    V = len(tv_ids)
    tv_to_row = {tv: i for i, tv in enumerate(tv_ids)}
    A = np.full((V, T), float(weights.alpha_ctx), dtype=np.float32)

    if target_cells:
        for tv, t in target_cells:
            i = tv_to_row.get(str(tv))
            tt = int(t)
            if i is not None and 0 <= tt < T:
                A[i, tt] = float(weights.alpha_gt)
    if ripple_cells:
        for tv, t in ripple_cells:
            i = tv_to_row.get(str(tv))
            tt = int(t)
            if i is not None and 0 <= tt < T and A[i, tt] != float(weights.alpha_gt):
                A[i, tt] = float(weights.alpha_rip)

    return {tv: A[i].astype(np.float64, copy=True) for tv, i in tv_to_row.items()}


def _build_cell_masks_2d(
    tv_ids: Sequence[str],
    T: int,
    target_cells: Optional[Iterable[Cell]],
    ripple_cells: Optional[Iterable[Cell]],
    tol_w: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Build boolean masks [V, T] for target and ripple cells and dilate them by
    +/- tol_w along the time axis.
    """
    tv_to_row = {str(tv): i for i, tv in enumerate(tv_ids)}
    V = len(tv_ids)
    tgt = np.zeros((V, T), dtype=bool)
    rip = np.zeros((V, T), dtype=bool)
    if target_cells:
        for tv, t in target_cells:
            i = tv_to_row.get(str(tv))
            tt = int(t)
            if i is not None and 0 <= tt < T:
                tgt[i, tt] = True
    if ripple_cells:
        for tv, t in ripple_cells:
            i = tv_to_row.get(str(tv))
            tt = int(t)
            if i is not None and 0 <= tt < T:
                rip[i, tt] = True

    if tol_w > 0:
        base_tgt = tgt.copy()
        base_rip = rip.copy()
        for dt in range(1, int(tol_w) + 1):
            tgt[:, dt:] |= base_tgt[:, :-dt]
            tgt[:, :-dt] |= base_tgt[:, dt:]
            rip[:, dt:] |= base_rip[:, :-dt]
            rip[:, :-dt] |= base_rip[:, dt:]

    return tgt, rip, tv_to_row


def _classify_flow_bins_from_masks(
    offsets_for_flow: np.ndarray,
    tgt_mask: np.ndarray,
    rip_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify bins using precomputed dilated cell masks shifted by per-TV median
    offsets. Precedence: target > ripple > context.
    """
    V, T = tgt_mask.shape
    is_gt = np.zeros(T, dtype=bool)
    is_rip = np.zeros(T, dtype=bool)

    def or_shift(acc: np.ndarray, row: np.ndarray, delta: int) -> None:
        if delta >= T or delta <= -T:
            return
        if delta >= 0:
            if delta == 0:
                acc |= row
            else:
                acc[delta:] |= row[:-delta]
        else:
            d = -delta
            acc[: T - d] |= row[d:]

    for i in range(V):
        or_shift(is_gt, tgt_mask[i], int(offsets_for_flow[i]))

    tmp = np.zeros(T, dtype=bool)
    for i in range(V):
        or_shift(tmp, rip_mask[i], int(offsets_for_flow[i]))
    is_rip = np.logical_and(tmp, ~is_gt)
    is_ctx = ~(is_gt | is_rip)
    return is_gt, is_rip, is_ctx


def _classify_flow_bins(
    flow_id: Any,
    T: int,
    tv_ids: Sequence[str],
    median_offsets: Mapping[Any, Mapping[str, float]],
    target_cells: Optional[Iterable[Cell]],
    ripple_cells: Optional[Iterable[Cell]],
    tol_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a given flow, classify each bin t ∈ [0..T) as GT/RIP/CTX based on
    whether t + Δ̄_{f→v*} maps within ±w to a target/ripple cell for some v*.

    Returns three boolean arrays of shape (T,): is_gt, is_rip, is_ctx.
    Precedence: target > ripple > context.
    """
    tgt = set((str(tv), int(t)) for (tv, t) in (target_cells or []))
    rip = set((str(tv), int(t)) for (tv, t) in (ripple_cells or []))

    is_gt = np.zeros(T, dtype=bool)
    is_rip = np.zeros(T, dtype=bool)
    is_ctx = np.ones(T, dtype=bool)  # start as context; overwrite below

    offs = median_offsets.get(flow_id, {})
    # Helper to check membership within tolerance
    def _in_cells(tv: str, t0: int, cells: set[Cell]) -> bool:
        if not cells:
            return False
        for dt in range(-abs(tol_w), abs(tol_w) + 1):
            tt = t0 + dt
            if tt < 0 or tt >= T:
                continue
            if (str(tv), int(tt)) in cells:
                return True
        return False

    for t in range(T):
        tagged_gt = False
        tagged_rip = False
        for tv in tv_ids:
            delta = float(offs.get(str(tv), 0.0))
            t_map = int(round(t + delta))
            if _in_cells(tv, t_map, tgt):
                tagged_gt = True
                break
            if _in_cells(tv, t_map, rip):
                tagged_rip = True
                # continue scanning in case a target is found; but precedence is set below
        if tagged_gt:
            is_gt[t] = True
            is_ctx[t] = False
        elif tagged_rip:
            is_rip[t] = True
            is_ctx[t] = False
        else:
            # Remains context
            pass

    return is_gt, is_rip, is_ctx


def _weights_for_flow_bins(
    flow_id: Any,
    T: int,
    tv_ids: Sequence[str],
    median_offsets: Mapping[Any, Mapping[str, float]],
    target_cells: Optional[Iterable[Cell]],
    ripple_cells: Optional[Iterable[Cell]],
    weights: ObjectiveWeights,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce per-bin arrays beta_class(f, t) and gamma_class(f, t) for t=0..T-1.
    Overflow bin T uses ctx weights implicitly by the calling code.
    """
    is_gt, is_rip, is_ctx = _classify_flow_bins(
        flow_id, T, tv_ids, median_offsets, target_cells, ripple_cells, weights.class_tolerance_w
    )
    beta = np.where(is_gt, weights.beta_gt, np.where(is_rip, weights.beta_rip, weights.beta_ctx)).astype(np.float64)
    gamma = np.where(is_gt, weights.gamma_gt, np.where(is_rip, weights.gamma_rip, weights.gamma_ctx)).astype(np.float64)
    return beta, gamma


def _compute_J_reg_and_J_tv(
    n_by_flow: Mapping[Any, np.ndarray],
    d_by_flow: Mapping[Any, np.ndarray],
    beta_gamma_by_flow: Mapping[Any, Tuple[np.ndarray, np.ndarray]],
    beta_ctx: float,
    gamma_ctx: float,
) -> Tuple[float, float]:
    """
    Compute J_reg and J_tv given per-flow schedules, demands, and per-bin
    (beta, gamma) arrays for t in 0..T-1. Overflow bin handled with ctx.
    """
    J_reg = 0.0
    J_tv = 0.0
    for f in n_by_flow.keys():
        n = n_by_flow[f]
        d = d_by_flow[f]
        T = n.size - 1
        beta, gamma = beta_gamma_by_flow[f]
        # Regularisation for 0..T-1
        J_reg += float(np.sum(beta * np.abs(n[:T] - d[:T])))
        # Overflow bin uses ctx
        J_reg += float(beta_ctx * abs(int(n[T]) - int(d[T])))
        # Total variation: t=1..T (including difference with overflow at T)
        diff = np.abs(n[1 : T + 1] - n[0:T])
        # Weight per position uses gamma for bins 1..T-1; use ctx at T (last)
        if T >= 2:
            J_tv += float(np.sum(gamma[1:] * diff[1:]))  # 1..T-1
        # Handle first difference (t=1) with its gamma[1]
        if T >= 1:
            J_tv += float(gamma[0] * diff[0])  # t=0→1 uses class at t=0
        # Overflow edge uses ctx
        if T >= 1:
            J_tv += float(gamma_ctx * diff[-1])
    return J_reg, J_tv


def _compute_J_share(
    n_by_flow: Mapping[Any, np.ndarray],
    d_by_flow: Mapping[Any, np.ndarray],
    theta: float,
) -> float:
    """
    Fair-share deviation across flows per bin, scaled by theta.
    Uses L1 distance of distributions (0.5 * sum |p - q|) per bin.
    """
    if theta <= 0:
        return 0.0
    # Assume all flows share the same T
    flows = list(n_by_flow.keys())
    if not flows:
        return 0.0
    T_plus_1 = n_by_flow[flows[0]].size
    T = T_plus_1 - 1
    # Stack per-bin vectors for t=0..T-1
    n_stack = np.stack([n_by_flow[f][:T] for f in flows], axis=0).astype(np.float64)
    d_stack = np.stack([d_by_flow[f][:T] for f in flows], axis=0).astype(np.float64)
    N_tot = np.sum(n_stack, axis=0)
    D_tot = np.sum(d_stack, axis=0)
    # Avoid division by zero: when total is 0, the share vector is undefined; treat as zeros
    denom_n = np.maximum(N_tot, 1.0)
    denom_d = np.maximum(D_tot, 1.0)
    p = n_stack / denom_n
    q = d_stack / denom_d
    l1 = 0.5 * np.sum(np.abs(p - q), axis=0)
    return float(theta * np.sum(l1))


def _compute_J_spill(n_by_flow: Mapping[Any, np.ndarray], eta: float) -> float:
    if eta <= 0:
        return 0.0
    spill = 0
    for f, arr in n_by_flow.items():
        spill += int(arr[-1])  # overflow bin at index T
    return float(eta * spill)


def _rolling_hour_sum_2d(O: np.ndarray, K: int) -> np.ndarray:
    """
    Forward-looking rolling sum to match capacity.rolling_hour_sum semantics.
    RH[:, t] = sum_{u=t}^{min(t+K, T)-1} O[:, u]
    """
    V, T = O.shape
    # cumsum with a zero prepended for easy range sums
    zero = np.zeros((V, 1), dtype=np.int64)
    cs = np.concatenate([zero, np.cumsum(O, axis=1, dtype=np.int64)], axis=1)  # [V, T+1]
    idx_end = np.arange(T)
    idx_end = np.minimum(idx_end + int(K), T)
    s_end = np.take(cs, idx_end, axis=1)      # [V, T]
    s_start = cs[:, :T]                       # [V, T]
    return s_end - s_start


def _compute_J_cap_fast(
    occ_by_tv: Mapping[str, np.ndarray],
    capacities_by_tv: Mapping[str, np.ndarray],
    alpha_by_tv: Mapping[str, np.ndarray],
    K: int,
) -> float:
    """
    Vectorized J_cap across all TVs using 2D arrays and a single rolling sum.
    """
    tvs = list(occ_by_tv.keys())
    if not tvs:
        return 0.0
    T = int(next(iter(occ_by_tv.values())).size)
    V = len(tvs)
    O = np.zeros((V, T), dtype=np.int32)
    C = np.zeros((V, T), dtype=np.float32)
    A = np.zeros((V, T), dtype=np.float32)
    for i, tv in enumerate(tvs):
        oi = np.asarray(occ_by_tv[tv])
        O[i, : min(T, oi.size)] = oi[:T]
        ci = capacities_by_tv.get(tv)
        if ci is not None:
            ci_arr = np.asarray(ci, dtype=np.float32)
            C[i, : min(T, ci_arr.size)] = ci_arr[:T]
        ai = alpha_by_tv.get(tv)
        if ai is not None:
            ai_arr = np.asarray(ai, dtype=np.float32)
            A[i, : min(T, ai_arr.size)] = ai_arr[:T]
    RH = _rolling_hour_sum_2d(O, int(K)).astype(np.float32, copy=False)
    exceed = RH - C
    np.maximum(exceed, 0.0, out=exceed)
    J_cap = float(np.sum(exceed * A, dtype=np.float64))
    return J_cap


def _compute_J_cap(
    occ_by_tv: Mapping[str, np.ndarray],
    capacities_by_tv: Mapping[str, np.ndarray],
    alpha_by_tv: Mapping[str, np.ndarray],
    K: int,
    *,
    audit_exceedances: bool = False, # if true, print audit lines for each exceedance cell
    indexer: Optional[TVTWIndexer] = None,
    target_cells: Optional[Iterable[Cell]] = None,
    ripple_cells: Optional[Iterable[Cell]] = None,
) -> float:
    """
    Compute weighted exceedance across TVs with provided alpha weights.
    """
    # Fast path: no auditing requested -> vectorized
    if not audit_exceedances:
        return _compute_J_cap_fast(occ_by_tv, capacities_by_tv, alpha_by_tv, K)

    J_cap = 0.0
    # Precompute classification sets for auditing
    tgt_cells = set((str(tv), int(t)) for (tv, t) in (target_cells or []))
    rip_cells = set((str(tv), int(t)) for (tv, t) in (ripple_cells or []))
    printed_header = False
    for tv_id, occ in occ_by_tv.items():
        T = occ.size
        cap = np.asarray(capacities_by_tv.get(tv_id, np.zeros(T, dtype=np.int64)), dtype=np.float64)
        if cap.size != T:
            # If capacity length mismatches, align by truncation/padding zeros
            if cap.size < T:
                cap = np.pad(cap, (0, T - cap.size), mode="constant", constant_values=0)
            else:
                cap = cap[:T]
        rh = rolling_hour_sum(occ.astype(np.int64, copy=False), int(K)).astype(np.float64)
        exceed = np.maximum(0.0, rh - cap)
        alpha = np.asarray(alpha_by_tv.get(tv_id, np.zeros(T, dtype=np.float64)), dtype=np.float64)
        if alpha.size != T:
            if alpha.size < T:
                alpha = np.pad(alpha, (0, T - alpha.size), mode="constant", constant_values=0.0)
            else:
                alpha = alpha[:T]
        contrib = alpha * exceed
        J_cap += float(np.sum(contrib))

        if audit_exceedances:
            # Print audit lines for each exceedance cell
            if not printed_header:
                printed_header = True
                bin_minutes = getattr(indexer, "time_bin_minutes", None)
                print("\n[Audit] Capacity exceedance details (J_cap):")
                print(f" - Rolling window K = {K} bins" + (f" (~{K * bin_minutes} minutes)" if bin_minutes else ""))
            time_map = getattr(indexer, "time_window_map", {}) if indexer is not None else {}
            any_for_tv = False
            for t in range(T):
                exc = float(exceed[t])
                if exc <= 0.0:
                    continue
                any_for_tv = True
                # Classification by membership in provided cell sets
                cell = (str(tv_id), int(t))
                if cell in tgt_cells:
                    cls = "target"
                elif cell in rip_cells:
                    cls = "ripple"
                else:
                    cls = "context"
                weight = float(alpha[t])
                rh_val = float(rh[t])
                cap_val = float(cap[t])
                contrib_val = float(contrib[t])
                human_time = time_map.get(int(t)) if isinstance(time_map, dict) else None
                time_str = human_time if human_time is not None else f"bin {t}"
                print(
                    f"   • TV '{tv_id}', {time_str}: class={cls.upper()}, weight α={weight:.6g}; "
                    f"rolling occupancy={rh_val:.6g}, capacity={cap_val:.6g}, exceedance={exc:.6g} -> contribution={contrib_val:.6g}"
                )
            if audit_exceedances and any_for_tv:
                subtotal = float(np.sum(contrib))
                print(f"   = Subtotal for TV '{tv_id}': {subtotal:.6g}")
    if audit_exceedances:
        print(f"[Audit] J_cap total: {J_cap:.6g}\n")
    return J_cap


# --------------------------------- Public API --------------------------------


def score(
    n_f_t: Mapping[Any, Union[Sequence[int], Mapping[int, int]]],
    *,
    flights_by_flow: Mapping[Any, Sequence[Any]],
    indexer: TVTWIndexer,
    capacities_by_tv: Mapping[str, np.ndarray],
    target_cells: Optional[Iterable[Cell]] = None,
    ripple_cells: Optional[Iterable[Cell]] = None,
    flight_list: Optional[object] = None,
    weights: Optional[ObjectiveWeights] = None,
    tv_filter: Optional[Iterable[str]] = None,
    audit_exceedances: bool = False,  # if true, print audit lines for each exceedance cell
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    """
    Evaluate the objective J for a given per-flow schedule matrix n_f_t.

    Parameters
    ----------
    n_f_t : Mapping[flow, Sequence[int] | Mapping[int,int]]
        Per-flow per-bin release counts including overflow bin at index T.
    flights_by_flow : Mapping[flow, Sequence[FlightSpec]]
        Flights per flow with requested (baseline) times at the controlled
        volume. Shapes as accepted by `assign_delays_flowful`.
    indexer : TVTWIndexer
        Time indexing helper.
    capacities_by_tv : Mapping[str, np.ndarray]
        Per-TV per-bin rolling-hour capacities C_v(t).
    target_cells, ripple_cells : Optional[Iterable[(tv_id, bin)]]
        Cell sets defining the attention regions for alpha/beta/gamma.
        Unspecified cells use context weights.
    flight_list : Optional[object]
        If provided, must expose `flight_metadata` as in optim.flight_list.
        Required to classify (f, t) pairs via median offsets and to compute
        occupancy after delays. When omitted, classification falls back to
        context for all bins, and occupancy is computed only for the delays
        mapping (missing flights treated as zero delay if present in metadata).
    weights : Optional[ObjectiveWeights]
        Objective weights; uses sensible defaults if None.
    tv_filter : Optional[Iterable[str]]
        Restrict occupancy and alpha weighting to these TVs. When None, uses
        TVs present in capacities_by_tv or in the cell sets.

    Returns
    -------
    (J_total, components, artifacts)
      - J_total: float
      - components: mapping with keys 'J_cap', 'J_delay', 'J_reg', 'J_tv',
        and optionally 'J_share', 'J_spill' when non-zero weights are set.
      - artifacts: useful intermediate results for debugging or incremental
        updates, including:
          * 'delays_min' (Dict[flight_id, minutes])
          * 'realised_start' (Dict[flight_id, datetime | {"bin": int}])
          * 'occupancy' (Dict[tv_id, np.ndarray])
          * 'demand' (Dict[flow, np.ndarray])
          * 'n' (Dict[flow, np.ndarray])
          * 'beta_gamma' (Dict[flow, (beta, gamma)])

    Examples
    --------
    Basic use mirroring the module-level example:

    >>> import numpy as np
    >>> import datetime as dt
    >>> from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
    >>> idx = TVTWIndexer(time_bin_minutes=30)
    >>> idx._tv_id_to_idx = {"V1": 0}; idx._idx_to_tv_id = {0: "V1"}; idx._populate_tvtw_mappings()
    >>> T = idx.num_time_bins
    >>> base = dt.datetime(2025, 1, 1)
    >>> flights_by_flow = {
    ...   "F": [
    ...     {"flight_id": "A1", "requested_dt": base.replace(hour=9)},
    ...     {"flight_id": "A2", "requested_dt": base.replace(hour=10)},
    ...   ]
    ... }
    >>> schedule = [0] * (T + 1); schedule[18] = 1; schedule[20] = 1
    >>> n_f_t = {"F": schedule}
    >>> class _Flights: ...
    >>> fl = _Flights()
    >>> fl.flight_metadata = {
    ...   "A1": {"occupancy_intervals": [{"tvtw_index": idx.get_tvtw_index("V1", 18), "entry_time_s": 0.0, "exit_time_s": 60.0}]},
    ...   "A2": {"occupancy_intervals": [{"tvtw_index": idx.get_tvtw_index("V1", 20), "entry_time_s": 0.0, "exit_time_s": 60.0}]},
    ... }
    >>> capacities_by_tv = {"V1": np.full(T, 10, dtype=int)}
    >>> J, comps, arts = score(
    ...   n_f_t,
    ...   flights_by_flow=flights_by_flow,
    ...   indexer=idx,
    ...   capacities_by_tv=capacities_by_tv,
    ...   target_cells=None, ripple_cells=None,
    ...   flight_list=fl,
    ... )
    >>> float(comps["J_cap"]) == 0.0 and float(comps["J_reg"]) == 0.0
    True
    """
    weights = weights or ObjectiveWeights()
    T = int(indexer.num_time_bins)
    K = int(indexer.rolling_window_size())

    # Normalize n_f_t to arrays and compute baseline demands d_f(t)
    n_by_flow: Dict[Any, np.ndarray] = {f: _to_len_T_plus_1_array(arr, T) for f, arr in n_f_t.items()}
    d_by_flow, fids_by_flow, bins_by_flow = _compute_demands(flights_by_flow, indexer)

    # SCHEDULE: FIFO per flow -> delays and realised starts
    import time
    if DEBUG_TIMING:
        time_start = time.time()
    delays_min, realised_start = assign_delays_flowful(flights_by_flow, n_by_flow, indexer)
    if DEBUG_TIMING:
        time_end = time.time(); print(f"assign_delays_flowful time: {time_end - time_start} seconds")

    # OCCUPANCY: after delays, only for TVs of interest
    # Determine TVs to compute: from tv_filter, or from union of cell sets, or from capacities
    if tv_filter is not None:
        tvs_of_interest = set(str(tv) for tv in tv_filter)
    else:
        tvs_of_interest = set(capacities_by_tv.keys())
        for s in (target_cells or []):
            tvs_of_interest.add(str(s[0]))
        for s in (ripple_cells or []):
            tvs_of_interest.add(str(s[0]))
        if not tvs_of_interest:
            tvs_of_interest = set(indexer.tv_id_to_idx.keys())

    occ_by_tv = compute_occupancy(
        flight_list if flight_list is not None else type("_Dummy", (), {"flight_metadata": {}})(),
        delays_min,
        indexer,
        tv_filter=tvs_of_interest,
    )

    # ALPHA cell weights per TV
    alpha_by_tv = _build_alpha_weights(indexer, target_cells, ripple_cells, weights, restrict_to_tvs=tvs_of_interest)

    # CLASSIFICATION for beta/gamma per (f, t)
    # TVs appearing in target/ripple sets guide mapping; if none, treat all as context
    attn_tv_ids = sorted({str(tv) for (tv, _t) in (target_cells or [])} | {str(tv) for (tv, _t) in (ripple_cells or [])})
    if flight_list is not None and attn_tv_ids:
        median_offsets = _compute_median_offsets(
            flights_by_flow, fids_by_flow, bins_by_flow, flight_list, indexer, attn_tv_ids
        )
    else:
        median_offsets = {f: {tv: 0.0 for tv in attn_tv_ids} for f in flights_by_flow.keys()}

    beta_gamma_by_flow: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
    if attn_tv_ids:
        tgt_mask, rip_mask, tv_row_map = _build_cell_masks_2d(
            attn_tv_ids, T, target_cells, ripple_cells, int(weights.class_tolerance_w)
        )
        for f in flights_by_flow.keys():
            offs_map = median_offsets.get(f, {})
            offs = np.array([int(round(float(offs_map.get(tv, 0.0)))) for tv in attn_tv_ids], dtype=np.int32)
            is_gt, is_rip, _is_ctx = _classify_flow_bins_from_masks(offs, tgt_mask, rip_mask)
            beta = np.where(is_gt, weights.beta_gt, np.where(is_rip, weights.beta_rip, weights.beta_ctx)).astype(np.float32)
            gamma = np.where(is_gt, weights.gamma_gt, np.where(is_rip, weights.gamma_rip, weights.gamma_ctx)).astype(np.float32)
            beta_gamma_by_flow[f] = (beta, gamma)
    else:
        # No attention TVs -> all context
        for f in flights_by_flow.keys():
            beta_gamma_by_flow[f] = (
                np.full(T, float(weights.beta_ctx), dtype=np.float32),
                np.full(T, float(weights.gamma_ctx), dtype=np.float32),
            )

    # J components ------------------------------------------------------------
    # Capacity exceedance
    import time
    if DEBUG_TIMING:
        time_start = time.time()
    J_cap = _compute_J_cap(
        occ_by_tv,
        capacities_by_tv,
        alpha_by_tv,
        K,
        audit_exceedances=audit_exceedances,
        indexer=indexer,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
    )
    if DEBUG_TIMING:
        time_end = time.time(); print(f"J_cap time: {time_end - time_start} seconds")

    if DEBUG_TIMING:
        time_start = time.time()
    # Delay cost
    total_delay_min = sum(int(v) for v in delays_min.values())
    J_delay = float(weights.lambda_delay) * float(total_delay_min)

    # Regularisation and smoothness
    J_reg, J_tv = _compute_J_reg_and_J_tv(
        n_by_flow, d_by_flow, beta_gamma_by_flow, weights.beta_ctx, weights.gamma_ctx
    )
    if DEBUG_TIMING:
        time_end = time.time(); print(f"J_reg, J_delay and J_tv time: {time_end - time_start} seconds")

    # Fair-share and spill
    if DEBUG_TIMING:
        time_start = time.time()
    J_share = _compute_J_share(n_by_flow, d_by_flow, weights.theta_share) if weights.theta_share > 0 else 0.0
    if DEBUG_TIMING:
        time_end = time.time(); print(f"J_share time: {time_end - time_start} seconds")

    time_start = time.time()
    J_spill = _compute_J_spill(n_by_flow, weights.eta_spill) if weights.eta_spill > 0 else 0.0

    J_total = J_cap + J_delay + J_reg + J_tv + J_share + J_spill

    components: Dict[str, float] = {
        "J_cap": float(J_cap),
        "J_delay": float(J_delay),
        "J_reg": float(J_reg),
        "J_tv": float(J_tv),
    }
    if weights.theta_share > 0:
        components["J_share"] = float(J_share)
    if weights.eta_spill > 0:
        components["J_spill"] = float(J_spill)

    artifacts: Dict[str, Any] = {
        "delays_min": delays_min,
        "realised_start": realised_start,
        "occupancy": occ_by_tv,
        "demand": d_by_flow,
        "n": n_by_flow,
        "beta_gamma": beta_gamma_by_flow,
        "alpha": alpha_by_tv,
    }

    return float(J_total), components, artifacts


__all__ = [
    "ObjectiveWeights",
    "score",
    "ScoreContext",
    "build_score_context",
    "score_with_context",
]
