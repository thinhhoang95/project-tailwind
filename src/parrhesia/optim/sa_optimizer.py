"""
Simulated annealing optimizer for single-stage, flow-centric scheduling.

This module ties together:
  - FlightList and flow partitioning (already produced externally)
  - Construction of per-flow requested times at a controlled volume
  - Objective evaluation via parrhesia.optim.objective.score
  - A feasibility-preserving SA loop with attention-biased sampling

Key ideas
---------
- Variables: integer release counts n_f(t) for each flow f and bin t ∈ [0..T],
  where T denotes the overflow bin (index == indexer.num_time_bins).
- Feasibility: non-anticipativity and completeness must hold for each flow.
- Moves: intra-flow shifts later, pull-forward (if feasible), short block
  smoothing, and optional overflow reduction. These preserve non-negativity
  and completeness; pull-forward checks cumulative feasibility.
- Attention bias: prefer bins classified as target or ripple based on the
  median-offset mapping used in the objective. We reuse the classification
  artifacts from `objective.score` to bias move proposals.

Public API
----------
- prepare_flow_scheduling_inputs(...):
    Build `flights_by_flow` and select a controlled volume per flow using the
    earliest-median policy across a provided hotspot set.

- run_sa(...):
    Execute the simulated-annealing loop and return the best schedule and
    objective breakdown. Designed for correctness first; incremental deltas
    can be added later if needed for scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import math
import random

import numpy as np

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from ..flows.flow_pipeline import collect_hotspot_flights
from project_tailwind.optimize.eval.flight_list import FlightList
from .objective import score, ObjectiveWeights, build_score_context, score_with_context
from ..fcfs.flowful import _normalize_flight_spec, preprocess_flights_for_scheduler


# ---------------------------- Controlled volumes ----------------------------


def _earliest_crossing_bins_for_flow(
    flow_flight_ids: Sequence[str],
    flight_list: FlightList,
    hotspot_ids: Sequence[str],
) -> Dict[str, List[int]]:
    """
    For each hotspot in `hotspot_ids`, collect earliest crossing bins for
    flights belonging to the flow. Returns a mapping
      hotspot_id -> list[int (bins)]
    Only flights that cross a hotspot contribute to its list.
    """
    # Use precomputed per-flight earliest bins if available to avoid rescanning
    idx = flight_list.indexer
    by_hotspot: Dict[str, List[int]] = {str(h): [] for h in hotspot_ids}
    allowed = set(str(h) for h in hotspot_ids)
    # Build a small cache for this flow
    for fid in flow_flight_ids:
        meta = flight_list.flight_metadata.get(fid)
        if not meta:
            continue
        earliest_bin_by_tv: Dict[str, int] = {}
        for iv in meta.get("occupancy_intervals", []) or []:
            tvtw_idx_raw = iv.get("tvtw_index")
            try:
                tvtw_idx = int(tvtw_idx_raw)
            except Exception:
                continue
            decoded = idx.get_tvtw_from_index(tvtw_idx)
            if not decoded:
                continue
            tv_id, tbin = decoded
            s_tv = str(tv_id)
            if s_tv not in allowed:
                continue
            tb = int(tbin)
            cur = earliest_bin_by_tv.get(s_tv)
            if cur is None or tb < cur:
                earliest_bin_by_tv[s_tv] = tb
        for h, b in earliest_bin_by_tv.items():
            by_hotspot[h].append(int(b))
    return by_hotspot


def _controlled_volume_for_flow(
    flow_flight_ids: Sequence[str],
    flight_list: FlightList,
    hotspot_ids: Sequence[str],
    *,
    earliest_bin_by_flight_by_hotspot: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> Optional[str]:
    """
    Earliest-median policy across the given hotspot set.

    For each hotspot h, compute the median of earliest crossing bins of flights
    in the flow that touch h. Pick the hotspot with the smallest median across
    h; tie-break by smallest IQR (p75 - p25) then by lexical TV id.
    Returns None if no flights touch any hotspot.
    """
    if earliest_bin_by_flight_by_hotspot is not None:
        # Aggregate from precomputed earliest bins per flight
        bins_by_h: Dict[str, List[int]] = {str(h): [] for h in hotspot_ids}
        for fid in flow_flight_ids:
            eb = earliest_bin_by_flight_by_hotspot.get(str(fid), {})
            for h in hotspot_ids:
                v = eb.get(str(h))
                if v is not None:
                    bins_by_h[str(h)].append(int(v))
    else:
        bins_by_h = _earliest_crossing_bins_for_flow(flow_flight_ids, flight_list, hotspot_ids)
    best: Optional[Tuple[float, float, str]] = None  # (median, iqr, tv_id)
    for h in hotspot_ids:
        vals = bins_by_h.get(str(h), []) or []
        if not vals:
            continue
        a = np.asarray(vals, dtype=np.float64)
        med = float(np.median(a))
        p25 = float(np.percentile(a, 25))
        p75 = float(np.percentile(a, 75))
        iqr = p75 - p25
        key = (med, iqr, str(h))
        if best is None or key < best:
            best = key
    return best[2] if best is not None else None


def prepare_flow_scheduling_inputs(
    *,
    flight_list: FlightList,
    flow_map: Mapping[str, int],
    hotspot_ids: Sequence[str],
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Optional[str]]]:
    """
    Construct `flights_by_flow` for scheduling under a controlled volume.

    Returns a mapping:
      - flights_by_flow[flow_id] = list of flight specs, each as
            { 'flight_id': str, 'requested_bin': int }
        where requested_bin is the earliest crossing bin at the chosen
        controlled volume for this flow; if a flight does not cross the
        chosen volume, we fall back to its earliest crossing among the
        provided hotspot set.
      - controlled_volume_by_flow[flow_id] = tv_id or None
    """
    # Group flights by flow
    flights_by_flow: Dict[int, List[str]] = {}
    for fid, f in flow_map.items():
        flights_by_flow.setdefault(int(f), []).append(str(fid))

    # Decoding helper to support different FlightList/indexer implementations
    idx_obj = getattr(flight_list, "indexer", None)
    decode = None
    if idx_obj is not None and hasattr(idx_obj, "get_tvtw_from_index"):
        decode = lambda j: idx_obj.get_tvtw_from_index(int(j))
    else:
        # Fallback path for FlightList variants that do not expose `.indexer`
        bins_per_tv = int(getattr(flight_list, "num_time_bins_per_tv"))
        idx_to_tv_id = getattr(flight_list, "idx_to_tv_id")
        def decode(j: int):
            tv_idx = int(j) // int(bins_per_tv)
            tbin = int(j) % int(bins_per_tv)
            tv_id = str(idx_to_tv_id[int(tv_idx)])
            return tv_id, tbin
    flights_specs_by_flow: Dict[int, List[Dict[str, Any]]] = {}
    controlled_by_flow: Dict[int, Optional[str]] = {}

    # Precompute earliest crossing per flight per hotspot
    earliest_bin_by_flight_by_hotspot: Dict[str, Dict[str, int]] = {}
    hotspot_set = set(str(h) for h in hotspot_ids)
    for fid, meta in flight_list.flight_metadata.items():
        d: Dict[str, int] = {}
        for iv in meta.get("occupancy_intervals", []) or []:
            try:
                tvtw_idx = int(iv.get("tvtw_index"))
            except Exception:
                continue
            decoded = decode(tvtw_idx)
            if not decoded:
                continue
            tv_id, tbin = decoded
            s_tv = str(tv_id)
            if s_tv not in hotspot_set:
                continue
            tb = int(tbin)
            cur = d.get(s_tv)
            if cur is None or tb < cur:
                d[s_tv] = tb
        earliest_bin_by_flight_by_hotspot[str(fid)] = d

    for flow_id, fids in flights_by_flow.items():
        ctrl = _controlled_volume_for_flow(
            fids, flight_list, list(hotspot_ids), earliest_bin_by_flight_by_hotspot=earliest_bin_by_flight_by_hotspot
        )
        controlled_by_flow[flow_id] = ctrl
        specs: List[Dict[str, Any]] = []
        for fid in fids:
            # Preferred: earliest bin at the controlled volume
            rb = None
            if ctrl is not None:
                rb = earliest_bin_by_flight_by_hotspot.get(fid, {}).get(ctrl)
            if rb is None:
                # Fallback: earliest across any hotspot in the set
                eb = earliest_bin_by_flight_by_hotspot.get(fid, {})
                if eb:
                    rb = min(int(x) for x in eb.values())
            if rb is None:
                # As last resort, use 0
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Flight {fid} has no crossing bin at controlled volume or hotspots, using bin 0. This can cause degradation of the objective.")
                rb = 0
            specs.append({"flight_id": fid, "requested_bin": int(rb)})
        flights_specs_by_flow[flow_id] = specs

    return flights_specs_by_flow, controlled_by_flow


# --------------------------- SA core and helpers ----------------------------


@dataclass
class SAParams:
    iterations: int = 1000
    warmup_moves: int = 50
    alpha_T: float = 0.95
    L: int = 50  # temperature update period
    seed: Optional[int] = 0
    attention_bias: float = 0.8  # probability to sample from target/ripple bins
    max_shift: int = 4  # maximum Δ for shift-later
    pull_max: int = 2   # maximum Δ for pull-forward
    smooth_window_max: int = 3  # maximum window length for smoothing


def _to_T_plus_1(arr: Iterable[int], T: int) -> np.ndarray:
    out = np.zeros(T + 1, dtype=np.int64)
    a = list(arr)
    out[: min(len(a), T + 1)] = np.asarray(a[: T + 1], dtype=np.int64)
    return out


def _build_initial_schedule_from_demands(demand_by_flow: Mapping[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
    return {f: np.array(arr, dtype=np.int64, copy=True) for f, arr in demand_by_flow.items()}


def _cumulative(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(arr.astype(np.int64, copy=False))


def _feasible_pull_forward(
    n: np.ndarray,
    d_cum: np.ndarray,
    src_t: int,
    dst_t: int,
    q: int,
) -> bool:
    if q <= 0:
        return False
    T = n.size - 1
    if not (0 <= dst_t < src_t <= T):
        return False
    if n[src_t] < q:
        return False
    # Check prefix feasibility at dst_t: R'_{≤ dst_t} = R_{≤ dst_t} + q ≤ D_{≤ dst_t}
    R_cum = _cumulative(n)
    return int(R_cum[dst_t]) + int(q) <= int(d_cum[dst_t])


def _apply_shift_later(n: np.ndarray, t: int, delta: int, q: int) -> Optional[Tuple[int, int, int]]:
    T = n.size - 1
    if not (0 <= t <= T - 1):
        return None
    dst = t + int(delta)
    if not (0 <= dst <= T):
        return None
    if n[t] < q or q <= 0:
        return None
    n[t] -= q
    n[dst] += q
    return (t, dst, q)


def _apply_pull_forward(n: np.ndarray, d_cum: np.ndarray, t: int, delta: int, q: int) -> Optional[Tuple[int, int, int]]:
    dst = t - int(delta)
    if dst < 0:
        return None
    if not _feasible_pull_forward(n, d_cum, t, dst, q):
        return None
    n[t] -= q
    n[dst] += q
    return (dst, t, q)


def _apply_smoothing(n: np.ndarray, d_cum: np.ndarray, t0: int, window: int, q: int) -> Optional[Tuple[int, int, int]]:
    # Pattern: +q at t0, -q at t0+window
    T = n.size - 1
    t1 = t0 + window
    if not (0 <= t0 < t1 <= T):
        return None
    if n[t1] < q or q <= 0:
        return None
    # Adding at t0 is a pull-forward from t1; check feasibility at t0
    if not _feasible_pull_forward(n, d_cum, t1, t0, q):
        return None
    n[t1] -= q
    n[t0] += q
    return (t0, t1, q)


def _propose_move(
    rng: random.Random,
    n_by_flow: Dict[Any, np.ndarray],
    d_by_flow: Dict[Any, np.ndarray],
    attn_bins_by_flow: Dict[Any, np.ndarray],
    params: SAParams,
) -> Optional[Tuple[Any, str, Tuple[int, int, int]]]:
    """
    Propose and apply a single in-place move on `n_by_flow` if feasible.

    Returns a tuple (flow_id, move_type, (src, dst, q)) on success; None if
    no feasible move found after a limited number of attempts.
    """
    flows = list(n_by_flow.keys())
    if not flows:
        return None
    # Candidate list of (f, t) from attention bins
    use_attention = rng.random() < float(params.attention_bias)
    for _ in range(128):  # bounded attempts
        if use_attention and any(
            isinstance(m, np.ndarray) and bool(m.any()) for m in attn_bins_by_flow.values()
        ):
            f = rng.choice(flows)
            attn_mask = attn_bins_by_flow.get(f)
            if isinstance(attn_mask, np.ndarray) and attn_mask.any():
                idxs = np.nonzero(attn_mask)[0].tolist()
                if not idxs:
                    t = rng.randrange(0, n_by_flow[f].size - 1)
                else:
                    t = rng.choice(idxs)
            else:
                t = rng.randrange(0, n_by_flow[f].size - 1)
        else:
            f = rng.choice(flows)
            t = rng.randrange(0, n_by_flow[f].size - 1)

        n = n_by_flow[f]
        d = d_by_flow[f]
        d_cum = _cumulative(d)
        move_kind = rng.choice(["shift_later", "pull_forward", "smooth"])  # overflow handled via shift/pull
        # Amount q from set {1, 2} up to available
        max_q = int(n[t]) if move_kind != "pull_forward" else int(n[t])
        if max_q <= 0 and move_kind != "pull_forward":
            continue
        q = min(rng.choice([1, 2]), max_q if max_q > 0 else 2)

        if move_kind == "shift_later":
            delta = rng.randint(1, int(params.max_shift))
            applied = _apply_shift_later(n, t, delta, q)
            if applied is not None:
                return (f, move_kind, applied)
        elif move_kind == "pull_forward":
            # Choose a source t_src ≥ 1 and dst = t_src - Δ
            if n[t] <= 0:
                continue
            delta = rng.randint(1, int(params.pull_max))
            applied = _apply_pull_forward(n, d_cum, t, delta, q)
            if applied is not None:
                return (f, move_kind, applied)
        else:  # smooth
            window = rng.randint(2, int(params.smooth_window_max))
            applied = _apply_smoothing(n, d_cum, t, window, q)
            if applied is not None:
                return (f, move_kind, applied)
    return None


def _attention_masks_from_artifacts(
    beta_gamma_by_flow: Mapping[Any, Tuple[np.ndarray, np.ndarray]],
    weights: ObjectiveWeights,
) -> Dict[Any, np.ndarray]:
    """
    Build boolean masks per flow for bins classified as target or ripple using
    the gamma weights from artifacts (target: == gamma_gt, ripple: == gamma_rip).
    """
    out: Dict[Any, np.ndarray] = {}
    g_gt = float(weights.gamma_gt)
    g_rip = float(weights.gamma_rip)
    for f, (_beta, gamma) in beta_gamma_by_flow.items():
        # gamma is shape (T,). Overflow handled separately; mark bins 0..T-1
        g = np.asarray(gamma, dtype=np.float64)
        mask = (np.isclose(g, g_gt) | np.isclose(g, g_rip)).astype(bool)
        out[f] = mask
    return out


def run_sa(
    *,
    flights_by_flow: Mapping[Any, Sequence[Any]],
    flight_list: FlightList,
    indexer: TVTWIndexer,
    capacities_by_tv: Mapping[str, np.ndarray],
    target_cells: Optional[Iterable[Tuple[str, int]]] = None,
    ripple_cells: Optional[Iterable[Tuple[str, int]]] = None,
    weights: Optional[ObjectiveWeights] = None,
    params: Optional[SAParams] = None,
    tv_filter: Optional[Iterable[str]] = None,
) -> Tuple[Dict[Any, np.ndarray], float, Dict[str, float], Dict[str, Any]]:
    """
    Execute a simulated annealing loop starting from n=d and return the best.

    Returns (n_best_by_flow, J_best, components, artifacts_of_best)
    where `n_best_by_flow` is a mapping to numpy int64 arrays of shape (T+1,).
    """
    params = params or SAParams()
    weights = weights or ObjectiveWeights()
    rng = random.Random(params.seed)

    # Compute demands directly from flights_by_flow -> d_f(t)
    def _compute_demands_from_flights(
        flights_by_flow: Mapping[Any, Sequence[Any]], indexer: TVTWIndexer
    ) -> Dict[Any, np.ndarray]:
        T = int(indexer.num_time_bins)
        out: Dict[Any, np.ndarray] = {}
        for f, specs in flights_by_flow.items():
            arr = np.zeros(T + 1, dtype=np.int64)
            for sp in specs or []:
                _fid, _rdt, rbin = _normalize_flight_spec(sp, indexer)
                rb = int(rbin)
                if 0 <= rb <= T:
                    arr[rb] += 1
            out[f] = arr
        return out

    d_by_flow = _compute_demands_from_flights(flights_by_flow, indexer)
    n_by_flow: Dict[Any, np.ndarray] = _build_initial_schedule_from_demands(d_by_flow)

    # Build scoring context and evaluate baseline with n=d via fast path
    context = build_score_context(
        flights_by_flow,
        indexer=indexer,
        capacities_by_tv=capacities_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        flight_list=flight_list,
        weights=weights,
        tv_filter=tv_filter,
    )
    J_best, comps_best, arts_best = score_with_context(
        n_by_flow,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=capacities_by_tv,
        flight_list=flight_list,
        context=context,
    )
    n_best = {f: np.array(v, dtype=np.int64, copy=True) for f, v in n_by_flow.items()}

    # Attention masks from artifacts
    attn_masks = _attention_masks_from_artifacts(arts_best.get("beta_gamma", {}), weights)

    # Warm-up to estimate temperature scale
    deltas: List[float] = []
    for _ in range(int(params.warmup_moves)):
        # Copy schedule for a trial move
        trial = {f: np.array(v, dtype=np.int64, copy=True) for f, v in n_by_flow.items()}
        mv = _propose_move(rng, trial, d_by_flow, attn_masks, params)
        if mv is None:
            continue
        J_trial, _c, _a = score_with_context(
            trial,
            flights_by_flow=flights_by_flow,
            capacities_by_tv=capacities_by_tv,
            flight_list=flight_list,
            context=context,
        )
        deltas.append(float(J_trial - J_best))
    sigma = float(np.std(np.asarray(deltas, dtype=np.float64))) if deltas else 1.0
    Tcur = 2.0 * sigma if sigma > 0 else 1.0

    # Main SA loop
    for it in range(int(params.iterations)):
        candidate = {f: np.array(v, dtype=np.int64, copy=True) for f, v in n_by_flow.items()}
        mv = _propose_move(rng, candidate, d_by_flow, attn_masks, params)
        if mv is None:
            continue
        J_new, comps_new, arts_new = score_with_context(
            candidate,
            flights_by_flow=flights_by_flow,
            capacities_by_tv=capacities_by_tv,
            flight_list=flight_list,
            context=context,
        )
        dJ = float(J_new - J_best)
        accept = dJ <= 0.0 or rng.random() < math.exp(-(max(dJ, 0.0)) / max(Tcur, 1e-9))
        if accept:
            n_by_flow = candidate
            if J_new <= J_best:
                J_best, comps_best, arts_best = J_new, comps_new, arts_new
                n_best = {f: np.array(v, dtype=np.int64, copy=True) for f, v in n_by_flow.items()}
        if (it + 1) % int(params.L) == 0:
            Tcur *= float(params.alpha_T)

    return n_best, float(J_best), comps_best, arts_best


__all__ = [
    "SAParams",
    "prepare_flow_scheduling_inputs",
    "run_sa",
]
