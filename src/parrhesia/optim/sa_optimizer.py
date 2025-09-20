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
import os
import logging
import math
import random

import numpy as np

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from ..flows.flow_pipeline import collect_hotspot_flights
from project_tailwind.optimize.eval.flight_list import FlightList
from .objective import score, ObjectiveWeights, build_score_context, score_with_context
from ..fcfs.flowful import _normalize_flight_spec, preprocess_flights_for_scheduler

logger = logging.getLogger(__name__)

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
    flight_ids: Optional[Iterable[str]] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Optional[str]]]:
    """
    Construct `flights_by_flow` for scheduling under a controlled volume.

    Parameters
    ----------
    flight_list : FlightList
        Provider for flight metadata and decoding helpers.
    flow_map : Mapping[str, int]
        Mapping from flight_id -> flow_id.
    hotspot_ids : Sequence[str]
        Hotspots considered for selecting the controlled volume.
    flight_ids : Optional[Iterable[str]], default=None
        Optional restriction to a subset of flights when computing earliest
        crossings. When omitted, all flights in `flight_list` are scanned.

    Returns
    -------
    Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Optional[str]]]
        Mapping:
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

    # Restrict iteration to the union of relevant flights when provided.
    subset: Optional[List[str]]
    if flight_ids is not None:
        subset = list(dict.fromkeys(str(fid) for fid in flight_ids))
    else:
        subset = None

    meta_items: Iterable[Tuple[str, Any]]
    if subset is None:
        meta_items = flight_list.flight_metadata.items()
    else:
        meta_map = flight_list.flight_metadata
        meta_items = ((fid, meta_map.get(fid)) for fid in subset)

    # Precompute earliest crossing per flight per hotspot, capturing entry_time_s
    # - earliest_bin_by_flight_by_hotspot: fid -> tv -> earliest bin (int)
    # - earliest_crossing_by_flight_by_tv: fid -> tv -> (earliest_bin, entry_time_s_at_that_bin)
    earliest_bin_by_flight_by_hotspot: Dict[str, Dict[str, int]] = {}
    earliest_crossing_by_flight_by_tv: Dict[str, Dict[str, Tuple[int, float]]] = {}
    hotspot_set = set(str(h) for h in hotspot_ids)

    for fid, meta in meta_items:
        if not meta:
            continue
        d_bin: Dict[str, int] = {}
        d_full: Dict[str, Tuple[int, float]] = {}
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
            # Normalize entry_time_s (seconds from takeoff)
            raw_entry = iv.get("entry_time_s", 0)
            try:
                entry_s = float(raw_entry)
            except Exception:
                entry_s = 0.0
            cur_bin = d_bin.get(s_tv)
            if cur_bin is None or tb < cur_bin:
                d_bin[s_tv] = tb
                d_full[s_tv] = (tb, float(entry_s))
        earliest_bin_by_flight_by_hotspot[str(fid)] = d_bin
        earliest_crossing_by_flight_by_tv[str(fid)] = d_full

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
                logger.warning(
                    "Flight %s has no crossing bin at controlled volume or hotspots, using bin 0. This can cause degradation of the objective.",
                    fid,
                )
                rb = 0

            # Attempt to enrich with requested_dt using takeoff_time + entry_time_s
            requested_dt = None
            try:
                meta = flight_list.flight_metadata.get(fid, {})
                takeoff_time = meta.get("takeoff_time")
            except Exception:
                takeoff_time = None

            if takeoff_time is not None and rb is not None:
                # Identify the TV used to select rb and its entry_time_s
                tv_to_pick: Optional[str] = None
                entry_s: Optional[float] = None
                if ctrl is not None:
                    pair = earliest_crossing_by_flight_by_tv.get(fid, {}).get(ctrl)
                    if pair is not None:
                        tv_to_pick = ctrl
                        b0, e0 = pair
                        # Sanity: ensure we're using the earliest bin at ctrl
                        if int(b0) == int(rb):
                            entry_s = float(e0)
                        else:
                            # If mismatch, prefer the entry_s corresponding to rb among hotspots
                            tv_to_pick = None
                if tv_to_pick is None:
                    # Fallback: pick the hotspot TV achieving the chosen rb
                    candidates = []
                    for tv, (b, e) in earliest_crossing_by_flight_by_tv.get(fid, {}).items():
                        if tv in hotspot_set and int(b) == int(rb):
                            candidates.append((tv, float(e)))
                    if candidates:
                        # Choose the smallest entry_s among candidates to represent within-bin ordering
                        tv_to_pick, entry_s = min(candidates, key=lambda x: float(x[1]))
                if entry_s is not None:
                    from datetime import timedelta
                    try:
                        requested_dt = takeoff_time + timedelta(seconds=float(entry_s))  # type: ignore[operator]
                    except Exception:
                        requested_dt = None

            spec: Dict[str, Any] = {"flight_id": fid, "requested_bin": int(rb)}
            if requested_dt is not None:
                # Provide within-bin time for minute-precise FIFO delays
                spec["requested_dt"] = requested_dt
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Enriched flight %s with requested_dt from entry offset; bin=%d", fid, int(rb)
                    )
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Flight %s missing takeoff/entry seconds for within-bin time; using bin-only (bin=%d)",
                        fid,
                        int(rb),
                    )
            specs.append(spec)
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
    verbose: bool = False  # if True (or env SA_VERBOSE truthy), print SA progress
    # Minutes to expand allowed change window relative to the min/max target bins.
    # Only bins within the expanded window are allowed to change during SA.
    rate_change_lower_bound_min: int = 0
    rate_change_upper_bound_min: int = 0


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
    allowed_mask: np.ndarray,
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
                # Restrict attention to allowed change bins
                T = n_by_flow[f].size - 1
                allowed_T = np.asarray(allowed_mask[:T], dtype=bool)
                idxs = np.nonzero(np.asarray(attn_mask, dtype=bool) & allowed_T)[0].tolist()
                if not idxs:
                    # Fall back to any allowed bin
                    allowed_idxs = np.nonzero(allowed_T)[0].tolist()
                    if not allowed_idxs:
                        continue
                    t = rng.choice(allowed_idxs)
                else:
                    t = rng.choice(idxs)
            else:
                # Fall back to any allowed bin
                T = n_by_flow[f].size - 1
                allowed_idxs = np.nonzero(np.asarray(allowed_mask[:T], dtype=bool))[0].tolist()
                if not allowed_idxs:
                    continue
                t = rng.choice(allowed_idxs)
        else:
            f = rng.choice(flows)
            # Uniform among allowed bins
            T = n_by_flow[f].size - 1
            allowed_idxs = np.nonzero(np.asarray(allowed_mask[:T], dtype=bool))[0].tolist()
            if not allowed_idxs:
                continue
            t = rng.choice(allowed_idxs)

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
            # Enforce allowed window at both source and destination
            dst = t + int(delta)
            T = n.size - 1
            if not (0 <= t <= T - 1):
                continue
            if not (0 <= dst <= T):
                continue
            if not (bool(allowed_mask[t]) and (dst < T and bool(allowed_mask[dst]))):
                # Disallow moves touching overflow or outside allowed window
                continue
            applied = _apply_shift_later(n, t, delta, q)
            if applied is not None:
                return (f, move_kind, applied)
        elif move_kind == "pull_forward":
            # Choose a source t_src ≥ 1 and dst = t_src - Δ
            if n[t] <= 0:
                continue
            delta = rng.randint(1, int(params.pull_max))
            dst = t - int(delta)
            if dst < 0:
                continue
            if not (bool(allowed_mask[t]) and bool(allowed_mask[dst])):
                continue
            applied = _apply_pull_forward(n, d_cum, t, delta, q)
            if applied is not None:
                return (f, move_kind, applied)
        else:  # smooth
            window = rng.randint(2, int(params.smooth_window_max))
            T = n.size - 1
            t1 = t + window
            if not (0 <= t < t1 <= T):
                continue
            if not (bool(allowed_mask[t]) and (t1 < T and bool(allowed_mask[t1]))):
                continue
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


def _compute_allowed_change_mask(
    indexer: TVTWIndexer,
    target_cells: Optional[Iterable[Tuple[str, int]]],
    params: SAParams,
) -> np.ndarray:
    """
    Build a boolean mask over time bins (length T+1 including overflow) that
    indicates which bins are allowed to change. Only bins in the expanded
    window [min_target_bin - lower_margin, max_target_bin + upper_margin]
    are allowed. The overflow bin (index T) is never allowed.

    If target_cells is None or empty, all non-overflow bins are allowed.
    """
    T = int(indexer.num_time_bins)
    mask = np.zeros(T + 1, dtype=bool)
    # Default: allow all non-overflow bins
    allow_all = True
    min_bin: Optional[int] = None
    max_bin: Optional[int] = None

    if target_cells is not None:
        for _tv, b in target_cells:
            try:
                tb = int(b)
            except Exception:
                continue
            if 0 <= tb < T:
                allow_all = False
                if min_bin is None or tb < min_bin:
                    min_bin = tb
                if max_bin is None or tb > max_bin:
                    max_bin = tb

    if allow_all or min_bin is None or max_bin is None:
        # Allow all non-overflow bins
        mask[:T] = True
        mask[T] = False
        return mask

    tbm = int(getattr(indexer, "time_bin_minutes", 30))
    lower_bins = int(math.ceil(float(params.rate_change_lower_bound_min) / float(max(tbm, 1))))
    upper_bins = int(math.ceil(float(params.rate_change_upper_bound_min) / float(max(tbm, 1))))
    lo = max(0, int(min_bin) - int(lower_bins))
    hi = min(T - 1, int(max_bin) + int(upper_bins))
    if lo <= hi:
        mask[lo : hi + 1] = True
    mask[T] = False
    return mask


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
    import time
    
    params = params or SAParams()
    weights = weights or ObjectiveWeights()
    rng = random.Random(params.seed)

    # Verbose flag: SAParams.verbose OR environment variable SA_VERBOSE in {1,true,yes,on}
    env_v = os.getenv("SA_VERBOSE", "").strip().lower()
    env_verbose = env_v in {"1", "true", "yes", "on", "y"}
    is_verbose = bool(getattr(params, "verbose", False)) or env_verbose

    if is_verbose:
        start_time = time.time()
        print(f"[SA] Starting SA optimization...")

    # Print optimization parameters
    
    print(f"[SA] Optimization parameters:")
    print(f"  iterations: {params.iterations}")
    print(f"  warmup_moves: {params.warmup_moves}")
    print(f"  alpha_T: {params.alpha_T}")
    print(f"  L: {params.L}")
    print(f"  seed: {params.seed}")
    print(f"  attention_bias: {params.attention_bias}")
    print(f"  max_shift: {params.max_shift}")
    print(f"  pull_max: {params.pull_max}")
    print(f"  smooth_window_max: {params.smooth_window_max}")
    print(f"  rate_change_lower_bound_min: {params.rate_change_lower_bound_min}")
    print(f"  rate_change_upper_bound_min: {params.rate_change_upper_bound_min}")

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

    if is_verbose:
        demand_start = time.time()
    d_by_flow = _compute_demands_from_flights(flights_by_flow, indexer)
    n_by_flow: Dict[Any, np.ndarray] = _build_initial_schedule_from_demands(d_by_flow)
    if is_verbose:
        demand_end = time.time()
        print(f"[SA] Demand computation time: {demand_end - demand_start:.3f}s")

    # Build scoring context and evaluate baseline with n=d via fast path
    if is_verbose:
        context_start = time.time()
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
    if is_verbose:
        context_end = time.time()
        print(f"[SA] Context building time: {context_end - context_start:.3f}s")

    if is_verbose:
        baseline_start = time.time()
    J_best, comps_best, arts_best = score_with_context(
        n_by_flow,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=capacities_by_tv,
        flight_list=flight_list,
        context=context,
    )
    n_best = {f: np.array(v, dtype=np.int64, copy=True) for f, v in n_by_flow.items()}
    if is_verbose:
        baseline_end = time.time()
        print(f"[SA] Baseline scoring time: {baseline_end - baseline_start:.3f}s")
        print(f"[SA] Baseline objective J0 = {J_best:.6f}")

    # Attention masks from artifacts
    if is_verbose:
        mask_start = time.time()
    attn_masks = _attention_masks_from_artifacts(arts_best.get("beta_gamma", {}), weights)
    # Allowed-change mask derived from target cells and SAParams margins
    allowed_mask = _compute_allowed_change_mask(indexer, target_cells, params)
    if is_verbose:
        mask_end = time.time()
        print(f"[SA] Attention mask time: {mask_end - mask_start:.3f}s")

    # Warm-up to estimate temperature scale
    deltas: List[float] = []
    successful_moves = 0
    failed_moves = 0
    
    if is_verbose:
        warmup_start = time.time()
        print(f"[SA] Starting warmup with {int(params.warmup_moves)} moves...")
    
    for warmup_iter in range(int(params.warmup_moves)):
        # Copy schedule for a trial move
        copy_start = time.time()
        trial = {f: np.array(v, dtype=np.int64, copy=True) for f, v in n_by_flow.items()}
        copy_end = time.time()

        if is_verbose:
            print(f"[SA] Copy time: {copy_end - copy_start:.6f}s")
        
        propose_start = time.time()
        mv = _propose_move(rng, trial, d_by_flow, attn_masks, allowed_mask, params)
        propose_end = time.time()

        if is_verbose:
            print(f"[SA] Propose time: {propose_end - propose_start:.6f}s")
        
        if mv is None:
            failed_moves += 1
            if is_verbose and warmup_iter % 10 == 0:
                print(f"[SA] Warmup {warmup_iter+1}/{int(params.warmup_moves)}: failed to propose move")
                print(f"[SA] Copy time: {copy_end - copy_start:.6f}s, Propose time: {propose_end - propose_start:.6f}s")
            continue
        
        score_start = time.time()
        J_trial, _c, _a = score_with_context(
            trial,
            flights_by_flow=flights_by_flow,
            capacities_by_tv=capacities_by_tv,
            flight_list=flight_list,
            context=context,
        )
        score_end = time.time()
        
        if is_verbose:
            print(f"[SA] Score time: {score_end - score_start:.6f}s")
        
        delta = float(J_trial - J_best)
        deltas.append(delta)
        successful_moves += 1
        
        if is_verbose and warmup_iter % 10 == 0:
            print(f"[SA] Warmup timing - Copy: {copy_end - copy_start:.6f}s, Propose: {propose_end - propose_start:.6f}s, Score: {score_end - score_start:.6f}s")
        
        if is_verbose and warmup_iter % 10 == 0:
            print(f"[SA] Warmup {warmup_iter+1}/{int(params.warmup_moves)}: J_trial={J_trial:.6f}, delta={delta:.6f}")
    
    sigma = float(np.std(np.asarray(deltas, dtype=np.float64))) if deltas else 1.0
    Tcur = 2.0 * sigma if sigma > 0 else 1.0
    
    if is_verbose:
        warmup_end = time.time()
        print(f"[SA] Warmup time: {warmup_end - warmup_start:.3f}s")
        print(f"[SA] Warmup complete: {successful_moves} successful moves, {failed_moves} failed moves")
        if deltas:
            print(f"[SA] Delta statistics: min={min(deltas):.6f}, max={max(deltas):.6f}, mean={np.mean(deltas):.6f}")
        print(f"[SA] Warmup: sigma={sigma:.6f}, T0={Tcur:.6f}, warmup_moves={int(params.warmup_moves)}")

    # Main SA loop
    if is_verbose:
        main_loop_start = time.time()
        print(f"[SA] Starting main SA loop with {int(params.iterations)} iterations...")
    
    for it in range(int(params.iterations)):
        candidate = {f: np.array(v, dtype=np.int64, copy=True) for f, v in n_by_flow.items()}
        mv = _propose_move(rng, candidate, d_by_flow, attn_masks, allowed_mask, params)
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
                if is_verbose:
                    print(f"[SA] it={it+1} improved best -> {J_new:.6f}")
                J_best, comps_best, arts_best = J_new, comps_new, arts_new
                n_best = {f: np.array(v, dtype=np.int64, copy=True) for f, v in n_by_flow.items()}
        if (it + 1) % int(params.L) == 0:
            Tcur *= float(params.alpha_T)
            if is_verbose:
                print(f"[SA] it={it+1} cooled: T={Tcur:.6f}, best={J_best:.6f}")

    if is_verbose:
        main_loop_end = time.time()
        total_end = time.time()
        print(f"[SA] Main loop time: {main_loop_end - main_loop_start:.3f}s")
        print(f"[SA] Total SA time: {total_end - start_time:.3f}s")
        print(f"[SA] Final best objective: {J_best:.6f}")

    return n_best, float(J_best), comps_best, arts_best


__all__ = [
    "SAParams",
    "prepare_flow_scheduling_inputs",
    "run_sa",
]
