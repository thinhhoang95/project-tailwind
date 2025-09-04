from __future__ import annotations

"""
Automatic rate adjustment via simulated annealing (SA).

Implements the plan in prompts/tailwind_api/detailed_plan_autorate.md.

Entry point: compute_automatic_rate_adjustment(payload: Mapping[str, Any]) -> dict

Request JSON (keys optional unless noted):
  - flows (required): mapping of flow-id -> list of flight IDs
  - targets (required): mapping of TV -> {"from": "HH:MM[:SS]", "to": "HH:MM[:SS]"}
  - ripples (optional): same schema as targets
  - auto_ripple_time_bins (optional): if > 0, overrides ripples by using union of
    footprints of flights across TVs with ±window dilation
  - indexer_path, flights_path, capacities_path (optional): artifact overrides
  - weights (optional): partial overrides for ObjectiveWeights
  - sa_params (optional): partial overrides for SAParams

Returns JSON with baseline vs optimized objectives, per-flow baseline n0 and
optimized n_opt arrays, baseline per-TV demand vectors, post-optimization
per-TV occupancy arrays, and per-flight delays in minutes under the optimized
schedule.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Sequence

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from parrhesia.optim.capacity import build_bin_capacities
from parrhesia.optim.objective import ObjectiveWeights, score
from parrhesia.optim.sa_optimizer import SAParams, prepare_flow_scheduling_inputs, run_sa
from parrhesia.optim.occupancy import compute_occupancy

from .base_evaluation import (
    _default_paths_from_root,
    _cells_from_ranges,
    _auto_ripple_cells_from_flows,
)
from .flows import _load_indexer_and_flights
from .resources import get_global_resources


def _coerce_int_flow_ids(flows_in: Mapping[Any, Any]) -> Dict[str, int]:
    """Deterministically coerce flow keys to integers.

    Numeric-like keys use their int value; others get assigned 0.. in sorted order.
    Returns mapping from original stringified key to int id.
    """
    flow_key_to_int: Dict[str, int] = {}
    next_id = 0
    for k in sorted((str(x) for x in flows_in.keys()), key=str):
        try:
            flow_key_to_int[k] = int(k)
        except Exception:
            flow_key_to_int[k] = next_id
            next_id += 1
    return flow_key_to_int


def _build_flow_map(
    flows_in: Mapping[Any, Any],
    fl: FlightList,
) -> Dict[str, int]:
    """Build flight_id -> flow_id mapping, ignoring unknown flights."""
    flow_key_to_int = _coerce_int_flow_ids(flows_in)
    flow_map: Dict[str, int] = {}
    for k, flights in flows_in.items():
        fid = flow_key_to_int[str(k)]
        seq: Iterable[Any] = flights if isinstance(flights, (list, tuple)) else []
        for flid in seq:
            sfl = str(flid)
            if sfl in fl.flight_metadata:
                flow_map[sfl] = fid
    return flow_map


def _per_tv_earliest_demands(
    *,
    idx: TVTWIndexer,
    fl: FlightList,
    T: int,
    flights_by_flow: Mapping[Any, Sequence[Mapping[str, Any]]],
    flow_map: Mapping[str, int],
    target_tv_ids: List[str],
    ripple_tv_ids: List[str],
) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[int]]]]:
    """
    Mirror of base_evaluation's step 5b to compute per-TV earliest-crossing
    demand vectors for targets and ripples.
    """
    # Precompute earliest crossing bin per flight for union(targets ∪ ripples)
    tv_union: set[str] = set(target_tv_ids) | set(ripple_tv_ids)
    earliest_bin_by_flight: Dict[str, Dict[str, int]] = {}
    if tv_union:
        allowed = set(str(x) for x in tv_union)
        for fid, meta in fl.flight_metadata.items():
            if str(fid) not in flow_map:
                continue
            d: Dict[str, int] = {}
            for iv in meta.get("occupancy_intervals", []) or []:
                try:
                    tvtw_idx = int(iv.get("tvtw_index"))
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
                cur = d.get(s_tv)
                if cur is None or tb < cur:
                    d[s_tv] = tb
            earliest_bin_by_flight[str(fid)] = d

    target_demands_by_flow: Dict[int, Dict[str, List[int]]] = {}
    ripple_demands_by_flow: Dict[int, Dict[str, List[int]]] = {}
    for f, specs in flights_by_flow.items():
        t_map: Dict[str, List[int]] = {tv: [0] * T for tv in target_tv_ids}
        r_map: Dict[str, List[int]] = {tv: [0] * T for tv in ripple_tv_ids}
        for sp in specs or []:
            fid = str(sp.get("flight_id"))
            eb = earliest_bin_by_flight.get(fid, {})
            for tv in target_tv_ids:
                b = eb.get(tv)
                if b is not None and 0 <= int(b) < T:
                    t_map[tv][int(b)] += 1
            for tv in ripple_tv_ids:
                b = eb.get(tv)
                if b is not None and 0 <= int(b) < T:
                    r_map[tv][int(b)] += 1
        target_demands_by_flow[int(f)] = t_map
        ripple_demands_by_flow[int(f)] = r_map

    return target_demands_by_flow, ripple_demands_by_flow


def compute_automatic_rate_adjustment(payload: Mapping[str, Any]) -> Dict[str, Any]:
    # 1) Load artifacts (paths may be overridden in payload)
    idx_path_default, fl_path_default, cap_path_default = _default_paths_from_root()

    explicit_idx = payload.get("indexer_path")
    explicit_fl = payload.get("flights_path")
    if explicit_idx or explicit_fl:
        idx_path = Path(explicit_idx or idx_path_default)
        fl_path = Path(explicit_fl or fl_path_default)
        idx, fl = _load_indexer_and_flights(indexer_path=idx_path, flights_path=fl_path)
    else:
        g_idx, g_fl = get_global_resources()
        if g_idx is not None and g_fl is not None:
            idx, fl = g_idx, g_fl  # type: ignore[assignment]
        else:
            idx_path = Path(idx_path_default)
            fl_path = Path(fl_path_default)
            idx, fl = _load_indexer_and_flights(indexer_path=idx_path, flights_path=fl_path)

    # 1a) Capacities: prefer explicit path; else reuse AppResources cache when available.
    cap_path = payload.get("capacities_path")
    if cap_path:
        capacities_by_tv = build_bin_capacities(str(cap_path), idx)
    else:
        capacities_by_tv = None
        try:
            from server_tailwind.core.resources import get_resources as _get_app_resources  # type: ignore
            _res = _get_app_resources()
            mat = _res.capacity_per_bin_matrix  # shape: [num_tvs, T]
            if mat is not None:
                capacities_by_tv = {}
                for tv_id, row_idx in _res.flight_list.tv_id_to_idx.items():
                    arr = mat[int(row_idx), :]
                    capacities_by_tv[str(tv_id)] = (arr * (arr >= 0.0)).astype(int)
                    capacities_by_tv[str(tv_id)][capacities_by_tv[str(tv_id)] == 0] = 9999
        except Exception:
            capacities_by_tv = None
        if capacities_by_tv is None:
            capacities_by_tv = build_bin_capacities(str(cap_path_default), idx)

    # 2) Parse and validate targets / ripples
    targets_in = payload.get("targets") or {}
    if not isinstance(targets_in, Mapping) or not targets_in:
        raise ValueError("'targets' is required and must be a non-empty mapping")
    target_cells, tvs = _cells_from_ranges(idx, targets_in)

    ripples_in = payload.get("ripples") or {}
    ripple_cells, _ = _cells_from_ranges(idx, ripples_in if isinstance(ripples_in, Mapping) else {})

    # 3) Build flow_map: flight_id -> int(flow_id)
    flows_in = payload.get("flows") or {}
    if not isinstance(flows_in, Mapping):
        raise ValueError("'flows' must be a mapping of flow-id -> [flight_id,...]")
    flow_map = _build_flow_map(flows_in, fl)

    # 4) Controlled volume and requested bins
    hotspot_ids = tvs  # strictly among targets
    flights_by_flow, ctrl_by_flow = prepare_flow_scheduling_inputs(
        flight_list=fl, flow_map=flow_map, hotspot_ids=hotspot_ids
    )

    # 5) Baseline n0 and demand
    T = int(idx.num_time_bins)
    n0: Dict[int, List[int]] = {}
    demand: Dict[int, List[int]] = {}
    for f, specs in flights_by_flow.items():
        arr = [0] * (T + 1)
        for sp in specs or []:
            try:
                rb = int(sp.get("requested_bin", 0))
            except Exception:
                rb = 0
            if 0 <= rb <= T:
                arr[rb] += 1
        n0[int(f)] = arr
        demand[int(f)] = arr[:T]

    # 5a) Auto-ripple override
    try:
        auto_w = int(payload.get("auto_ripple_time_bins", 0))
    except Exception:
        auto_w = 0
    if auto_w > 0:
        ripple_cells = _auto_ripple_cells_from_flows(idx, fl, flow_map.keys(), auto_w)

    # 5b) Per-TV baseline earliest-crossing demand (same as base_evaluation)
    target_tv_ids = list(dict.fromkeys(str(tv) for tv in tvs))
    ripple_tv_ids = sorted({str(tv) for (tv, _b) in ripple_cells})
    target_demands_by_flow, ripple_demands_by_flow = _per_tv_earliest_demands(
        idx=idx,
        fl=fl,
        T=T,
        flights_by_flow=flights_by_flow,
        flow_map=flow_map,
        target_tv_ids=target_tv_ids,
        ripple_tv_ids=ripple_tv_ids,
    )
    # Restrict scoring to TVs of interest only (targets ∪ ripples)
    tv_filter = set(target_tv_ids) | set(ripple_tv_ids)

    # 6) Baseline objective (n0)
    weights = ObjectiveWeights(**(payload.get("weights") or {}))
    J0, comps0, _arts0 = score(
        n0,
        flights_by_flow=flights_by_flow,
        indexer=idx,
        capacities_by_tv=capacities_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        flight_list=fl,
        weights=weights,
        tv_filter=tv_filter,
    )

    # 7) Run simulated annealing to optimize
    sa_kwargs = dict(payload.get("sa_params") or {})
    # Filter only accepted fields to avoid TypeError
    params = SAParams(**{k: v for k, v in sa_kwargs.items() if k in SAParams.__dataclass_fields__})
    n_best, J_star, comps_star, arts_star = run_sa(
        flights_by_flow=flights_by_flow,
        flight_list=fl,
        indexer=idx,
        capacities_by_tv=capacities_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        weights=weights,
        params=params,
        tv_filter=tv_filter,
    )

    # 8) Post-optimization per-flow occupancy on target/ripple TVs
    delays = arts_star.get("delays_min", {})  # flight_id -> minutes

    def _occ_for_flow(fid_list: List[str]) -> Dict[str, List[int]]:
        sub_meta = {fid: fl.flight_metadata[fid] for fid in fid_list if fid in fl.flight_metadata}
        class _SubFL:
            pass
        sub = _SubFL()
        sub.flight_metadata = sub_meta
        delays_sub = {fid: int(delays.get(fid, 0)) for fid in fid_list}
        occ = compute_occupancy(sub, delays_sub, idx, tv_filter=set(target_tv_ids) | set(ripple_tv_ids))
        return {tv: occ.get(tv, []).tolist() for tv in (set(target_tv_ids) | set(ripple_tv_ids))}

    target_occ_by_flow: Dict[int, Dict[str, List[int]]] = {}
    ripple_occ_by_flow: Dict[int, Dict[str, List[int]]] = {}
    for f, specs in flights_by_flow.items():
        fids = [str(sp.get("flight_id")) for sp in (specs or [])]
        occ = _occ_for_flow(fids)
        target_occ_by_flow[int(f)] = {tv: occ.get(tv, [0] * T) for tv in target_tv_ids}
        ripple_occ_by_flow[int(f)] = {tv: occ.get(tv, [0] * T) for tv in ripple_tv_ids}

    # 9) Assemble response
    flows_out: List[Dict[str, Any]] = []
    for f in sorted(flights_by_flow.keys(), key=lambda x: int(x)):
        n_opt_arr = n_best.get(f)
        n_opt_list = (
            list(map(int, n_opt_arr.tolist())) if hasattr(n_opt_arr, "tolist") else list(map(int, (n_opt_arr or [])))
        )
        flows_out.append(
            {
                "flow_id": int(f),
                "controlled_volume": (str(ctrl_by_flow.get(f)) if ctrl_by_flow.get(f) is not None else None),
                "n0": n0[int(f)],
                "demand": demand[int(f)],
                "n_opt": n_opt_list,
                "target_demands": target_demands_by_flow.get(int(f), {}),
                "ripple_demands": ripple_demands_by_flow.get(int(f), {}),
                "target_occupancy_opt": target_occ_by_flow.get(int(f), {}),
                "ripple_occupancy_opt": ripple_occ_by_flow.get(int(f), {}),
            }
        )

    improvement_abs = float(J0 - J_star)
    improvement_pct = (improvement_abs / J0 * 100.0) if J0 != 0 else 0.0
    # Per-flight delays (minutes) under the optimized schedule
    delays_out: Dict[str, int] = {str(fid): int(v) for fid, v in (delays or {}).items()}
    return {
        "num_time_bins": T,
        "tvs": list(hotspot_ids),
        "target_cells": [(str(tv), int(b)) for (tv, b) in target_cells],
        "ripple_cells": [(str(tv), int(b)) for (tv, b) in ripple_cells],
        "flows": flows_out,
        "delays_min": delays_out,
        "objective_baseline": {"score": float(J0), "components": comps0},
        "objective_optimized": {"score": float(J_star), "components": comps_star},
        "improvement": {"absolute": improvement_abs, "percent": round(improvement_pct, 2)},
        "weights_used": asdict(weights),
        "sa_params_used": asdict(params),
    }


__all__ = ["compute_automatic_rate_adjustment"]
