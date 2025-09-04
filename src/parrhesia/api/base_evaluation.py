from __future__ import annotations

"""
Base evaluation for flows: build scheduling inputs, baseline n0, and score.

Implements the plan in docs/plans/plan_post_flow_extraction_reprepare_flow_scheduling_inputs.md.

Entry point: compute_base_evaluation(payload: dict) -> dict

Input payload schema (keys are optional unless noted):
  - flows (required): mapping of flow-id -> list of flight IDs
  - targets (required): mapping of TV id -> {"from": "HH:MM:SS", "to": "HH:MM:SS"}
  - ripples (optional): mapping like targets; used for reduced weights
  - indexer_path (optional): path to tvtw_indexer.json
  - flights_path (optional): path to so6_occupancy_matrix_with_times.json
  - capacities_path (optional): path to capacities GeoJSON
  - weights (optional): partial overrides for ObjectiveWeights

Returns JSON with fields:
  - num_time_bins, tvs, target_cells, ripple_cells
  - flows: [{flow_id, controlled_volume, n0, demand, target_demands, ripple_demands}]
  - objective: { score, components }
  - weights_used: effective weights
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from datetime import datetime

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from parrhesia.optim.capacity import build_bin_capacities
from parrhesia.optim.objective import ObjectiveWeights, score
from parrhesia.optim.sa_optimizer import prepare_flow_scheduling_inputs
from .flows import _load_indexer_and_flights  # reuse helper for defaults
from .resources import get_global_resources


def _parse_time_hms(s: str) -> datetime:
    """Parse HH:MM[:SS] into a datetime on an arbitrary reference day."""
    ss = str(s).strip()
    # Accept HH:MM as well; default seconds to 0
    fmt = "%H:%M:%S" if ss.count(":") == 2 else "%H:%M"
    dt = datetime.strptime(ss, fmt)
    # Normalize to a fixed date (year/month/day irrelevant for binning)
    return dt.replace(year=2025, month=1, day=1)


def _cells_from_ranges(
    idx: TVTWIndexer, ranges: Mapping[str, Mapping[str, Any]]
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Convert {tv: {from,to}} mapping to explicit (tv, bin) cells.

    Returns (cells, tvs_considered). Invalid TVs are ignored.
    """
    cells: List[Tuple[str, int]] = []
    tvs: List[str] = []
    for tv, win in (ranges or {}).items():
        tv_id = str(tv)
        if tv_id not in idx.tv_id_to_idx:
            continue
        try:
            t_from = _parse_time_hms(str(win.get("from")))
            t_to = _parse_time_hms(str(win.get("to")))
        except Exception:
            # Skip malformed
            continue
        bins = idx.bin_range_for_interval(t_from, t_to)
        for b in bins:
            cells.append((tv_id, int(b)))
        tvs.append(tv_id)
    return cells, tvs


def _default_paths_from_root() -> Tuple[Path, Path, Path]:
    # Mirrors logic in flows._load_indexer_and_flights
    here = Path(__file__).resolve()
    root = None
    for p in here.parents:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            root = p
            break
    if root is None:
        root = here.parents[3]
    idx_path = root / "data" / "tailwind" / "tvtw_indexer.json"
    fl_path = root / "data" / "tailwind" / "so6_occupancy_matrix_with_times.json"
    cap_path = root / "data" / "cirrus" / "wxm_sm_ih_maxpool.geojson"
    return idx_path, fl_path, cap_path


def compute_base_evaluation(payload: Mapping[str, Any]) -> Dict[str, Any]:
    # 1) Load artifacts (paths may be overridden in payload)
    idx_path_default, fl_path_default, cap_path_default = _default_paths_from_root()

    # Prefer explicit paths when provided; otherwise use globally registered
    # resources if available (set at server startup), else fall back to defaults.
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
        # Try to use shared resources from server_tailwind if available
        capacities_by_tv = None
        try:
            from server_tailwind.core.resources import get_resources as _get_app_resources  # type: ignore

            _res = _get_app_resources()
            mat = _res.capacity_per_bin_matrix  # shape: [num_tvs, T]
            if mat is not None:
                # Build mapping aligned to indexer tv ordering
                capacities_by_tv = {}
                # Ensure we iterate TVs in the indexer's map for consistency
                for tv_id, row_idx in _res.flight_list.tv_id_to_idx.items():
                    arr = mat[int(row_idx), :]
                    # Replace missing capacity markers (-1) with zeros
                    capacities_by_tv[str(tv_id)] = (arr * (arr >= 0.0)).astype(int)
                    capacities_by_tv[str(tv_id)][capacities_by_tv[str(tv_id)] == 0] = 9999
        except Exception:
            capacities_by_tv = None

        if capacities_by_tv is None:
            # Fallback to project default path
            capacities_by_tv = build_bin_capacities(str(cap_path_default), idx)

    # 2) Parse and validate targets / ripples
    targets_in = payload.get("targets") or {}
    if not isinstance(targets_in, Mapping) or not targets_in:
        raise ValueError("'targets' is required and must be a non-empty mapping")
    target_cells, tvs = _cells_from_ranges(idx, targets_in)  # tvs are from targets

    ripples_in = payload.get("ripples") or {}
    ripple_cells, _ = _cells_from_ranges(idx, ripples_in if isinstance(ripples_in, Mapping) else {})

    # 3) Build flow_map: flight_id -> int(flow_id)
    flows_in = payload.get("flows") or {}
    if not isinstance(flows_in, Mapping):
        raise ValueError("'flows' must be a mapping of flow-id -> [flight_id,...]")
    # Determine deterministic int IDs for flow keys
    flow_key_to_int: Dict[str, int] = {}
    next_id = 0
    for k in sorted((str(x) for x in flows_in.keys()), key=str):
        try:
            flow_key_to_int[k] = int(k)
        except Exception:
            flow_key_to_int[k] = next_id
            next_id += 1

    # Build map, ignoring unknown flight IDs
    flow_map: Dict[str, int] = {}
    for k, flights in flows_in.items():
        fid = flow_key_to_int[str(k)]
        seq: Iterable[Any] = flights if isinstance(flights, (list, tuple)) else []
        for flid in seq:
            sfl = str(flid)
            if sfl in fl.flight_metadata:
                flow_map[sfl] = fid
            else:
                # Unknown flight id; skip
                pass

    # 4) Controlled volume selection and requested bins
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

    # 5a) Per-TV demand vectors for targets and ripples (length T, no overflow)
    target_tv_ids = list(dict.fromkeys(str(tv) for tv in tvs))  # preserve order from targets
    ripple_tv_ids = sorted({str(tv) for (tv, _b) in ripple_cells})

    # Precompute earliest crossing bin per flight for union(targets ∪ ripples)
    tv_union: set[str] = set(target_tv_ids) | set(ripple_tv_ids)
    earliest_bin_by_flight: Dict[str, Dict[str, int]] = {}
    if tv_union:
        allowed = set(str(x) for x in tv_union)
        for fid, meta in fl.flight_metadata.items():
            if str(fid) not in flow_map:  # limit to flights considered in flows
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
                cur = d.get(s_tv)
                tb = int(tbin)
                if cur is None or tb < cur:
                    d[s_tv] = tb
            earliest_bin_by_flight[str(fid)] = d

    # Build per-flow per-TV demand arrays
    target_demands_by_flow: Dict[int, Dict[str, List[int]]] = {}
    ripple_demands_by_flow: Dict[int, Dict[str, List[int]]] = {}
    for f, specs in flights_by_flow.items():
        t_map: Dict[str, List[int]] = {tv: [0] * T for tv in target_tv_ids}
        r_map: Dict[str, List[int]] = {tv: [0] * T for tv in ripple_tv_ids}
        for sp in specs or []:
            fid = str(sp.get("flight_id"))
            eb = earliest_bin_by_flight.get(fid, {})
            # Targets
            for tv in target_tv_ids:
                b = eb.get(tv)
                if b is not None and 0 <= int(b) < T:
                    t_map[tv][int(b)] += 1
            # Ripples
            for tv in ripple_tv_ids:
                b = eb.get(tv)
                if b is not None and 0 <= int(b) < T:
                    r_map[tv][int(b)] += 1
        target_demands_by_flow[int(f)] = t_map
        ripple_demands_by_flow[int(f)] = r_map

    # 6) Score baseline
    weights = ObjectiveWeights(**(payload.get("weights") or {}))
    J, components, _arts = score(
        n0,
        flights_by_flow=flights_by_flow,
        indexer=idx,
        capacities_by_tv=capacities_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        flight_list=fl,
        weights=weights,
    )

    # 7) Assemble response
    flows_out: List[Dict[str, Any]] = []
    for f in sorted(flights_by_flow.keys(), key=lambda x: int(x)):
        flows_out.append(
            {
                "flow_id": int(f),
                "controlled_volume": (str(ctrl_by_flow.get(f)) if ctrl_by_flow.get(f) is not None else None),
                "n0": n0[int(f)],
                "demand": demand[int(f)],
                "target_demands": target_demands_by_flow.get(int(f), {}),
                "ripple_demands": ripple_demands_by_flow.get(int(f), {}),
            }
        )

    return {
        "num_time_bins": T,
        "tvs": list(hotspot_ids),
        "target_cells": [(str(tv), int(b)) for (tv, b) in target_cells],
        "ripple_cells": [(str(tv), int(b)) for (tv, b) in ripple_cells],
        "flows": flows_out,
        "objective": {"score": float(J), "components": components},
        "weights_used": asdict(weights),
    }


__all__ = ["compute_base_evaluation"]
