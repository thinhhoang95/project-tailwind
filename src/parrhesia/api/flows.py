from __future__ import annotations

"""
Flow computation wrapper for the /flows API.

This module loads the TVTW indexer and the full flight list JSON, collects
flights relevant to the requested traffic volumes and time bins, clusters them
into flows, selects a controlled volume per flow, and assembles a response that
includes per-flow demand profiles and per-flight details.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from datetime import datetime, timedelta

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from parrhesia.flows.flow_pipeline import (
    collect_hotspot_flights,
    build_global_flows,
)
from parrhesia.optim.sa_optimizer import prepare_flow_scheduling_inputs
from .resources import get_global_resources


def _project_root_from_here() -> Path:
    """Best-effort project root discovery.

    This module now lives under `src/parrhesia/api/flows.py` as part of the
    `parrhesia` package. We try to locate the repository root by walking up
    until we find a marker like `pyproject.toml` or `.git`. Fallback is three
    levels up from this file (api -> parrhesia -> src -> project root).
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # Fallback: .../src/parrhesia/api -> parents[3] == project root
    try:
        return here.parents[3]
    except Exception:
        return here.parent.parent.parent


def _load_indexer_and_flights(
    *,
    indexer_path: Optional[Path] = None,
    flights_path: Optional[Path] = None,
) -> Tuple[TVTWIndexer, FlightList]:
    """
    Load the TVTW indexer and FlightList.

    Preference order:
    1) Use globally-registered resources (set via parrhesia.api.resources.set_global_resources)
    2) If explicit paths are provided, load from those
    3) Fallback to project-relative defaults under data/tailwind
    """
    # 1) Try shared resources first
    g_idx, g_fl = get_global_resources()
    if g_idx is not None and g_fl is not None:
        return g_idx, g_fl  # type: ignore[return-value]

    # 2) Explicit paths take precedence when provided
    if indexer_path is not None and flights_path is not None:
        idx = TVTWIndexer.load(str(indexer_path))
        fl = FlightList.from_json(str(flights_path), idx)
        return idx, fl

    # 3) Fallback to project defaults
    root = _project_root_from_here()
    print(f'Warning: Falling back to project defaults under data/tailwind')
    idx_path = indexer_path or (root / "data" / "tailwind" / "tvtw_indexer.json")
    fl_path = flights_path or (root / "data" / "tailwind" / "so6_occupancy_matrix_with_times.json")
    idx = TVTWIndexer.load(str(idx_path))
    fl = FlightList.from_json(str(fl_path), idx)
    return idx, fl


def _demand_profile(
    flights_specs: Sequence[Mapping[str, Any]],
    *,
    num_time_bins: int,
) -> List[int]:
    """
    Build demand profile array (length == num_time_bins) by counting flights per
    requested bin.
    """
    arr = [0] * int(num_time_bins)
    for sp in flights_specs or []:
        try:
            rb = int(sp.get("requested_bin", 0))
        except Exception:
            rb = 0
        if 0 <= rb < num_time_bins:
            arr[rb] += 1
    return arr


def _earliest_crossing_time_for_requested_bin(
    *,
    flight_id: str,
    requested_bin: int,
    allowed_tvs: Optional[Iterable[str]],
    flight_list: FlightList,
) -> Optional[datetime]:
    """
    Find earliest crossing datetime for a flight at the requested bin. If
    `allowed_tvs` is provided, restrict to those TVs; otherwise consider any of
    the TVs in the request context.
    """
    meta = getattr(flight_list, "flight_metadata", {}).get(str(flight_id))
    if not meta:
        return None
    takeoff: datetime = meta.get("takeoff_time")  # type: ignore[assignment]
    if not isinstance(takeoff, datetime):
        return None
    allowed = set(str(tv) for tv in (allowed_tvs or [])) if allowed_tvs is not None else None
    # Decoder for tvtw_index -> (tv_id, bin)
    idx_obj = getattr(flight_list, "indexer", None)
    if idx_obj is not None and hasattr(idx_obj, "get_tvtw_from_index"):
        decode = lambda j: idx_obj.get_tvtw_from_index(int(j))
    else:
        bins_per_tv = int(getattr(flight_list, "num_time_bins_per_tv"))
        idx_to_tv_id = getattr(flight_list, "idx_to_tv_id")
        def decode(j: int):
            tv_idx = int(j) // int(bins_per_tv)
            tbin = int(j) % int(bins_per_tv)
            tv_id = str(idx_to_tv_id.get(int(tv_idx))) if isinstance(idx_to_tv_id, dict) else None
            if tv_id is None:
                return None
            return tv_id, tbin
    best_dt: Optional[datetime] = None
    for iv in meta.get("occupancy_intervals", []) or []:
        try:
            tvtw_idx = int(iv.get("tvtw_index"))
        except Exception:
            continue
        decoded = decode(tvtw_idx)
        if not decoded:
            continue
        tv_id, tbin = decoded
        if int(tbin) != int(requested_bin):
            continue
        if allowed is not None and str(tv_id) not in allowed:
            continue
        # Convert entry time in seconds to a datetime using timedelta
        raw_entry_s = iv.get("entry_time_s", 0)
        try:
            entry_s = float(raw_entry_s)
        except Exception:
            entry_s = 0.0
        entry_dt = takeoff + timedelta(seconds=entry_s)
        if best_dt is None or entry_dt < best_dt:
            best_dt = entry_dt
    return best_dt


def compute_flows(
    *,
    tvs: Sequence[str],
    timebins: Optional[Sequence[int]] = None,
    indexer_path: Optional[str] = None,
    flights_path: Optional[str] = None,
    threshold: Optional[float] = None,
    resolution: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Core implementation for the /flows endpoint.

    Parameters
    - tvs: list of traffic volume IDs (strings).
    - timebins: optional list of time-bin indices (global filter applied to all TVs).
    - threshold: optional Jaccard cutoff in [0,1] for Leiden graph.
    - resolution: optional Leiden resolution parameter (>0).

    Returns a JSON-serializable dict with:
    - num_time_bins
    - tvs, timebins
    - flows: list of { flow_id, controlled_volume, demand, flights: [...] }
      where each flight is { flight_id, requested_bin, earliest_crossing_time }
    """
    if not tvs:
        return {
            "num_time_bins": 0,
            "tvs": [],
            "timebins": [],
            "flows": [],
        }

    # Load artifacts
    idx_path = Path(indexer_path) if indexer_path else None
    fl_path = Path(flights_path) if flights_path else None
    idx, fl = _load_indexer_and_flights(indexer_path=idx_path, flights_path=fl_path)

    hotspot_ids = [str(tv).strip() for tv in tvs if str(tv).strip()]
    windows = [int(x) for x in (timebins or [])] if timebins is not None else None

    # Union selection and flow partitioning
    union_ids, _meta = collect_hotspot_flights(fl, hotspot_ids, active_windows=windows)
    # Leiden parameters (fallback to defaults if not provided)
    _thr = 0.1 if threshold is None else float(threshold)
    _res = 1.0 if resolution is None else float(resolution)

    flow_map = build_global_flows(
        fl,
        union_ids,
        hotspots=hotspot_ids,
        trim_policy="earliest_hotspot",
        leiden_params={"threshold": _thr, "resolution": _res, "seed": 0},
        direction_opts={
            "mode": "coord_cosine",
            # tv_centroids is optional; if not provided here, no reweighting occurs.
            # Server layer can inject a mapping {tv_id: (lat, lon)} when available.
        },
    )

    # Controlled volume and requested bins per flight in each flow
    flights_by_flow, ctrl_by_flow = prepare_flow_scheduling_inputs(
        flight_list=fl,
        flow_map=flow_map,
        hotspot_ids=hotspot_ids,
    )

    # Assemble response
    T = int(idx.num_time_bins)
    out_flows: List[Dict[str, Any]] = []
    for flow_id, specs in sorted(flights_by_flow.items(), key=lambda kv: int(kv[0])):
        ctrl = ctrl_by_flow.get(flow_id)
        # Flights with earliest crossing times
        flights_payload: List[Dict[str, Any]] = []
        for sp in specs:
            fid = str(sp.get("flight_id"))
            try:
                rb = int(sp.get("requested_bin", 0))
            except Exception:
                rb = 0
            # Prefer earliest crossing at the controlled volume; fallback to any hotspot
            allowed_tvs = [ctrl] if ctrl else hotspot_ids
            dt = _earliest_crossing_time_for_requested_bin(
                flight_id=fid,
                requested_bin=rb,
                allowed_tvs=allowed_tvs,
                flight_list=fl,
            )
            flights_payload.append(
                {
                    "flight_id": fid,
                    "requested_bin": rb,
                    "earliest_crossing_time": (dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, datetime) else None),
                }
            )

        out_flows.append(
            {
                "flow_id": int(flow_id),
                "controlled_volume": str(ctrl) if ctrl is not None else None,
                "demand": _demand_profile(specs, num_time_bins=T),
                "flights": flights_payload,
            }
        )

    return {
        "num_time_bins": T,
        "tvs": hotspot_ids,
        "timebins": list(windows) if isinstance(windows, list) else [],
        "flows": out_flows,
    }


__all__ = ["compute_flows"]
