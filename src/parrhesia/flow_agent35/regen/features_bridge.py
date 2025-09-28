"""Bridge utilities around :mod:`parrhesia.metaopt` feature extraction."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from parrhesia.metaopt.feats.flow_features import FlowFeaturesExtractor

def build_flow_features_extractor(
    indexer: Any,
    flight_list: Any,
    capacities_by_tv: Mapping[str, np.ndarray],
    travel_minutes_map: Mapping[str, Mapping[str, float]],
    *,
    autotrim_from_ctrl_to_hotspot: bool = False,
) -> FlowFeaturesExtractor:
    """Instantiate a :class:`FlowFeaturesExtractor` with shared caches."""

    return FlowFeaturesExtractor(
        indexer=indexer,
        flight_list=flight_list,
        capacities_by_tv=capacities_by_tv,
        travel_minutes_map=travel_minutes_map,
        autotrim_from_ctrl_to_hotspot=autotrim_from_ctrl_to_hotspot,
    )


def extract_features_for_flows(
    extractor: FlowFeaturesExtractor,
    hotspot_tv: str,
    timebins: Sequence[int],
    *,
    flows_payload: Optional[Mapping[str, Any]] = None,
    direction_opts: Optional[Mapping[str, Any]] = None,
) -> Dict[int, FlowFeatures]:
    """Compute flow features for the provided hotspot window."""

    return extractor.compute_for_hotspot(
        hotspot_tv=hotspot_tv,
        timebins=timebins,
        flows_payload=flows_payload,
        direction_opts=direction_opts,
    )


def _extract_requested_bin_from_metadata(
    flight_id: str,
    control_tv: Optional[str],
    *,
    flight_list: Any,
    indexer: Any,
) -> Optional[int]:
    """Best-effort reconstruction of the control-time requested bin for a flight."""

    if not control_tv:
        return None
    meta = getattr(flight_list, "flight_metadata", {}) or {}
    entry = meta.get(str(flight_id))
    if not entry:
        return None
    best: Optional[int] = None
    for iv in entry.get("occupancy_intervals", []) or []:
        tvtw_idx = iv.get("tvtw_index")
        if tvtw_idx is None:
            continue
        try:
            tvtw = indexer.get_tvtw_from_index(int(tvtw_idx))
        except Exception:
            continue
        if not tvtw:
            continue
        tv_id, tbin = tvtw
        if str(tv_id) != str(control_tv):
            continue
        try:
            tbin_int = int(tbin)
        except Exception:
            continue
        if best is None or tbin_int < best:
            best = tbin_int
    return best


def baseline_demand_by_flow_from_payload(
    indexer: Any,
    *,
    flows_payload: Optional[Mapping[str, Any]] = None,
    flow_to_flights: Optional[Mapping[str, Sequence[str]]] = None,
    flight_list: Optional[Any] = None,
    features: Optional[Mapping[int, Any]] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Sequence[Mapping[str, Any]]]]:
    """Construct per-flow demand histograms aligned with the indexer bins."""

    T = int(getattr(indexer, "num_time_bins"))
    demand: Dict[int, np.ndarray] = {}
    flights_by_flow: Dict[int, Sequence[Mapping[str, Any]]] = {}

    if flows_payload:
        for fobj in flows_payload.get("flows", []) or []:
            try:
                flow_id = int(fobj.get("flow_id"))
            except Exception:
                continue
            specs_out = []
            arr = np.zeros(T + 1, dtype=np.int64)
            for sp in fobj.get("flights", []) or []:
                fid = sp.get("flight_id")
                if fid is None:
                    continue
                try:
                    rbin = int(sp.get("requested_bin", 0))
                except Exception:
                    rbin = 0
                rbin = max(0, min(T, rbin))
                arr[rbin] += 1
                specs_out.append({"flight_id": str(fid), "requested_bin": rbin})
            if arr.sum() == 0:
                continue
            demand[int(flow_id)] = arr
            flights_by_flow[int(flow_id)] = tuple(specs_out)
        if demand:
            return demand, flights_by_flow

    if not flow_to_flights:
        raise ValueError("Missing flows payload and flow_to_flights mapping; cannot reconstruct demand")
    if features is None:
        raise ValueError("features mapping is required when reconstructing from flow_to_flights")
    if flight_list is None:
        raise ValueError("flight_list is required when reconstructing from flow_to_flights")

    for key, flight_ids in flow_to_flights.items():
        try:
            flow_id = int(key)
        except Exception:
            continue
        flow_score = features.get(flow_id) if features is not None else None
        control_tv = getattr(flow_score, "control_tv_id", None)
        arr = np.zeros(T + 1, dtype=np.int64)
        specs_out = []
        for fid in flight_ids or []:
            rbin = _extract_requested_bin_from_metadata(str(fid), control_tv, flight_list=flight_list, indexer=indexer)
            if rbin is None:
                continue
            rbin = max(0, min(T, int(rbin)))
            arr[rbin] += 1
            specs_out.append({"flight_id": str(fid), "requested_bin": rbin})
        if arr.sum() == 0:
            continue
        demand[flow_id] = arr
        flights_by_flow[flow_id] = tuple(specs_out)

    if not demand:
        raise ValueError("Unable to reconstruct baseline demand for any flow")

    return demand, flights_by_flow


def coverage_i(
    *,
    tGl: int,
    tGu: int,
    timebins_h: Sequence[int],
    indexer: Any,
) -> float:
    """Compute the fraction of hotspot bins covered by the control window."""

    if not timebins_h:
        return 0.0
    T = int(getattr(indexer, "num_time_bins"))
    start = max(0, int(min(timebins_h)))
    end = min(T - 1, int(max(timebins_h)))
    ctl_start = max(0, int(tGl))
    ctl_end = min(T - 1, int(tGu))
    overlap_start = max(start, ctl_start)
    overlap_end = min(end, ctl_end)
    if overlap_end < overlap_start:
        return 0.0
    covered = overlap_end - overlap_start + 1
    return float(covered) / float(len(timebins_h))
