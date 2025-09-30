"""Hotspot exceedance helpers."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from parrhesia.metaopt.base_caches import build_base_caches


def compute_hotspot_exceedance(
    indexer: Any,
    flight_list: Any,
    capacities_by_tv: Mapping[str, np.ndarray],
    hotspot_tv: str,
    timebins_h: Sequence[int],
    *,
    caches: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute rolling-hour exceedance statistics for a hotspot."""

    if not timebins_h:
        raise ValueError("timebins_h is empty; hotspot window is required")
    caches_local = dict(caches or {})
    if not caches_local:
        caches_local = build_base_caches(
            flight_list=flight_list,
            capacities_by_tv=capacities_by_tv,
            indexer=indexer,
        )
    rolling_occ = np.asarray(caches_local.get("rolling_occ_by_bin"))
    hourly_capacity_matrix = np.asarray(caches_local.get("hourly_capacity_matrix"))
    bins_per_hour = int(caches_local.get("bins_per_hour", int(indexer.rolling_window_size())))
    row_map = getattr(flight_list, "tv_id_to_idx", {})
    if str(hotspot_tv) not in row_map:
        raise KeyError(f"Hotspot TV {hotspot_tv} not present in flight list")
    # We only compute the exceedance for the hotspot TV
    row = int(row_map[str(hotspot_tv)])
    occ_row = rolling_occ[row]
    cap_row = hourly_capacity_matrix[row]
    timebins = [int(b) for b in timebins_h]
    D_vec: list[float] = [] # D_vec only contains the exceedance for the hotspot TV and the designated timebins
    for b in timebins:
        if b < 0 or b >= occ_row.shape[0]:
            continue
        hour_idx = min(cap_row.shape[0] - 1, max(0, b // bins_per_hour))
        hourly_cap = float(cap_row[hour_idx])
        exceed = max(0.0, float(occ_row[b]) - hourly_cap)
        D_vec.append(exceed)
    if not D_vec:
        D_vec = [0.0]
    arr = np.asarray(D_vec, dtype=np.float64)
    D_total = float(arr.sum())
    D_peak = float(arr.max()) if arr.size else 0.0
    D_q95 = float(np.quantile(arr, 0.95)) if arr.size else 0.0
    return {
        "D_vec": arr,
        "D_total": D_total,
        "D_peak": D_peak,
        "D_q95": D_q95,
    }
