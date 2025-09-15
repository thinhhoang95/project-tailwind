from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from typing import Any as _Any  # avoid heavy imports at module import time


def _reshape_by_tv(vec: np.ndarray, num_tvs: int, T: int) -> np.ndarray:
    """Reshape a length (num_tvs*T) vector to matrix [num_tvs, T]."""
    arr = np.asarray(vec)
    if arr.size != num_tvs * T:
        raise ValueError(f"Vector length {arr.size} != num_tvs*T ({num_tvs*T})")
    return arr.reshape((num_tvs, T))


def _hour_of_bin(T: int, bins_per_hour: int) -> np.ndarray:
    """Vector of hour index per bin [0..T-1]."""
    h = np.arange(T, dtype=np.int32) // int(bins_per_hour)
    h[h >= 24] = 23  # clip safety
    return h


def build_base_caches(
    flight_list: _Any,
    capacities_by_tv: Mapping[str, np.ndarray],
    indexer: _Any,
) -> Dict[str, Any]:
    """
    Build base caches used by feature engineering.

    Returns a dict with:
      - 'occ_base': 1D np.ndarray length V*T (pre-regulation occupancy counts)
      - 'num_tvs': int, 'T': int, 'bins_per_hour': int
      - 'tv_row_of_tvtw': 1D np.ndarray length V*T mapping tvtw -> row index
      - 'hour_of_tvtw': 1D np.ndarray length V*T mapping tvtw -> hour index [0..23]
      - 'cap_per_bin': 1D np.ndarray length V*T with per-bin capacity (hourly/ bins_per_hour)
      - 'hourly_capacity_matrix': np.ndarray shape [V, 24]
      - 'hourly_occ_base': np.ndarray shape [V, 24]
      - 'slack_per_bin': 1D np.ndarray length V*T with max(hourly_capacity − rolling_occ, 0)
      - 'slack_per_bin_matrix': np.ndarray shape [V, T]
      - 'rolling_occ_by_bin': np.ndarray shape [V, T] with forward-looking rolling-hour occupancy
      - 'excess_per_bin_matrix': np.ndarray shape [V, T] with max(rolling_occ - hourly_cap, 0)
      - 'hourly_excess_bool': np.ndarray shape [V, T] with (rolling_hour_occ - hourly_cap > 0)
    """
    num_tvs = len(flight_list.tv_id_to_idx)
    T = int(indexer.num_time_bins)
    bins_per_hour = int(60 // int(indexer.time_bin_minutes))
    if bins_per_hour <= 0:
        raise ValueError("Invalid time_bin_minutes; bins_per_hour <= 0")

    # Base occupancy per TVTW
    occ_base = np.asarray(flight_list.get_total_occupancy_by_tvtw(), dtype=np.float64)
    if occ_base.size != num_tvs * T:
        # Some FlightList variants track num_tvtws; use that to infer V
        raise ValueError("Occupancy vector size mismatch with indexer bins")

    # Build per-TV hourly capacity matrix from per-bin capacities
    hourly_capacity_matrix = np.zeros((num_tvs, 24), dtype=np.float64)
    cap_per_bin = np.zeros(num_tvs * T, dtype=np.float64)
    for tv_id, row in flight_list.tv_id_to_idx.items():
        arr = np.asarray(capacities_by_tv.get(tv_id, np.zeros(T, dtype=np.float64)), dtype=np.float64)
        # `arr` is expected to repeat the hourly capacity value across bins within the hour
        for h in range(24):
            start = h * bins_per_hour
            if start >= T:
                break
            hourly_capacity_matrix[int(row), h] = float(arr[start]) if arr.size > start else 0.0
            end = min(start + bins_per_hour, T)
            per_bin_val = hourly_capacity_matrix[int(row), h] / float(bins_per_hour)
            cap_per_bin[row * T + start : row * T + end] = per_bin_val

    # Hour index per bin and tv_row_of_tvtw mapping
    hour_bins = _hour_of_bin(T, bins_per_hour)
    hour_of_tvtw = np.tile(hour_bins, reps=num_tvs)
    tv_row_of_tvtw = np.repeat(np.arange(num_tvs, dtype=np.int32), repeats=T)

    # Base hourly occupancy and per-bin capacity aligned by hour
    occ_mat = _reshape_by_tv(occ_base, num_tvs, T)
    hourly_occ_base = np.zeros((num_tvs, 24), dtype=np.float64)
    for h in range(24):
        start = h * bins_per_hour
        end = min(start + bins_per_hour, T)
        if start >= end:
            continue
        hourly_occ_base[:, h] = occ_mat[:, start:end].sum(axis=1)

    cap_by_bin_hour = hourly_capacity_matrix[:, hour_bins]

    # Rolling-hour occupancy and hourly excess indicator per bin
    # Compute forward-looking rolling sum of width K=bins_per_hour
    K = bins_per_hour
    # cumsum trick per row
    csum = np.cumsum(occ_mat, axis=1)
    # prepend zero column
    csum = np.concatenate([np.zeros((num_tvs, 1), dtype=np.float64), csum], axis=1)
    idx = np.arange(T, dtype=np.int32)
    end_idx = np.minimum(idx + K, T)
    roll = csum[:, end_idx] - csum[:, :T]
    hourly_excess_bool = (roll - cap_by_bin_hour) > 0.0
    # Exceedance magnitude per bin (rolling occupancy minus hourly capacity, clamped at 0)
    excess_per_bin_matrix = np.maximum(roll - cap_by_bin_hour, 0.0)

    # Slack per bin matrix [V, T]: rolling-hour slack relative to hourly capacity
    slack_per_bin_matrix = np.maximum(cap_by_bin_hour - roll, 0.0)
    # Flatten to [V*T]
    slack_per_bin = slack_per_bin_matrix.reshape(-1)

    # Slack per bin matrix [V, T] already computed above

    return {
        "occ_base": occ_base.astype(np.float64, copy=False),
        "num_tvs": int(num_tvs),
        "T": int(T),
        "bins_per_hour": int(bins_per_hour),
        "tv_row_of_tvtw": tv_row_of_tvtw.astype(np.int32, copy=False),
        "hour_of_tvtw": np.tile(_hour_of_bin(T, bins_per_hour), reps=num_tvs).astype(np.int32, copy=False),
        "cap_per_bin": cap_per_bin.astype(np.float64, copy=False),
        "hourly_capacity_matrix": hourly_capacity_matrix.astype(np.float64, copy=False),
        "hourly_occ_base": hourly_occ_base.astype(np.float64, copy=False),
        "slack_per_bin": slack_per_bin.astype(np.float64, copy=False),
        "slack_per_bin_matrix": slack_per_bin_matrix.astype(np.float64, copy=False),
        "rolling_occ_by_bin": roll.astype(np.float64, copy=False),
        "excess_per_bin_matrix": excess_per_bin_matrix.astype(np.float64, copy=False),
        "hourly_excess_bool": hourly_excess_bool.astype(np.bool_, copy=False),
    }


def attention_mask_from_cells(
    hotspot: Tuple[str, int],
    ripple_cells: Optional[Sequence[Tuple[str, int]]] = None,
    weights: Optional[Sequence[float]] = None,
    tv_id_to_idx: Optional[Mapping[str, int]] = None,
    T: Optional[int] = None,
) -> Dict[Tuple[int, int], float]:
    """
    Build a sparse attention mask θ for given (tv_id, bin) cells.

    Returns a dict keyed by (tv_row, bin) -> weight. The hotspot cell gets
    weight 1.0 if not provided; ripple cells may have provided weights or 0.5
    by default.
    """
    items: List[Tuple[Tuple[int, int], float]] = []
    if tv_id_to_idx is None or T is None:
        # Build minimal mapping later in consumers by checking keys
        tv_map = None
    else:
        tv_map = {str(k): int(v) for k, v in tv_id_to_idx.items()}

    def _row(tv: str) -> Optional[int]:
        if tv_map is None:
            return None
        return tv_map.get(str(tv))

    h_tv, h_bin = hotspot
    r = _row(h_tv)
    if r is not None and h_bin is not None:
        items.append(((int(r), int(h_bin)), 1.0))

    if ripple_cells:
        if weights is None or len(weights) != len(ripple_cells):
            w = [0.5] * len(ripple_cells)
        else:
            w = [float(x) for x in weights]
        for (tv, b), ww in zip(ripple_cells, w):
            rr = _row(tv)
            if rr is not None:
                items.append(((int(rr), int(b)), float(ww)))

    return dict(items)
