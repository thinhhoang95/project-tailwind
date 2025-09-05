"""
Capacity utilities: build per-TV per-bin capacity series and rolling sums.

As per docs/plans/flowful_sa.md, we parse hourly capacity strings from a
GeoJSON, map them to the TVTW binning, and provide a helper for rolling-hour
occupancy sums.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any
import json
import re
import numpy as np

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


_HOUR_RANGE_RE = re.compile(
    r"^\s*(?P<h1>\d{1,2})(?::(?P<m1>\d{2}))?\s*-\s*(?P<h2>\d{1,2})(?::(?P<m2>\d{2}))?\s*$"
)


def _parse_hour_range_to_bins(key: str, indexer: TVTWIndexer) -> Tuple[int, int]:
    """
    Parse an hour range like "6:00-7:00" into [start_bin, end_bin) indices.

    If minutes are omitted (e.g., "6-7"), they default to ":00". The mapping
    assigns the capacity value to all bins whose start time falls within the
    half-open interval [start, end) in minutes since midnight.
    """
    m = _HOUR_RANGE_RE.match(str(key))
    if not m:
        raise ValueError(f"Unrecognized hour range format: {key!r}")
    h1 = int(m.group("h1"))
    h2 = int(m.group("h2"))
    m1 = int(m.group("m1")) if m.group("m1") is not None else 0
    m2 = int(m.group("m2")) if m.group("m2") is not None else 0
    start_min = h1 * 60 + m1
    end_min = h2 * 60 + m2
    if not (0 <= start_min <= 1440 and 0 <= end_min <= 1440):
        raise ValueError(f"Hour range out of bounds: {key!r}")
    # Map to bin indices by bin start time
    start_bin = start_min // indexer.time_bin_minutes
    end_bin = end_min // indexer.time_bin_minutes
    # Clip to the day's bins
    start_bin = max(0, min(indexer.num_time_bins, start_bin))
    end_bin = max(0, min(indexer.num_time_bins, end_bin))
    return int(start_bin), int(end_bin)


def build_bin_capacities(
    geojson_path: str,
    indexer: TVTWIndexer,
) -> Dict[str, np.ndarray]:
    """
    Build per-TV per-bin capacity arrays C_v(t) from a GeoJSON file.

    The GeoJSON is expected to be a FeatureCollection where each feature's
    properties include:
      - "traffic_volume_id": TV identifier (string)
      - "capacity": mapping of hour ranges (e.g., "6:00-7:00") to integers

    Policy for outside provided hours: bins with no provided hour range are
    filled with 0.

    Returns a mapping: tv_id -> numpy array of shape (T,), where T is the
    number of time bins in the indexer, containing integer capacities.
    """
    # Initialize all TVs to zeros by default
    T = indexer.num_time_bins
    capacities: Dict[str, np.ndarray] = {
        tv_id: np.zeros(T, dtype=np.int64) for tv_id in indexer.tv_id_to_idx.keys()
    }

    # Load GeoJSON (avoid geopandas dependency; parse JSON directly)
    with open(geojson_path, "r") as f:
        data = json.load(f)

    features: List[Dict[str, Any]] = data.get("features", []) or []
    for feat in features:
        props = feat.get("properties", {}) or {}
        tv_id = props.get("traffic_volume_id")
        if not tv_id or tv_id not in capacities:
            # Skip TVs not present in the indexer mapping
            continue
        cap_map = props.get("capacity") or {}
        if not isinstance(cap_map, dict):
            continue
        arr = capacities[tv_id]
        for hour_key, val in cap_map.items():
            try:
                v = int(val)
            except Exception:
                # Skip non-integer capacity values
                continue
            try:
                start_bin, end_bin = _parse_hour_range_to_bins(str(hour_key), indexer)
            except Exception:
                # Skip malformed hour strings but continue others
                continue
            if end_bin <= start_bin:
                continue
            arr[start_bin:end_bin] = v
        capacities[tv_id] = arr

    return capacities


def rolling_hour_sum(occ_by_bin: np.ndarray, K: int) -> np.ndarray:
    """
    Compute rolling-hour sums over window size K along the last axis.

    Supports inputs of shape (T,) or (..., T). For bins near the end where a
    full K-length window would exceed the horizon, the sum is computed over the
    available tail (i.e., zeros are assumed beyond T).
    """
    arr = np.asarray(occ_by_bin)
    if K <= 0:
        raise ValueError("K must be positive")
    T = arr.shape[-1]
    # cumsum with a zero prepended for easy range sums
    zero = np.zeros(arr.shape[:-1] + (1,), dtype=arr.dtype)
    csum = np.concatenate([zero, np.cumsum(arr, axis=-1)], axis=-1)
    # For each position t, end index is min(t+K, T)
    idx_end = np.arange(T)
    idx_end = np.minimum(idx_end + K, T)
    s_end = np.take(csum, idx_end, axis=-1)
    s_start = csum[..., :T]
    return s_end - s_start


__all__ = ["build_bin_capacities", "rolling_hour_sum"]

