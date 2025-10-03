from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from server_tailwind.core.resources import AppResources, get_resources

# NOTICE FOR MAINTAINERS AND CODING AGENTS: This hotspot segment extractor is a direct reimplementation of the `get_hotspot_segments` in `network_evaluator_for_api.py`
# if you read this, please be aware that any changes to this file may cause inconsistency in the /hotspots API endpoint.
# We have plan to migrate the API endpoint to this module in the future.

def _rolling_hour_forward(matrix_2d: np.ndarray, window_bins: int) -> np.ndarray:
    """Forward-looking rolling sum with window size ``window_bins`` along axis=1.

    For each bin j, result[:, j] = sum(matrix_2d[:, j : j+window_bins]) truncated at end.
    Mirrors NetworkEvaluator._apply_rolling_hour_forward semantics.
    """
    # If window is 1 bin or less, no rolling sum is needed.
    if window_bins <= 1:
        return matrix_2d.astype(np.float32, copy=False)
    num_tvs, num_bins = matrix_2d.shape
    # Use cumsum for efficient rolling sum calculation.
    # Pad with a column of zeros on the left for correct calculation at the start.
    cs = np.cumsum(
        np.concatenate(
            [np.zeros((num_tvs, 1), dtype=np.float32), matrix_2d.astype(np.float32, copy=False)],
            axis=1,
        ),
        axis=1,
        dtype=np.float64,
    )
    # Calculate the rolling sum for each bin.
    out = np.empty_like(matrix_2d, dtype=np.float32)
    for j in range(num_bins):
        # The end of the window is truncated at the last bin.
        j2 = min(num_bins, j + window_bins)
        # The sum over a window is the difference of the cumulative sums.
        out[:, j] = cs[:, j2] - cs[:, j]
    return out


def extract_hotspot_segments_from_resources(
    *, threshold: float = 0.0, resources: Optional[AppResources] = None
) -> List[Dict[str, Any]]:
    """Detect hotspot segments using shared AppResources artifacts.

    - Pulls occupancy from ``res.flight_list``.
    - Computes a forward-looking 60-minute rolling sum per TV.
    - Subtracts ``res.capacity_per_bin_matrix`` (hourly cap repeated per bin).
    - Merges contiguous overloaded bins into segments.

    Returns a list of segment dicts compatible with regen usage.
    """
    # Get application resources, loading them if necessary.
    res = (resources or get_resources()).preload_all()
    fl = res.flight_list

    num_tvs = len(fl.tv_id_to_idx)
    if num_tvs == 0:
        return []

    # Get total occupancy and dimensions.
    total_occ = fl.get_total_occupancy_by_tvtw().astype(np.float32, copy=False)
    num_tvtws = int(fl.num_tvtws)
    bins_per_tv = int(num_tvtws // num_tvs)
    if bins_per_tv <= 0:
        return []

    # Reshape to [num_tvs, bins_per_tv] with stable row order.
    # This creates a matrix where each row corresponds to a traffic volume.
    per_tv = np.zeros((num_tvs, bins_per_tv), dtype=np.float32)
    tv_items = sorted(fl.tv_id_to_idx.items(), key=lambda kv: kv[1])
    for tv_id, row_idx in tv_items:
        start = int(row_idx) * bins_per_tv
        end = start + bins_per_tv
        per_tv[int(row_idx), :] = total_occ[start:end]

    # Rolling-hour window size (ceil to cover full 60 min)
    window_bins = max(1, int(np.ceil(60.0 / float(int(fl.time_bin_minutes)))))
    # Compute the forward-looking rolling sum of occupancy.
    rolling = _rolling_hour_forward(per_tv, window_bins)

    # Capacity per bin matrix from resources
    cap = res.capacity_per_bin_matrix
    if cap is None or cap.shape != (num_tvs, bins_per_tv):
        raise RuntimeError("capacity_per_bin_matrix is missing or has unexpected shape")

    # Helper to create a time label from a bin offset.
    def _label(bin_offset: int) -> str:
        start_total_min = int(bin_offset * int(fl.time_bin_minutes))
        return f"{(start_total_min // 60) % 24:02d}:{start_total_min % 60:02d}"

    segments: List[Dict[str, Any]] = []
    thr = float(threshold)

    # Iterate TVs in row order for stable output
    for tv_id, row_idx in tv_items:
        r = int(row_idx)
        cap_row = cap[r, :]
        roll_row = rolling[r, :]

        # Calculate the difference between rolling occupancy and capacity.
        # Where capacity is invalid (<0), mark diff as -inf to prevent inclusion in hotspots.
        valid = cap_row >= 0.0
        diff = np.where(valid, roll_row - cap_row, -np.inf)
        # Bins are considered overloaded if the difference exceeds the threshold.
        overloaded = diff > thr

        # Find contiguous segments of overloaded bins.
        i = 0
        while i < bins_per_tv:
            # Skip non-overloaded bins.
            if not overloaded[i]:
                i += 1
                continue
            # Found the start of a segment.
            start_bin = i
            # Find the end of the segment.
            j = i + 1
            while j < bins_per_tv and overloaded[j]:
                j += 1
            end_bin = j - 1  # inclusive

            # Extract data for the identified segment.
            seg_slice = slice(start_bin, end_bin + 1)
            seg_diff = diff[seg_slice]
            seg_roll = roll_row[seg_slice]
            seg_cap = cap_row[seg_slice]

            # Calculate statistics for the segment.
            max_excess = float(np.max(seg_diff)) if seg_diff.size > 0 else 0.0
            sum_excess = float(np.sum(seg_diff[seg_diff > -np.inf])) if seg_diff.size > 0 else 0.0
            peak_rolling = float(np.max(seg_roll)) if seg_roll.size > 0 else 0.0
            cap_min = float(np.min(seg_cap)) if seg_cap.size > 0 else -1.0
            cap_max = float(np.max(seg_cap)) if seg_cap.size > 0 else -1.0

            # Append segment information to the list.
            segments.append(
                {
                    "traffic_volume_id": str(tv_id),
                    "start_bin": int(start_bin),  # bin offset within TV (day-relative)
                    "end_bin": int(end_bin),      # inclusive
                    "start_label": _label(start_bin),
                    "end_label": _label(end_bin),
                    "time_bin_minutes": int(fl.time_bin_minutes),
                    "window_minutes": 60,
                    "max_excess": max_excess,
                    "sum_excess": sum_excess,
                    "peak_rolling_count": peak_rolling,
                    "capacity_stats": {"min": cap_min, "max": cap_max},
                }
            )
            # Continue search from the end of the current segment.
            i = j

    # Sort segments primarily by max_excess (descending), then by TV ID and start bin for stability.
    segments.sort(
        key=lambda s: (
            -float(s.get("max_excess", 0.0)),
            str(s.get("traffic_volume_id", "")),
            int(s.get("start_bin", 0)),
        )
    )
    return segments


def segment_to_hotspot_payload(seg: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a segment into a hotspot payload usable by regen.

    Regen expects window bins as [start, end_exclusive]; segments are inclusive.
    """
    return {
        "control_volume_id": str(seg["traffic_volume_id"]),
        # Convert inclusive end_bin to exclusive for regen's expectation.
        "window_bins": [int(seg["start_bin"]), int(seg["end_bin"]) + 1],
        "metadata": {},
        "mode": "inventory",
    }

