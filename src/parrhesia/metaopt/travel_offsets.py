from __future__ import annotations

from typing import Dict, Mapping, Optional


def minutes_to_bin_offsets(
    travel_minutes: Mapping[str, Mapping[str, float]],
    time_bin_minutes: int,
) -> Dict[str, Dict[str, int]]:
    """
    Convert nested mapping of minutes[src][dst] -> float to integer bin offsets.
    Rounds to nearest integer number of bins using time_bin_minutes.
    """
    out: Dict[str, Dict[str, int]] = {}
    denom = max(1, int(time_bin_minutes))
    for src, rows in (travel_minutes or {}).items():
        d: Dict[str, int] = {}
        for dst, m in (rows or {}).items():
            try:
                bins = int(round(float(m) / float(denom)))
            except Exception:
                bins = 0
            d[str(dst)] = int(bins)
        out[str(src)] = d
    return out


def flow_offsets_from_ctrl(
    control_tv_id: Optional[str],
    tv_id_to_idx: Mapping[str, int],
    bin_offsets: Mapping[str, Mapping[str, int]],
) -> Optional[Dict[int, int]]:
    """
    Build τ_{G,s} for a flow controlled at `control_tv_id`.

    Returns mapping tv_row_index -> offset_bins from control to that TV. If
    control_tv_id is None or not present in bin_offsets, returns None.
    """
    if control_tv_id is None:
        return None
    row_map = {str(k): int(v) for k, v in tv_id_to_idx.items()}
    ctrl = str(control_tv_id)
    row_offsets = bin_offsets.get(ctrl)
    if row_offsets is None:
        return None
    out: Dict[int, int] = {}
    for tv, row in row_map.items():
        off = row_offsets.get(tv)
        if off is not None:
            out[int(row)] = int(off)
    return out

