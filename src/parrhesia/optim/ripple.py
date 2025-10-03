"""Shared helpers for deriving ripple cells from flight footprints."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def compute_auto_ripple_cells(
    *,
    indexer,
    flight_list,
    flight_ids: Iterable[str],
    window_bins: int,
) -> List[Tuple[str, int]]:
    """Union of TVs × bins from flight footprints with optional ±window dilation.

    Mirrors the behaviour previously embedded within API and regen entry points.
    Unknown flights or malformed occupancy intervals are ignored.
    """

    try:
        w = int(window_bins)
    except Exception:
        w = 0
    if w < 0:
        w = 0

    T = int(indexer.num_time_bins)
    bins_by_tv: Dict[str, set[int]] = {}
    flight_metadata = getattr(flight_list, "flight_metadata", {}) or {}

    for fid in flight_ids:
        meta = flight_metadata.get(str(fid))
        if not meta:
            continue
        for iv in (meta.get("occupancy_intervals") or []):
            try:
                tvtw_idx = int(iv.get("tvtw_index"))
            except Exception:
                continue
            decoded = indexer.get_tvtw_from_index(tvtw_idx)
            if not decoded:
                continue
            tv_id, tbin = decoded
            tv_key = str(tv_id)
            bins_by_tv.setdefault(tv_key, set())
            try:
                tb = int(tbin)
            except Exception:
                continue
            if 0 <= tb < T:
                bins_by_tv[tv_key].add(tb)

    if w > 0:
        for tv in list(bins_by_tv.keys()):
            base_bins = bins_by_tv[tv]
            expanded: set[int] = set()
            for b in base_bins:
                start = max(0, b - w)
                end = min(T - 1, b + w)
                for t in range(start, end + 1):
                    expanded.add(t)
            bins_by_tv[tv] = expanded

    cells: List[Tuple[str, int]] = []
    for tv in sorted(bins_by_tv.keys()):
        for b in sorted(bins_by_tv[tv]):
            cells.append((tv, int(b)))
    return cells


__all__ = ["compute_auto_ripple_cells"]


