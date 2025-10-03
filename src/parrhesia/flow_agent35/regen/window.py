"""Window selection logic."""
from __future__ import annotations

import math

from .types import Bundle, Window


def select_window_for_bundle(
    bundle: Bundle,
    *,
    bins_per_hour: int,
    total_bins: int,
    margin_hours: float = 0.25,
    min_window_hours: float = 1.0,
) -> Window:
    """Select a robust regulation window covering all flows in the bundle."""

    if not bundle.flows:
        raise ValueError("bundle is empty")
    start = min(fs.diagnostics.tGl for fs in bundle.flows)
    end = max(fs.diagnostics.tGu for fs in bundle.flows)
    margin_bins = int(math.ceil(float(margin_hours) * float(bins_per_hour)))
    min_bins = int(max(1, math.ceil(float(min_window_hours) * float(bins_per_hour))))
    start_adj = max(0, int(start) - margin_bins)
    end_adj = min(total_bins - 1, int(end) + margin_bins)
    if end_adj < start_adj:
        end_adj = start_adj
    duration = end_adj - start_adj + 1
    if duration < min_bins:
        deficit = min_bins - duration
        start_adj = max(0, start_adj - deficit // 2)
        end_adj = min(total_bins - 1, start_adj + min_bins - 1)
    return Window(start_bin=int(start_adj), end_bin=int(end_adj))
