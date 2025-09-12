from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from typing import Any as _Any


def build_flow_g0(flight_list: _Any, flight_ids: Sequence[str]) -> np.ndarray:
    """
    Compute total occupancy vector g0 across all TVs and bins for the given flow (list of flights).
    Returns a 1D float64 array of length num_tvs * T.
    """
    if not flight_ids:
        return np.zeros(int(getattr(flight_list, 'num_tvtws', 0)), dtype=np.float64)
    g0 = np.zeros(int(getattr(flight_list, 'num_tvtws')), dtype=np.float64)
    for fid in flight_ids:
        try:
            g0 += np.asarray(flight_list.get_occupancy_vector(str(fid)), dtype=np.float64)
        except Exception:
            # Skip invalid flight ids
            continue
    return g0


def build_xG_series(
    flights_by_flow: Mapping[int, Sequence[Mapping[str, object]]],
    ctrl_by_flow: Mapping[int, Optional[str]],
    flow_id: int,
    num_time_bins_per_tv: int,
) -> np.ndarray:
    """
    Build per-bin activity time series x_G(t) at the flow's controlled volume row.

    Uses the 'requested_bin' field for each flight spec in the flow as the per-flight
    requested time at the controlled volume; x_G is a histogram of requested_bin values.
    """
    specs = list(flights_by_flow.get(int(flow_id), []) or [])
    T = int(num_time_bins_per_tv)
    x = np.zeros(T, dtype=np.float64)
    if not specs:
        return x
    for sp in specs:
        rb = sp.get("requested_bin")
        try:
            b = int(rb)
        except Exception:
            continue
        if 0 <= b < T:
            x[b] += 1.0
    return x
