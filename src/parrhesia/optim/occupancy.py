"""
Occupancy computation after applying per‑flight delays.

This module provides a single function `compute_occupancy` that aggregates
per‑TV, per‑bin occupancy from SO6‑style occupancy intervals after shifting
each flight by its assigned delay.

Inputs follow the plan in docs/plans/flowful_sa.md:
  - `flight_list`: supplies `flight_metadata` with fields
      { takeoff_time, occupancy_intervals: [{tvtw_index, entry_time_s, exit_time_s}], ... }
  - `delays`: mapping from `flight_id` to integer delay minutes
  - `indexer`: `TVTWIndexer` for TVTW decoding and time‑bin helpers
  - `tv_filter`: optional iterable of TV identifiers (strings) to restrict
     computation to a subset of volumes of interest

Output is a mapping from `traffic_volume_id` to a numpy array of shape (T,),
where T is `indexer.num_time_bins`, containing per‑bin flight counts. Each
interval contributes 1 to its decoded bin (from `tvtw_index`), shifted by any
whole‑bin delay.

Implementation is optimized to skip intervals whose TV is not in `tv_filter`.
Unknown TVTWs or malformed intervals are ignored.

Examples
--------
Minimal end‑to‑end example with a stub indexer and two flights:

>>> import numpy as np
>>> from parrhesia.optim.occupancy import compute_occupancy
>>> class _StubIndexer:
...     def __init__(self):
...         self.time_bin_minutes = 15
...         self.num_time_bins = 4
...         self.tv_id_to_idx = {"TV1": 0}
...     def get_tvtw_from_index(self, idx):
...         # For the example, decode every index as ("TV1", base_bin = idx)
...         return ("TV1", int(idx))
>>> class _Flights:
...     flight_metadata = {
...         "F1": {
...             "occupancy_intervals": [
...                 {"tvtw_index": 1, "entry_time_s": 0.0, "exit_time_s": 600.0}
...             ]
...         },
...         "F2": {
...             "occupancy_intervals": [
...                 {"tvtw_index": 0, "entry_time_s": 10.0, "exit_time_s": 300.0}
...             ]
...         },
...     }
>>> occ = compute_occupancy(_Flights(), {"F1": 15, "F2": 0}, _StubIndexer(), tv_filter=["TV1"])  # 15‑min = +1 bin shift
>>> sorted(occ.keys())
['TV1']
>>> occ["TV1"].tolist()  # F2 at bin 0, F1 base bin 1 shifted to bin 2
[1, 0, 1, 0]
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional
from datetime import timedelta

import numpy as np

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


def compute_occupancy(
    flight_list,
    delays: Mapping[str, int],
    indexer: TVTWIndexer,
    tv_filter: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per‑TV per‑bin occupancy after applying per‑flight delays.

    Parameters
    ----------
    flight_list : object
        Must expose `flight_metadata` (mapping flight_id -> dict) and contain
        for each flight:
          - `takeoff_time`: datetime (naive UTC)
          - `occupancy_intervals`: list of dicts with keys
                `tvtw_index` (int), `entry_time_s` (float), `exit_time_s` (float)
    delays : Mapping[str, int]
        Mapping flight_id -> delay in minutes (integers). Missing flights imply
        zero delay.
    indexer : TVTWIndexer
        Provider of TVTW decode and time bin helpers. Determines the number of
        time bins per day.
    tv_filter : Optional[Iterable[str]]
        If provided, restrict occupancy aggregation to these traffic_volume_ids.
        When None, aggregate for all TVs known to the indexer.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping `tv_id` -> occupancy array of shape (T,), dtype=int64.

    Notes
    -----
    - Each interval contributes 1 to the bin indicated by its decoded
      `tvtw_index` time bin, adjusted by whole-bin shifts implied by the
      assigned delay. The shift is computed as:
        shift_bins = floor((entry_s + delay_sec)/Δ) - floor(entry_s/Δ),
      where Δ = indexer.time_bin_minutes * 60.
    - Intervals with unknown/undecodable `tvtw_index` are skipped.
    - If `tv_filter` is provided, only TVs in the filter are returned.
    - Bins beyond the day horizon are ignored.

    Examples
    --------
    Basic usage with a tiny stub indexer and two flights:

    >>> import numpy as np
    >>> class _StubIndexer:
    ...     def __init__(self):
    ...         self.time_bin_minutes = 15
    ...         self.num_time_bins = 4
    ...         self.tv_id_to_idx = {"TV1": 0}
    ...     def get_tvtw_from_index(self, idx):
    ...         return ("TV1", int(idx))
    >>> class _Flights:
    ...     flight_metadata = {
    ...         "F1": {"occupancy_intervals": [{"tvtw_index": 1, "entry_time_s": 0.0, "exit_time_s": 600.0}]},
    ...         "F2": {"occupancy_intervals": [{"tvtw_index": 0, "entry_time_s": 10.0, "exit_time_s": 300.0}]},
    ...     }
    >>> occ = compute_occupancy(_Flights(), {"F1": 15, "F2": 0}, _StubIndexer(), tv_filter=["TV1"])  # +1 bin for F1
    >>> occ["TV1"].tolist()
    [1, 0, 1, 0]

    Restricting to TVs via `tv_filter` ensures only those keys are returned:

    >>> occ = compute_occupancy(_Flights(), {"F1": 15}, _StubIndexer(), tv_filter=["TV1"])  # only TV1 present
    >>> set(occ.keys()) == {"TV1"}
    True
    """
    # Establish which TVs to compute for
    if tv_filter is not None:
        tv_of_interest = set(str(tv) for tv in tv_filter)
    else:
        tv_of_interest = set(indexer.tv_id_to_idx.keys())

    T = int(indexer.num_time_bins)
    bin_len_s = int(indexer.time_bin_minutes) * 60
    occ: Dict[str, np.ndarray] = {tv: np.zeros(T, dtype=np.int64) for tv in tv_of_interest}

    # Iterate all flights and accumulate presence bins per interval
    for fid, meta in getattr(flight_list, "flight_metadata", {}).items():
        # Delay in seconds (allow negatives defensively)
        delay_min = delays.get(fid, 0)
        try:
            delay_sec = int(delay_min) * 60
        except Exception:
            delay_sec = 0

        for iv in meta.get("occupancy_intervals", []) or []:
            try:
                tvtw_idx = int(iv.get("tvtw_index"))
            except Exception:
                continue
            decoded = indexer.get_tvtw_from_index(tvtw_idx)
            if not decoded:
                continue
            tv_id, base_bin = decoded
            if tv_id not in tv_of_interest:
                continue

            entry_s = iv.get("entry_time_s", 0.0)
            try:
                entry_s = float(entry_s)
            except Exception:
                entry_s = 0.0
            # Whole-bin shift induced by the delay at this entry offset
            # Use floor to count boundary crossings when adding delay
            before = int(entry_s // bin_len_s)
            after = int((entry_s + delay_sec) // bin_len_s)
            shift = after - before
            b = int(base_bin) + int(shift)
            if 0 <= b < T:
                arr = occ.get(tv_id)
                if arr is not None:
                    arr[b] += 1

    return occ


__all__ = ["compute_occupancy"]
