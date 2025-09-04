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

from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from weakref import WeakKeyDictionary
import math

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


# Cache of decoded footprints per flight_list keyed by the flight_list object
# Weakly referenced to avoid memory leaks across runs.
_OCC_FOOTPRINTS_CACHE: "WeakKeyDictionary" = WeakKeyDictionary()

DEBUG_PROFILE_TIMING = False

# Optional Numba acceleration
_HAS_NUMBA = True
try:
    from numba import njit  # type: ignore

    @njit(cache=True)
    def _accumulate_2d_numba(comp_rows: np.ndarray, bins: np.ndarray, V: int, T: int) -> np.ndarray:  # pragma: no cover
        out = np.zeros((V, T), dtype=np.int64)
        n = comp_rows.shape[0]
        for i in range(n):
            r = int(comp_rows[i])
            b = int(bins[i])
            if 0 <= r < V and 0 <= b < T:
                out[r, b] += 1
        return out

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def _build_footprints_for_flight_list(flight_list, indexer: TVTWIndexer) -> Dict[str, object]:
    """
    Precompute per-flight interval footprints in a vectorizable numeric form.

    For each flight, produce three aligned int32 arrays:
      - rows: TV row indices (0..num_tvs-1)
      - base_bins: decoded base time-bin for the interval (0..T-1)
      - entry_mods: entry_time_s modulo bin length (seconds in [0, bin_len_s))

    Returns a dict with keys:
      { 'T': int, 'bin_len_s': int, 'per_flight': Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] }
    """
    T = int(indexer.num_time_bins)
    bin_len_s = int(indexer.time_bin_minutes) * 60
    num_rows = len(getattr(indexer, 'idx_to_tv_id', indexer.tv_id_to_idx))

    per_flight: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    fm = getattr(flight_list, "flight_metadata", {}) or {}
    for fid, meta in fm.items():
        rows_list = []
        base_bins_list = []
        entry_mods_list = []
        for iv in meta.get("occupancy_intervals", []) or []:
            tvtw_raw = iv.get("tvtw_index")
            if tvtw_raw is None:
                continue
            try:
                tvtw_idx = int(tvtw_raw)
            except Exception:
                continue
            # Arithmetic decode: global_idx = row * T + base_bin
            row = tvtw_idx // T
            base_bin = tvtw_idx - row * T  # faster than % for positives

            # Validate row against indexer
            if row < 0 or row >= num_rows:
                continue

            entry_raw = iv.get("entry_time_s", 0)
            try:
                # Use floor to match original boundary-crossing semantics
                entry_floor = int(math.floor(float(entry_raw)))
                entry_mod = entry_floor % bin_len_s
            except Exception:
                entry_mod = 0

            rows_list.append(int(row))
            base_bins_list.append(int(base_bin))
            entry_mods_list.append(int(entry_mod))

        if rows_list:
            per_flight[str(fid)] = (
                np.asarray(rows_list, dtype=np.int32),
                np.asarray(base_bins_list, dtype=np.int32),
                np.asarray(entry_mods_list, dtype=np.int32),
            )
        else:
            per_flight[str(fid)] = (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

    return {"T": T, "bin_len_s": bin_len_s, "per_flight": per_flight}


def _get_or_build_footprints_cache(flight_list, indexer: TVTWIndexer) -> Dict[str, object]:
    """
    Retrieve cached footprints for this flight_list compatible with the current
    indexer configuration (T and bin length). Rebuild if incompatible or missing.
    """
    cache = _OCC_FOOTPRINTS_CACHE.get(flight_list)
    T = int(indexer.num_time_bins)
    bin_len_s = int(indexer.time_bin_minutes) * 60
    if not cache or cache.get("T") != T or cache.get("bin_len_s") != bin_len_s:
        cache = _build_footprints_for_flight_list(flight_list, indexer)
        _OCC_FOOTPRINTS_CACHE[flight_list] = cache
    return cache


def clear_occupancy_cache(flight_list: Optional[object] = None) -> None:
    """Clear cached footprints. If a flight_list is provided, evict only that entry."""
    try:
        if flight_list is None:
            _OCC_FOOTPRINTS_CACHE.clear()
        else:
            _OCC_FOOTPRINTS_CACHE.pop(flight_list, None)
    except Exception:
        pass

def compute_occupancy(
    flight_list,
    delays: Mapping[str, int],
    indexer: TVTWIndexer,
    tv_filter: Optional[Iterable[str]] = None,
    flight_filter: Optional[Iterable[str]] = None,
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
    flight_filter : Optional[Iterable[str]]
        If provided, restrict computation to these flight ids only. When None,
        process all flights present in `flight_list.flight_metadata`.

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
    import time 

    # Establish which TVs to compute for (maintain deterministic order)
    if tv_filter is not None:
        tv_ids_requested = list(dict.fromkeys(str(tv) for tv in tv_filter))
    else:
        tv_ids_requested = list(indexer.tv_id_to_idx.keys())

    # Numeric dimensions and helpers
    T = int(indexer.num_time_bins)
    bin_len_s = int(indexer.time_bin_minutes) * 60
    tv_id_to_row = indexer.tv_id_to_idx
    num_rows = len(getattr(indexer, 'idx_to_tv_id', indexer.tv_id_to_idx))

    # Build compact mapping for the rows we care about
    # row_to_compact[row] -> [0..V-1] or -1 if not selected
    if DEBUG_PROFILE_TIMING:
        start_time = time.time()
    row_to_compact = np.full(num_rows, -1, dtype=np.int32)
    tv_ids: list[str] = []
    for tv in tv_ids_requested:
        r = tv_id_to_row.get(str(tv))
        if r is None:
            continue
        row_idx = int(r)
        if 0 <= row_idx < num_rows and row_to_compact[row_idx] < 0:
            row_to_compact[row_idx] = len(tv_ids)
            tv_ids.append(str(tv))
    if DEBUG_PROFILE_TIMING:
        end_time = time.time()
        print(f"Time taken to build compact mapping: {end_time - start_time} seconds")

    V = len(tv_ids)
    if V == 0:
        return {}

    # Precompute delays in seconds
    if DEBUG_PROFILE_TIMING:
        start_time = time.time()
    delays_sec: Dict[str, int] = {}
    for k, v in (delays or {}).items():
        try:
            delays_sec[str(k)] = int(v) * 60
        except Exception:
            delays_sec[str(k)] = 0
    if DEBUG_PROFILE_TIMING:
        end_time = time.time()
        print(f"Time taken to precompute delays: {end_time - start_time} seconds")
    
    # Retrieve or build cached footprints (rows, base_bins, entry_mods) per flight
    if DEBUG_PROFILE_TIMING:
        start_time = time.time()
    cache = _get_or_build_footprints_cache(flight_list, indexer)
    if DEBUG_PROFILE_TIMING:
        end_time = time.time()
        print(f"Time taken to retrieve or build cached footprints: {end_time - start_time} seconds")
    
    per_flight = cache["per_flight"]  # type: ignore[index]

    # Accumulate all contributing (compact_row, bin) pairs across flights
    rows_all: list[np.ndarray] = []
    bins_all: list[np.ndarray] = []
    if DEBUG_PROFILE_TIMING:
        start_time = time.time()
    # Determine which flights to process
    if flight_filter is not None:
        flight_ids_requested = [str(fid) for fid in flight_filter if str(fid) in per_flight]
    else:
        flight_ids_requested = list(per_flight.keys())
    for fid in flight_ids_requested:
        triple = per_flight.get(str(fid))  # type: ignore[assignment]
        if triple is None:
            continue
        rows_i, base_bins_i, entry_mods_i = triple  # each is np.ndarray[int32]
        if rows_i.size == 0:
            continue
        delay_sec = int(delays_sec.get(str(fid), 0))

        # Vectorized shift and bin computation per interval
        # shift = floor((entry_mod + delay_sec) / bin_len_s)
        # b = base_bin + shift
        shift_i = (entry_mods_i.astype(np.int64, copy=False) + int(delay_sec)) // bin_len_s
        bins_i = base_bins_i.astype(np.int64, copy=False) + shift_i

        # Valid bin range
        valid_mask = (bins_i >= 0) & (bins_i < T)
        if not np.any(valid_mask):
            continue

        # Map rows to compact rows and filter to selected TVs
        comp_rows_i = row_to_compact[rows_i]
        allowed_mask = comp_rows_i >= 0
        if not np.any(allowed_mask):
            continue

        mask = valid_mask & allowed_mask
        if not np.any(mask):
            continue

        rows_all.append(comp_rows_i[mask].astype(np.int64, copy=False))
        bins_all.append(bins_i[mask].astype(np.int64, copy=False))
    
    if DEBUG_PROFILE_TIMING:
        end_time = time.time()
        print(f"Time taken to accumulate all contributing (compact_row, bin) pairs across flights: {end_time - start_time} seconds")
    
    # If no contributing intervals, return zeros for requested TVs
    if not rows_all:
        return {tv: np.zeros(T, dtype=np.int64) for tv in tv_ids}

    comp_rows = np.concatenate(rows_all)
    bins = np.concatenate(bins_all)

    # Build occupancy matrix either via numba or via bincount
    if _HAS_NUMBA and comp_rows.size >= 100000:
        occ_matrix = _accumulate_2d_numba(comp_rows.astype(np.int64), bins.astype(np.int64), int(V), int(T))
    else:
        lin_idx = comp_rows * T + bins
        counts_flat = np.bincount(lin_idx, minlength=V * T)
        occ_matrix = counts_flat.reshape((V, T))

    # Assemble result mapping (include zeros for any requested TVs that didn't map)
    result: Dict[str, np.ndarray] = {}
    for i, tv in enumerate(tv_ids):
        # Ensure int64 dtype as documented
        result[tv] = occ_matrix[i].astype(np.int64, copy=False)

    # Preserve behavior: unknown requested TVs appear with zero arrays
    missing = [tv for tv in tv_ids_requested if str(tv) not in set(tv_ids)]
    for tv in missing:
        result[str(tv)] = np.zeros(T, dtype=np.int64)

    return result


__all__ = ["compute_occupancy", "clear_occupancy_cache"]
