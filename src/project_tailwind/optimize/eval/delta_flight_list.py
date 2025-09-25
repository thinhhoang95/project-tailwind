from __future__ import annotations

from typing import Dict, Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from datetime import datetime, timedelta


class DeltaFlightList:
    """
    A lightweight, read-only view over a base FlightList that overlays
    per-flight integer delays (in minutes) without mutating the base data.

    Only the methods and attributes required by the network evaluator are
    implemented. All other attributes are delegated to the base.
    """

    def __init__(self, base_flight_list, delays_by_flight: Dict[str, int] | None = None):
        self._base = base_flight_list
        self._delays = delays_by_flight or {}

        # Simple attribute proxies used by NetworkEvaluator and other readers
        self.time_bin_minutes = self._base.time_bin_minutes
        self.tv_id_to_idx = self._base.tv_id_to_idx
        self.num_tvtws = self._base.num_tvtws
        self.num_flights = self._base.num_flights
        self.flight_ids = self._base.flight_ids

        # Cache for total occupancy computation
        self._cached_total_occupancy: np.ndarray | None = None

        # Proxy that reflects delayed takeoff times without mutating base
        self.flight_metadata = _FlightMetadataProxy(self._base.flight_metadata, self._delays)

    # --- Public API expected by NetworkEvaluator ---------------------------------
    def get_total_occupancy_by_tvtw(self) -> np.ndarray:
        """
        Return the total occupancy vector as base + per-flight delta shifts.
        """
        if self._cached_total_occupancy is not None:
            return self._cached_total_occupancy

        # Base occupancy
        total = self._base.get_total_occupancy_by_tvtw().astype(np.float32, copy=True)

        if not self._delays:
            self._cached_total_occupancy = total
            return total

        # Accumulate deltas from each delayed flight
        delta_accumulator = np.zeros_like(total)
        for flight_id, delay_min in self._delays.items():
            if delay_min == 0:
                continue

            # Original vector and shifted vector
            orig = self._base.get_occupancy_vector(flight_id)
            shifted = self._base.shift_flight_occupancy(flight_id, delay_min)

            # Add (shifted - orig)
            # Using inplace ops to minimize allocations
            delta_accumulator += (shifted - orig)

        total += delta_accumulator
        self._cached_total_occupancy = total
        return total

    def get_occupancy_vector(self, flight_id: str) -> np.ndarray:
        """
        Return occupancy vector for a specific flight, shifted if delayed.
        """
        delay_min = self._delays.get(flight_id, 0)
        if delay_min == 0:
            return self._base.get_occupancy_vector(flight_id)
        return self._base.shift_flight_occupancy(flight_id, delay_min)

    def iter_hotspot_crossings(
        self,
        hotspot_ids: Sequence[str],
        active_windows: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None,
    ) -> Iterable[Tuple[str, str, datetime, int]]:
        """Yield hotspot crossings with TVTW bins adjusted by per-flight delays."""

        hotspot_set = {str(h) for h in hotspot_ids}
        if not hotspot_set:
            return

        global_windows: Optional[set[int]] = None
        per_hotspot: Optional[Dict[str, set[int]]] = None
        if active_windows is None:
            pass
        elif isinstance(active_windows, dict):
            per_hotspot = {
                str(k): {int(x) for x in (v or [])}
                for k, v in active_windows.items()
            }
        else:
            global_windows = {int(x) for x in (active_windows or [])}

        num_time_bins = int(getattr(self.indexer, "num_time_bins", 0)) or int(
            getattr(self._base, "num_time_bins", getattr(self._base, "num_time_bins_per_tv", 0))
        )
        if num_time_bins <= 0:
            num_time_bins = 1

        bin_minutes = int(self.time_bin_minutes)

        for fid, meta in self.flight_metadata.items():
            try:
                takeoff = meta.get("takeoff_time")
            except Exception:
                takeoff = None
            if takeoff is None:
                continue
            delay_min = int(self._delays.get(fid, 0))
            shift_bins = 0
            if delay_min != 0:
                shift_bins = delay_min // bin_minutes
                if delay_min % bin_minutes != 0:
                    shift_bins += 1

            intervals = meta.get("occupancy_intervals", []) or []
            for iv in intervals:
                try:
                    tvtw_idx = int(iv.get("tvtw_index"))
                except Exception:
                    continue
                decoded = self.indexer.get_tvtw_from_index(tvtw_idx)
                if not decoded:
                    continue
                tv_id, time_idx = decoded
                if str(tv_id) not in hotspot_set:
                    continue

                shifted_idx = int(time_idx) + int(shift_bins)
                if shifted_idx < 0:
                    shifted_idx = 0
                elif shifted_idx >= num_time_bins:
                    shifted_idx = num_time_bins - 1

                allowed = True
                if per_hotspot is not None:
                    allowed_set = per_hotspot.get(str(tv_id))
                    allowed = allowed_set is not None and shifted_idx in allowed_set
                elif global_windows is not None:
                    allowed = shifted_idx in global_windows
                if not allowed:
                    continue

                entry_s = iv.get("entry_time_s", 0)
                try:
                    entry_s = float(entry_s)
                except Exception:
                    entry_s = 0.0
                entry_dt = takeoff + timedelta(seconds=float(entry_s))
                yield (str(fid), str(tv_id), entry_dt, shifted_idx)

    # --- Optional helpers for compatibility --------------------------------------
    def get_matrix_shape(self) -> tuple:
        return self._base.get_matrix_shape()

    def copy(self):
        """
        Cheap logical copy of the view. Note this intentionally does NOT deep-copy
        the base data to keep memory bounded. Consumers that rely on comparing to
        an "original" should be aware that this is a view copy.
        """
        return DeltaFlightList(self._base, dict(self._delays))

    # Delegate common attribute access to base to minimize surface area changes
    def __getattr__(self, item: str) -> Any:
        return getattr(self._base, item)

    def __repr__(self) -> str:
        return f"DeltaFlightList(base={self._base!r}, delayed_flights={len(self._delays)})"


class _FlightMetadataProxy:
    """
    Dict-like view over base flight metadata that overlays delayed takeoff times
    for flights listed in the delays mapping. Other keys are passed through.
    """

    def __init__(self, base_metadata: Dict[str, Any], delays_by_flight: Dict[str, int]):
        self._base = base_metadata
        self._delays = delays_by_flight

    def __getitem__(self, flight_id: str) -> Dict[str, Any]:
        meta = self._base[flight_id]
        delay_min = self._delays.get(flight_id, 0)
        if delay_min:
            # Return a shallow copy with adjusted takeoff_time
            adjusted = dict(meta)
            adjusted_takeoff = adjusted["takeoff_time"] + timedelta(minutes=delay_min)
            adjusted["takeoff_time"] = adjusted_takeoff
            return adjusted
        return meta

    def __contains__(self, key: str) -> bool:
        return key in self._base

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def keys(self) -> Iterable[str]:
        return self._base.keys()

    def items(self):  # optional convenience
        for k in self._base.keys():
            yield k, self.__getitem__(k)

    def __iter__(self):  # for dict-like behavior
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)


