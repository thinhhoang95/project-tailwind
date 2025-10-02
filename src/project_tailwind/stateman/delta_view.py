"""Delta occupancy view construction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse

from project_tailwind.optimize.eval.flight_list import FlightList

from .delay_assignment import DelayAssignmentTable
from .types import OccupancyIntervalDict
from .utils import (
    build_sparse_delta,
    decode_tvtw_indices,
    encode_tvtw_indices,
    to_float_array,
    to_int64_array,
)

logger = logging.getLogger(__name__)


def _compute_shift_bins(entry_seconds: np.ndarray, delay_seconds: int, bin_seconds: int) -> np.ndarray:
    """
    Compute how many discrete time bins each entry shifts after applying a delay.
    
    Parameters:
    	entry_seconds (np.ndarray): Entry times in seconds for each interval.
    	delay_seconds (int): Delay in seconds applied to each entry.
    	bin_seconds (int): Duration of a single time bin in seconds.
    
    Returns:
    	np.ndarray: Integer array (same shape as `entry_seconds`) with the per-entry bin shift (new_bin_index - old_bin_index).
    """

    if delay_seconds == 0 or entry_seconds.size == 0:
        return np.zeros(entry_seconds.shape, dtype=np.int64)
    entry_bins = np.floor(entry_seconds / bin_seconds).astype(np.int64)
    new_entry_bins = np.floor((entry_seconds + delay_seconds) / bin_seconds).astype(np.int64)
    return new_entry_bins - entry_bins


@dataclass(slots=True)
class DeltaOccupancyView:
    """Vectorized representation of incremental occupancy changes."""

    num_tvtws: int
    delta_counts_sparse: sparse.csr_matrix
    per_flight_new_intervals: Dict[str, List[OccupancyIntervalDict]]
    delays: DelayAssignmentTable
    regulation_id: Optional[str] = None

    _changed_flights: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """
        Validate the constructed DeltaOccupancyView and initialize internal changed-flight list.
        
        Ensures `delta_counts_sparse` is a 1x`num_tvtws` row vector and, if `_changed_flights`
        was not provided, sets it to the list of keys from `per_flight_new_intervals`.
        
        Raises:
            ValueError: If `delta_counts_sparse` does not have shape `(1, num_tvtws)`.
        """
        if self.delta_counts_sparse.shape != (1, self.num_tvtws):
            raise ValueError("delta_counts_sparse must be a row vector matching num_tvtws")
        if self._changed_flights is None:
            self._changed_flights = list(self.per_flight_new_intervals.keys())

    @classmethod
    def from_delay_table(
        cls,
        flights: FlightList,
        delays: DelayAssignmentTable,
        *,
        regulation_id: Optional[str] = None,
    ) -> "DeltaOccupancyView":
        """
        Builds a DeltaOccupancyView that captures incremental occupancy changes after applying the provided delay assignments.
        
        Parameters:
            flights (FlightList): Source flight list containing per-flight metadata, occupancy intervals, and time-bin configuration used to compute TVTW indices and bin shifts.
            delays (DelayAssignmentTable): Delay assignments (minutes) keyed by flight ID; only nonzero delays are applied to produce updated intervals.
            regulation_id (Optional[str]): Optional regulation identifier to attach to the resulting view.
        
        Returns:
            DeltaOccupancyView: Instance containing:
                - a sparse 1xnum_tvtws delta vector representing moved occupancy (delta_counts_sparse),
                - per_flight_new_intervals mapping flight IDs to their updated occupancy intervals (with tvtw_index, entry_time_s, exit_time_s),
                - a copy of the provided delays,
                - the supplied regulation_id,
                - the list of flight IDs whose occupancy changed.
        """
        num_tvtws = int(flights.num_tvtws)
        num_time_bins = int(flights.num_time_bins)
        bin_seconds = int(flights.time_bin_minutes) * 60

        per_flight_new_intervals: Dict[str, List[OccupancyIntervalDict]] = {}
        changed_flights: List[str] = []
        old_columns: List[np.ndarray] = []
        new_columns: List[np.ndarray] = []

        for flight_id, delay_minutes in delays.nonzero_items():
            metadata = flights.flight_metadata.get(flight_id)
            if metadata is None:
                logger.warning("Flight %s not found in FlightList. Skipping delay.", flight_id)
                continue
            intervals = metadata.get("occupancy_intervals") or []
            if not intervals:
                continue
            old_cols = to_int64_array(iv.get("tvtw_index", 0) for iv in intervals)
            if old_cols.size == 0:
                continue
            entry_seconds = to_float_array(iv.get("entry_time_s", 0.0) for iv in intervals)
            exit_seconds = to_float_array(iv.get("exit_time_s", 0.0) for iv in intervals)

            delay_seconds = int(delay_minutes) * 60
            shift_bins = _compute_shift_bins(entry_seconds, delay_seconds, bin_seconds)

            tv_indices, old_time_indices = decode_tvtw_indices(old_cols, num_time_bins)
            new_time_indices = old_time_indices + shift_bins
            valid_mask = (new_time_indices >= 0) & (new_time_indices < num_time_bins)
            if not np.any(valid_mask):
                per_flight_new_intervals[flight_id] = []
                old_columns.append(old_cols)
                changed_flights.append(flight_id)
                continue

            new_cols = encode_tvtw_indices(tv_indices[valid_mask], new_time_indices[valid_mask], num_time_bins)
            new_columns.append(new_cols)
            old_columns.append(old_cols)

            new_entry_seconds = entry_seconds[valid_mask] + delay_seconds
            new_exit_seconds = exit_seconds[valid_mask] + delay_seconds
            per_flight_new_intervals[flight_id] = [
                {
                    "tvtw_index": int(col),
                    "entry_time_s": float(new_entry),
                    "exit_time_s": float(new_exit),
                }
                for col, new_entry, new_exit in zip(new_cols, new_entry_seconds, new_exit_seconds)
            ]
            changed_flights.append(flight_id)

        if old_columns:
            old_concat = np.concatenate(old_columns)
        else:
            old_concat = np.empty(0, dtype=np.int64)
        if new_columns:
            new_concat = np.concatenate(new_columns)
        else:
            new_concat = np.empty(0, dtype=np.int64)

        delta_sparse = build_sparse_delta(num_tvtws, old_concat, new_concat)

        return cls(
            num_tvtws=num_tvtws,
            delta_counts_sparse=delta_sparse,
            per_flight_new_intervals=per_flight_new_intervals,
            delays=delays.copy(),
            regulation_id=regulation_id,
            _changed_flights=changed_flights,
        )

    def changed_flights(self) -> List[str]:
        """
        List flight IDs whose occupancy changed.
        
        Returns:
            A list of flight IDs that had occupancy changes. The returned list is a shallow copy of the internal changed-flight list.
        """

        return list(self._changed_flights)

    def as_dense_delta(self, dtype: np.dtype | type = np.int64) -> np.ndarray:
        """
        Produce the delta vector in dense 1-D form.
        
        Parameters:
            dtype (np.dtype | type): Desired NumPy dtype for the returned array.
        
        Returns:
            np.ndarray: Dense 1-D array of length equal to the view's number of TVTW indices, with the requested dtype.
        """

        return np.asarray(self.delta_counts_sparse.toarray()).ravel().astype(dtype, copy=False)

    def stats(self) -> Dict[str, int]:
        """
        Compute basic statistics for this delta occupancy view.
        
        Returns:
            stats (Dict[str, int]): Mapping with:
                - num_changed_flights: Number of flights whose occupancy changed.
                - total_delay_minutes: Sum of delay minutes for the changed flights.
                - nonzero_entries: Number of nonzero entries in the sparse delta vector.
        """

        total_delay = sum(self.delays.get(fid, 0) for fid in self._changed_flights)
        nnz = int(self.delta_counts_sparse.count_nonzero())
        return {
            "num_changed_flights": len(self._changed_flights),
            "total_delay_minutes": total_delay,
            "nonzero_entries": nnz,
        }


__all__ = ["DeltaOccupancyView"]
