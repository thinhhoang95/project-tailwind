"""Internal helper utilities for state transition computations."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np
from scipy import sparse


def to_int64_array(values: Iterable[object]) -> np.ndarray:
    """
    Create a 1D NumPy array of dtype int64 from an iterable of values.
    
    Parameters:
        values (Iterable[object]): Iterable of values that can be converted to integers.
    
    Returns:
        np.ndarray: 1D array of dtype `int64` containing the input values converted to integers.
    """

    return np.asarray(list(values), dtype=np.int64)


def to_float_array(values: Iterable[object]) -> np.ndarray:
    """
    Create a 1D NumPy array from an iterable of values.
    
    Returns:
        A 1D NumPy ndarray of dtype `float64` containing the provided values.
    """

    return np.asarray(list(values), dtype=np.float64)


def decode_tvtw_indices(tvtw_indices: np.ndarray, num_time_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Split flat TVTW indices into their TV and time components.
    
    Parameters:
        tvtw_indices: Array of global indices encoded as (tv * num_time_bins + time).
        num_time_bins: Number of time bins used per TV when encoding the global indices.
    
    Returns:
        (tv_indices, time_indices): Tuple where `tv_indices` contains the decoded TV component for each input index and `time_indices` contains the decoded time component (values in 0..num_time_bins-1).
    """

    tv_indices = tvtw_indices // num_time_bins
    time_indices = tvtw_indices % num_time_bins
    return tv_indices, time_indices


def encode_tvtw_indices(tv_indices: np.ndarray, time_indices: np.ndarray, num_time_bins: int) -> np.ndarray:
    """
    Map TV (target-vertex) indices and time-bin indices to flat TVTW indices.
    
    Parameters:
        tv_indices (np.ndarray): Array of TV indices (0-based).
        time_indices (np.ndarray): Array of time-bin indices (0-based), shape-broadcastable with `tv_indices`.
        num_time_bins (int): Number of time bins per TV.
    
    Returns:
        np.ndarray: Array of contiguous TVTW indices computed as `tv_indices * num_time_bins + time_indices`.
    """

    return tv_indices * num_time_bins + time_indices


def build_sparse_delta(num_tvtws: int, old_cols: np.ndarray, new_cols: np.ndarray) -> sparse.csr_matrix:
    """
    Create a 1×num_tvtws CSR matrix representing removals and additions at specified column indices.
    
    The returned sparse row has -1 at indices from `old_cols`, +1 at indices from `new_cols`, and aggregates duplicate entries by summation.
    
    Parameters:
        num_tvtws (int): Total number of TVTW columns (the number of columns in the resulting row).
        old_cols (np.ndarray): 1-D integer array of column indices to decrement (removals).
        new_cols (np.ndarray): 1-D integer array of column indices to increment (additions).
    
    Returns:
        sparse.csr_matrix: A CSR matrix with shape (1, num_tvtws) and dtype int64 containing the delta values.
    """

    if old_cols.size == 0 and new_cols.size == 0:
        return sparse.csr_matrix((1, num_tvtws), dtype=np.int64)
    data_parts = []
    indices_parts = []
    if old_cols.size:
        data_parts.append(np.full(old_cols.shape, -1, dtype=np.int64))
        indices_parts.append(old_cols)
    if new_cols.size:
        data_parts.append(np.full(new_cols.shape, 1, dtype=np.int64))
        indices_parts.append(new_cols)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.int64)
    indices = np.concatenate(indices_parts) if indices_parts else np.empty(0, dtype=np.int64)
    indptr = np.array([0, data.size], dtype=np.int64)
    delta = sparse.csr_matrix((data, indices, indptr), shape=(1, num_tvtws), dtype=np.int64)
    delta.sum_duplicates()
    return delta


def iter_nonzero_delays(pairs: Sequence[tuple[str, int]]) -> Iterator[tuple[str, int]]:
    """
    Yield pairs whose delay value is greater than zero.
    
    Parameters:
        pairs (Sequence[tuple[str, int]]): Sequence of (flight_id, delay) pairs.
    
    Returns:
        Iterator[tuple[str, int]]: An iterator over the input pairs with `delay` greater than 0.
    """

    for flight_id, delay in pairs:
        if delay > 0:
            yield flight_id, delay
