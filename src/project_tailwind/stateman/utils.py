"""Internal helper utilities for state transition computations."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import numpy as np
from scipy import sparse


def to_int64_array(values: Iterable[object]) -> np.ndarray:
    """Return a 1D ``int64`` numpy array from an iterable of values."""

    return np.asarray(list(values), dtype=np.int64)


def to_float_array(values: Iterable[object]) -> np.ndarray:
    """Return a 1D ``float64`` numpy array from an iterable of values."""

    return np.asarray(list(values), dtype=np.float64)


def decode_tvtw_indices(tvtw_indices: np.ndarray, num_time_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Split global TVTW indices into TV indices and time indices."""

    tv_indices = tvtw_indices // num_time_bins
    time_indices = tvtw_indices % num_time_bins
    return tv_indices, time_indices


def encode_tvtw_indices(tv_indices: np.ndarray, time_indices: np.ndarray, num_time_bins: int) -> np.ndarray:
    """Combine TV indices and time indices into contiguous TVTW indices."""

    return tv_indices * num_time_bins + time_indices


def build_sparse_delta(num_tvtws: int, old_cols: np.ndarray, new_cols: np.ndarray) -> sparse.csr_matrix:
    """Construct a 1xN CSR delta vector from removed and added column indices."""

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
    """Yield only the pairs with strictly positive delay values."""

    for flight_id, delay in pairs:
        if delay > 0:
            yield flight_id, delay
