from datetime import timedelta
from typing import List, Dict, Any
import numpy as np

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList


def delay_operator(
    flight_id: str,
    delay_min: int,
    state: FlightList,
    indexer: TVTWIndexer,
) -> None:
    """
    Applies a ground delay to a flight in-place within the state.
    Optimized for maximum performance.

    Args:
        flight_id: The identifier of the flight to be delayed.
        delay_min: The ground delay in minutes.
        state: The FlightList object to be modified.
        indexer: An instance of TVTWIndexer to resolve indices.
    """
    if delay_min == 0:
        return

    # Cache frequently accessed data
    flight_meta = state.flight_metadata[flight_id]
    flight_data_entry = state.flight_data[flight_id]
    
    # 1. Update takeoff_time (minimal overhead)
    delay_delta = timedelta(minutes=delay_min)
    new_takeoff_time = flight_meta["takeoff_time"] + delay_delta
    
    flight_meta["takeoff_time"] = new_takeoff_time
    flight_data_entry["takeoff_time"] = new_takeoff_time.isoformat()

    # 2. Fast path for sub-bin delays
    time_bin_shift = delay_min // indexer.time_bin_minutes
    if time_bin_shift == 0:
        return

    # 3. Vectorized interval processing
    occupancy_intervals = flight_meta["occupancy_intervals"]
    num_intervals = len(occupancy_intervals)
    
    if num_intervals == 0:
        return
    
    # Pre-allocate arrays for batch processing
    tvtw_indices = np.empty(num_intervals, dtype=np.int32)
    entry_times = np.empty(num_intervals, dtype=np.float32)
    exit_times = np.empty(num_intervals, dtype=np.float32)
    
    # Extract data in single pass
    for i, interval in enumerate(occupancy_intervals):
        tvtw_indices[i] = interval["tvtw_index"]
        entry_times[i] = interval["entry_time_s"]
        exit_times[i] = interval["exit_time_s"]
    
    # Batch convert indices to tv_ids and time windows
    # Use vectorized operations if indexer supports it
    if hasattr(indexer, 'get_tvtw_from_indices_batch'):
        # If batch method exists, use it
        tv_ids, time_windows = indexer.get_tvtw_from_indices_batch(tvtw_indices)
    else:
        # Fallback to manual vectorization with pre-allocated arrays
        tv_ids = np.empty(num_intervals, dtype=np.int32)
        time_windows = np.empty(num_intervals, dtype=np.int32)
        
        # Use indexer's internal lookup tables if available
        if hasattr(indexer, '_index_to_tvtw'):
            # Direct lookup table access (much faster)
            for i, idx in enumerate(tvtw_indices):
                tv_id, tw = indexer._index_to_tvtw[idx]
                tv_ids[i] = tv_id
                time_windows[i] = tw
        else:
            # Fall back to method calls
            for i, idx in enumerate(tvtw_indices):
                tv_id, tw = indexer.get_tvtw_from_index(idx)
                tv_ids[i] = tv_id
                time_windows[i] = tw
    
    # Vectorized time window shift
    new_time_windows = (time_windows + time_bin_shift) % indexer.num_time_bins
    
    # Batch convert back to indices
    if hasattr(indexer, 'get_tvtw_indices_batch'):
        new_tvtw_indices = indexer.get_tvtw_indices_batch(tv_ids, new_time_windows)
    else:
        # Optimized single-pass conversion
        new_tvtw_indices = np.empty(num_intervals, dtype=np.int32)
        
        # Use indexer's internal lookup if available
        if hasattr(indexer, '_tvtw_to_index'):
            for i in range(num_intervals):
                key = (tv_ids[i], new_time_windows[i])
                new_tvtw_indices[i] = indexer._tvtw_to_index.get(key, -1)
        else:
            for i in range(num_intervals):
                new_tvtw_indices[i] = indexer.get_tvtw_index(tv_ids[i], new_time_windows[i])
    
    # Check for invalid indices
    if np.any(new_tvtw_indices < 0):
        invalid_idx = np.where(new_tvtw_indices < 0)[0][0]
        raise ValueError(
            f"No global index found for (tv_id, time_window_idx) = "
            f"({tv_ids[invalid_idx]}, {new_time_windows[invalid_idx]})"
        )
    
    # Build new intervals list efficiently
    new_occupancy_intervals = [
        {
            "tvtw_index": int(new_tvtw_indices[i]),
            "entry_time_s": entry_times[i],
            "exit_time_s": exit_times[i],
        }
        for i in range(num_intervals)
    ]
    
    # Update both metadata and flight_data (by reference, no copy)
    flight_meta["occupancy_intervals"] = new_occupancy_intervals
    flight_data_entry["occupancy_intervals"] = new_occupancy_intervals
    
    # 4. Efficient occupancy vector update
    # Instead of creating full zero vector, directly update with sparse representation
    if hasattr(state, 'update_flight_occupancy_sparse'):
        # If sparse update method exists, use it
        state.update_flight_occupancy_sparse(flight_id, new_tvtw_indices)
    else:
        # Optimized dense update - only allocate if necessary
        num_tvtws = state.get_matrix_shape()[1]
        
        # Use pre-allocated buffer if available
        if hasattr(state, '_temp_occupancy_buffer'):
            new_occupancy_vector = state._temp_occupancy_buffer
            new_occupancy_vector.fill(0)
        else:
            new_occupancy_vector = np.zeros(num_tvtws, dtype=np.float32)
        
        # Set indices directly using advanced indexing
        if num_intervals > 0:
            new_occupancy_vector[new_tvtw_indices] = 1.0
        
        state.update_flight_occupancy(flight_id, new_occupancy_vector)


# Alternative: Batch delay operator for multiple flights
import numpy as np
from datetime import timedelta
from typing import Dict

# --- ✂️ helper ---------------------------------------------------------------

def _ensure_indexer_cache(indexer):
    """
    Lazily build two NumPy lookup tables on the indexer the first time we
    need them so later calls become O(1) array slicing instead of millions
    of Python calls.
    """
    if getattr(indexer, "_tv_id_by_idx", None) is None:
        n_idx = len(indexer._tv_id_to_idx) * indexer.num_time_bins            # <- whatever your size is
        # Fast vectorised inverse mapping: idx -> (tv_id, tw)
        tv_id_by_idx = np.arange(n_idx, dtype=np.int32) // indexer.num_time_bins
        tw_by_idx     = np.arange(n_idx, dtype=np.int32) %  indexer.num_time_bins

        # Forward mapping (tv_id, tw) -> idx
        # Shape: (num_tvs, num_time_bins)
        num_tvs = len(indexer._tv_id_to_idx)
        idx_by_tv_tw = (
            np.arange(num_tvs, dtype=np.int32).reshape(-1, 1) * indexer.num_time_bins +
            np.arange(indexer.num_time_bins, dtype=np.int32)[None, :]
        )

        indexer._tv_id_by_idx = tv_id_by_idx
        indexer._tw_by_idx    = tw_by_idx
        indexer._idx_by_tv_tw = idx_by_tv_tw


# --- 🚀 vectorised batch delay -----------------------------------------------

def batch_delay_operator(
    flight_delays: Dict[str, int],
    state: "FlightList",
    indexer: "TVTWIndexer",
) -> None:
    if not flight_delays:
        return

    # ------------------------------------------------------------------ step 0
    active_items   = [(fid, d) for fid, d in flight_delays.items() if d > 0]
    if not active_items:
        return

    _ensure_indexer_cache(indexer)  # build the look-up tables once
    tv_id_by_idx, tw_by_idx, idx_by_tv_tw = (
        indexer._tv_id_by_idx,
        indexer._tw_by_idx,
        indexer._idx_by_tv_tw,
    )

    # ------------------------------------------------------------------ step 1
    # Collect every interval *once* into flat NumPy buffers
    flight_id_arr, offset_arr, shift_arr, entry_s_arr, exit_s_arr = [], [], [], [], []

    intervals_flat = []          # keeps original dicts so we can mutate in-place
    for fid, delay_min in active_items:
        meta = state.flight_metadata[fid]

        # 1 a. update take-off
        td = timedelta(minutes=delay_min)
        meta["takeoff_time"] += td
        state.flight_data[fid]["takeoff_time"] = meta["takeoff_time"].isoformat()

        # 1 b. collect occupancy intervals if we need to shift them
        bins = delay_min // indexer.time_bin_minutes
        if bins:
            ivs = meta["occupancy_intervals"]
            start = len(intervals_flat)
            intervals_flat.extend(ivs)

            flight_id_arr.append(fid)
            offset_arr.append(start)
            shift_arr.append(bins)

            # Stash entry/exit once so we can rebuild without per-element dict work
            entry_s_arr.extend([iv["entry_time_s"] for iv in ivs])
            exit_s_arr.extend([iv["exit_time_s"]  for iv in ivs])

    if not intervals_flat:
        return

    # ------------------------------------------------------------------ step 2
    # Vectorised idx -> (tv_id, tw)
    tvtw_idx_np = np.fromiter(
        (iv["tvtw_index"] for iv in intervals_flat),
        dtype=np.int32,
        count=len(intervals_flat),
    )
    tv_ids = tv_id_by_idx[tvtw_idx_np]
    tws = tw_by_idx[tvtw_idx_np]

    # ------------------------------------------------------------------ step 3
    # One pass per *flight*, but on slice views—no Python loops over intervals
    entry_s_np = np.asarray(entry_s_arr, dtype=np.int32)
    exit_s_np  = np.asarray(exit_s_arr,  dtype=np.int32)
    num_tvtws  = state.get_matrix_shape()[1]

    for fid, start, bins in zip(flight_id_arr, offset_arr, shift_arr):
        # slice view for this flight
        sl       = slice(start, start + len(state.flight_metadata[fid]["occupancy_intervals"]))
        new_tws  = (tws[sl] + bins) % indexer.num_time_bins
        new_idx  = idx_by_tv_tw[tv_ids[sl], new_tws]

        # build list-of-dicts without Python comprehension
        new_intervals = [
            {"tvtw_index": int(idx), "entry_time_s": int(ent), "exit_time_s": int(ext)}
            for idx, ent, ext in zip(new_idx, entry_s_np[sl], exit_s_np[sl])
        ]

        # metadata + raw record
        state.flight_metadata[fid]["occupancy_intervals"] = new_intervals
        state.flight_data[fid]["occupancy_intervals"]     = new_intervals

        # sparse occupancy update: only touch changed bins
        state.clear_flight_occupancy(fid)                 # (assumes you have one)
        state.add_flight_occupancy(fid, new_idx)
    
    # Finalize all updates in one batch conversion from LIL to CSR
    if hasattr(state, 'finalize_occupancy_updates'):
        state.finalize_occupancy_updates()

