from typing import List
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


def delay_operator(
    occupancy_vector: List[int], delay_min: int, indexer: TVTWIndexer
) -> List[int]:
    """
    Applies a ground delay to a flight's occupancy vector.

    Args:
        occupancy_vector: A list of TVTW indices representing the flight's trajectory.
        delay_min: The ground delay in minutes.
        indexer: An instance of TVTWIndexer to resolve indices.

    Returns:
        A new occupancy vector with the delay applied.
    """
    if delay_min == 0:
        return occupancy_vector

    time_bin_shift = delay_min // indexer.time_bin_minutes
    if time_bin_shift == 0:
        # If the delay is smaller than the time bin, no change in the occupancy vector
        return occupancy_vector

    new_occupancy_vector = []
    for tvtw_index in occupancy_vector:
        tv_id, time_window_idx = indexer.get_tvtw_from_index(tvtw_index)
        if tv_id is not None:
            new_time_window_idx = time_window_idx + time_bin_shift
            
            # We need to handle cases where the new time window wraps around to the next day,
            # but for now we assume it stays within the same day's 
            # available time bins.
            new_time_window_idx = new_time_window_idx % indexer.num_time_bins

            new_tvtw_index = indexer.get_tvtw_index(tv_id, new_time_window_idx)
            if new_tvtw_index is not None:
                new_occupancy_vector.append(new_tvtw_index)
            else:
                # This case should ideally not happen if the indexer is consistent.
                # It means there is no global index for the new (tv_id, time_window_idx)
                # which would be strange. We will just drop it from the vector for now.
                raise ValueError(f"No global index found for (tv_id, time_window_idx) = ({tv_id}, {new_time_window_idx})")
        else:
            # Handle case where tvtw_index is not found in the indexer
            raise ValueError(f"TVTW index {tvtw_index} not found in the indexer")
            
    return new_occupancy_vector

