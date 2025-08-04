import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional

from project_tailwind.rqs.bitmap_querying_system import RouteQuerySystem
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

def find_alternative_route(
    origin: str,
    destination: str,
    takeoff_time: datetime,
    overloaded_tvtws: List[int],
    rqs: RouteQuerySystem,
    tvtw_indexer: TVTWIndexer
) -> Optional[Tuple[str, np.ndarray, float]]:
    """
    Finds an alternative route for a flight to avoid a list of overloaded TVTWs.

    Args:
        origin: The origin airport ICAO code.
        destination: The destination airport ICAO code.
        takeoff_time: The scheduled takeoff time of the flight.
        overloaded_tvtws: A list of TVTW indices that the flight should avoid. These indices are
                          for the flight's actual takeoff time.
        rqs: An initialized RouteQuerySystem instance. The routes in the RQS are based
             on a nominal midnight takeoff.
        tvtw_indexer: An initialized TVTWIndexer instance.

    Returns:
        A tuple containing (route_string, new_impact_vector, distance) for the best
        alternative route, or None if no alternative is found. The returned impact
        vector is adjusted for the flight's actual takeoff time.
    """
    time_bin_minutes = tvtw_indexer.time_bin_minutes
    num_time_bins = tvtw_indexer.num_time_bins
    
    takeoff_offset_minutes = takeoff_time.hour * 60 + takeoff_time.minute
    takeoff_offset_bins = takeoff_offset_minutes // time_bin_minutes

    banned_reference_tvtws = []
    for tvtw_idx in overloaded_tvtws:
        tvtw_tuple = tvtw_indexer.get_tvtw_from_index(tvtw_idx)
        if tvtw_tuple is None:
            continue
        
        tv_id, time_window_idx_actual = tvtw_tuple
        tv_idx = tvtw_indexer._tv_id_to_idx.get(tv_id)
        
        if tv_idx is None:
            continue

        time_window_idx_ref = (time_window_idx_actual - takeoff_offset_bins + num_time_bins) % num_time_bins
        
        ref_tvtw_idx = tv_idx * num_time_bins + time_window_idx_ref
        banned_reference_tvtws.append(ref_tvtw_idx)

    alternative_routes = rqs.get_routes_avoiding_OD(origin, destination, banned_reference_tvtws)

    if not alternative_routes:
        return None

    best_route = min(alternative_routes, key=lambda x: x[1])
    best_route_str, best_distance = best_route

    ref_impact_vector = rqs.get_vector(best_route_str)
    if ref_impact_vector is None:
        return None

    actual_impact_vector = []
    for ref_tvtw_idx in ref_impact_vector:
        ref_tvtw_idx = int(ref_tvtw_idx)
        tv_idx = ref_tvtw_idx // num_time_bins
        time_window_idx_ref = ref_tvtw_idx % num_time_bins
        
        time_window_idx_actual = (time_window_idx_ref + takeoff_offset_bins) % num_time_bins
        
        actual_tvtw_idx = tv_idx * num_time_bins + time_window_idx_actual
        actual_impact_vector.append(actual_tvtw_idx)

    return (best_route_str, np.array(actual_impact_vector, dtype=np.uint32), best_distance)
