
import datetime
from typing import List, Dict, Any

def assign_delays(
    entrants: List[Dict[str, Any]],
    capacity: int,
    window_end: datetime.datetime,
    epsilon_s: int = 1,
) -> List[Dict[str, Any]]:
    """
    Assigns delays to entrants in a CASA window if capacity is exceeded.

    This function implements the "push excess to the end of the window (FIFO)" logic
    from the C-CASA re-adaptation plan. It revises the crossing time for flights
    that exceed the window's capacity.

    Args:
        entrants (List[Dict[str, Any]]): A list of CASA event dictionaries, sorted
                                         by 't_entry_star'. Each dictionary must
                                         contain at least 't_entry_star'.
        capacity (int): The number of flights allowed in this window.
        window_end (datetime.datetime): The end time of the CASA window.
        epsilon_s (int): A small buffer in seconds to push flights strictly
                         beyond the window end time.

    Returns:
        List[Dict[str, Any]]: The list of entrants with updated 't_entry_star'
                              for any delayed flights. The list itself is modified
                              in-place, but also returned for clarity.
    """
    if len(entrants) <= capacity:
        return entrants

    # The first 'capacity' flights are allowed. The rest are delayed.
    flights_to_delay = entrants[capacity:]

    # All excess flights are pushed to at least epsilon_s seconds after the window ends.
    push_to_time = window_end + datetime.timedelta(seconds=epsilon_s)

    for event in flights_to_delay:
        # Set the revised crossing time to be the later of its current scheduled time
        # or the time it's being pushed to.
        revised_time = max(event["t_entry_star"], push_to_time)
        event["t_entry_star"] = revised_time

    return entrants
