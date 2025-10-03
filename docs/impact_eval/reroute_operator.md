# Reroute Operator Documentation

The `find_alternative_route` function is an operator designed to find a new flight path between an origin and destination that avoids a given set of congested airspace volumes (TVTWs). It queries a pre-computed database of routes to find a viable, non-congested alternative.

## Algorithm Details

A key challenge in rerouting is that the available routes in the `RouteQuerySystem` (RQS) are stored relative to a nominal takeoff time (e.g., midnight), while the congestion information (`overloaded_tvtws`) is for a flight's *actual* takeoff time. The algorithm must therefore translate between these two time reference frames.

The algorithm is executed in four main stages:

1.  **Translate to Reference Frame**: The operator first translates the list of `overloaded_tvtws` (which are in the "actual time" frame) into the RQS's "reference time" frame.
    *   It calculates a `takeoff_offset_bins` value, representing the time difference between the flight's actual takeoff and the nominal midnight takeoff, measured in time bins.
    *   For each `tvtw_idx` in `overloaded_tvtws`, it resolves the index to its `tv_id` and `time_window_idx_actual`.
    *   It then subtracts the `takeoff_offset_bins` from `time_window_idx_actual` to find the corresponding `time_window_idx_ref`. This gives the time window in the reference frame that needs to be avoided.
    *   These reference TVTWs are collected into a list of `banned_reference_tvtws`.

2.  **Query for Alternative Routes**: The function calls the `rqs.get_routes_avoiding_OD` method. It passes the flight's `origin`, `destination`, and the `banned_reference_tvtws`. The RQS searches its database for all routes that connect the origin and destination without passing through any of the banned reference TVTWs.

3.  **Select Best Route**: If the query returns no alternative routes, the function returns `None`. Otherwise, it selects the best available route from the results. The "best" route is defined as the one with the shortest great-circle distance.

4.  **Translate Back to Actual Frame**: Once the best alternative route is chosen, its impact vector (which is in the reference frame) must be translated back to the actual time frame of the flight.
    *   This process is the reverse of Step 1. The operator iterates through the chosen route's reference impact vector.
    *   For each reference TVTW index, it adds the `takeoff_offset_bins` to the time window component to get the actual time window.
    *   These new TVTW indices are collected to form the `actual_impact_vector`.

Finally, the function returns the new route's string representation, its `actual_impact_vector`, and its distance.

## Inputs and Outputs

### Inputs

-   `origin` (`str`): The ICAO code of the origin airport (e.g., "KSFO").
-   `destination` (`str`): The ICAO code of the destination airport (e.g., "KJFK").
-   `takeoff_time` (`datetime`): The scheduled takeoff time of the flight.
-   `overloaded_tvtws` (`List[int]`): A list of TVTW indices that are congested and must be avoided. These indices are relative to the flight's actual `takeoff_time`.
-   `rqs` (`RouteQuerySystem`): An initialized instance of the route query system containing a database of nominal routes.
-   `tvtw_indexer` (`TVTWIndexer`): An initialized instance of the TVTW indexer.

### Outputs

-   `Optional[Tuple[str, np.ndarray, float]]`: A tuple containing the following information for the best alternative route found:
    -   `route_string` (`str`): The string representation of the new route.
    -   `new_impact_vector` (`np.ndarray`): The impact vector for the new route, adjusted for the actual takeoff time.
    -   `distance` (`float`): The great-circle distance of the new route.
-   Returns `None` if no suitable alternative route can be found.

## Usage Example

```python
import numpy as np
from datetime import datetime
from project_tailwind.impact_eval.operators.reroute import find_alternative_route

# Assume mock RQS and TVTWIndexer classes for this example
class MockTVTWIndexer:
    def __init__(self, time_bin_minutes, num_time_bins):
        self.time_bin_minutes = time_bin_minutes
        self.num_time_bins = num_time_bins
        self._tv_id_to_idx = {'TV_A': 0, 'TV_B': 1, 'TV_C': 2}
        self._tv_idx_to_id = {v: k for k, v in self._tv_id_to_idx.items()}

    def get_tvtw_from_index(self, index):
        tv_idx = index // self.num_time_bins
        time_idx = index % self.num_time_bins
        tv_id = self._tv_idx_to_id.get(tv_idx)
        return tv_id, time_idx

class MockRQS:
    def get_routes_avoiding_OD(self, origin, dest, banned_tvtws):
        print(f"Searching for routes from {origin} to {dest} avoiding {banned_tvtws}")
        # In our scenario, the banned reference TVTW will be 100.
        if 100 in banned_tvtws:
            # Return an alternative route that is known to be clear.
            # Tuple format: (route_string, distance)
            return [("ROUTE_ALT_C", 2600.0)]
        return []

    def get_vector(self, route_str):
        if route_str == "ROUTE_ALT_C":
            # This route's vector in the *reference* frame.
            # It passes through TV_C (index 2) at reference time 5.
            # Reference index = 2 * 96 + 5 = 197.
            return np.array([197])
        return None

# 1. Initialization
tvtw_indexer = MockTVTWIndexer(time_bin_minutes=15, num_time_bins=96)
rqs = MockRQS()

# 2. Flight Details
origin = "KSFO"
destination = "KJFK"
# Takeoff at 8:00 AM. This is 8 * (60/15) = 32 time bins after midnight.
takeoff_time = datetime(2025, 1, 1, 8, 0, 0)

# 3. Congestion Info
# Let's say the original route would pass through TV_B (index 1) at actual time 36.
# This corresponds to an actual TVTW index of: 1 * 96 + 36 = 132.
overloaded_tvtws = [132]
print(f"Overloaded TVTWs (actual time): {overloaded_tvtws}")

# 4. Find an alternative route
alternative = find_alternative_route(
    origin=origin,
    destination=destination,
    takeoff_time=takeoff_time,
    overloaded_tvtws=overloaded_tvtws,
    rqs=rqs,
    tvtw_indexer=tvtw_indexer
)

# 5. Print results
if alternative:
    route_str, impact_vector, distance = alternative
    print(f"Found alternative route: {route_str}")
    print(f"Distance: {distance} miles")
    print(f"New Impact Vector (actual time): {impact_vector}")

# Expected output:
# Overloaded TVTWs (actual time): [132]
# Searching for routes from KSFO to KJFK avoiding [100]
# Found alternative route: ROUTE_ALT_C
# Distance: 2600.0 miles
# New Impact Vector (actual time): [229]
```
