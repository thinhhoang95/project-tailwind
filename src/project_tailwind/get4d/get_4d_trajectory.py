from typing import List, Dict, Any, Tuple
import networkx as nx
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def _haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the earth (specified in decimal degrees).
    Returns distance in nautical miles.
    """
    R = 6371.0  # Radius of Earth in kilometers
    
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance_km = R * c
    return distance_km / 1.852  # Convert km to nautical miles

def get_4d_trajectory(
    route: str,
    nodes_graph: nx.Graph = None,
    graph_path: str = None,
    eta_and_distance_climb_table: List[Tuple[float, float, float]] = None,
    eta_and_distance_descend_table: List[Tuple[float, float, float]] = None,
    cruise_ground_speed_kts: float = 460.0,
    cruise_altitude_ft: float = 35000.0,
) -> Tuple[List[float], List[float]]:
    """
    Calculates the estimated time of arrival (ETA) and altitude at each waypoint of a given route.

    Args:
        route: A string representing the route, e.g., "LFPG RESMI TOU LFBO".
        nodes_graph: A networkx graph containing waypoint data. If None, it's loaded from graph_path.
        graph_path: Path to the GML file with waypoint data.
        eta_and_distance_climb_table: Performance table for climb.
        eta_and_distance_descend_table: Performance table for descent.
        cruise_ground_speed_kts: Cruise ground speed in knots.
        cruise_altitude_ft: Cruise altitude in feet.

    Returns:
        A tuple containing two lists:
        - altitudes_ft: Altitude in feet at each waypoint.
        - elapsed_times_s: Elapsed time in seconds from takeoff at each waypoint.
    """
    # 1. Parse route string
    waypoint_names = [wp.upper() for wp in route.split() if wp]
    if not waypoint_names:
        return [], []

    # 2. Load waypoint geometry
    if nodes_graph is None:
        if graph_path is None:
            raise ValueError("Either nodes_graph or graph_path must be provided.")
        nodes_graph = nx.read_gml(graph_path)

    waypoints = []
    for name in waypoint_names:
        if name not in nodes_graph.nodes:
            raise ValueError(f"Waypoint '{name}' not found in the nodes graph.")
        node = nodes_graph.nodes[name]
        waypoints.append({'name': name, 'lat': node['lat'], 'lon': node['lon']})

    # 3. Build horizontal distance array
    cum_dist_wp_nm = [0.0]
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i+1]
        dist = _haversine_distance_nm(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
        cum_dist_wp_nm.append(cum_dist_wp_nm[-1] + dist)
    route_len_nm = cum_dist_wp_nm[-1]

    # 4. Extract climb & descent performance data
    if not eta_and_distance_climb_table or not eta_and_distance_descend_table:
        raise ValueError("Climb and descent performance tables must be provided.")

    climb_dist_data = np.array([row[2] for row in eta_and_distance_climb_table])
    climb_alt_data = np.array([row[0] for row in eta_and_distance_climb_table])
    climb_eta_data = np.array([row[1] for row in eta_and_distance_climb_table])

    # The descent table is sorted by altitude descending (TOD to landing).
    # Distance-to-go is therefore also descending. `np.interp` requires
    # monotonically increasing x-values, so we must reverse the table.
    d_descent = eta_and_distance_descend_table[0][2]
    
    desc_table_reversed = eta_and_distance_descend_table[::-1]

    desc_dist_to_go_data = np.array([row[2] for row in desc_table_reversed])
    desc_alt_data = np.array([row[0] for row in desc_table_reversed])
    desc_time_to_go_data = np.array([abs(row[1]) for row in desc_table_reversed])

    d_climb = climb_dist_data[-1]
    
    # 5. Locate TOC and TOD and build final waypoint list
    toc_dist = d_climb
    tod_dist = route_len_nm - d_descent

    events = []
    for i, wp in enumerate(waypoints):
        events.append({'dist': cum_dist_wp_nm[i], 'wp': wp})

    if toc_dist < tod_dist:
        events.append({'dist': toc_dist, 'type': 'TOC'})
        events.append({'dist': tod_dist, 'type': 'TOD'})
    
    events.sort(key=lambda x: x['dist'])

    final_waypoints_with_dist = []
    seen_dists = set()
    for event in events:
        if event['dist'] not in seen_dists:
            if 'wp' in event:
                final_waypoints_with_dist.append({'dist': event['dist'], 'waypoint': event['wp']})
            else: # Synthetic waypoint (TOC/TOD)
                for i in range(len(cum_dist_wp_nm) - 1):
                    if cum_dist_wp_nm[i] <= event['dist'] < cum_dist_wp_nm[i+1]:
                        p1 = waypoints[i]
                        p2 = waypoints[i+1]
                        leg_dist = cum_dist_wp_nm[i+1] - cum_dist_wp_nm[i]
                        fraction = (event['dist'] - cum_dist_wp_nm[i]) / leg_dist if leg_dist > 0 else 0.0
                        synth_wp = {
                            'name': event['type'],
                            'lat': p1['lat'] + fraction * (p2['lat'] - p1['lat']),
                            'lon': p1['lon'] + fraction * (p2['lon'] - p1['lon'])
                        }
                        final_waypoints_with_dist.append({'dist': event['dist'], 'waypoint': synth_wp})
                        break
            seen_dists.add(event['dist'])

    final_cum_dists = [item['dist'] for item in final_waypoints_with_dist]
    
    # 6. Generate altitude profile
    altitudes_ft = []
    if toc_dist < tod_dist:
        for d in final_cum_dists:
            if d <= toc_dist:
                alt = np.interp(d, climb_dist_data, climb_alt_data)
            elif d > tod_dist:
                dist_to_go = route_len_nm - d
                alt = np.interp(dist_to_go, desc_dist_to_go_data, desc_alt_data)
            else:
                alt = cruise_altitude_ft
            altitudes_ft.append(alt)
    else: # Overlapping climb/descent
        for d in final_cum_dists:
            alt_climb = np.interp(d, climb_dist_data, climb_alt_data)
            dist_to_go = route_len_nm - d
            alt_descent = np.interp(dist_to_go, desc_dist_to_go_data, desc_alt_data)
            altitudes_ft.append(min(alt_climb, alt_descent))

    # 7. Generate elapsed-time profile
    elapsed_times_s = []
    if toc_dist < tod_dist:
        time_at_toc = np.interp(toc_dist, climb_dist_data, climb_eta_data)
        time_at_tod = time_at_toc + ((tod_dist - toc_dist) / cruise_ground_speed_kts * 3600)
        total_descent_time = desc_time_to_go_data[-1]

        for d in final_cum_dists:
            if d <= toc_dist:
                eta = np.interp(d, climb_dist_data, climb_eta_data)
            elif d > tod_dist:
                dist_to_go = route_len_nm - d
                time_to_go = np.interp(dist_to_go, desc_dist_to_go_data, desc_time_to_go_data)
                time_from_tod = total_descent_time - time_to_go
                eta = time_at_tod + time_from_tod
            else: # Cruise
                eta = time_at_toc + ((d - toc_dist) / cruise_ground_speed_kts * 3600)
            elapsed_times_s.append(eta)
    else: # Overlapping climb/descent - this is a simplification
        # This case requires finding the intersection point and is complex.
        # A simple time calculation is difficult without a clear single profile.
        # Returning altitudes but empty times for this complex case.
        # A full implementation would solve for the intersection altitude and time.
        raise NotImplementedError("Overlapping climb/descent is not implemented yet.")
        return altitudes_ft, []

    # 8. Return
    return altitudes_ft, elapsed_times_s


if __name__ == '__main__':
    from project_tailwind.vnav.vnav_performance import Performance, get_eta_and_distance_climb, get_eta_and_distance_descent
    from project_tailwind.vnav.vnav_profiles_rev1 import (
        NARROW_BODY_JET_CLIMB_PROFILE,
        NARROW_BODY_JET_DESCENT_PROFILE,
        NARROW_BODY_JET_CLIMB_VS_PROFILE,
        NARROW_BODY_JET_DESCENT_VS_PROFILE,
    )

    # Example Usage
    route_str = "LFPG LFPB LFPY LFCR"
    cruise_alt = 35000.0
    cruise_gs = 460.0

    # 1. Create a sample graph for the waypoints
    G = nx.Graph()
    G.add_node("LFPG", lat=49.0097, lon=2.5479)
    G.add_node("LFPB", lat=48.7233, lon=2.3794)
    G.add_node("LFPY", lat=48.3539, lon=2.2386)
    G.add_node("LFCR", lat=44.2206, lon=4.7656)

    # 2. Setup performance model
    perf_model = Performance(
        climb_speed_profile=NARROW_BODY_JET_CLIMB_PROFILE,
        descent_speed_profile=NARROW_BODY_JET_DESCENT_PROFILE,
        climb_vertical_speed_profile=NARROW_BODY_JET_CLIMB_VS_PROFILE,
        descent_vertical_speed_profile=[(alt, -vs) for alt, vs in NARROW_BODY_JET_DESCENT_VS_PROFILE],
        cruise_altitude_ft=cruise_alt,
        cruise_speed_kts=cruise_gs,
    )

    # 3. Generate climb and descent tables
    climb_table = get_eta_and_distance_climb(perf_model, origin_airport_elevation_ft=0)
    descent_table = get_eta_and_distance_descent(perf_model, destination_airport_elevation_ft=0)
    
    # 4. Get the 4D trajectory
    altitudes, times = get_4d_trajectory(
        route=route_str,
        nodes_graph=G,
        eta_and_distance_climb_table=climb_table,
        eta_and_distance_descend_table=descent_table,
        cruise_ground_speed_kts=cruise_gs,
        cruise_altitude_ft=cruise_alt,
    )

    # 5. Print results
    print(f"4D Trajectory for route: {route_str}")
    print("-" * 50)
    # Re-parse route to get waypoint names for printing
    waypoints_for_print = [wp.upper() for wp in route_str.split() if wp]
    # This part needs adjustment to get the final waypoint list including synthetic ones
    # For now, we'll just print the raw lists. A more advanced printout would need the final waypoint list from the function.
    
    print(f"{'Altitude (ft)':>15} | {'Elapsed Time (s)':>20}")
    print("-" * 50)
    for alt, t in zip(altitudes, times):
        print(f"{alt:15.2f} | {t:20.2f}")

