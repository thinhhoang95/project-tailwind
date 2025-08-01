Please write a Python function that returns the estimated time of arrival (ETA) of the aircraft at each waypoint given in a route, taking into account the performance.

# Input
- `route`: for example: `LFPG RESMI TOU LFBO`
- `nodes_graph` (Nodes only): The waypoints can be read from the graph in `D:/project-akrav/data/graphs/ats_fra_nodes_only.gml` networkx gml file. The node id is directly the waypoint name, and each node has two attributes: lat and lon. There are no edges in the graph (in compliance with free route airspace context). Default to None, then read_gml, if not None then use the route graph directly.
- `eta_and_distance_climb_table` and `eta_and_distance_descend_table`, two tables produced from `vnav_performance.py`'s `get_eta_and_distance_climb` and `get_eta_and_distance_descent` methods.
```python
# Example: descent performance table
#   Altitude (ft) |  ETATO (s) |    Distance (nm)
# ------------------------------------------------
#          35,000 |    -1300.0 |           141.50
#          28,000 |     -880.0 |            89.00
#          24,000 |     -640.0 |            59.00
#          20,000 |     -560.0 |            35.00
#          10,000 |     -360.0 |            15.00
#           1,000 |        0.0 |             0.00
```
- The cruise ground speed. Default to 460kts.
- The cruise altitude. In feet. For example: 35000.

# Output
Two lists: a list of altitude at each waypoint, and a list of elapsed time in seconds from takeoff.

# Algorithm

1. **Parse the route string** into an ordered list of waypoint identifiers (ICAO codes for the origin and destination airports plus any intermediate fixes). Split on whitespace, drop empty tokens and convert everything to upper-case.

2. **Load waypoint geometry**
   - If `nodes_graph` is `None`, read the GML graph with `networkx.read_gml(...)` using the path given in the input description.
   - Otherwise reuse the supplied `nodes_graph` argument.
   Each node’s `lat` and `lon` attributes give the geographical position of the waypoint.

3. **Build the horizontal distance array**
   - Convert successive waypoint coordinates to great-circle distance (e.g. haversine formula) in nautical miles.
   - Produce `cum_dist_wp[i]`: cumulative along-track distance from take-off up to (and including) waypoint *i*. The total route length is `route_len = cum_dist_wp[-1]`.

4. **Extract climb & descent performance data**
   - `eta_and_distance_climb_table` and `eta_and_distance_descend_table` are produced by `vnav_performance.get_eta_and_distance_*` and contain triples `(alt_ft, eta_sec, dist_nm)`.
   - For climb the distance column is **distance flown since take-off**; the last row therefore represents the Top-of-Climb (TOC).
   - For descent the distance column is **distance-to-go** from Top-of-Descent (TOD) to landing; the first row corresponds to cruise altitude, the last row to runway elevation.
   - Define
     ```python
     d_climb   = eta_and_distance_climb_table[-1][2]   # nm from take-off to TOC
     d_descent = eta_and_distance_descend_table[-1][2] # nm from TOD to landing
     ```

5. **Locate TOC and TOD on the route**
   - TOC: first point where `cum_dist_wp ≥ d_climb`.
   - TOD: first point where `cum_dist_wp ≥ route_len − d_descent`.
   - If a transition falls inside a leg, insert a synthetic waypoint by linear interpolation of latitude, longitude and cumulative distance so that TOC / TOD become explicit elements of the sequence.

6. **Generate the altitude profile**
   - **Climb segment (origin → TOC)**: interpolate altitude linearly between climb-table knot points using the along-track distance column.
   - **Cruise segment (TOC → TOD)**: constant altitude = `cruise_altitude`.
   - **Descent segment (TOD → destination)**: interpolate altitude between descent-table knot points, but using **distance-to-go** = `route_len − cum_dist_wp` when looking up the table.

7. **Generate the elapsed-time profile**
   - **Climb**: for every waypoint before TOC, obtain elapsed time by linear interpolation in the climb table (`dist_nm ↔ eta_sec`).
   - **Cruise**: starting with the elapsed time at TOC, accumulate `leg_time = (leg_distance_nm / cruise_ground_speed_kts) * 3600` for each cruise leg.
   - **Descent**: for waypoints after TOD, interpolate elapsed time in the descent table using distance-to-go, then compute `elapsed_time_wp = time_at_TOD + interpolated_time_from_TOD`.

8. **Return** two parallel lists (including any inserted TOC/TOD waypoints):
   - `altitudes_ft`: aircraft altitude when crossing each waypoint.
   - `elapsed_times_s`: cumulative time, in seconds, since brake-release / take-off.
