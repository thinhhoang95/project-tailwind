Please plan for the most optimal approach, then implement the code that **derives the count contribution vector** for each routing options described in all the CSV files in `output/representatives_filtered` directory.

The starting point is the code in `src/project_tailwind/traffic_counting/impact_vector_derivation.py`, which currently clones the traffic counting process. More details about this code can be found in `docs/traffic_count/traffic_counter_documentation.md`. 

1. We would like to assign an index to each traffic volume time window (called a TVTW). For example, traffic volume `EBBUELS1` from `00:00-00:30` is one TVTW (the time starts from take off time: 00:00 is the takeoff time). We note this assignment into an external file, and write some helper functions (in a separate python module) that allows quickly conversion between the index and the tuple (traffic volume, time window) - in human readable format.

2. We would like to repurpose the counting code in `impact_vector_derivation` so that it will produce **a sequence of TVTW indices that the count related to the route can be counted towards to**. 

*For example: given the route `LFPG RESMI TOU LFBO`, the computation will give a sequence `121 242 193 207 206` where 121 is the TVTW of the Bordeaux Traffic Volume LFBBZ1 for instance, at takeoff time, 00:00-00:30 for example.* 

## For context
- The waypoints can be read from the graph in `D:/project-akrav/data/graphs/ats_fra_nodes_only.gml` networkx gml file. The node id is directly the waypoint name, and each node has two attributes: lat and lon. There are no edges in the graph (in compliance with free route airspace context).
