Based on the example in @test_hotspot_flow_retrieval.py , can you help me plan to implement a script in @cache_flow_extract.py (currently empty) that:

1. It will get all the hotspots (tvtw_index, traffic_volume_id string, time_bin index) and export these information into a CSV file.

The second CSV file involves the two following tasks:

1. For each hotspot, retrieve all the traffic flows: including the upstream reference traffic volume ID (a string), the hotspot volume ID, the average pairwise scores and the final score, and all the flight identifiers. Note that each row should indicate one flow, and the flight identifiers are separated by whitespace.
2. Pre-compute the cache values **for each flow**, which will be very helpful for downstream tasks:
    1. CountOver(t): the count of the number of overloaded sectors that the flow contributes to.
    2. SumOver(t): the total number of excess flights that the flow contributes to.
    3. MinSlack(t): of all the TVTWs that the flow (group of flights) traverses, what is the minimum slack (if capacity is still larger than current total occupancy count, slack is the difference between the two; otherwise it is zero).

All of these are indexed by t, in which t = 0 is the original, undelayed flow; t = 1 means every original arrival time bin index of each flight is incremented by 1. For instance: if the original flight in the group pass through section s at time \tau_s, then at t=1, you take the occupancy value at time 1+\tau_s (in unit of time bins). We need to compute from t=0 to t=15 (for your context, each time bin is 15 minutes, so this is equivalent to 4 hours).

Because you have a massive occupancy matrix, it will be tremendously helpful to think about vectorization, and perfoming simultaneous computation of many cache values (CountOver, SumOver, MinSlack) at once. 

It will then save the results (the hotspots the flows-with cached indices) in a designated output directory as two csv files.