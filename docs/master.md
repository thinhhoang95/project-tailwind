# Part 1: Reroute Impact/Occupancy Precomputation

From the historical data, the idea is that we will find a more consolidated list of routes, which we can use later for rapid lookup when it comes to finding **rerouting** solutions.

1. We run group_by_city_pairs_mp.py to obtain a list of CSV files: for example: LFLF contains all flights originating and landing inside the French airspace. This will help us build a look up system more efficienty later.

2. We then run grouper_auto.py to perform clustering. It is based on Leiden alg for community detection. It helps slim down the set of all historical routes to key routes to help computing the impact vector.

3. Afterwards, we run the filter_spurrious script to get rid of the "deformed trajectories" that have length > 1.5 times the great circle length. This will help get rid of the faulty trajectories from data processing.

4. We then run impact_vector_computation.py to compute the traffic volume time window contribution for each routing option. This is the heavy part.

# Part 2: Flight Plan Impact/Occupancy Precomputation

Next, from the original flight plan data (so6 from NEST for instance), we compute the similar "impact vector" (or later in formal documents referred to as *occupancy vector*) for each flight. This will be the baseline to be manipulated through delays and reroutes (which use the estimated vector computed in part 1).

1. This is done through the code `impact_vector_so6.py` module.

# Part 3: Occupancy Operators

### Delay Operator

Given an *occupancy* (or impact) vector of a flight, the amount of delay given in minutes, return the new, adjusted occupancy vector.

