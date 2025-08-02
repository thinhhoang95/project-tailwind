1. We run group_by_city_pairs_mp.py to obtain a list of CSV files: for example: LFLF contains all flights originating and landing inside the French airspace. This will help us build a look up system more efficienty later.

2. We then run grouper_auto.py to perform clustering. It is based on Leiden alg for community detection. It helps slim down the set of all historical routes to key routes to help computing the impact vector.

3. Afterwards, we run the filter_spurrious script to get rid of the "deformed trajectories" that have length > 1.5 times the great circle length. This will help get rid of the faulty trajectories from data processing.

4. We then run impact_vector_computation.py to compute the traffic volume time window contribution for each routing option. This is the heavy part.

