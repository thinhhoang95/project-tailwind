1. We run group_by_city_pairs_mp.py to obtain a list of CSV files: for example: LFLF contains all flights originating and landing inside the French airspace. This will help us build a look up system more efficienty later.

2. We then run grouper_auto.py to perform clustering. It is based on Leiden alg for community detection. It helps slim down the set of all historical routes to key routes to help computing the impact vector.

