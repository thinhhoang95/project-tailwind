## Part 1: Reroute Impact/Occupancy Precomputation

From the historical data, the idea is that we will find a more consolidated list of routes, which we can use later for rapid lookup when it comes to finding **rerouting** solutions.

1. We run group_by_city_pairs_mp.py to obtain a list of CSV files: for example: LFLF contains all flights originating and landing inside the French airspace. This will help us build a look up system more efficienty later.

2. We then run grouper_auto.py to perform clustering. It is based on Leiden alg for community detection. It helps slim down the set of all historical routes to key routes to help computing the impact vector.

3. Afterwards, we run the filter_spurrious script to get rid of the "deformed trajectories" that have length > 1.5 times the great circle length. This will help get rid of the faulty trajectories from data processing.

4. We then run impact_vector_computation_with_entry_times.py to compute the traffic volume time window contribution for each routing option. This is the heavy part.

## Part 2: Flight Plan Impact/Occupancy Precomputation

Next, from the original flight plan data (so6 from NEST for instance), we compute the similar "impact vector" (or later in formal documents referred to as *occupancy vector*) for each flight. This will be the baseline to be manipulated through delays and reroutes (which use the estimated vector computed in part 1).

1. This is done through the code `impact_vector_so6_with_entry_times.py` module.

> Attention: if you change the traffic volume definitions (geojson file), you ought to delete the tvtw_indexer.json and rerun the impact_vector_so6_with_entry_times.py again.

## Part 3: Occupancy Operators

## Premises
### Delay and Reroute Operator

Given an *occupancy* (or impact) vector of a flight, the amount of delay given in minutes, return the new, adjusted occupancy vector. This is realized through `delay.py` in `operators`.

Likewise the *reroute* or impact vector of a flight in `reroute.py`, same folder.

---
# The DeepFlow Regulation Plan using ALNS
After theoretical study, the goal is to adopt ALNS to optimize over regulations rather than flights. To do this, we need flow extraction.

## ALNS
See `docs/walkthrough.md` for complete description of the framework. The script can be fired off from `orchestrator.py`.

## Network Evaluator
The helper class will be in charge with computing overload and pulling out the list of flights occupying which hotspot using a Flow Extractor (see below).

It is also capable of getting the raw count for each traffic volume time window (TVTW). See `retrieve_raw_tvtw_count.py` test script. This should be best used in conjunction with project-cirrus's capacity faker (`capacity_faker.py`) script which reproduces a traffic volume definition geojson with the actual count, from which we can manually induce hotspots.

## Flow Extraction
The flow extraction script is `flow_x/flow_extractor.py`. You can check out the docs in `flow_extractor.md`. A complete example can be found in `tests/test_network_evaluator.py`.

# Scenarios
From project-cirrus, we launch the scenario_gen module `main.py` to generate multiple scenarios from the original `geojson` file. The outputs are scenario geojson files, such as summer good wx well staffed low/medium/high. Low/medium/high indicate the cut levels: low is easier to solve than medium, and medium is easier to solve than high. It requires a traffic_volumes_with_capacity.geojson file.

There is absolutely no need to recompute the occupancy vectors for each flight: the scenarios just change the capacity values, and do NOT tamper with the indexing system (tvtw_indexer.json) as well as the time bins nor the definition of the traffic volumes. 