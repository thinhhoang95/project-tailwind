Please help me plan for the implementation of a *flow extraction algorithm*.

# High-level Goal

Given the flight list that is retrieved from the hotspot via the method `get_hotspot_flights` from the NetworkEvaluator, use the Leiden Algorithm to perform community detection on the basis of the Jaccard distance by comparing the **traffic volume footprint** of each flight.

All Flow Extraction logic will be implemented in the directory `src/project_tailwind/subflows`, though necessary extensions of other modules might be required.

To achieve this task, it might be necessary to perform the following subtasks:

1. Extend the `flight_list.py` module to add a method to quickly retrieve the *traffic volume footprint* (see definition below), given the `flight_identifiers`. It is only sufficient to return the **index of the traffic volumes**, no need for the `traffic_volume_id` in string.

2. Implement the similarity matrix computation method: `compute_jaccard_similarity` to return the Jaccard similarity matrix for all the `flight_identifiers` given. However, if a particular hotspot's traffic volume index is given, known as the `hotspot_tv_index`, then compute the Jaccard similarity from the first traffic volume to the designated traffic volume only.

    > Note: the given `hotspot_tv_index` is the index of the hotspot traffic volume. For example: the traffic volume `EGLSW13` will have the (row) index of 221, according to `tvtw_indexer.py`, and the traffic volume footprint is [216, 372, 529, 221, 335, 337, ...] then we only take the [216, 372, 529, 221] for computation of the similarity matrix. It is NOT the same as we take the 220th element of the footprint array.

3. From the similarity matrix, run the Leiden community detection algorithm with `leidenalg` to return the cluster/community assignment for each flight. There is an argument to control the `threshold` to building the graph from the similarity matrix.

## Output requirement

> A dictionary that assigns flight_identifier to the community index.

4. Write a wrapper that bundle steps 2 and 3 and can be called externally to get the community assignment.

## Definitions
- A traffic volume footprint is a set of traffic volumes that each flight passes through.

# Test
Please write a unit test with synthetic data to ensure the implementation is sound and correct.

# Requirements
- Use vectorization when it fits to increase performance.