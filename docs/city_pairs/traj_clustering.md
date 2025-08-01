Can you help me implement the following algorithm to build a dictionary of key routing options for each airport pair.

# Goal Context Description
The directory `output/city_pairs/grouped` currently contains a lot of CSV files like `EGLF.csv`, which already grouped flights originating from London and arriving in France.

An example for each csv file is as follows:
```csv
flight_identifier,route
345248VLG1CM,EGKK SFD HAWKE LFPO
3944E1AFR96RF,EGLL OTSID LFXU LFPB LFPG
```

Now, for each airport pair, for example: `EGKK → LFPO`, there could be one flight (many cases), several or a lot (many cases too).

- If there is only one flight for the airport pair, just write it out directly.
- If there are more than 2 flights for each airport pair, we rely on community detection to detect the number of trajectory clusters (or communities), and the **representative trajectory** for each community. Think of this like dimensionality reduction to massively speed up the lookup process that we are building.

# Instructions
We will focus only on the **implementation of the community detection pipeline first**. The module is called `path_airport_pair_grouping.py` which will be implemented in `src/project_tailwind/city_pairs/ap_grouper` directory. There are two tasks to be fulfilled:
1. Compute the pairwise distance between each trajectory in the set. This will give a distance matrix.
2. Perform community detection using `leidenalg`.

### Part 1: Compute the Pairwise distance
1. Write a `compute_distance.py` file inside `src/project_tailwind/city_pairs/ap_grouper` directory. Use `shapely` to compute the distance between any two given paths, given as a list of 2-tuple geographical coordinates (latitude and longitude). You might need to plan a bit on this, because we are not working directly with Euclidean coordinates, but geographical coordinates. Let me know the optimal approach you think might be best suitable.

    > You also need to recommend a threshold value.

2. Write a `community_detection.py` file that takes the distance matrix, run the Leiden algorithm, and return the list of communities. For each community, pick the "most appropriate candidate" for the community. This will be the path that represents the whole group.

3. Finally, write the `grouper.py` which takes a pandas dataframe of the csv file (already filtered by the airport pair), the `networkx` nodes-only graph (gml file, each node contains a pair of lat and lon attributes). It will convert the `real_waypoints` to the list of 2-tuple geographical coordinates that is appropriate for the `compute_distance` module.