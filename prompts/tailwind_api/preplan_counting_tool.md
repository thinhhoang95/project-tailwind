Can you help me plan for the API to support the **occupancy counting tool** front-end interface. 

# Inputs

#### Filtering Parameters
- `traffic_volume_ids`: filtering for list of traffic volume IDs.
- `from_time_str`, `to_time_str`: *from* and *to* time in format HHMMSS. For example: 071226 is 07:12:26.

#### Clustering or Categorization Parameters
- (Optional) an object detailing `{"flow_id": [flight_ids]}`.

#### Data Source
- The input source would need to contain three information: the flight identifier, the occupancy vector for each flight. The current `so6_occupancy_matrix_with_times.json` satisfies this condition, but the feeding source might not limit to this json file (e.g., thanks to optimization, we might have a revised occupancy matrix).

# Outputs

- For each traffic volume ID, a vector of N (for example: N = 96 bins if we are seeing 15-minute time length bin) of occupancy count for each time bin.

- If clustering/categorization is supplied, for each traffic volume, **for each category/cluster/flow**, a vector of N of occupancy count for each time bin.

# Instructions

Can you inspect the current occupancy related source and:

1. Propose a unified input format for Data Source: ideally, the current loaded JSON can be used directly (as loading the file could be expensive in memory and time). But nevertheless, a data structure of this should be detailed so other modules providing counting outputs must adhere to.

2. Propose a unified output format for the task.

3. Give detailed examples.