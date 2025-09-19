Can you help me plan for an extra parameter  for the `/autorate_occupancy` endpoint that will sort the traffic volumes in the result by three modes:

1. `total_count`: ranks by total count per whole day.

2. `total_excess`: ranks by exceedances (i.e., demand - capacity).

3. `total_changes`: by the sum of absolute changes between post optimization and pre optimization occupancy counts.

For 1 and 2, the logic should be consistent with `/original_counts`'s `rank_by` parameters. You may inspect how these were handled for planning.

For 3, we require the client to provide us with an object that pairs the traffic volume to the sum of absolute change, then we sort by that. 