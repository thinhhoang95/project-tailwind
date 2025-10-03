Can you help me plan to refactor the API endpoint for `/hotspots` so that we use a sliding stride rolling-hour occupancy count to detect hotspot instead?

# Context

In the current implementation, we check the hourly cumulative count at the beginning of each hour, for example: 07:00-08:00, 08:00-09:00. A more correct approach is to compare the rolling-hour occupancy count **for each time bin** (for example, in the current set-up, it is 15 minutes. But please do not use a hard-coded value, it follows the time-bin length probably set elsewhere in the code, because this value comes from `tvtw_indexer`).

That means you have a cumulative count at 08:15 for occupancy values from 08:15-09:15, another one at 08:30 for 08:30-09:30...

# Required Output

In the current format, but the duration should be the contiguous period which there is a continuous presence of overload. For example, by the definition above, if you have overload at 08:15 (for the period fromo 08:15-09:15), and at 08:30, then the output should be 08:15 to 08:30. If the periods are non-contiguous, break them down into multiple entries where each is a contiguous period.