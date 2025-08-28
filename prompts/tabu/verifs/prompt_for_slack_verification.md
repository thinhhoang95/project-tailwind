# Goals

Can you help me write a verification script to validate the calculation of "slack valley" for a given flight.

# Description

## Footprint Check
We follow the reference script `verify_hotspot_multiplicity.py` up to the hotspot selection. Then you take the first 10 flights of the hotspot.

You then retrieve the traffic volumes that the flight pass through `flight_list`'s `get_occupancy_vector` and the time (hour level) that the flight passes through. You then compare this against the `FlightFeature`'s `footprint` (via `get_footprint` method). Just print out all traffic volumes for both ways and mark OK if they match (just the set of traffic volumes only, not the time).

## Slack Value Check
Then, for each traffic volume, you sum the traffic the occupancy to obtain the hourly occupancy count, this is the hourly demand, you retrieve:
- The hourly occupancy from the NetworkEvaluator.
- The hourly capacity from the NetworkEvaluator.

You check whether the hourly occupancy and the manually calculated occupancy matches, and from there you may compute the slack value = capacity - hourly_occupancy (non-negative only).

Then you aggregate for all traffic volumes, take the 5 percentile value, and compare it against slack_min_p5 of `flight_features`. If the two methods show the same results, output OK.