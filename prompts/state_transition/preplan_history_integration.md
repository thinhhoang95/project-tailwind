# Goal

The context is Air Traffic Flow Management (ATFM) where you can create regulations to delay a set of flights to ensure dynamic capacity balancing (DCB). A regulation consists of a (control) traffic volume, a flight list to be targeted with delays, and the allowed (hourly) entry rate. A regulation plan could consist of many regulations.

Our goal is to plan for a "Git-like" feature that stores the regulation commit history, applying regulations (to delay flights), and "move the network state" to the after-effect of delay applications. This is done through the `stateman` module, which essentially allows "stepping" occupancy and metadata information of the `FlightList`, which were already realized in the implementation of `flight_list_with_delta.py`.

# Some questions

1. Before delving into detailed plan, I would like to know whether a "replacement" of `FlightList` with `FlightListWithDelta` would work (i.e., all the API functions would remain compatible to call, as if this is a proper flight list with just delay applied - i.e., no inconsistency: we do not want some functions to use the "old" and others to use the "new" flight list). 