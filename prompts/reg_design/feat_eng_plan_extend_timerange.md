I would like your plan on extending the current code in `flows_metaopt_end_to_end.py` to support **multi-time-bins** flow extraction and feature calculations.

# Context

Currently, we are allowed to only enter one time bin for the hotspot:

```python
# 3) Choose hotspot and convert time to bin index
hotspot_tv = "LSGL13W"
# For 15-minute bins, 09:00–09:15 starts at bin = 9*60/15 = 36
start_h, start_m = 9, 0
bin_size = indexer.time_bin_minutes
hotspot_bin = (start_h * 60 + start_m) // bin_size
```

I would like that we can enter the *begin and end time*, for example: `09:00` to `10:00`, and it will:

1. Work out the (contiguous) time bin range automatically.

2. The features are accumulated **by sum over each time bin**, that is the current procedure computes each feature for one time bin, now for many time bins, we just sum them together.

# Requirements
The current code heavily uses caching and vectorization. Please ensure that the revised 