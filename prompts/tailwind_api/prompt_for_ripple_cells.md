Please help me plan to automatically select the ripple cells if requested.

# Context

In `base_evaluation.py` and the corresponding API endpoint `/base_evaluation`, currently the ripple cells are assumed to be empty. 

I would like to:

1. Modify the API to take in an optional parameter called `auto_ripple_time_bins`, which defaults to 0.

# Auto Ripple Cells

If `auto_ripple_time_bins` is set to an integer, positive value, we will automatically retrieve the ripple cells before proceeding to perform base evaluation. Please implement this feature in a separate function.

The goal is that given all the flights from all flows, we collect all traffic volumes that each of them cross through (informally speaking, the "union set of footprints" with the time bins). Then you add plus and minus `auto_ripple_time_bins` for extended robustness. Afterwards, you pass the ripple cells to the rest of the evaluation framework, as usual.

You may need to look out for duplicates, and contiguous-ize the time bins. Find the beginning, the end for each traffic_volume_id for the "from"-"to" range for each individual ripple traffic volume ID.
