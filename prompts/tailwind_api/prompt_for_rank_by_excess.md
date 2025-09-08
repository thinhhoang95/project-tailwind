Can you help me plan (just plan only, without writing any code yet) for another `rank_by` mode for the `/original_count` API endpoint described in `API_README_COUNTS.md`.

# The new mode: `total_excess`

Following the similar computation (to ensure consistency in logic) in `/hotspots` API endpoint, we are going to add a `rank_by` `total_excess` option to put the hotspots with highest amount of overload on the top.

The excess count should follow the logic as described in `/hotspot` endpoint: capacity (object): { tv_id: [float, ...] } capacity per bin aligned to the same bins as counts for the ranked top‑50 TVs. Values are the hourly capacity value repeated across bins in that hour; -1 indicates capacity not available.

If you think it makes sense, please modularize this so the code stays clean and readable. 