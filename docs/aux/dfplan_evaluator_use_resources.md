The goal is to switch `dfplan_evaluator.py` to prefer the process-wide `AppResources` (indexer, flight list, capacities, and TV GDF). Raise an Exception if resources are not available.

### What you will change
- Prefer `get_resources()` for `indexer` and `flight_list`; do not fall back if resources are not available. Raise an Exception.
- Use the resources-backed `RegulationParser` (`regulation_parser_with_resources.py`) if present; otherwise raise an Exception.
- Keep capacities logic as-is: it already uses resources’ `capacity_per_bin_matrix`. But please remove the fallback logic when the resources is not provided, but raise an Exception instead.

- Note: This approach intentionally does not call `preload_all()`; if resources weren’t preloaded by the caller, lazy access may initialize them from default paths. In the server path this is fine since resources are set up before evaluator runs.

- Also add a small runtime guard to assert `res.indexer.num_time_bins` matches `flight_list.num_time_bins_per_tv`, to harden consistency checks.

- You should also add docstring.