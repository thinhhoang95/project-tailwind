**Inputs & Outputs**

Inputs
- Occupancy matrix: Sparse structure of flights × (TV, time-bin) used by `FlightList`.
  - Example: `so6_occupancy_matrix_with_times.json`.
- TVTW indexer: Maps traffic volume IDs and time information to indices and bin size.
  - Example: `tvtw_indexer.json`.
- Capacities GeoJSON: Provides hourly capacity per traffic volume.
  - Example: `wxm_sm_ih_maxpool.geojson`.

Where they are used
- `FlightList(...)`: Parses occupancy + links to the same indexer file.
- `TVTWIndexer.load(...)`: Supplies `num_time_bins`, `time_bin_minutes`, `tv_id_to_idx`, decoding helpers.
- `NetworkEvaluator(...)`: Uses capacities and flight list to expose hotspots and capacity time series.

Outputs
- `PlanState.plan`: List of `RegulationSpec` committed by the search.
- `RunInfo`:
  - `commits`: Number of regulations added.
  - `total_delta_j`: Sum of objective improvements (negative is better).
  - `log_path`: Path to JSONL log file (if logging enabled).
  - `summary`: Final global objective, components, artifacts, and `num_flows`.
  - `action_counts`: Aggregated counts of action selections across all simulations.

Example wiring
```python
idx = TVTWIndexer.load(".../tvtw_indexer.json")
fl = FlightList(".../so6_occupancy_matrix_with_times.json", ".../tvtw_indexer.json")
caps_gdf = gpd.read_file(".../wxm_sm_ih_maxpool.geojson")
evalr = NetworkEvaluator(caps_gdf, fl)
```

Tips
- Ensure the `FlightList` and `TVTWIndexer` point to matching bin counts and bin size; the `RateFinder` relies on consistent `time_bin_minutes` to translate hourly rates into per-bin quotas.

