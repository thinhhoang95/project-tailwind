Please help me plan for an API endpoint to trigger Simulated Annealing optimization, based on the pipeline `pipeline.py` and `sa_optimizer.py`.

# Inputs

The input is the same as for `/base_evaluation`. If auto ripple time bins is defined, then the ripple cells (traffic volumes + time bins) are automatically filled as well.

#### JSON body
- **flows** (required, object): Mapping of flow-id -> list of flight IDs. Flow IDs may be strings or numeric; they are coerced to integers deterministically.
- **targets** (required, object): Mapping `TV_ID -> {"from": "HH:MM[:SS]", "to": "HH:MM[:SS]"}`. Defines attention cells for target TVs.
- **ripples** (optional, object): Same schema as `targets`. Defines secondary attention cells.
- **auto_ripple_time_bins** (optional, integer, default 0): If greater than 0, ripple cells are computed automatically as the union of TVTW footprints of all flights in all flows, dilated by ±`auto_ripple_time_bins` along time. When provided and > 0, this overrides `ripples`.
- **indexer_path** (optional, string): Override path to `tvtw_indexer.json`. Default: `data/tailwind/tvtw_indexer.json`.
- **flights_path** (optional, string): Override path to `so6_occupancy_matrix_with_times.json`. Default: `data/tailwind/so6_occupancy_matrix_with_times.json`.
- **capacities_path** (optional, string): Override path to capacities GeoJSON. Default: `data/cirrus/wxm_sm_ih_maxpool.geojson`.
- **weights** (optional, object): Partial overrides for `ObjectiveWeights` (e.g., `{"alpha_gt": 10.0, "lambda_delay": 0.1}`).

Validation errors (HTTP 400) are returned if:
- **flows** is missing or not an object
- **targets** is missing or empty
- Time ranges are malformed (HH:MM or HH:MM:SS required)

Unknown items are ignored gracefully:
- Unknown TV IDs in `targets`/`ripples` are dropped
- Unknown flight IDs in `flows` are ignored

# Instructions

1. Please run the simulated annealing optimizer to optimize the objective function. Please plan in detail for inputs, outputs, usage. 

2. Return the result that includes:

    1. Post-optimization demand array for each flow for the control volume.

    2. Post-optimization target demands and ripple demands (similarly in `/base_evaluation`).

