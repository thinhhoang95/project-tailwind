**MCTSAgent**

- Purpose: Orchestrates the full loop from hotspot discovery to committing regulations and computing the final objective.
- Location: `src/parrhesia/flow_agent/agent.py`

How it works
- Discovers hotspots: Uses `HotspotInventory.build_from_segments(...)` to create candidate hotspot payloads and seeds them into `PlanState.metadata['hotspot_candidates']`.
- Initializes search: Creates `CheapTransition` and `MCTS` with `RateFinder` injected, then logs `run_start` via `SearchLogger` if provided.
- Runs search-and-commit loop: Calls `mcts.run(state)` to get a fully-evaluated `CommitRegulation` action; extracts `delta_j` (objective improvement) from the action’s `diagnostics` and appends a `RegulationSpec` into the plan.
- Repeats for up to `min(mcts_cfg.commit_depth, max_regulations)` regulations or stops early when no improvement (`delta_j >= 0`).
- Computes final objective: Builds a global schedule from the committed regulations and calls `score_with_context(...)` to produce the final score and components.
- Records `RunInfo`: Summarizes commits, total improvement, `log_path`, `summary`, and aggregated `action_counts` from MCTS.

Key inputs
- `evaluator: NetworkEvaluator`, `flight_list: FlightList`, `indexer: TVTWIndexer`.
- `mcts_cfg: MCTSConfig`, `rate_finder_cfg: RateFinderConfig`, `discovery_cfg: HotspotDiscoveryConfig`.
- Optional `logger: SearchLogger`, `max_regulations: int`.

Key outputs
- `(final_state: PlanState, run_info: RunInfo)`; `final_state.plan` contains `RegulationSpec` entries with `committed_rates` and diagnostics.

Examples
- Minimal run
```python
from pathlib import Path
import geopandas as gpd
from parrhesia.flow_agent import MCTSAgent, MCTSConfig, RateFinderConfig, HotspotDiscoveryConfig, SearchLogger
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator

occupancy = "path/to/so6_occupancy_matrix_with_times.json"
indexer = "path/to/tvtw_indexer.json"
caps = "path/to/wxm_sm_ih_maxpool.geojson"

idx = TVTWIndexer.load(indexer)
fl = FlightList(occupancy, indexer)
caps_gdf = gpd.read_file(caps)
evalr = NetworkEvaluator(caps_gdf, fl)

logger = SearchLogger.to_timestamped("output/flow_agent_runs")
agent = MCTSAgent(
    evaluator=evalr,
    flight_list=fl,
    indexer=idx,
    mcts_cfg=MCTSConfig(max_sims=24, commit_depth=1, seed=0),
    rate_finder_cfg=RateFinderConfig(use_adaptive_grid=True, max_eval_calls=128),
    discovery_cfg=HotspotDiscoveryConfig(top_hotspots=5, top_flows=3),
    logger=logger,
    max_regulations=1,
)

final_state, info = agent.run()
print(info.commits, info.total_delta_j, info.summary)
logger.close()
```

Notes
- MCTSAgent intentionally bypasses stage guards when materializing the final `RegulationSpec` after MCTS to ensure robustness even if the search ended from a non-canonical stage.
- Final objective recomputes a global `n_f(t)` schedule consistent with committed rates; for per-flow vs blanket modes it uses distinct flow keys so flows from different hotspots/windows cannot collide.

**Hotspot Discovery (Detailed)**

- Inputs
  - `evaluator.get_hotspot_segments(threshold)`: Candidate congestion segments per traffic volume (TV) with `[start_bin, end_bin]` and severity/overload.
  - `TVTWIndexer`: Filters to TVs known by the indexer and provides bin math.
  - `FlightList`: Supplies flight/TV crossing events for computing flow entrants.

- Pipeline
  - Pull segments and rank: Filter to known TVs (i.e., listed in the indexer), then sort by `severity` (or `overload` fallback). Keep `top_hotspots`.
  - Preload crossings: Cache per-TV flight crossings once via `FlightList.iter_hotspot_crossings` (or metadata fallback) for speed.
  - For each segment
    - Normalize window: Convert closed range to half-open `(t0, t1+1)`; ensure `t1 > t0`.
    - Collect flights that cross the TV in the window bins.
    - Build global flows: `build_global_flows(...)` clusters flights; then restrict to the controlled TV with `prepare_flow_scheduling_inputs(...)`.
    - Trim flows: Sort flows by entrants in window; keep `top_flows` and cap `max_flights_per_flow` per flow.
    - Compute proxies/prior: For each kept flow, build a histogram of entrants aligned to the window (`flow_proxies`), and compute `hotspot_prior = total entrants`.
  - Serialize: `HotspotInventory.to_candidate_payloads(...)` returns JSON-able dicts ready to seed `PlanState`.

- Candidate payload schema (what MCTS consumes)
```python
{
  "control_volume_id": "TV123",
  "window_bins": [t0, t1],           # half-open
  "candidate_flow_ids": ["F1","F2"],
  "mode": "per_flow",
  "metadata": {
    "flow_to_flights": {"F1": ["AA10","AA20"], "F2": ["BA30"]},
    "flow_proxies": {"F1": [..len=t1-t0..], "F2": [..]}
  },
  "hotspot_prior": 42.0               # entrants-based severity
}
```

- Seeding into PlanState
```python
descs = inventory.build_from_segments(
    threshold=float(cfg.threshold),
    top_hotspots=int(cfg.top_hotspots),
    top_flows=int(cfg.top_flows),
    max_flights_per_flow=int(cfg.max_flights_per_flow),
    leiden_params=cfg.leiden_params,
    direction_opts=cfg.direction_opts,
)
candidates = inventory.to_candidate_payloads(descs)
state = PlanState()
state.metadata["hotspot_candidates"] = candidates
```

- How MCTS uses candidates
  - At stage `select_hotspot`, MCTS enumerates one `PickHotspot(...)` action per item in `state.metadata['hotspot_candidates']` and uses `hotspot_prior` to form priors.
  - When a `PickHotspot` is taken, `CheapTransition` sets a `HotspotContext` and copies `metadata['flow_proxies']` so `z_hat` shaping is aligned with the window.

