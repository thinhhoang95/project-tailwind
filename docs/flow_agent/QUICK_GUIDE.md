# MCTS Flow Agent — Quick Guide

This guide shows how to run the MCTS-based flow agent end-to-end: discover hotspots, select flows, optimize rates, and compute a global objective. It also explains key components, configuration, logging, and tips.

- For detailed internals, see: docs/flow_agent/mcts.md
- New end-to-end agent: src/parrhesia/flow_agent/agent.py
- Benchmark example: tests/flow_agent/mcts_benchmarking/test_mcts_1.py

## Overview

The agent searches flow regulations to reduce the network objective (exceedance + delays + regularization). It:
- Discovers hotspot candidates (TV, time window) and clusters flights to flows
- Selects flows for the chosen hotspot and commits optimal rates
- Repeats across hotspots up to a chosen budget or STOP
- Computes a final global objective and writes a JSONL search trace

## Key Components

- PlanState (src/parrhesia/flow_agent/state.py)
  - Tracks the current plan (committed regulations), hotspot context, stage, and a residual proxy `z_hat` for shaping.
- Actions (src/parrhesia/flow_agent/actions.py)
  - NewRegulation, PickHotspot, AddFlow, RemoveFlow, Continue, Back, CommitRegulation, Stop.
- CheapTransition (src/parrhesia/flow_agent/transition.py)
  - Symbolic state transitions + maintains `z_hat` using per‑flow proxies.
- RateFinder (src/parrhesia/flow_agent/rate_finder.py)
  - Fast per‑hotspot rate optimization; evaluates candidates via `parrhesia.optim.objective`.
- MCTS (src/parrhesia/flow_agent/mcts.py)
  - PUCT + progressive widening + shaped rewards. Enumerates Idle → Hotspot → Flows → Confirm.
- HotspotInventory (src/parrhesia/flow_agent/hotspot_discovery.py)
  - Builds descriptors: TV/window, candidate flows, flow proxies, severity prior.
- SearchLogger (src/parrhesia/flow_agent/logging.py)
  - JSONL trace writer with timestamped events.
- MCTSAgent (src/parrhesia/flow_agent/agent.py)
  - Orchestrates discovery → MCTS search → multi‑commit → final global objective.

## Prerequisites

Artifacts:
- Occupancy with times: `output/so6_occupancy_matrix_with_times.json` (or `data/tailwind/...`)
- TVTW indexer: `output/tvtw_indexer.json` (or `data/tailwind/...`)
- Traffic volume capacities (GeoJSON): e.g., `data/cirrus/wxm_sm_ih_maxpool.geojson`

Environment (example):
```bash
conda activate silverdrizzle
python -V
```

## Quick Start (End‑to‑End Agent)

```python
from pathlib import Path
import geopandas as gpd
from parrhesia.flow_agent import (
    MCTSAgent, MCTSConfig, RateFinderConfig, SearchLogger,
    HotspotDiscoveryConfig,
)
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

# Paths to artifacts
project_root = Path.cwd()
occupancy_path = project_root / "output" / "so6_occupancy_matrix_with_times.json"
indexer_path   = project_root / "output" / "tvtw_indexer.json"
caps_path      = project_root / "output" / "wxm_sm_ih_maxpool.geojson"

# Load artifacts
indexer = TVTWIndexer.load(str(indexer_path))
flight_list = FlightList(str(occupancy_path), str(indexer_path))
caps_gdf = gpd.read_file(str(caps_path))
evaluator = NetworkEvaluator(caps_gdf, flight_list)

# Configure
mcts_cfg = MCTSConfig(max_sims=24, commit_depth=2, commit_eval_limit=6, seed=0)
rf_cfg   = RateFinderConfig(use_adaptive_grid=True, max_eval_calls=128)
disc_cfg = HotspotDiscoveryConfig(
    threshold=0.0, top_hotspots=10, top_flows=4, max_flights_per_flow=20,
    leiden_params={"threshold": 0.3, "resolution": 1.0, "seed": 0},
    direction_opts={"mode": "none"},
)
logger = SearchLogger.to_timestamped("output/flow_agent_runs")

# Run agent (limit to 1–2 regulations for a fast pass)
agent = MCTSAgent(
    evaluator=evaluator,
    flight_list=flight_list,
    indexer=indexer,
    mcts_cfg=mcts_cfg,
    rate_finder_cfg=rf_cfg,
    discovery_cfg=disc_cfg,
    logger=logger,
    max_regulations=1,
)
state, info = agent.run()
logger.close()

print("commits:", info.commits)
print("final objective:", info.summary.get("objective"))
print("log path:", info.log_path)
# Committed regs are in: state.plan
```

What you get:
- One or more `RegulationSpec` objects in `state.plan`
- `info.summary` with final global objective/components/artifacts
- JSONL trace at `info.log_path` with `run_start`, `after_commit`, `run_end` events

## How It Connects (Data Flow)

1) Discovery
- `NetworkEvaluator.get_hotspot_segments(threshold)` → TV/time windows
- For each segment:
  - `collect_hotspot_flights` → union of flights touching (TV, window)
  - `build_global_flows` → cluster flights to flows
  - `prepare_flow_scheduling_inputs` → per‑flow flight specs for control TV
  - Build per‑flow proxies (entrants histograms)
  - Package a `HotspotDescriptor` with `hotspot_prior`, `flow_to_flights`, `flow_proxies`

2) Search
- `PlanState.metadata["hotspot_candidates"]` seeded from descriptors
- `MCTS` enumerates actions per stage:
  - idle: `NewRegulation`, `Stop`
  - select_hotspot: `PickHotspot` (one per candidate), `Stop`
  - select_flows: `AddFlow`/`RemoveFlow`/`Continue` + `Stop`
  - confirm: `CommitRegulation`/`Back`/`Stop`
- Priors: `PickHotspot` uses `hotspot_prior`; `AddFlow` uses proxy mass
- RateFinder invoked only at commit; results cached

3) Final Objective
- `MCTSAgent` assembles global `flights_by_flow` from committed regulations
- Builds capacities per TV; creates a `ScoreContext` via `build_score_context`
- Constructs schedules from committed rates; calls `score_with_context`

## Configuration Knobs

- `MCTSConfig` (src/parrhesia/flow_agent/mcts.py)
  - `max_sims`, `commit_depth`, `commit_eval_limit`, `c_puct`, `alpha/k0/k1`
- `RateFinderConfig` (src/parrhesia/flow_agent/rate_finder.py)
  - `rate_grid`, `use_adaptive_grid`, `max_eval_calls`, `passes`
- `HotspotDiscoveryConfig` (src/parrhesia/flow_agent/hotspot_discovery.py)
  - `threshold`, `top_hotspots`, `top_flows`, `max_flights_per_flow`, Leiden/direction options
- `MCTSAgent(..., max_regulations=...)`
  - Hard cap on number of regulations per run

## Logging and Trace

- `SearchLogger.to_timestamped(dir)` creates `run_YYYYMMDD_HHMMSS.jsonl`
- Events:
  - `run_start`: includes config and candidate count
  - `after_commit`: regulation summary and `delta_j`
  - `run_end`: final objective summary
- JSONL is newline‑delimited; safe for numpy/datetime types

## Tips

- Start with small budgets: `max_sims≈12–24`, `commit_depth≈1–2`, `commit_eval_limit≈3–6`
- Restrict discovery: `top_hotspots≈5–10`, `top_flows≈3–5`, `max_flights_per_flow≈10–20`
- Use `direction_opts={"mode": "none"}` to skip direction reweighting for speed
- Keep `seed=0` for reproducibility

## Run the Quick Smoke Test

```bash
conda run -n silverdrizzle pytest tests/flow_agent/mcts_benchmarking/test_mcts_1.py -s -q
```

If artifacts are missing, the test will be skipped.

---

For deeper details (Priors, Shaping, Caching, Action signatures), read: docs/flow_agent/mcts.md.

