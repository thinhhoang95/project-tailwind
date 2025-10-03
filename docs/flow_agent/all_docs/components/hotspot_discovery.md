**Hotspot Discovery**

- Purpose: Turn evaluator segments and flight crossings into actionable hotspot candidates with flow groupings and priors.
- Location: `src/parrhesia/flow_agent/hotspot_discovery.py`

Config
- `HotspotDiscoveryConfig` fields:
  - `threshold`: Minimum severity to include in `get_hotspot_segments(threshold=...)`.
  - `top_hotspots`: Max segments to consider after ranking by severity/overload.
  - `top_flows`: Max number of candidate flows per hotspot.
  - `max_flights_per_flow`: Cap flights listed per flow.
  - `leiden_params`: Community detection parameters for `build_global_flows`.
  - `direction_opts`: Options for direction handling in flow clustering.

Artifacts
- `HotspotDescriptor`:
  - `control_volume_id`, `window_bins` [half-open), `candidate_flow_ids`.
  - `hotspot_prior`: Entrants-based severity used as a prior in MCTS.
  - `mode`: Usually `per_flow`.
  - `metadata`:
    - `flow_to_flights`: Map of flow → flight IDs for the window.
    - `flow_proxies`: Per-flow histograms aligned to the window indicating entrants by bin.

How it works
1) Pull and filter segments
- Calls `evaluator.get_hotspot_segments(threshold)` and filters to TVs known to the `TVTWIndexer`.
- Ranks segments by `severity` or `overload` when available; keeps `top_hotspots`.

2) Preload crossings
- Caches per-TV flight crossings once (using `FlightList.iter_hotspot_crossings` when available, or a metadata fallback) to speed per-segment processing.

3) Build descriptors
- For each segment `[start_bin, end_bin]`, converts to a half-open window `(t0, t1+1)` and gathers all flights crossing the TV within bins.
- Clusters to global flows via `build_global_flows(...)` and then restricts to the controlled TV using `prepare_flow_scheduling_inputs(...)`.
- Trims to the top flows by entrants and caps flights per flow.
- Computes entrants histograms per flow aligned with the window to form `flow_proxies`; prior = total entrants across kept flows.

4) Serialize for `PlanState`
- `HotspotInventory.to_candidate_payloads(...)` converts descriptors to JSON-able dicts for `PlanState.metadata['hotspot_candidates']`.

Examples
- Build candidate payloads
```python
from parrhesia.flow_agent.hotspot_discovery import HotspotInventory, HotspotDiscoveryConfig

inv = HotspotInventory(evaluator=evalr, flight_list=fl, indexer=idx)
cfg = HotspotDiscoveryConfig(top_hotspots=5, top_flows=3, max_flights_per_flow=15,
                             leiden_params={"threshold": 0.3, "resolution": 1.0, "seed": 0},
                             direction_opts={"mode": "none"})

descs = inv.build_from_segments(**cfg.__dict__)
candidates = inv.to_candidate_payloads(descs)
# Seed into PlanState.metadata['hotspot_candidates'] for MCTS
```

Tips
- Windows are normalized to half-open; be careful when comparing `[t0, t1]` from segments to `(t0, t1+1)` in descriptors.
- The discovery stage is the main source of flow priors and proxies used by MCTS for better early guidance.

