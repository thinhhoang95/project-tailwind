### Goal
Design a full-fledged MCTS agent that autonomously:
- picks hotspots and windows,
- clusters and selects flows for the chosen hotspot,
- chooses commit(s) and rates,
- repeats across hotspots up to budgets or STOP,
- logs a detailed, timestamped search trace and a final run summary,
- computes objective values strictly via `parrhesia.optim.objective` (as in `tests/flow_agent/test_rate_finder.py`).

### High-level architecture
- Extend current single-regulation MCTS to cover the full pipeline with multi-commit trajectories.
- Introduce a thin orchestration layer that prepares hotspot candidates and per-hotspot flow data before search, reuses `RateFinder` for commits, and calls `CheapTransition` for symbolic steps.

### New/updated components
- New: `src/parrhesia/flow_agent/agent.py`
  - Class `MCTSAgent`
  - Responsibilities:
    - Discover hotspot candidates (using capacity metadata and hotspot detection; `NetworkEvaluator.get_hotspot_segments` may be used purely for discovery/capacity extraction, not for objective scoring).
    - For each candidate, build flows and light-weight proxies.
    - Seed `PlanState` metadata with these candidates.
    - Construct `CheapTransition`, `RateFinder`, and `MCTS`, then run to produce commits.
    - Loop commits until STOP or budgets.
    - Provide a final global objective computation using `parrhesia.optim.objective.build_score_context` and `score_with_context` across all committed regulations (no ALNS `NetworkPlanMove`/legacy evaluators).
- New: `src/parrhesia/flow_agent/hotspot_discovery.py`
  - `HotspotDescriptor` (tv_id, window_bins, severity/prior, candidate_flow_ids, metadata including `flow_to_flights`, `flow_proxies`, any precomputed entrants counts).
  - `HotspotInventory` (builds and caches descriptors; progressive fetching when needed).
- Update: `src/parrhesia/flow_agent/mcts.py`
  - Extend action enumeration to include idle/select_hotspot stages.
  - Add priors for hotspot selection.
  - Keep progressive widening over hotspots and flows.
  - Support multi-commit rollouts with `commit_depth > 1`.
  - Optional logging hooks.
- New: `src/parrhesia/flow_agent/logging.py`
  - `SearchLogger` with JSONL file writer and helpers for compact, structured tracing and final summaries.
- Optional: `src/parrhesia/flow_agent/priors.py`
  - Centralize heuristic priors for actions (hotspot severity, flow mass, etc.).

### Action space and stages
- Stages in `PlanState` already exist: `idle`, `select_hotspot`, `select_flows`, `confirm`, `stopped`.
- Actions (existing + new usage):
  - `NewRegulation`: from `idle` to `select_hotspot`.
  - `PickHotspot(tv_id, window_bins, candidate_flow_ids, mode, metadata)`: into `select_flows` with proxies initialized; metadata includes `flow_to_flights` and `flow_proxies`.
  - `AddFlow(flow_id)`, `RemoveFlow(flow_id)`, `Continue`: as-is.
  - `Back`: from `confirm` to `select_flows`, or from `select_flows` back to `select_hotspot` (already supported).
  - `CommitRegulation(committed_rates, diagnostics)`: evaluated by `RateFinder`, appended to `plan`, then returns to `idle`.
  - `Stop`: can be chosen at `idle`, `select_hotspot`, `select_flows`, or `confirm`.
- MCTS enumeration updates:
  - `idle`: enumerate `NewRegulation`, `Stop`.
  - `select_hotspot`: enumerate `PickHotspot` for inventory items (controlled by progressive widening) and `Stop`.
  - `select_flows` and `confirm`: already implemented.

### Hotspot and flow discovery pipeline
- Inputs: `FlightList`, `TVTWIndexer`, and capacity metadata. `NetworkEvaluator.get_hotspot_segments` can be used for hotspot/window detection and to load per-TV hourly capacities, but NOT for objective scoring.
- Steps per discovery pass:
  - Get hotspot segments via `NetworkEvaluator.get_hotspot_segments(threshold=...)`.
  - For each segment: select `traffic_volume_id`, `start_bin:end_bin` window (merge to contiguous windows if needed).
  - Collect candidate flights crossing the TV and time window.
  - Cluster to flows with `build_global_flows`; gather flights per flow with `prepare_flow_scheduling_inputs`.
  - Trim by heuristics (top-K flows by size/impact; per-flow max flights).
  - Build proxies per flow via RateFinder entrants (same as test helper), store as arrays over the window.
  - Compute descriptor severity/prior (e.g., peak overload approximation, entrants sum, capacity gap proxy).
  - Save descriptor to `HotspotInventory`.
- Inventory injected into `PlanState.metadata["hotspot_candidates"]` as a list of minimal dicts containing:
  - `control_volume_id`, `window_bins`, `candidate_flow_ids`, `metadata` with `flow_to_flights` and `flow_proxies`, and `hotspot_prior`.

### Priors and progressive widening
- Priors
  - `PickHotspot`: proportional to `hotspot_prior` from inventory (e.g., entrants-in-window, capacity gap, or evaluator-reported overload severity). Temperature-scaled softmax.
  - `AddFlow`: already uses proxy mass sum; keep as-is.
  - `RemoveFlow`, `Continue`, `Back`, `CommitRegulation`, `Stop`: use current constants, tune empirically.
- Progressive widening
  - At `select_hotspot`: widen by highest-prior hotspots not yet expanded.
  - At `select_flows`: keep current widening over flows.

### Potential shaping
- Keep the current φ(s) on the active hotspot window (z_hat maintained by `CheapTransition`), with γ=1.
- For `idle` and `select_hotspot` stages (no active z_hat), φ(s)=0; exploration relies on priors.
- Optionally consider a future extension to maintain a global residual proxy across committed TVs for multi-commit shaping; out of scope for first iteration.

### Commit evaluation and caching
- Keep current `RateFinder` integration and caching keyed by:
  - plan key, `control_volume_id`, `window_bins`, selected flows list, and mode.
- Internally, `RateFinder` already uses `parrhesia.optim.objective.build_score_context` and `score_with_context` for evaluation (see `tests/flow_agent/test_rate_finder.py`).
- Budgets:
  - `commit_eval_limit` per MCTS run remains a hard cap.
  - `max_sims`, `max_time_s`, and `commit_depth` control total scope.

### Final/global objective computation (no ALNS)
- After the agent finishes, compute a single global objective for the entire committed plan using `parrhesia.optim.objective`:
  - Aggregate the committed regulations into a combined per-flow schedule `n_f_t` and a unified `flights_by_flow` mapping. To keep flows disjoint across TVs, namespace flow ids per TV (e.g., `"TV1::42"`).
  - Build `capacities_by_tv` by merging the hourly capacities for all TVs appearing in the regulations (reuse the same loading logic used by `RateFinder._build_capacities_for_tv`).
  - Construct a `ScoreContext` via `build_score_context(flights_by_flow, indexer=..., capacities_by_tv=..., target_cells=..., ripple_cells=..., flight_list=..., weights=..., tv_filter=all_tvs_in_plan)`.
  - Evaluate `score_with_context(n_f_t, flights_by_flow=..., capacities_by_tv=..., flight_list=..., context=...)`.
  - Log the resulting `J_total`, components, and artifacts in the run summary. Do not use ALNS `NetworkPlanMove` or any legacy `NetworkEvaluator` pathways for objective scoring.

### Orchestration loop (agent)
- Initialize:
  - Build `HotspotInventory` (optionally lazily, or full upfront).
  - Seed `PlanState` in `idle`, and set `metadata["hotspot_candidates"]`.
  - Instantiate `CheapTransition` with a union of all proxies for fallback (it prefers metadata overrides per hotspot), and `RateFinder` with `FlightList` and `TVTWIndexer` (capacities loaded as in `RateFinder`; do not depend on legacy ALNS evaluators for scoring).
- Run:
  - Call `mcts.run(state, commit_depth=N)` to produce a first commit action; apply via `transition.step`.
  - Repeat to reach desired commits or until STOP is selected.
  - Optionally refresh inventory after each commit (capacities/loads may shift).
- Outputs:
  - Final `PlanState.plan` list of `RegulationSpec` with committed rates and diagnostics; logging summary and a final global objective computed via `parrhesia.optim.objective`.

### Logging and tracing
- File format: JSON Lines, timestamped file name like output/flow_agent_runs/mcts_YYYYmmdd_HHMMSSZ.log
- `SearchLogger` API
  - `log_run_start(config, seed, data_hashes, budgets)`
  - `log_inventory(candidates_summary)` top M hotspots with priors/severity
  - `log_sim_start(sim_id)`
  - `log_node(key, stage, N, Q, m_allow, widened_count)`
  - `log_selection(stage, candidates=[{sig, P, N, Q, U}], chosen_sig)`
  - `log_step(action_kind, stage_from, stage_to)`
  - `log_commit_eval(signature, flows, diagnostics)` (delta_j, final_objective, eval_calls, timing_seconds, rates)
  - `log_best_commit_update(delta_j, rates, at_depth)`
  - `log_global_objective(J_total, components)` computed via `parrhesia.optim.objective`
  - `log_run_summary(summary_dict)`
- Run summary includes:
  - wall clock totals, sims run, eval calls, cache hits
  - number of committed regulations
  - per-regulation breakdown:
    - tv_id, window, mode, flows with (#flights, chosen rate)
    - per-commit delta_j from `RateFinder` and diagnostics
  - aggregate delta_j (sum of commits) and a final global objective computed with `score_with_context`
  - top hotspots considered and how often expanded

### Interfaces and minimal edits
- `PlanState.metadata["hotspot_candidates"]`: list of dicts with keys:
  - `control_volume_id`, `window_bins`, `candidate_flow_ids`, `mode`, `metadata` (`flow_to_flights`, `flow_proxies`), `hotspot_prior`.
- `mcts.py`
  - Update `_enumerate_actions` to:
    - idle: `[NewRegulation(), Stop()]`
    - select_hotspot: `[PickHotspot(... for top-K per widening ...), Stop()]`
  - Update `_compute_priors` to read `hotspot_prior` for `PickHotspot` candidates.
  - Add signatures/decoding for `("new_reg",)` and `("hotspot", tv_id, t0, t1)`; reconstruct action with payload from `PlanState.metadata`.
  - Add optional `logger` calls in key points (simulate start, selection, commit eval, best update).
  - Keep commit caching as-is.
- `transition.py`
  - Already supports `NewRegulation`, `PickHotspot`, flow edits, `Continue`, `Back`, `CommitRegulation`, `Stop`. No changes expected.
- `rate_finder.py`
  - No functional change; we will rely on existing `diagnostics`.

### Heuristic details
- Hotspot prior candidates:
  - Simple initial choice: entrants sum in window, optionally weighted by estimated capacity gap median; fallback to number of affected flights.
  - Normalize to [0,1] logits before temperature softmax.
- Flow trimming (to control branching):
  - Keep top 3–5 flows by entrants count; cap flights per flow (e.g., 10–20).
  - Store both full and trimmed in descriptor to allow future widening of flows set if needed.

### Budgets and performance
- Defaults:
  - `commit_depth`: 2–3
  - `max_sims`: 32–64
  - `max_time_s`: 30–60 seconds
  - `commit_eval_limit`: 5–8 per run
- Caching:
  - Keep `RateFinder` caches across commits; key includes plan’s canonical key, so cache is stable across MCTS simulations.
  - Inventory cached per tv_id+window to avoid repeated flow extraction.

### Testing plan
- Unit tests
  - Inventory building on tiny synthetic `FlightList` (1–2 TVs) → descriptors include `flow_to_flights`, `flow_proxies`.
  - MCTS enumeration coverage for stages idle/select_hotspot/select_flows/confirm, priors sorted, and widening limits respected.
- Integration tests (slow, skipped if artifacts missing)
  - End-to-end: build inventory from real artifacts (as in `test_mcts_real_data_smoke`), run `MCTSAgent` with small budgets, assert:
    - At least one commit produced
    - Sum of `delta_j` < 0
    - Global objective computed via `parrhesia.optim.objective` is consistent and logged
    - Log file created with required keys
- Trace integrity
  - Validate JSONL schema per event type.

### Open points and assumptions
- We’ll lean on the existing `z_hat` shaping only for the active hotspot stage; global shaping is optional future work.
- Flow clustering cost can be significant; we’ll cache per (tv_id, window) and cap flows and flights.
- Global final objective check will exclusively use `parrhesia.optim.objective` helpers; legacy ALNS components (`NetworkPlanMove`, legacy `NetworkEvaluator` pathways for scoring) are not used.

### Deliverables (edits)
- New:
  - `src/parrhesia/flow_agent/agent.py` (MCTSAgent)
  - `src/parrhesia/flow_agent/hotspot_discovery.py`
  - `src/parrhesia/flow_agent/logging.py`
  - Optional: `src/parrhesia/flow_agent/priors.py`
- Update:
  - `src/parrhesia/flow_agent/mcts.py` (enumeration, priors, signatures, optional logging hook)
  - `docs/flow_agent/mcts.md` (add sections for hotspot selection, full agent usage, logging, and global objective via `parrhesia.optim.objective`)

### Example run sketch
```python
agent = MCTSAgent(
  evaluator=NetworkEvaluator(caps_gdf, flight_list),
  flight_list=flight_list,
  indexer=indexer,
  mcts_cfg=MCTSConfig(max_sims=48, commit_depth=2, commit_eval_limit=6, seed=0),
  discovery_cfg=HotspotDiscoveryConfig(threshold=0.0, top_hotspots=10, top_flows=4, max_flights_per_flow=20),
  logger=SearchLogger.to_timestamped("output/flow_agent_runs")
)
result_state, run_info = agent.run()
# result_state.plan → committed regulations; run_info.log_path → JSONL trace
```

- I’ll proceed next by scaffolding the new files, extending `mcts.py` to enumerate hotspots, and wiring the logger hooks, then add a minimal integration test harness.