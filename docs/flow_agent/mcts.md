# Flow Agent MCTS for ATFM

This document describes the MCTS implementation for the Air Traffic Flow Management (ATFM) flow-planning agent. It summarizes the design in prompts/flow_agent/fourth_plan_mcts.md and documents the implemented module used by tests and examples in this repository.

Contents
- Overview
- Concepts and Data Flow
- API Reference
- Priors and Progressive Widening
- Potential Shaping (phi)
- Commit Evaluation and Caching
- Hotspot Selection
- Full Agent Usage
- Tips and Performance Notes
- Limitations and Next Steps

## Overview
The agent searches action sequences that build a single regulation (flow selection → confirmation → commit) and aims to reduce the network objective (excess traffic + delays + regularization). The search uses Monte Carlo Tree Search (MCTS) with:

- PUCT selection with heuristic priors
- Progressive widening to control branching
- Shaped rewards using a cheap potential function φ(s) computed from a residual proxy z_hat
- Real commit evaluation via RateFinder (rate optimization + evaluator) executed sparingly and cached

The code is compact and focused to keep commit calls within a small budget while still guiding the search effectively.

## Concepts and Data Flow

Core types
- PlanState (src/parrhesia/flow_agent/state.py): the mutable planning state. Holds the current plan, hotspot context, and z_hat (proxy array for shaping in the active hotspot window).
- CheapTransition (src/parrhesia/flow_agent/transition.py): deterministic symbolic transition model that updates PlanState on non-commit actions and maintains z_hat using flow proxies.
- RateFinder (src/parrhesia/flow_agent/rate_finder.py): fast per-hotspot rate optimization and objective evaluation. Called only when committing a regulation.
- MCTS (src/parrhesia/flow_agent/mcts.py): the search engine tying these together.

High-level search loop
1. Start at a PlanState where a hotspot has been picked and flows are known (stage "select_flows").
2. Enumerate actions (AddFlow/RemoveFlow/Continue, and Stop or Back/Commit in later stages).
3. Select actions with PUCT using priors and progressive widening.
4. For Commit actions, run RateFinder once (capped by a budget) and cache the result; for others, step via CheapTransition.
5. Use shaped rewards r' = r_base + Δφ with γ=1. At terminal, bootstrap with v = −φ(s_terminal).
6. Backup along the path and repeat up to simulation/time budgets.

## API Reference

Module: src/parrhesia/flow_agent/mcts.py

Classes
- MCTSConfig
  - c_puct: float = 2.0 — PUCT exploration constant
  - alpha: float = 0.7, k0: int = 4, k1: float = 1.0 — widening schedule m(s) = k0 + k1·N(s)^alpha
  - widen_batch_size: int = 2 — expand new children in small batches (implemented as 1-at-a-time deterministic expansion)
  - commit_depth: int = 1 — max commits per simulation
  - max_sims: int = 24, max_time_s: float = 20.0 — simulation and time budgets
  - commit_eval_limit: int = 3 — hard cap on RateFinder calls per MCTS.run()
  - priors_temperature: float = 1.0 — temperature for softmax priors
  - phi_scale: float = 1.0 — scale for potential φ(s)
  - seed: int = 0 — RNG seeding for determinism

- MCTS(transition: CheapTransition, rate_finder: RateFinder, config: Optional[MCTSConfig] = None)
  - run(root: PlanState, *, max_sims: Optional[int] = None, commit_depth: Optional[int] = None) -> CommitRegulation
    - Executes MCTS from the given PlanState. Returns a CommitRegulation action with committed_rates and diagnostics (including RateFinder info) that improved or matched the objective.

Internal structures
- TreeNode: stores per-node stats (N, W, Q), priors P(action_signature), and cached φ(s).
- EdgeStats: per (parent_state, action_signature): N, W, Q tracked for PUCT.

## Priors and Progressive Widening

Priors
- Derived heuristically from flow “proxies” (histograms of entrants within the hotspot window). A larger mass of entrants yields a higher prior for AddFlow(flow_id).
- For PickHotspot actions, priors are proportional to a supplied `hotspot_prior` in `PlanState.metadata["hotspot_candidates"]`.
- RemoveFlow is mildly discouraged initially; Continue, Commit, Back, Stop receive neutral/low priors.
- Softmax with temperature combines to a probability distribution used by PUCT.

Progressive widening
- Each node permits at most m(s) = k0 + k1·N(s)^alpha children. When new visits arrive, the highest-prior unexpanded action is added first.

## Potential Shaping (phi)

Definition
- φ(s) = −||z_hat_+(s)||^2, where z_hat_+ is z_hat clipped at 0. z_hat is a per-minute proxy over the active hotspot window held in PlanState.

Maintenance
- CheapTransition updates z_hat when flows are added/removed. Typically proxies represent the per-bin demand attributable to a given flow in the window.

Reward
- Per step shaped reward r' = r_base + Δφ with γ=1. Non-commit steps have r_base = 0. Commits use r_base = −ΔJ (negative improvement is good). Terminal bootstrap value is v = −φ(s_terminal).

## Commit Evaluation and Caching

- CommitRegulation triggers a RateFinder.find_rates(...) call with the current state’s hotspot, window, and selected flows (mapping flow_id → flight IDs) supplied via hotspot context metadata.
- Results (rates, ΔJ, diagnostics) are cached with a stable key to avoid recomputation across simulations.
- A strict commit_eval_limit throttles the number of evaluator calls per MCTS.run() to keep end-to-end runtime manageable.

## Hotspot Selection

- The state machine includes these stages: `idle`, `select_hotspot`, `select_flows`, `confirm`, `stopped`.
- Action enumeration has been extended to:
  - `idle`: `NewRegulation`, `Stop`
  - `select_hotspot`: one `PickHotspot` per entry in `PlanState.metadata["hotspot_candidates"]`, plus `Stop`
  - `select_flows` and `confirm`: unchanged
- Candidates must be injected as a list of dicts with keys `control_volume_id`, `window_bins`, `candidate_flow_ids`, `mode`, `metadata` (must include `flow_to_flights` and `flow_proxies`), and `hotspot_prior`.

## Full Agent Usage

High-level agent wiring the full pipeline is available in `src/parrhesia/flow_agent/agent.py` via `MCTSAgent`.

Example:
```python
from parrhesia.flow_agent import MCTSAgent, MCTSConfig, RateFinderConfig, SearchLogger, HotspotDiscoveryConfig
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator

agent = MCTSAgent(
    evaluator=NetworkEvaluator(caps_gdf, flight_list),
    flight_list=flight_list,
    indexer=indexer,
    mcts_cfg=MCTSConfig(max_sims=48, commit_depth=2, commit_eval_limit=6, seed=0),
    rate_finder_cfg=RateFinderConfig(use_adaptive_grid=True),
    discovery_cfg=HotspotDiscoveryConfig(threshold=0.0, top_hotspots=10, top_flows=4, max_flights_per_flow=20),
    logger=SearchLogger.to_timestamped("output/flow_agent_runs"),
)
state, info = agent.run()
```

The agent:
- Discovers hotspot candidates from `NetworkEvaluator.get_hotspot_segments`.
- Builds flows and flow proxies per hotspot and seeds `PlanState.metadata`.
- Runs MCTS to select flows and commit rates (multi-commit supported by `commit_depth`).
- Computes a final global objective using `parrhesia.optim.objective` and logs a JSONL trace if a logger is provided.

## Usage Guide (Quickstart)

Prerequisites
- PlanState set to a hotspot selection stage with candidate_flow_ids and window_bins.
- Metadata for the hotspot context must include:
  - flow_to_flights: Dict[str, Sequence[str]] — maps each flow to its flight IDs (for RateFinder).
  - flow_proxies: Dict[str, Sequence[float]] — per-flow entrant histograms over the hotspot window (for shaping and priors).

Typical setup (pseudocode)
```python
from parrhesia.flow_agent import (
    PlanState, NewRegulation, PickHotspot, CheapTransition,
    MCTS, MCTSConfig, RateFinder, RateFinderConfig
)

# 1) Prepare RateFinder from NetworkEvaluator + FlightList
rf = RateFinder(evaluator=evaluator, flight_list=flight_list, indexer=indexer,
                config=RateFinderConfig(rate_grid=tuple(), use_adaptive_grid=True))

# 2) Build proxies (e.g., entrants per bin) and flow_to_flights
proxies = {flow_id: np.array([...], dtype=float) for flow_id in candidate_flows}
flow_to_flights = {flow_id: list_of_flight_ids}

# 3) Initialize state and transition
transition = CheapTransition(flow_proxies=proxies)
state = PlanState()
state, _, _ = transition.step(state, NewRegulation())
state, _, _ = transition.step(state, PickHotspot(
    control_volume_id=tv_id,
    window_bins=(t0, t1),
    candidate_flow_ids=tuple(sorted(candidate_flows)),
    metadata={"flow_to_flights": flow_to_flights, "flow_proxies": proxies},
))

# 4) Run MCTS to obtain a commit action
mcts = MCTS(transition=transition, rate_finder=rf, config=MCTSConfig(max_sims=12, commit_depth=1))
commit_action = mcts.run(state)

# 5) Apply commit to the plan
state, is_commit, _ = transition.step(state, commit_action)
assert is_commit
```

## Tips and Performance Notes

- Commit cost: Limit commit_eval_limit and rely on caching and transpositions to contain runtime.
- Priors: If proxies are not provided, priors degrade to near-uniform; shaping still provides useful guidance.
- Determinism: Use MCTSConfig.seed to keep action selection and expansion order stable.
- RateFinder grids: The adaptive grid can be enabled (use_adaptive_grid=True) to keep candidate rates small and focused.

## Limitations and Next Steps

- The current action set assumes a prior hotspot selection; extending candidate generation to include picking hotspots and a STOP policy is straightforward.
- We use a heuristic potential and priors; a learned value function or policy could improve guidance.
- Multi-commit trajectories per simulation are supported via commit_depth > 1 but the example tests focus on a single-commit improvement.
