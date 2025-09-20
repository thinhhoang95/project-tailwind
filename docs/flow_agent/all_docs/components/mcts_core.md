**MCTS Core**

- Purpose: Tree search that chooses actions to construct and commit effective regulations.
- Location: `src/parrhesia/flow_agent/mcts.py`

Main classes
- `MCTSConfig`: Search hyperparameters.
  - `c_puct`: Exploration constant in PUCT.
  - `alpha, k0, k1, widen_batch_size`: Progressive widening parameters; allowed children `m_allow = k0 + k1 * N(parent)^alpha`.
  - `commit_depth`: Max number of `CommitRegulation` actions allowed per simulation path.
  - `max_sims`, `max_time_s`: Budgets for number of simulations and wall-clock time.
  - `commit_eval_limit`: Legacy limiter for commit evaluation calls; effective limit is by `max_sims` and caching.
  - `priors_temperature`: Softmax temperature for priors (when applied).
  - `phi_scale`: Scales potential shaping signal.
  - `seed`: RNG seed for stable tie-breaking.
- `TreeNode`: Node statistics and structure.
  - `P`: Prior per action-signature.
  - `edges`: `EdgeStats` with `N, W, Q` per action.
  - `children`: Map action-signature → child node key.
  - `phi`: Cached state potential (from `PlanState.z_hat`).
- `EdgeStats`: Visit/value stats per edge; `Q = W / max(1, N)`.

How it works
1) Simulation loop
- Starts at root `PlanState`; looks up/creates `TreeNode` keyed by `state.canonical_key()`.
- Expansion via progressive widening: ranks candidate actions by computed priors, expands the best not-yet-expanded until the allowed width.
- Selection via PUCT over expanded children only: `score = Q + c_puct * P * sqrt(N_parent)/(1+N_edge)`.
- Commit evaluation: When selection chooses `CommitRegulation`, runs `_evaluate_commit(state)` once to get rates and `delta_j` via `RateFinder`. The immediate reward uses `r_base = -delta_j`.
- Shaped reward: Adds potential difference `phi(next) - phi(curr)` to the base reward. Leaf bootstrap returns `-phi(leaf)`.
- Termination: Path ends on terminal state or when `commit_depth` is consumed; remaining value bootstraps from leaf potential.
- Backup: Accumulates total return along the path into nodes and edges.

2) Priors and candidates
- `_enumerate_actions(state)`: Yields valid next actions depending on `PlanState.stage`.
  - `idle`: `NewRegulation`, `Stop`.
  - `select_hotspot`: One `PickHotspot` per candidate in `state.metadata['hotspot_candidates']`, plus `Stop`.
  - `select_flows`: `AddFlow` for not-yet-selected candidates, `RemoveFlow` for selected, `Continue` if any selected, plus `Stop`.
  - `confirm`: `CommitRegulation`, `Back`, `Stop`.
- `_compute_priors(...)`: Scores actions using hotspot and flow metadata.
  - `AddFlow`: Sums the flow’s proxy histogram (proportional to entrants in window).
  - `PickHotspot`: Reads `hotspot_prior` from candidate payloads and normalizes.

3) Commit caching and best commit
- `_commit_eval_cache`: Memoizes `(rates, delta_j, info)` by a signature of the selected flows and hotspot window.
- `_best_commit`: Tracks the best evaluated commit across all sims; `run(...)` returns its `CommitRegulation` action.

Examples
- Understanding progressive widening
```python
# With alpha=0.7, k0=4, k1=1, node.N=25 → m_allow ≈ 4 + 1 * (25 ** 0.7) ≈ 4 + 9.5 ≈ 13
# The node will expand up to 13 children (by prior order) before selection uses PUCT among them.
```

- Shaping intuition with `phi`
```python
# If adding a flow increases overload proxy z_hat, phi becomes more negative, so (phi_sp - phi_s) is negative.
# The search then prefers actions that drive z_hat down (toward zero), guiding it to feasible sets of flows.
```

Key integration points
- `CheapTransition.step(...)`: Applies actions and updates `PlanState` stages and `z_hat`.
- `RateFinder.find_rates(...)`: Expensive evaluation invoked only at commit; candidates are cached.
- `PlanState.canonical_key()`: Stable, JSON-based key for nodes and caches.

