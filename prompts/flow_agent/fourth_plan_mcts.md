# Preambles
For missing information, consult the `second_plan.md` file.

### Assumptions to guide planning
- Commit step costs 6–10s right now (rate finding + evaluation). We’ll throttle commit evaluations per search and rely on transpositions and caching to amortize costs.
- Shaping uses ϕ(s) from `CheapTransition` residuals; γ=1; ϕ(terminal)=0 as in the plan.
- v1 priors are heuristic; no learned value function.

## MCTS scaffolding tasks (implementation plan)

### 1) Core scaffolding and config
- • goal: Create `src/parrhesia/flow_agent/mcts.py` with the core loop and types.
- • deliver:
  - `MCTSConfig`: c_puct, α, k0, k1, priors_temperature, commit_depth D, max_sims, max_time_s, commit_eval_limit, seed, widen_batch_size.
  - `TreeNode`: state_key, N, W, Q, P (per-action), children map, widened_count, cached_phi.
  - `EdgeStats`: per (parent_key, action_signature): N, W, Q.
  - `Search`: holds transposition table, RNG, timers, counters.
  - Public `MCTS.run(plan_state, budget_sims, commit_depth) -> best_action`.
- • accept: Module imports cleanly; `MCTS.run` signature matches plan; no side effects.

### 2) Action space and candidate generation
- • goal: Centralize action enumeration per decision mode with guardrails.
- • deliver:
  - `ActionGenerator` functions using `actions.py`: candidates for ChooseRegulationMode, SelectHotspot, SelectFlows, CommitRegulation, STOP.
  - STOP always available; enforce hard limits (e.g., N_MAX_COMMITS from `PlanState`).
  - Rank candidates (unexpanded) via priors to support progressive widening.
  - Canonical `action_signature(action)` for edge keys.
- • accept: Deterministic candidate sets given fixed seed; signatures stable; unit tests for idempotence.

### 3) Priors integration (v1 heuristic)
- • goal: Minimal `priors.py` for probability mass over actions.
- • deliver:
  - `score_prior(state, candidate_actions) -> dict[action->P]` using simple features (hotspot load share, flights-in-window, overlap).
  - Softmax with temperature; fallback to uniform if features missing.
- • accept: Sums to 1, stable under permutation; tests on a few synthetic states.

### 4) Progressive widening
- • goal: Control branching with N-dependent budget.
- • deliver:
  - Widening schedule m(s) = k0 + k1 * N(s)^α; expand top-ranked not-yet-expanded actions by priors up to m(s).
  - `widen_batch_size` to expand in small chunks.
- • accept: In tests, number of children ≤ m(s) over time; widening monotone.

### 5) Selection (PUCT)
- • goal: Navigate tree using PUCT and priors.
- • deliver:
  - Compute U(s,a) = Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a)).
  - Tie-breaking deterministic under seed.
  - Respect `commit_depth D`: forbid selecting commit actions once per-simulation commit quota reached.
- • accept: Unit tests with controlled P, N confirm selected argmax matches expectation.

### 6) Expansion and leaf handling
- • goal: Create new nodes; bootstrap leaf values.
- • deliver:
  - On first visit to a state_key, create `TreeNode`, cache ϕ(s), compute priors, initialize edge stats.
  - Leaf bootstrap value v_leaf = -ϕ(s_leaf) (no learned v).
- • accept: New nodes have P, zeroed stats, cached ϕ; test that bootstrap equals -ϕ.

### 7) Transition integration (non-commit vs commit)
- • goal: Step environment along path.
- • deliver:
  - Non-commit actions: use `CheapTransition.step` to get s′, isCommit=False, isTerminal.
  - Commit actions: invoke a single commit evaluation:
    - Prefer `NetworkEvaluator.deltaJ_commit(s, a)` if available; else call `RateFinder.find_rates(...)` and apply ΔJ and plan update consistently with the test reference path.
  - Reward at commit: r_base = -ΔJ; Non-commit r_base = 0.
  - Shaping: add Δϕ = ϕ(s′) - ϕ(s) for every step; ϕ(terminal)=0.
  - Cache commit evaluations keyed by (plan_key, action_signature); honor `commit_eval_limit`.
- • accept: On synthetic env, one commit per sim when budget allows; cache hit reduces repeated evaluator calls.

### 8) Backup with shaped returns
- • goal: Update stats along the path.
- • deliver:
  - Accumulate return with γ=1 using shaped rewards r′ = r_base + Δϕ at each step.
  - Update both edge-level (N,W,Q) and node-level counts.
- • accept: Numeric tests ensure shaped backup equals sum of local r′; Q = W/N.

### 9) Transpositions
- • goal: Merge statistics across action orders.
- • deliver:
  - Transposition table keyed by `PlanState.canonical_key()`.
  - Edge stats keyed by (parent_key, action_signature) to keep correct edge-level Q.
  - When revisiting a known state, reuse node P, N, W, Q and widen counters.
- • accept: Tests show visit counts accumulate regardless of path order; canonical key invariants hold.

### 10) Termination and rollout policy
- • goal: Stop conditions and default rollout.
- • deliver:
  - Terminal on STOP or when `PlanState` indicates N_MAX_COMMITS reached.
  - If depth limit reached without commit, rollout ends with v = -ϕ(s_leaf).
  - Optional truncation if time budget exceeded mid-sim (graceful abort of current sim).
- • accept: Deterministic termination in tests; no infinite loops.

### 11) Final action selection at root
- • goal: Policy for action to execute now.
- • deliver:
  - Choose argmax by N(s0,a); tie-break on Q.
  - Return STOP if its N dominates or if all Q gains negligible.
- • accept: Unit tests on fixed priors/N select expected action.

### 12) Budgeting and performance controls
- • goal: Make search feasible with 6–10s commits.
- • deliver:
  - Config defaults for current environment:
    - sims: 12–40; time cap: 10–30s; D: 1–2; commit_eval_limit: 2–4 per run; c_puct ~ 2.0; α ~ 0.7; k0 ~ 4; k1 tuned small.
  - Early-stop if last K sims add < ε average improvement at root.
- • accept: Dry-run with stubs completes under caps; counters reflect throttling.

### 13) Instrumentation and logging
- • goal: Observability.
- • deliver:
  - Metrics: sims, wall time, widen expansions, transposition hits, commit eval calls, cache hit rate, average Q at root, ΔΣz, ΔΣz^2 if available.
  - Structured logging toggled by config verbosity.
- • accept: Metrics dumped at end of `MCTS.run`; stable keys for dashboards.

### 14) Testing
- • goal: Confidence without expensive commits.
- • deliver:
  - Unit tests for PUCT math, widening schedule, shaped backup, transpositions, determinism (fixed seed).
  - Integration test with stub evaluator/rate_finder (constant-time) to validate end-to-end selection and single-commit behavior.
  - Optional slow smoke: reuse your real-data commit fixture behind `pytest.mark.slow`.
- • accept: All unit tests pass; integration test deterministic within tolerance.

### 15) Docs and config surface
- • goal: Usability.
- • deliver:
  - Docstrings for `mcts.py` public APIs.
  - Config reference in `prompts/flow_agent/second_plan.md` appendix or a `README.md`.
  - Example: how to run `MCTS.run(...)` from a planning entrypoint.
- • accept: Docs render; example runs with stubs.

## Risks and mitigations
- • high commit cost: Throttle with `commit_eval_limit`, rely on transpositions and caching; bias commit evaluation to root and near-root nodes first.
- • action explosion: Progressive widening + priors ranking + strict guards on candidate generation.
- • reward scale drift: Clip ϕ and Δϕ as per plan; keep γ=1; log scales to tune c_puct.

## Suggested initial defaults (tuned for current costs)
- **c_puct**: 2.0
- **α**: 0.7, **k0**: 4, **k1**: 1.0
- **commit_depth**: 1–2
- **max_sims**: 24
- **max_time_s**: 20
- **commit_eval_limit**: 3
- **priors_temperature**: 1.0
- **widen_batch_size**: 2
- **seed**: 0

### Immediate next steps
- Implement Tasks 1–3 (core, action generation, priors) and add unit tests.
- Wire Tasks 4–8 (widening, PUCT, expansion, transitions, backup) behind stub evaluator to validate behavior and performance.
- Integrate real commit path with strict `commit_eval_limit` and caching; add the slow smoke test.

- Verified pre-MCTS components exist in `flow_agent` and there’s no `mcts.py`. The plan above sequences the MCTS build with clear deliverables and acceptance checks, and includes performance controls to accommodate current 6–10s commit costs.

# Appendix
## phi(s)
Here’s the idea in plain terms:

* **What is $\hat z$?**
  A fast, rough picture of **how much overload remains** in the chosen hotspot over time if we *were* to regulate the currently selected flows.
  Think of a minute-by-minute array over the hotspot window:

  $$
  \hat z_{t} \approx \text{(projected demand into the hotspot at }t\text{)} - \text{(capacity at }t\text{)}\quad(\ge 0)
  $$

  We use $\hat z$ only to *guide the search* (potential shaping), not to score commits (those use the real evaluator).

* **What do we precompute?**
  For every candidate flow $f$ and hotspot $h$, a small histogram $H_f[t]$: “how many flights from $f$ would hit this hotspot at minute $t$ (within $[t_0,t_1)$).”

* **Initialization (at a node):**

  $$
  \hat z_t \leftarrow z^{\text{base}}_t
  $$

  where $z^{\text{base}}$ comes from the current plan (no new regulation yet).

* **When you add a flow to regulate (SelectFlows.AddFlow$f$):**
  We assume (optimistically but safely clipped) that regulating $f$ can *remove* its contribution in-window, so we **subtract its histogram**:

  $$
  \hat z_t \leftarrow \max\!\big(0,\ \hat z_t - \beta \, H_f[t]\big)
  $$

  with $\beta \in (0,1]$. For v1, just set $\beta=1$. (This is a first-order proxy; it ignores exact rates/FIFO delays.)

* **When you remove a flow from the selection (SelectFlows.RemoveFlow$f$):**
  Put its contribution back:

  $$
  \hat z_t \leftarrow \hat z_t + \beta \, H_f[t]
  $$

* **Potential and shaped reward:**
  We track the scalar

  $$
  \phi(s) = -\theta_2 \sum_t \hat z_t^2\quad(\theta_2>0)
  $$

  and use $\Delta\phi=\phi(s')-\phi(s)$ to shape non-commit steps. Because we touched only a few minutes, we update $\sum_t \hat z_t^2$ **incrementally** (subtract old squares for touched bins, apply the change, add new squares).

* **At commit:**
  We run the real **rate\_finder + evaluator** to get the true $\Delta J$. After applying the commit, we **reset** $\hat z$ to the new baseline $z^{\text{base}}$ from the updated plan. (No shaping at terminal: $\phi(\text{terminal})=0$.)

* **Why this works (v1):**
  It’s cheap (array adds/subtracts), monotone (never makes overload look worse when you add a regulated flow), and gives the MCTS a smooth “less overload is better” hill to climb—without paying evaluator cost until commit.