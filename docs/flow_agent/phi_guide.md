### Simulation and Evaluation in the Flow MCTS: what happens, where, and how phi(s) is used

Below is a code-accurate walkthrough of the simulation loop, potential-shaping, when “full evaluation” happens, how results are backed up, and what the agent does after a commit. References are from the files under `src/parrhesia/flow_agent`.

### 1) What phi(s) is and where it’s used

- phi(s) is a potential function computed from the state’s `z_hat` (a proxy vector over the active hotspot window). It is negative the squared L2 norm of the positive part of `z_hat`, scaled by `phi_scale`. More overload → more negative phi.
```999:1006:src/parrhesia/flow_agent/mcts.py
def _phi(self, state: PlanState) -> float:
    z = state.z_hat
    if z is None:
        return 0.0
    arr = np.asarray(z, dtype=float)
    # Use only positive overload and scale
    val = -float(np.dot(np.maximum(arr, 0.0), np.maximum(arr, 0.0)))
    return self.cfg.phi_scale * val
```

- Each newly created search node caches `phi(s)` as `node.phi`. It’s also used for bootstrap values:
```991:997:src/parrhesia/flow_agent/mcts.py
def _create_node(self, state: PlanState) -> TreeNode:
    key = state.canonical_key()
    node = TreeNode(key=key)
    node.phi = self._phi(state)
    self.nodes[key] = node
    return node
```

- The `z_hat` that phi(s) reads is maintained by the symbolic transition model `CheapTransition`. When you add or remove a flow, a precomputed per-flow proxy histogram (entrants-by-bin across the hotspot window) is added to or subtracted from `z_hat`, with optional decay and clipping:
```71:79:src/parrhesia/flow_agent/transition.py
if isinstance(action, AddFlow):
    guard_can_add_flow(next_state, action.flow_id)
    context = next_state.hotspot_context
    assert context is not None  # guarded above
    self._decay(next_state)
    proxy = self._lookup_proxy(action.flow_id, context)
    self._apply_proxy(next_state, proxy, sign=1.0)
    next_state.hotspot_context = context.add_flow(action.flow_id)
    return next_state, False, False
```
```193:200:src/parrhesia/flow_agent/transition.py
def _apply_proxy(self, state: PlanState, proxy: np.ndarray, *, sign: float) -> None:
    if state.z_hat is None:
        state.z_hat = np.zeros(proxy.shape, dtype=float)
    if state.z_hat.shape != proxy.shape:
        state.z_hat = np.zeros(proxy.shape, dtype=float)
    state.z_hat += sign * proxy
    np.clip(state.z_hat, -self.clip_value, self.clip_value, out=state.z_hat)
```

### 2) Shaped rewards inside a simulation

- At each step, the immediate reward is:
  - r_base = 0 for non-commit actions, or r_base = −ΔJ for commit actions (see below).
  - delta_phi = phi(s’) − phi(s).
  - Shaped one-step reward r = r_base + delta_phi.
```845:853:src/parrhesia/flow_agent/mcts.py
phi_s = node.phi
phi_sp = self._phi(next_state)
delta_phi = phi_sp - phi_s
r_shaped = r_base + delta_phi
total_return += r_shaped
```

- When reaching a fresh leaf (first-time visit) or a node with no children, the bootstrap value is taken as v = −phi(s_leaf) and backed up immediately for the path that led there:
```520:546:src/parrhesia/flow_agent/mcts.py
v = -node.phi
...
self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="leaf_bootstrap")
self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="leaf_bootstrap")
return v
```
```612:627:src/parrhesia/flow_agent/mcts.py
if not node.children:
    v = -node.phi
    ...
    self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="no_children")
    self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="no_children")
    return v
```

- When a simulation terminates (either a terminal action like `Stop`, or the simulation hits the per-simulation commit quota), it adds a final leaf term v_leaf = −phi(s_T) to the accumulated shaped return and backs up the full sum:
```942:977:src/parrhesia/flow_agent/mcts.py
leaf_v = -self._phi(state)
total_return += leaf_v
...
self._backup(
    path,
    total_return,
    sim_index=sim_index,
    step_index=step_index,
    reason="terminal_or_budget",
)
return total_return
```

Interpretation:
- For paths that end due to terminal/quota, the backed-up return equals sum of commit returns plus a telescoping of phi that anchors at −phi(s_start).  
- For early leaf expansion cases, the backed-up value is just −phi(s_leaf), not the “path sum.” This is intentional in the implementation and means early expansions are valued by the leaf’s potential alone.

### 3) Where “full evaluation” happens (CommitRegulation → RateFinder)

- When `CommitRegulation` is selected by MCTS, the code performs a full evaluation via `RateFinder.find_rates`. The returned ΔJ (objective delta: candidate − baseline) is converted to an immediate base reward r_base = −ΔJ.
```812:835:src/parrhesia/flow_agent/mcts.py
commit_action, delta_j = self._evaluate_commit(state, sim_index=sim_index, step_index=step_index)
...
action = commit_action
r_base = -float(delta_j)
...
if (not banned_commit) and (self._best_commit is None or delta_j < self._best_commit[1]):
    self._best_commit = (commit_action, float(delta_j))
```

- Inside `_evaluate_commit`, the full evaluation is the `RateFinder.find_rates(...)` call; this returns the committed rates, ΔJ, and diagnostics. There’s a per-run budget `commit_eval_limit` and a local cache keyed by the full state signature plus TV, window, and flows:
```1293:1303:src/parrhesia/flow_agent/mcts.py
with self._timed("mcts.rate_finder.find_rates"):
    rates, delta_j, info = self.rate_finder.find_rates(
        plan_state=state,
        control_volume_id=str(ctx.control_volume_id),
        window_bins=tuple(int(b) for b in ctx.window_bins),
        flows=flows,
        mode="per_flow" if ctx.mode == "per_flow" else "blanket",
    )
self._commit_calls += 1
self._last_delta_j = float(delta_j)
self._commit_eval_cache[base_key] = (rates, delta_j, info)
```

- What `RateFinder` actually evaluates:
  - Builds a baseline objective for the selected `(TV, window, flows)` using a cached “score context.”
  - Sweeps candidate rates (per-flow or blanket), constructs a release schedule, and re-scores the objective.
  - ΔJ = objective_candidate − objective_baseline.
```513:526:src/parrhesia/flow_agent/rate_finder.py
with self._timed("rate_finder.score_with_context.baseline"):
    objective, components, artifacts = score_with_context(
        context.d_by_flow,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=capacities_by_tv,
        flight_list=self._base_flight_list,
        context=context,
    )
baseline = _BaselineResult(
    objective=float(objective),
    components=components,
    artifacts=artifacts,
)
```
```691:699:src/parrhesia/flow_agent/rate_finder.py
delta_j = float(objective) - float(baseline_obj)
result = _CandidateResult(
    delta_j=delta_j,
    objective=float(objective),
    components=components,
    artifacts=artifacts,
)
self._candidate_cache_store(signature, result)
return result
```

- Interaction of ΔJ and phi during backup:
  - At the commit step, the one-step reward is r_base + delta_phi = (−ΔJ) + [phi(s’) − phi(s)].
  - If the simulation then terminates (or hits quota), it adds −phi(s_T) and backs up the accumulated total. In that case, the phi terms telescope to a constant offset anchored at the start, and the net contribution of phi along the path cancels except for that anchor; the ΔJ’s from commit steps remain as true “task reward.”  
  - For early-leaf bootstrap cases, the backup is just −phi(s_leaf) (no ΔJ unless a commit occurred before).

### 4) After full evaluation (outer agent behavior): restart from a new root

- The outer `MCTSAgent.run()` loops up to `commit_depth` (and bounded by `max_regulations`), each time:
  1) Ensures `state.stage="idle"`.
  2) Calls `mcts.run(state)` to pick a single `CommitRegulation` with rates.
  3) Materializes that regulation into `state.plan`, bans duplicates, and removes that hotspot candidate.
  4) If ΔJ ≥ 0 (no improvement), it stops early. Otherwise, it repeats to propose the next regulation.
```193:201:src/parrhesia/flow_agent/agent.py
for _ in range(max(1, limit)):
    # Ensure we are in idle to start a new regulation
    state.stage = "idle"
    try:
        with self._timed("agent.mcts.run"):
            commit_action = mcts.run(state)
```
```308:316:src/parrhesia/flow_agent/agent.py
regulation = RegulationSpec(
    control_volume_id=ctrl,
    window_bins=wb,
    flow_ids=flow_ids,
    mode="per_flow" if mode == "per_flow" else "blanket",
    committed_rates=rates_to_store,
    diagnostics=dict(commit_action.diagnostics or {}),
)
state.plan.append(regulation)
```

- Important implications:
  - Yes, after each successful “full evaluation”/commit, the agent restarts search from a new root that includes the committed regulation in `state.plan`. Previous regulations cannot be “undone” in subsequent iterations; there is no remove-regulation action in this planner. Each outer iteration plans the next regulation on top of the already-committed plan.
  - The MCTS instance is reused across iterations, but the root state key changes (because `state.plan` changed). Commit-evaluation caching is also keyed by the state’s canonical key; so results are not reused across different roots.

### 5) Where phi(s) appears in the loop (recap)

- Compute and cache at node creation; used as bootstrap:
  - `_create_node(...): node.phi = _phi(state)`
  - Bootstrap returns: `v = -node.phi`
- Shaped immediate reward each transition:
  - r = r_base + [phi(s’) − phi(s)]
  - Commit sets r_base = −ΔJ; otherwise r_base = 0
- Terminal/quota end adds −phi(s_T) and backs up total
- Early leaf/no-children backs up only −phi(s_current)

### 6) Practical knobs in `run_agent.py`

- The example sets budgets high enough to see multiple commits:
  - `mcts_cfg.commit_depth=128`, `commit_eval_limit=16`, `max_actions=512`
  - The agent is created with `max_regulations=128`, so it can iterate multiple times, each time re-rooting at the updated plan.

### 7) Backup mechanics

- Only at the end of a simulation pass is a value backed up along the path. The backup routine increments visit counts and averages Q = W/N on nodes and edges:
```1437:1464:src/parrhesia/flow_agent/mcts.py
def _backup(self, path: Sequence[Tuple[str, Tuple]], value: float, ...):
    self._inc_action("backup", ...)
    v = float(value)
    for depth, (node_key, sig) in enumerate(path):
        node = self.nodes.get(node_key)
        if node is None:
            continue
        node.N += 1
        node.W += v
        node.Q = node.W / max(1, node.N)
        est = node.edges.get(sig)
        if est is None:
            est = EdgeStats()
            node.edges[sig] = est
        est.N += 1
        est.W += v
        est.Q = est.W / max(1, est.N)
```

### 8) Summary of the flow

- Simulation steps:
  - Selection with PUCT; progressive widening; step with shaped reward r = r_base + Δphi.
- Full evaluation:
  - On `CommitRegulation`, call `RateFinder.find_rates(...)`, get ΔJ, set r_base = −ΔJ.
- Backup:
  - Terminal/quota: back up the accumulated shaped return plus −phi(s_T).
  - Early leaf/no-children: back up −phi(s_leaf) alone.
- After commit:
  - Outer agent appends the committed regulation to `state.plan`, bans duplicates, and restarts search from the updated root. Past regulations are not undone.

- Impact of phi(s):
  - Guides the search cheaply between commits via Δphi.
  - Provides bootstrap values at leaves.
  - At terminal/quota endings, phi terms telescope, leaving the sum of commit rewards (−ΔJ) anchored by −phi at the start; at early leaves, the leaf bootstrap −phi(s_leaf) is used directly.


- Committed regulation plan is built incrementally in `state.plan`; each new regulation is proposed by a fresh MCTS run from the new root with previously committed regs fixed.

- Full evaluation happens only on `CommitRegulation` via `RateFinder.find_rates(...)`; r_base = −ΔJ is added to shaped reward and then included in the backup.


- phi(s) is computed from `z_hat` maintained by `CheapTransition` using flow entrants histograms; it appears at node creation, as delta between consecutive states for shaped rewards, and as a leaf bootstrap or terminal anchor.

# FAQ

### 1. Shaped Reward's Explicit Expression

- Per step the shaped reward is r = r_base + [phi(s') − phi(s)]. For non‑commit actions r_base = 0; for a CommitRegulation step r_base = −ΔJ returned by the evaluator.

- At terminal/quota end they add −phi(s_T). Therefore for a complete simulation that reaches terminal/quota, the backed‑up value is
  G = Σ_commits (−ΔJ_i) − phi(s_start).
  The phi terms telescope and cancel except for the start anchor. So your expression “−ΔJ + phi(s_T) − phi(start)” is missing the final −phi(s_T); the correct net is −ΔJ − phi(s_start) for a single‑commit path, or the sum over commits for multiple.

### 2. Re-rooting after commit will make the objective $\Delta J$ useless?

- Re‑rooting after a commit does not make the “real evaluation objective” useless. Within each outer iteration, −ΔJ is the only non‑shaping reward and it is exactly what drives which commit action MCTS selects. After the agent executes that commit, the state (and therefore the true objective for subsequent decisions) changes, so the search correctly restarts from a new root that includes the committed regulation. Old Qs aren’t needed; the previous ΔJ signals were already used to choose the commit.

#### Nuances

- Early leaf/no‑children backups use only −phi(s_leaf); any −ΔJ accumulated earlier in that partial rollout is intentionally not included in that early bootstrap. Once simulations reach terminal/quota from those areas, the backed‑up values include the Σ(−ΔJ_i) term and the phi anchor.

- **Because −phi(s_start) is a constant offset shared by all complete trajectories from the same root, it doesn’t affect which action is preferred; the −ΔJ terms do.**

### 3. About the role of the outer-loop and inner-loop

#### Understanding 1:

After seeing the "real" objective \Delta J upon committing, the agent cannot undo the regulation it decided to commit, so the "lesson" is only available from the second commit onwards, but there is nothing that could be done about the first regulation, which had become the permanent part of the plan. In this sense, the plan is always "less optimal/faulty", especially the first several regulations.

#### Understanding 2:

The agent might attempt to evaluate for real objective \Delta J several time, then pick the "best commit", in this sense, the first regulations and the last regulations are of equal chance to be "optimal".

Which understanding is correct according to this framework?

#### Answer:

> Understanding 2 is the right mental model, with a couple of caveats.

What actually happens in one outer iteration
- The inner MCTS run performs many simulations. Whenever a CommitRegulation edge is visited, it calls the expensive evaluator to get ΔJ and:
  - uses r_base = −ΔJ for that step’s reward, and
  - updates self._best_commit if this ΔJ is the best seen so far.
- The inner run continues (subject to max_actions and commit_eval_limit), potentially evaluating many different commits. ΔJ values influence the search immediately via backup.
- When the inner run stops, mcts.run returns the best commit found (the one with the lowest ΔJ). If no improving commit was found (best ΔJ ≥ 0), the outer loop stops without committing.
- The outer agent then materializes that single best commit into the plan and re-roots. Past commits cannot be undone.

Implications
- A commit is not taken “mid-simulation.” It’s chosen after the inner run finishes, from among the evaluated candidates. So the first regulation is the best one the inner loop could find under its budgets, not merely the first it happened to try.
- Because the planner is sequential and irreversible, the overall plan is greedily constructed and not guaranteed globally optimal. Each regulation is “optimal” only conditional on the regs already committed. Early regs are not inherently worse than later ones, but they can constrain what is achievable afterward.

So:
- Understanding 2 is correct for how a single regulation is chosen within an outer iteration.
- Understanding 1 is correct only in the sense that once a regulation is committed it cannot be undone; however, the “lesson” from ΔJ is already used within the same inner run to pick the first regulation, not only from the second commit onward.