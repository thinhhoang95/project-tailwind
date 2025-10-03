# Current Implementation's End-to-End Walkthrough

In the current implementation of Monte Carlo Tree Search (MCTS), we attempt to find a *regulation plan* which consists of many regulations. Each regulation consists of a control traffic volume, a time window, one or several "flows" (sets of flights), and the allowed hourly rate.

### Inner loop: MCTS to find regulations

In short, we start from an empty plan. The inner loop starts. We add the first regulation by first choosing a hotspot. A hotspot is a traffic volume where demand exceeds capacity, and the time window is set to be equal to the hotspot duration. The agent could then add new flows, remove flows, until they run out of inner loop budgets (for example: actions, time...), and they have to go to the "confirm stage", from there they can choose to commit. If commit is selected, then a coarse rate finding process is launched, the agent receives a **local reward**, which only measures the exceedances of the current selected hotspot (see `docs/flow_agent/fixes/dot_five_fixes/rate_finder_local_obj.md`). After getting the local reward, backpropagation happens to update the Q values of the visited states, and the MCTS ends. This completes one iteration of the inner loop, and the outer loop advances.

### Outer loop: environment step

- Hotspot inventory is NOT recomputed after each commitment. It’s built once (per outer “master” run) and then candidates are pruned as they’re used or banned.
```649:662:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
descs = self.inventory.build_from_segments(...)
candidates = self.inventory.to_candidate_payloads(descs)
```
```294:321:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
def _remove_hotspot_candidate(...):
    ... if (tv, t0, t1) == target: removed = True
    ... if removed: state.metadata["hotspot_candidates"] = filtered
```

### Final Evaluation

When the outer loop runs out of budget, it will trigger the final evaluation that will yield a global objective function, but **no backpropagation of reward will be performed at the end of the episode**.


### Inner-loop and outer-loop budgets and exact stop conditions

- Inner loop (MCTS) budgets
  - per-simulation commit budget:
```490:492:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
for sig, est in node.edges.items():
    if sig[0] == "commit" and commits_used >= commit_depth:
        continue
```
```655:661:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
if is_terminal or commits_used >= commit_depth:
    leaf_key = state.canonical_key()
    leaf_v = -self._phi(state)
    total_return += leaf_v
    self._dbg(
        f"[MCTS/simulate] terminal_or_budget commits_used={commits_used}/{commit_depth} leaf_bootstrap={float(leaf_v):.3f} return={float(total_return):.3f} path_len={len(path)}"
    )
```

- Outer loop (agent) budgets/stop
  - number of master runs, time, total inner MCTS actions:
```371:409:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
if self.outer_max_runs is not None and master_runs_executed >= self.outer_max_runs:
    if self._global_stats is not None:
        try:
            self._global_stats.on_limit_hit(
                "outer_max_run_param",
                {
                    "outer_runs_completed": int(master_runs_executed),
                    "outer_max_run_param": int(self.outer_max_runs),
                },
            )
        except Exception:
            pass
    break
if self.outer_max_time_s is not None and (now - master_start) >= float(self.outer_max_time_s):
    if self._global_stats is not None:
        try:
            self._global_stats.on_limit_hit(
                "outer_max_time",
                {
                    "elapsed_s": float(now - master_start),
                    "outer_max_time": float(self.outer_max_time_s),
                },
            )
        except Exception:
            pass
    break
if self.outer_max_actions is not None and master_actions_total >= int(self.outer_max_actions):
    if self._global_stats is not None:
        try:
            self._global_stats.on_limit_hit(
                "outer_max_actions",
                {
                    "actions_total": int(master_actions_total),
                    "outer_max_actions": int(self.outer_max_actions),
                },
            )
        except Exception:
            pass
    break
```
  - how many inner commits are attempted in one master run (outer loop budget):
```278:286:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
commit_depth_limit = int(max(0, self.mcts_cfg.commit_depth))
configured_inner_commit_limit = (
    int(self.max_inner_loop_commits_and_evals)
    if self.max_inner_loop_commits_and_evals is not None
    else None
)
outer_loop_budget = (
    int(max(0, configured_inner_commit_limit)) if configured_inner_commit_limit is not None else commit_depth_limit
)
```
```492:503:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
for _ in range(max(1, int(outer_loop_budget))):
    state.stage = "idle"
    action_counts_this_run: Dict[str, int] = {}
    combined_action_counts: Dict[str, int] = {}
    run_stats_this_run: Dict[str, Any] = {}
    try:
        with self._timed("agent.mcts.run"):
            commit_action = mcts.run(
                state,
                run_index=(master_run_index * max(1, int(outer_loop_budget))) + run_idx,
            )
            action_counts_this_run = dict(mcts.action_counts)
```
  - early stop if no local improvement (ΔJ ≥ 0):
```970:986:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
if self.early_stop_no_improvement and delta_j >= 0.0:
    if self.debug_logger is not None:
        try:
            self.debug_logger.event(
                "outer_stop_no_improvement",
                {
                    "delta_j": float(delta_j),
                    "commits": int(commits),
                    "limit": int(outer_loop_budget),
                    "outer_master_run_index": int(master_run_index + 1),
                },
            )
        except Exception:
            pass
    outer_stop_reason = "no_improvement"
    outer_stop_info = {
        "delta_j": float(delta_j),
        "commits": int(commits),
        "limit": int(outer_loop_budget),
    }
```
  - stop when total inner-commits limit reached (if configured):
```1063:1066:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
if configured_inner_commit_limit is not None and commits >= configured_inner_commit_limit:
    final_stop_reason = "inner_commit_limit_reached"
elif configured_inner_commit_limit is None and commits >= commit_depth_limit:
    final_stop_reason = "commit_depth_limit_reached"
```

*Point of confusion:* The inner commit depth is a per-MCTS-run search horizon; the outer inner_commit_limit is a per-master-run cap on how many regulations are actually added. **Example:** commit_depth=1, inner_commit_limit=3 → each MCTS run looks only one commit deep, but the agent will attempt up to three separate commits (three runs of MCTS) in that outer run.

  - after inner-loop(s), compute final objective (no backprop):
```1056:1058:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
with self._timed("agent.final_objective"):
    final_summary = self._compute_final_objective(state)
```

- Optional note (unchanged behavior): The MCTS also blocks evaluating a commit when the budget is exhausted at action time:
```561:568:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
if isinstance(action, CommitRegulation):
    if commits_used >= commit_depth:
        v = -node.phi
        self._dbg(
            f"[MCTS/simulate] commit_blocked budget commits_used={commits_used}/{commit_depth} -> bootstrap={float(v):.3f}"
        )
        self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="commit_blocked")
        self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="commit_blocked")
        return v
```

## Other observations

- Contrary to conventional MCTS in game-playing, there is no roll-out phase in the current implementation. It is **MCTS with leaf evaluation (no rollout)**.

- Consider recomputing hotspot inventory (and carrying forward “assigned delay” effects) inside a simulation so that deeper lookahead becomes meaningful.

## Additional caveats and confirmations

- **Single-pass MCTS per outer attempt**: Each `run(...)` performs one simulate pass and returns the best commit found in that single pass (not many simulations accumulating statistics as in classic MCTS).
```341:360:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
def run(self, root: PlanState, *, commit_depth: Optional[int] = None, run_index: Optional[int] = None) -> CommitRegulation:
    depth_limit = int(commit_depth if commit_depth is not None else self.cfg.commit_depth)
    self._best_commit = None
    ...
    self._simulate(root, depth_limit, sim_index=1)
    if self._best_commit is None:
        raise RuntimeError("MCTS did not evaluate any commit")
    return self._best_commit[0]
```

- **Reward shaping and local scope**: MCTS uses shaped reward r = −ΔJ + Δφ, where ΔJ is the local objective delta from the rate finder at the selected TV/window, and φ is a residual proxy over the hotspot window.
```572:599:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
commit_action, delta_j = self._evaluate_commit(state, sim_index=sim_index, step_index=step_index)
...
r_base = -float(delta_j)
...
phi_s = node.phi
phi_sp = self._phi(next_state)
delta_phi = phi_sp - phi_s
r_shaped = r_base + delta_phi
total_return += r_shaped
```
```637:645:/mnt/d/project-tailwind/src/parrhesia/flow_agent/rate_finder.py
target_set = set(target_cells)
target_arg = target_set if target_set else None

if context is None:
    with self._timed("rate_finder.build_score_context"):
        context = build_score_context(
            flights_by_flow,
            indexer=self._indexer,
            capacities_by_tv=capacities_by_tv,
```

- **No rollout; leaf/bootstrap uses −φ**: When a node is newly expanded or no viable moves remain or terminal/budget is hit, value bootstraps from −φ.
```395:403:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
if node is None:
    node = self._create_node(state)
    created_nodes.add(key)
    ...
    v = -node.phi
    ...
    self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="leaf_bootstrap")
    return v
```

- **Async commit evaluation path**: If a commit evaluation job is scheduled (or pending), the commit action returns immediately with diagnostics and ΔJ = 0 until the result is available.
```971:1003:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
if self._commit_executor is not None:
    ...
    future_job = self._commit_executor.submit(...)
    ...
    commit_action = CommitRegulation(
        committed_rates={},
        diagnostics={"rate_finder": pending_diag},
    )
    return commit_action, 0.0
```

- **Duplicate/banned regulation handling**: Commits that duplicate an existing regulation or are banned are short-circuited and marked as banned in diagnostics.
```913:937:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
info = {
    "control_volume_id": str(ctx.control_volume_id),
    "window_bins": [window_norm[0], window_norm[1]],
    "mode": mode_norm,
    "banned_regulation": True,
    "reason": "duplicate_in_plan",
}
commit_action = CommitRegulation(
    committed_rates={},
    diagnostics={"rate_finder": info, "banned_regulation": True},
)
return commit_action, 0.0
```

- **Final vs system-wide objective outputs**: After building the plan-domain final objective, a separate system-wide objective (across all TVs) can be computed and included in outputs.
```1511:1518:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
def _compute_system_objective(self, state: PlanState) -> Dict[str, Any]:
    """
    Compute objective over ALL TVs in the indexer (global, not scoped).

    Scheduling is applied only to flows that belong to the current plan's
    regulated TVs/windows (if any). Unregulated TVs/windows contribute via
    baseline occupancy (zero-delay) captured by the context.
```

# Implications and Proposed Changes

1. In the whole two step search framework, at any moment, the agent will only attempt to locally optimize the situation to **dissipate the exceedances at the selected hotspot as control volumes for each flows**. There is no global cohesion or coherence between regulations.

    > We might need to adjust where the reward will appear and how it needs to backpropagate. Currenty, as detailed in the document `docs/flow_agent/fixes/dot_five_fixes/three_kinds_of_scopes.md`, the current code base relies on different kind of objective functions at different parts of the code. This is wrong, and inconsistent. In the new approach, we will only use **one and only one global objective function** for consistency. Nevertheless, for computational efficiency, we might rely on smaller **delta-views** to update the global objectives, but **regardless of how it is computed, only the global objective should be used**.
    >
    > We will also need to rework on the back-propagation strategy: the general idea is that the agent will receive absolutely no reward until the very final step of evaluating the whole plan. Before: it receives a reward signal at every regulation commit, but no reward at the end of the "episode". The MCTS happens within regulations, at the end of the whole plan. In the new approach, we will make MCTS backpropagate at the end of the episode. This is more faithful to MCTS, and more correct. In other words, we will do-away with the two-loop structure. The entire search will be organized into one MCTS simulation, and the search can be restarted from canonical partial plans anywhere in the tree.

2. There is no explicit mechanism that “prevents flights that already had their delay assigned from getting another round of delay assignment” between inner commits. The agent stores regulations and only computes a global schedule/objective at the end (and during commit evaluations), but it doesn’t persist “assigned delay flags” back into the candidate discovery or mutate the inventory between commits.

    > Between two outer loop iterations, based on the partial plan so far, we will need to recompute the hotspot inventory, redetect the hotspots. The hotspots could change (some might get extinguished, others may light up). The flights got assigned delays, so the flow component fundamentally changes to. It's like a different problem after each regulation is committed. The current code does NOT currently handle recomputing the hotspot inventory, as well as preventing the flights that already had their delay assigned from getting another round of delay assignment.
    >
    > As a result, the new approach should correctly recompute the hotspots, (and as a result, new flows), it's like solving the new problem after the environment has evolved.

3. The current implementation also does not allow the agent to "return to earlier regulation planning states to continue discovery from there". Once a regulation is committed, it is locked in, and the only point for the MCTS to return to after the end of one episode is the beginning, at another root (which is an empty regulation plan). The reworked framework, like mentioned above, will behave more like a one-big-MCTS-loop.

4. One important question is: if in the new framework, the search returns to a previous node to expand, then how do we deal with the "inner budget control?"

> We add the "commit depth" as part of the state definition: (plan_so_far, depth). For example: if max commit_depth = H, then at depth d the remaining is H-d.

# Instructions

1. Could you help me revise the proposed changes. Do you think the proposed changes are in the right direction?

2. What sort of problems do you find that could derail the implementation process? For me I'm quite concerned about the agent getting stuck in some cycles that it cannot break out (i.e., insufficient guardrails to prevent these suprious or reward hacking).