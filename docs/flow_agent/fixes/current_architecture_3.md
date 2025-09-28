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

## Inner-loop and outer-loop budgets and exact stop conditions

- Inner loop (MCTS) budgets
  - max simulations:
```464:474:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
for i in range(sims):
```
  - time budget:
```465:477:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
if now > t_end: ... stop_reason = "time_budget_exhausted"; break
```
  - global action budget across selection/expansion/step/backup:
```404:415:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
self._actions_done += 1
if self._action_budget is not None and self._actions_done >= self._action_budget:
    raise MCTS._ActionBudgetExhausted()
```
```877:881:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
# expansion counts as one action
self._inc_action("expand", ...)
```
```1151:1153:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
# selection counts as one action
self._inc_action("select", ...)
```
```1227:1232:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
# environment/state transition counts as one action
self._inc_action("step", ...)
```
```2082:2084:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
# backprop counts as one action
self._inc_action("backup", ...)
```
  - per-simulation commit budget:
```931:949:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
if sig[0] == "commit" and commits_used >= commit_depth: continue
```
```1343:1391:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
if is_terminal or commits_used >= commit_depth: ... reason="commit_budget" ... return total_return
```
  - commit evaluation budget (limits calls into RateFinder):
```1750:1762:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
outstanding = self._commit_calls + self._pending_commit_count()
if outstanding >= int(self.cfg.commit_eval_limit):
    rates, delta_j, info = ({}, 0.0, {"reason": "eval_budget_exhausted"})
```
  - “min unique commit evaluations” exploration control (forces exploration/backtracking until a minimum number of distinct commit signatures are evaluated):
```991:1016:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
if selection_records and len(self._commit_eval_cache) < int(self.cfg.min_unique_commit_evals):
    ... choose action by prior weights (forced explore) ...
```
```1021:1047:/mnt/d/project-tailwind/src/parrhesia/flow_agent/mcts.py
if best_sig[0] == "commit" and len(self._commit_eval_cache) < int(self.cfg.min_unique_commit_evals):
    ... forced backtrack for unique commit ...
```

- RateFinder budgets (per commit evaluation)
  - evaluation call limit:
```315:318:/mnt/d/project-tailwind/src/parrhesia/flow_agent/rate_finder.py
eval_call_limit = min(int(self.config.max_eval_calls), int(flow_count) * (int(rate_grid_len) + 3))
```
  - returns diagnostics including eval_calls and stopped_early:
```587:605:/mnt/d/project-tailwind/src/parrhesia/flow_agent/rate_finder.py
"eval_calls": eval_calls, ... "stopped_early": stopped_early
```

- Outer loop (agent) budgets/stop
  - number of master runs, time, total inner MCTS actions:
```842:883:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
if self.outer_max_runs is not None and master_runs_executed >= self.outer_max_runs: break
if self.outer_max_time_s is not None and (now - master_start) >= ...: break
if self.outer_max_actions is not None and master_actions_total >= ...: break
```
  - how many inner commits are attempted in one master run (outer loop budget):
```747:756:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
commit_depth_limit = int(max(0, self.mcts_cfg.commit_depth))
configured_inner_commit_limit = (int(self.max_inner_loop_commits_and_evals) if ... else None)
outer_loop_budget = (int(max(0, configured_inner_commit_limit)) if ... else commit_depth_limit)
```
  - inner-loop repetition within a master run (adds at most `outer_loop_budget` regulations):
```969:987:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
for _ in range(max(1, int(outer_loop_budget))):
    ... commit_action = mcts.run(...)
```
  - early stop if no local improvement (ΔJ ≥ 0):
```1460:1511:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
if self.early_stop_no_improvement and delta_j >= 0.0:
    outer_stop_reason = "no_improvement"; break
```
  - stop when inner-commit limit reached (if configured):
```1515:1545:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
if self.max_inner_loop_commits_and_evals is not None and commits >= self.max_inner_loop_commits_and_evals:
    ... on_limit_hit("inner_commit_limit_reached") ...
```
  - after inner-loop(s), compute final objective (no backprop):
```1546:1552:/mnt/d/project-tailwind/src/parrhesia/flow_agent/agent.py
with self._timed("agent.final_objective"):
    final_summary = self._compute_final_objective(state)
```

## Other observations

- Contrary to conventional MCTS in game-playing, there is no roll-out phase in the current implementation. It is **MCTS with leaf evaluation (no rollout)**.

- Consider recomputing hotspot inventory (and carrying forward “assigned delay” effects) inside a simulation so that deeper lookahead becomes meaningful.

# Implications and Proposed Changes

1. In the whole two step search framework, at any moment, the agent will only attempt to locally optimize the situation to **dissipate the exceedances at the selected hotspot as control volumes for each flows**. There is no global cohesion or coherence between regulations.

    > We might need to adjust where the reward will appear and how it needs to backpropagate. Currenty, as detailed in the document `docs/flow_agent/fixes/dot_five_fixes/three_kinds_of_scopes.md`, the current code base relies on different kind of objective functions at different parts of the code. This is wrong, and inconsistent. In the new approach, we will only use **one and only one global objective function** for consistency. Nevertheless, for computational efficiency, we might rely on smaller **delta-views** to update the global objectives, but **regardless of how it is computed, only the global objective should be used**.
    >
    > We will also need to rework on the back-propagation strategy: the general idea is that the agent will receive absolutely no reward until the very final step of evaluating the whole plan. Before: it receives a reward signal at every regulation commit, but no reward at the end of the "episode". The MCTS happens within regulations, at the end of the whole plan. In the new approach, we will make MCTS backpropagate at the end of the episode. This is more faithful to MCTS, and more correct.

2. There is no explicit mechanism that “prevents flights that already had their delay assigned from getting another round of delay assignment” between inner commits. The agent stores regulations and only computes a global schedule/objective at the end (and during commit evaluations), but it doesn’t persist “assigned delay flags” back into the candidate discovery or mutate the inventory between commits.

    > Between two outer loop iterations, based on the partial plan so far, we will need to recompute the hotspot inventory, redetect the hotspots. The hotspots could change (some might get extinguished, others may light up). The flights got assigned delays, so the flow component fundamentally changes to. It's like a different problem after each regulation is committed. The current code currently handles recomputing the hotspot inventory, as well as preventing the flights that already had their delay assigned from getting another round of delay assignment.
    >
    > As a result, the new approach should correctly recompute the hotspots, (and as a result, new flows), it's like solving the new problem after the environment has evolved.

3. The current implementation also does not allow the agent to "return to earlier regulation planning states to continue discovery from there". Once a regulation is committed, it is locked in, and the only point for the MCTS to return to after the end of one episode is the beginning, at another root (which is an empty regulation plan).

# Instructions

1. Please dig into the code to verify my understanding of the whole framework. If there is an "if" (for example, my stated statement is only true if some variable is set to something), please mention it in detail. 

2. Please be specific about the budget controls: the conditions for the inner loop and the outer loop to stop and advance.

3. Code citation is very welcomed.