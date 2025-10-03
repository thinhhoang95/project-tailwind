Short answer
- The big risks are: exploding action space, sparse/late rewards, expensive global evaluations, non-stationarity from recomputation, cycles/transpositions, and async/staleness issues. You’ll need progressive widening, potential-based shaping (policy-invariant), aggressive caching/incremental schedule updates, canonicalization/transposition tables, and clear validity guards.

Below is a concrete revision that keeps your intent but reduces the chance of getting stuck or derailed.

1) Revised proposal: a coherent one-tree design

What to keep conceptually
- Keep commit_depth as the horizon (max number of regulations).
- Keep rate finding as a candidate-generation helper, but it cannot define its own objective; it should use the global objective as the scoring target.
- Keep diagnostics, async evaluation option, and budgets; just reframe them around one MCTS.

Core changes to adopt
- One MCTS over whole plans
  - State: S = (plan_so_far as an orderless canonical set of regulations, remaining_depth = H − |plan_so_far|, schedule_context, hotspot_inventory, caches).
  - Transition: apply a regulation, re-schedule globally for all regulated flows, recompute hotspots.
  - Terminal: remaining_depth == 0 or no admissible actions.
  - Return/value: −J_global(S_terminal) relative to a fixed baseline J0, or equivalently J0 − J_global(S_terminal). No separate “local” objectives.

- Action generation = macro-actions only
  - Each action is “commit one regulation” (choose TV/window/flows/rates).
  - Generate actions from the current state’s hotspot inventory. Recompute inventory after each transition.
  - Do not include “remove regulation” or “tweak flow” micro-steps as separate actions in the main tree; use them inside the regulation proposal generator (see below).
  - Prevent duplicates and banned regs via canonical checks at generation time.

- Regulation proposal generator (inside node expansion)
  - Given the current schedule/hotspots, propose top-K regulations via a fast local search:
    - choose hotspot → pick one of several window variations,
    - choose flow subsets (contributors + a few nearby expansions),
    - choose a small set of candidate rate vectors.
  - Score proposals using the global objective delta (re-schedule regulated flows subset, but evaluate with the same global objective).
  - Keep only the top-K proposals as children and grow K via progressive widening with visit count.

- Global objective only, with optional potential-based shaping
  - Primary return is always based on the global objective J over the full indexer (or plan-domain if that is the defined global J).
  - To reduce sparsity without breaking correctness, you can keep shaping but make it potential-based:
    - r’(s, a, s’) = r_global(s, a, s’) + Φ(s’) − Φ(s) with γ=1,
    - where r_global is zero until terminal and then r_terminal = J0 − J(s_T), or you may use per-step r_global = J(s) − J(s’) if you want dense rewards.
    - Φ(s) must be derived from the same global goal (e.g., residual exceedance mass over all TVs/time after scheduling). This preserves policy invariance, stops reward hacking from mismatched objectives, and keeps your “one objective” rule intact.

- Leaf evaluation and rollout policy
  - No random rollout needed. Use leaf evaluation = −J(s_leaf) at budget boundary or terminal.
  - If leaf evaluation is too slow, cache J(s) and hotpots per canonical plan, and consider a cheap heuristic V_leaf(s) (admissible or consistent with J) for early bootstrapping, gradually replaced by exact J as time allows (best-effort evaluation).

- Progressive widening and priors
  - Progressive widening over actions to prevent branching explosion: expand at most K = c · N^α children at a node (N = node visits, 0<α<1).
  - Use a prior over actions (PUCT) from your proposal generator’s score or a learned/predictive model to bias the tree policy.

- Budgets and anytime behavior
  - Replace the two-loop structure with a single tree search loop that runs until time/simulation/action budget.
  - Keep your “outer_max_time/actions” as the top-level budget; “commit_depth” is still the horizon and is embedded in the state.
  - Maintain an incumbent best plan at the root. When budget ends, return the incumbent (best J found), not the last leaf.

- Canonicalization and transpositions
  - Canonical plan key should be orderless (e.g., sorted set of normalized regulations) so that permutations of commit order map to the same state.
  - Use a transposition table keyed by this canonical key to reuse J(s), schedule_context, hotspot inventory, and children lists across the tree.

- Hotspot inventory as a function of state
  - After each action (i.e., after re-scheduling), recompute the hotspot inventory from the new scheduled demand/capacity.
  - Flights’ previously assigned delays are not “frozen”; they’re recomputed jointly for all regulated flows, which avoids double-assignment artifacts and keeps the system consistent.

- Async evaluation (optional)
  - If commit evaluation is async, mark edges as pending and use optimistic/pessimistic bounds and virtual loss.
  - On completion, merge results into the node’s children and update Q/N. Ensure versioning so late results don’t poison newer states.

What this removes or simplifies
- No inner loop per-hotspot MCTS that backprops local returns.
- No separate inventory lifecycle outside the state transition; it’s recomputed per-node.
- No final “episode reward only at the end of outer loop”; reward is attached to leaf states in the one MCTS.

2) Likely pitfalls and concrete guardrails

A. Action-space blow-up and search starvation
- Problem: Recomputing hotspots each state can yield many TVs, many windows, many flow subsets, many rate vectors.
- Guardrails:
  - Progressive widening with tiered generators:
    - Tier 1: for each top-M hotspots by exceedance mass, 1–3 tight windows, 1–3 flow sets, 1–2 rate choices → at most a few dozen children.
    - Tier 2 (unlocked with visits): window expansions/shifts, more flow variants, more rate choices.
  - Cap per-hotspot children and total children. Admit more only as N grows.
  - Stochastic sub-sampling of similar proposals; cluster near-duplicates and keep one.
  - Early reject candidates that do not improve a cheap upper bound on J (e.g., if quick bound says delta-J cannot be < ε).

B. Sparse/late reward and credit assignment
- Problem: If you use only terminal −J, the tree can be slow to learn.
- Guardrails:
  - Use potential-based shaping Φ derived from residual exceedance mass or a tight surrogate of final J. This is policy-invariant and densifies the signal.
  - Or adopt step-wise r_global = J(s) − J(s’) (still global, not local), which is dense and consistent.

C. Expensive global evaluation per node
- Problem: Re-scheduling and recomputing J at every node is costly.
- Guardrails:
  - Caching:
    - Transposition table keyed by canonical plan → store J(s), schedule_context, hotspot inventory.
    - LRU cache for schedule_context to keep memory bounded.
  - Incremental/delta scheduling:
    - When adding a regulation, re-solve only the affected flights/windows if your solver supports warm starts or decomposition.
    - Warm starts for the rate finder; re-use contexts, segment caches.
  - Dual-mode evaluation:
    - First pass: cheap approximate evaluation for expansion ordering.
    - Second pass (on promising children or on leaf): exact evaluation to commit Q.

D. Non-stationarity and staleness
- Problem: Async jobs and mutable caches can cause inconsistent values.
- Guardrails:
  - Make the environment deterministic for a given plan key. All derived artifacts must be a pure function of the plan.
  - Version your caches; don’t reuse a schedule_context past its plan key.
  - For async edges, store a “pending” child with a timestamp and tie it to a specific parent state revision. Drop or re-evaluate if stale.

E. Cycles, duplicates, and transpositions
- Problem: Multiple paths to the same plan; cycles via add/remove; duplicate regulations.
- Guardrails:
  - Only allow “add regulation” actions; no “remove/tweak existing” at the macro level. Tweaks happen inside the proposal generator when producing a new, distinct regulation.
  - Orderless canonical key for plans, transposition table merge.
  - Duplicate/banned checks at action-gen; maintain a per-state banned set with reasons in diagnostics.

F. Reward hacking from mismatched objectives
- Problem: If any module uses a different objective, the agent will exploit the mismatch.
- Guardrails:
  - One and only one J_global. Rate finder scores with that J, hotspot inventory uses it for selection thresholds, and leaf evaluation uses it.
  - If you keep shaping, ensure it is potential-based with Φ derived from J. No ad hoc “local” scores.

G. Thrashing and complexity creep
- Problem: Many small regulations that micro-adjust the same flows.
- Guardrails:
  - Add explicit plan complexity cost in J (e.g., λ · number_of_regulations or λ · total edges).
  - Minimum spacing constraints between regulations that target the same TV/window family.
  - Tie-breakers that prefer simpler plans when ΔJ is within a small tolerance.

H. Early-stop/no-improvement logic
- Problem: In the old code, early stop after a local commit could halt exploration too soon.
- Guardrails:
  - In the one-tree setup, stop by budget or by “no incumbent improvement for T seconds / S simulations.”
  - Always return the best incumbent plan found so far.

I. Reproducibility and determinism
- Problem: Debugging is hard without determinism.
- Guardrails:
  - Deterministic candidate ranking with seeded RNGs.
  - Canonical sorting when ties occur.
  - Capture seeds and environment versions in diagnostics.

J. Memory growth
- Problem: Tree and caches can explode.
- Guardrails:
  - Limit max nodes; prune with least-recently-visited, keep full stats at the root and a frontier.
  - Periodically re-root the tree at the current best incumbent (tree reuse).

3) Concrete design details to de-risk implementation

- State and key
  - key = hash(sorted(regulations_normalized)), include remaining_depth implicitly via |plan|.
  - Store: J(s), schedule_context, hotspot inventory, children list, visit stats (N, W, Q), bounds.

- Tree policy (PUCT)
  - Select child maximizing Q + c_puct · P(a|s) · sqrt(ΣN)/ (1 + N_a).
  - P(a|s) from proposal generator’s approximate delta-J, normalized per node.

- Node expansion
  - Build hotspot inventory (top-M by exceedance mass).
  - For each hotspot, generate a small candidate set; evaluate cheaply; keep best few; add children lazily.

- Evaluation
  - Exact: compute schedule for all regulated flows and objective J(s).
  - Approx: quick bound based on residual exceedances, rate limits, and previously cached slacks.

- Backprop
  - Use total return equal to J0 − J(s_terminal) plus potential shaping if enabled.
  - Average (or max) backups; be consistent across the codebase.

- Budgets
  - mcts.max_time_s, mcts.max_sims, mcts.max_expansions, horizon H.
  - Remove “outer loop” semantics; keep a thin coordinator that runs MCTS with budgets and returns the incumbent.

- Asynchronous evaluation (optional)
  - Use virtual loss to spread threads.
  - For pending children, use an optimistic bound to allow exploration without committing to a value until done.

4) On “commit depth” in the state
- Yes, encode remaining_depth = H − |plan| in state. That cleanly imposes the horizon inside the tree policy. You can still parameterize H from config (previous commit_depth).

5) Migration plan (low-risk)
- Phase 1: Unify objectives
  - Make rate_finder and any scoring use J_global only. Remove local-only objective paths. Keep the two-loop for now.
- Phase 2: Recompute hotspot inventory post-commit
  - After each commit in the current outer loop, recompute schedule and rebuild inventory. Ensure no double assignment artifacts.
- Phase 3: Introduce canonical plan keys and transposition cache
  - Cache J(s), schedule_context per plan key. Unit test determinism and equality across different commit orders.
- Phase 4: Convert to single MCTS with multi-sim runs
  - Implement node/data structures, progressive widening, proposal generator. Keep async disabled initially.
- Phase 5: Add optional shaping Φ and async evaluation
  - Validate policy invariance (A/B vs no-Φ) and convergence speed.
- Phase 6: Memory/time hardening
  - LRU caches, pruning policies, re-rooting, reproducibility harness.

6) Minimal tests to add
- Plan equivalence: different commit orders → identical J and key.
- Duplicate/banned regulation filters work across states.
- Hotspot recomputation reflects scheduled delays; extinguished hotspots disappear, new ones can appear.
- Potential shaping invariance: best plan with and without Φ is the same (up to tie-breaking).
- Async correctness: late results do not update the wrong node; pending edges converge to the same Q as sync.
- Monotonic capacity: no state violates TV capacities post-schedule.

Bottom line
- Your proposed changes are in the right direction, but don’t throw away all intermediate feedback. Use global-only evaluation, and if you need denser signals, add potential-based shaping derived from that same global objective.
- The main engineering challenges are controlling branching, making evaluation cheap enough, and keeping values consistent in a dynamic, cached environment. Progressive widening, canonicalization/transpositions, incremental scheduling, and cautious async are your best tools to avoid cycles and reward hacking.