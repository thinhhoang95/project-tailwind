## Objectives and Scopes in the Flow Agent

This note explains the different objective values you may see in the console and JSONL logs, and clarifies their scopes. Use this to interpret the numbers consistently and avoid mixing domains.

### Terms
- **Objective (J)**: The scalar cost returned by the evaluator. Lower is better. Components include `J_cap`, `J_delay`, `J_reg`, `J_tv` (and optionally `J_share`, `J_spill`).
- **ΔJ (delta J)**: Change in objective for a local commit evaluation. The reward used in MCTS for a commit is `r_base = −ΔJ`.

- **Component math**: The reported objective is the sum of its components: `J_total = J_cap + J_delay + J_reg + J_tv (+ J_share + J_spill when enabled)`.

### Scopes
There are three relevant scopes:

1) System Baseline (run_start)
- What: Objective over the union of all hotspot candidate TVs/windows discovered at the beginning of the run, with no regulations applied (baseline schedule = release at requested bins).
- Where logged: `run_start` event.
- Keys: `objective` plus `objective_scope = "system_candidates_union"`.
- Purpose: A broad, system-level baseline for the session’s candidate set. It is NOT the same domain as the final plan unless the final plan covers all the same TVs/windows.

2) Candidate/Local (rate finder and live panel)
- What: Objective and ΔJ evaluated by `RateFinder` for a single TV/window under consideration (local domain).
- Where shown: the live panel titled “Best Candidate (Local)”; logged via `global_best_update` snapshots.
- Keys: `best_objective_value`, `best_delta_j`, with hints `best_scope = "local_candidate"`.
- Purpose: Guides MCTS during search. This is the learning signal; it is local by design and faster to compute.

3) Final Plan (run_end)
- What: Objective computed from the current plan’s committed regulations. Domain = TVs/windows that actually received a regulation (plan domain). Unregulated TVs/windows are not added to the domain by default.
- Where logged: `run_end` event.
- Keys: `objective` plus `objective_scope = "plan_domain"`.
- Purpose: Selects the best inner run at the end and is the authoritative cost for the delivered plan.

4) System-wide (all TVs)
- What: Objective over ALL TVs known to the indexer (global, not scoped to candidates or plan). Scheduling is applied only to flows in the plan’s regulated TVs/windows (if any); other TVs contribute via baseline occupancy.
- Where logged: attached to `run_end` payload as `system_objective` and optionally to `run_start` as `system_objective` (both labeled with `system_objective_scope = "all_tvs"`). The runner prints this as “System objective (all TVs)”.
- Purpose: Provides an apples-to-apples system metric across the entire network regardless of which TVs/windows were considered or regulated.

### Console strings (clarified)
- Best panel title: “Best Candidate (Local)”. Rows show “Candidate objective (local)” and “Best ΔJ (local)”.
- Final line: “Final plan objective (plan domain): A → B (ΣΔJ_local: …)”. Here, A is computed as `B − sum(ΔJ_local)` and is only a convenience estimate of the plan’s starting point in the local-ΔJ sense; it is not the system baseline.
- System line: “System objective (all TVs): X”. This is the system-wide metric comparable to the system-wide baseline when available.

### Why numbers can differ a lot
- The system baseline (run_start) usually spans many TVs/windows, so its objective can be much larger than the final plan objective that only covers the regulated TVs/windows.
- The live “Best Candidate (Local)” panel shows a local number (single TV/window), so it is typically much smaller than system-level values.
- The system-wide (all TVs) objective includes the entire network; it can go up or down even when the plan-domain objective improves, because impacts outside the plan domain are also counted.

### Current learning signal in MCTS
- Commit reward uses the local ΔJ from `RateFinder` (scope: single TV/window), with potential-based shaping (Δphi) during simulation. The final global objective is not used in backprop.

### Migration path (optional)
- If you need a “system final objective” comparable to the `run_start` baseline, compute the final plan on the same union-of-candidates domain and log it alongside `run_end`. This is slower but yields apples-to-apples A → B (Δ) across a fixed domain.

### What to compare (and what not to)
- **OK to compare**:
  - `system_objective (all_tvs)` at run start vs `system_objective (all_tvs)` at run end → true system-level change.
  - Plan-domain objectives between different plans only when their domains (regulated TVs/windows) are identical; otherwise, prefer the system-wide metric for fairness.
- **Not comparable**:
  - `system_candidates_union` (run_start baseline) vs `plan_domain` (run_end). These are different scopes/domains.
  - Local candidate objective/ΔJ vs any system-level objective.

### Mapping typical log lines to scopes
- `run_start.objective` with `objective_scope = "system_candidates_union"` → union-of-candidates baseline.
- `global_best_update.best_objective_value` with `best_scope = "local_candidate"` → local TV/window metric; also shows fields like `best_N_flights_flows` and `best_N_entrants_flows` for scale context.
- `run_end.objective` with `objective_scope = "plan_domain"` → final plan-domain objective; components sum to this value.
- `run_end.system_objective` with `system_objective_scope = "all_tvs"` (and optionally `system_initial_objective`) → system-wide metric; the runner prints this as “System objective (all TVs)”.

### Where these numbers come from (code pointers)
- Plan-domain objective and components are produced in `src/parrhesia/flow_agent/agent.py` by `_compute_final_objective(...)`, using `parrhesia.optim.objective.score_with_context(...)`.
- System-wide objective is computed in `src/parrhesia/flow_agent/agent.py` by `_compute_system_objective(...)` (scope: all TVs) and attached to `run_end` (and sometimes `run_start`).
- Console prints for the plan-domain and system-wide lines are in `examples/flow_agent/run_agent.py`.
- Objective component math (`J_cap`, `J_delay`, `J_reg`, `J_tv`, optional `J_share`, `J_spill`) is implemented in `src/parrhesia/optim/objective.py`.

### Quick interpretation example
- A large number at `run_start.objective` (scope = `system_candidates_union`) reflects the union of candidates and will usually dwarf plan-domain numbers.
- A smaller number at `run_end.objective` (scope = `plan_domain`, with a components breakdown) reflects cost only over TVs/windows that were actually regulated.
- A “System objective (all TVs)” printed by the runner (and logged as `system_objective`) is the true network-wide figure; compare it to the system-wide initial to gauge net impact, even if the plan-domain objective improved.


