### Quick confirmations from the current code
- Plan and hotspot state are managed in `PlanState` and `HotspotContext`:
```109:140:src/parrhesia/flow_agent/state.py
@dataclass
class PlanState:
    """Snapshot of the planning process prior to MCTS."""

    plan: List[RegulationSpec] = field(default_factory=list)
    hotspot_context: Optional[HotspotContext] = None
    z_hat: Optional[np.ndarray] = None
    stage: StageLiteral = "idle"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "PlanState":
        clone = PlanState(
            plan=list(self.plan),
            hotspot_context=self.hotspot_context,
            z_hat=None if self.z_hat is None else np.array(self.z_hat, copy=True),
            stage=self.stage,
            metadata=dict(self.metadata),
        )
        return clone

    def canonical_key(self) -> str:
        """Return a stable JSON key for caching purposes."""
        payload = {
            "plan": [reg.to_canonical_dict() for reg in self.plan],
            "hotspot_context": self.hotspot_context.to_canonical_dict()
            if self.hotspot_context
            else None,
            "stage": self.stage,
            "z_hat": self._z_hat_signature(),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
```
- The transition currently supports a mid-episode confirm stage:
```91:96:src/parrhesia/flow_agent/transition.py
if isinstance(action, Continue):
    guard_can_continue(next_state)
    next_state.stage = "confirm"
    next_state.metadata["awaiting_commit"] = True
    return next_state, False, False
```
- MCTS uses per-step shaping and local commit evaluation (not terminal-only):
```1233:1237:src/parrhesia/flow_agent/mcts.py
phi_s = node.phi
phi_sp = self._phi(next_state)
delta_phi = phi_sp - phi_s
r_shaped = r_base + delta_phi
total_return += r_shaped
```
- The MCTS docstring notes commit evaluation is integrated:
```98:103:src/parrhesia/flow_agent/mcts.py
- State mutations happen only through `CheapTransition.step`, which returns a copy.
- Commit evaluation integrates RateFinder once per commit action and caches results.
- Flow proxies for shaping are provided via `HotspotContext.metadata['flow_proxies']`.
- Flight grouping per flow (for commits) is provided via `HotspotContext.metadata['flow_to_flights']`.
```

## Proposed redesign: single-loop, end-of-episode reward
Goal: abolish the inner loop, make each episode the whole plan (multiple regulations), emit zero reward until terminal, then compute the global, unscoped objective once and backpropagate.

### 1) State model changes
- Plan-level state (`PlanState`)
  - Keep `plan: List[RegulationSpec]`.
  - Add episode bookkeeping:
    - `reg_count: int` (derived from `len(plan)`).
    - `episode_actions: int` (incremented per low-level action to enforce action budget).
    - `episode_done: bool` (terminal flag).
    - `episode_meta: Dict[str, Any]` with cached cheap stats for gating (e.g., `capacity_by_tv_window`, `factor`, `max_removals_per_reg`, `max_regulations`).
  - We remove `z_hat` completely from the code since we will not use reward shaping. However, we keep `phi` should there be a need for implementation in the future.

- Per-regulation context (`HotspotContext`)
  - Keep `control_volume_id`, `window_bins`, `candidate_flow_ids`, `selected_flow_ids`, `mode`, `metadata`.
  - Add fields/metadata:
    - `removals_used: int` (counter).
    - `flight_counts_by_flow: Dict[str, int]` (from `flow_to_flights` lengths; used for priors and gating).
    - `capacity_estimate: float` (avg or peak capacity across the window for the selected TV; used for gating).
    - Optionally `entrants_by_flow: Dict[str,int]` if available; otherwise rely on flight counts.
  - Stages: simplify to two inter-regulation stages:
    - `"pick_hotspot"` and `"build_regulation"` (no confirm/reward step mid-way).
    - Drop the "confirm" mid-episode reward path; `CommitRegulation` will not evaluate.

### 2) Transition model changes
- Keep `CheapTransition.step(state, action)` but adjust:
  - `AddFlow`: unchanged mutation; zero reward semantics handled in MCTS (no base reward).
  - `RemoveFlow`:
    - Enforce `removals_used < max_removals_per_reg`. If exhausted, disallow the action.
  - (Old) `Continue` (we will rename to `FinalizeRegulation` to avoid confusions):
    - Gate by capacity-proxy rule:
      - Let `total_flights = sum(flight_counts_by_flow[f] for f in selected_flow_ids)`.
      - Let `threshold = entrances_to_occupancy_dwelling_factor * capacity_estimate`.
      - Only allow to proceed to “ready-to-commit” if `total_flights >= threshold`.
    - No stage `confirm`, no mid-way reward; go directly to a commit-able state.
  - `CommitRegulation`:
    - Append a `RegulationSpec` to `plan` with the chosen hotspot and flows; do not evaluate or shape reward.
    - Reset `hotspot_context` to `None` and stage to `"pick_hotspot"`.
  - `NewRegulation` and `PickHotspot`:
    - `PickHotspot` sets `stage = "build_regulation"`.
    - Preload `flight_counts_by_flow`, `capacity_estimate`, and optionally per-flow priors into context metadata.
  - `Stop`:
    - Mark `episode_done = True`; no evaluation happens here; evaluation is done once at MCTS leaf by the search, not the transition.

### 3) MCTS changes
- Rewards
  - Set per-step base reward to 0 for all non-terminal actions.
  - Optionally retain a very small potential-shaping term `phi` strictly for exploration bias, with `phi_scale ≈ 0.0–0.1`. Do not let it dominate.
  - Terminal reward: only when `episode_done` or when depth/regulation limit met. Compute single scalar value as negative global objective: `R_terminal = -J_global(plan)`.

- Terminal and depth control
  - Terminal conditions:
    - `Stop` chosen, or
    - `len(plan) >= max_regulations`, or
    - `episode_actions >= max_episode_actions`.
  - No intermediate commit-evaluation budget; the only evaluation budget is at episode end.

- Priors and widening
  - Hotspot selection prior: proportional to recent exceedances or a lightweight heuristic (e.g., local capacity violations map), with Dirichlet noise.
  - Flow selection prior: proportional to `flight_counts_by_flow` (sqrt or log transform), normalized; add Dirichlet noise using `flow_dirichlet_epsilon/alpha`.
  - Action-level prior temperature set lower for structured choices, higher for exploratory ones.

- Action set per stage
  - At `"pick_hotspot"`: `PickHotspot` for each candidate, or `Stop` when plan is acceptable or budgets constrained.
  - At `"build_regulation"`: `AddFlow`, `RemoveFlow`, `FinalizeRegulation`, `CommitRegulation`, `Back` (optional).
  - Disallow `Continue`’s legacy “confirm” route.

- Caching and value target
  - Remove commit-level caches (`_commit_eval_cache`) from MCTS; evaluation now only at terminal via a single plan evaluation function.
  - Node value `Q` represents expected terminal return (no sum of intermediate rewards).
  - Keep progressive widening; ensure node keys remain stable via `PlanState.canonical_key()`.

### 4) Global objective evaluation at leaf
- New evaluator API (pure function), e.g., `evaluate_plan_global(plan, evaluator, flight_list, indexer, rate_finder_cfg) -> float`:
  - For each committed `RegulationSpec`, derive per-flow `n_f(t)` and compute rates. You may:
    - Option A (pure end-of-episode): run rate search across all regulations jointly or per regulation sequentially but without feeding back partial rewards during search.
    - Option B (cheap heuristic): derive rates via a deterministic proxy (min cap, flow weights) for the MCTS value estimate; re-evaluate precisely outside training for logging.
  - Use existing safespill/global scoring pipeline to get `J_total`. The agent already has a final objective routine:
```1776:1788:src/parrhesia/flow_agent/agent.py
def _compute_final_objective(self, state: PlanState) -> Dict[str, Any]:
    # Build global flights_by_flow by stitching descriptors per (tv, window)
    flights_by_flow: Dict[str, List[Dict[str, Any]]] = {}
    capacities_by_tv: Dict[str, np.ndarray] = {}
    T = int(self.indexer.num_time_bins)

    def _ensure_caps(tv: str) -> None:
        if tv not in capacities_by_tv:
            caps = self.rate_finder._build_capacities_for_tv(tv)
            capacities_by_tv.update(caps)

    # Helper: entrants → flight specs with requested_bin for deterministic scheduling
```
  - Adapt this into a pure callable that accepts `plan` and returns scalar `J_global`. MCTS will call only at terminal.

### 5) Agent orchestration changes (`MCTSAgent.run`)
- Discovery remains: build `HotspotInventory`, annotate candidates with:
  - `flow_to_flights` and `flight_counts_by_flow`.
  - `capacity_estimate` per (TV, window), e.g., average capacity over window bins.
  - Optional `entrants_by_flow`.
- Build a single root `PlanState` with `stage="pick_hotspot"`, budgets in `episode_meta`, and instantiate one `MCTS` in episode mode:
  - `max_sims`/`max_time_s`/`max_actions` control the search.
  - Remove the inner loop of “run a search per regulation then commit”; instead, run one search that returns a best full plan.
- Output:
  - The returned best plan is the policy extracted path at root (argmax-N or max-Q child chain until terminal). Then evaluate once (for reporting) and export.

### 6) Budgets and loop prevention
- Episode-level budgets
  - `max_regulations_in_plan` (caps number of commits).
  - `max_episode_actions` (caps total selection/expansion/step/backup actions).
  - `outer_max_time_s` and `outer_max_runs` (runner-level, keep as is).
- Within-regulation budgets
  - `max_removals_per_reg`: disallow further `RemoveFlow` after threshold.
  - Continue gating via `entrances_to_occupancy_dwelling_factor * capacity_estimate`.
- Loop detection
  - Keep visited-key checks in MCTS; optionally drop back/forward oscillations at the same node by banning reversals for K steps.
  - Penalize or prune repeating flow sets within the same hotspot.

### 7) Priors and noise
- Hotspot priors: from simple exceedance heuristic; add Dirichlet noise with `hotspot_dirichlet_epsilon/alpha`.
- Flow priors: proportional to `flight_counts_by_flow` (softmax over log counts), with Dirichlet noise using `flow_dirichlet_*` from config.
- Action-class priors: allow small prior to `Stop` to end early when no promising hotspots.

### 8) Config surface
- Extend `MCTSConfig` or add `EpisodeConfig`:
  - `max_regulations_in_plan: int`
  - `max_episode_actions: Optional[int]`
  - `entrances_to_occupancy_dwelling_factor: float`
  - `max_removals_per_reg: int`
  - Priors/noise knobs already present (`flow_dirichlet_*`, etc.); reuse.

### 9) Logging/telemetry
- Remove per-commit ΔJ and local objective reporting from inner loop UI. Track:
  - Episode sims, node counts, plan length.
  - Best terminal reward and decoded `J_global`.
  - Top actions and stage summaries remain useful for debugging, but ΔJ(local) metrics should be marked as N/A.
- Keep export and plan validation at the end (unchanged path in `run_agent.py`).

### 10) Step-by-step implementation plan (edits)
- State and guards
  - Add fields to `PlanState` and `HotspotContext`; update `actions.guard_*` to enforce gating and removal budgets.
- Transition
  - Remove “confirm” path; enforce `FinalizeRegulation` gating; keep `CommitRegulation` zero-reward behavior.
- MCTS
  - Remove commit-time evaluation plumbing and caches.
  - Set per-step reward to 0; calculate value only at terminal via `evaluate_plan_global`.
  - Implement priors from counts and noise.
- Agent
  - Build candidate metadata (`flow_to_flights`, `flight_counts_by_flow`, `capacity_estimate`) before search.
  - Replace per-regulation inner runs with a single episode search.
  - Use final evaluator for reporting/export.
- Config and wiring
  - Extend config and pass into agent/mcts/transition.
- Tests
  - Unit tests for guards/gating.
  - MCTS simulation returns zero cumulative reward for partial paths; non-zero only at terminal.
  - End-to-end: verify a small fixture produces a plan with improved global objective vs empty plan.

### Acceptance criteria
- No mid-episode reward/evaluation: all non-terminal steps contribute zero reward; only the terminal plan evaluation produces the learning signal.
- Continue gating works from flow counts and capacity factor; removal budget is enforced.
- The selected plan improves the global unscoped objective vs baseline in at least one smoke test.
- Tree search remains stable with priors/noise; no oscillatory add/remove loops beyond budgets.
- Existing export/validation flows remain functional.

### Notes on computation strategy
- For speed, use a cheap rates heuristic during search for the terminal value, and re-evaluate precisely once outside the search for reporting. Keep a flag to switch to full evaluation inside the search if performance allows.

- - -

Summary
- Redesign to a single MCTS episode over the whole plan: zero per-step reward, terminal reward = negative global objective.
- Add gating and budgets in `actions`/`transition` using flow counts and capacity estimates; enforce a removal budget.
- Update priors based on flow crowding and add Dirichlet noise.
- Move all evaluation to leaf: remove commit-time scoring; adapt agent to run one search and evaluate once at the end.