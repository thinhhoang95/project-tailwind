Below is a concise review of Plan A and Plan B, followed by a consolidated, achievable plan that keeps MCTS as the core and targets a working prototype as early as possible. The final plan assumes a single unique rate per flow within a regulation window (no multi-bin schedules). A “blanket cap” mode (one rate for all flows in a regulation) is supported as a trivial subset.

Review: strengths and weaknesses

Plan A
- Strengths
  - Fast, deterministic rate_finder using queue-cap simulation + coordinate descent on a discrete rate grid. This is simple, reproducible, cache-friendly, and minimizes evaluator calls.
  - Clear MCTS action/state design with commit-only J evaluation and potential-based shaping for dense signals.
  - Explicit transpositions via canonical plan keys.
  - Testing targets and tight performance budgets oriented to a quick prototype.
- Weaknesses
  - Mentions templates/schedules; could imply multiple bins, which conflicts with “one rate per flow” and adds complexity.
  - Assumes potential (ϕ) calibration online; if ϕ is off, guidance can mislead search (mitigation needed).
  - AdjustExisting flow/regulation state adds scope that may not be necessary for the first working prototype.

Plan B
- Strengths
  - End-to-end packaging: CLI, config, logging, telemetry, caches, deliverables, risks, and milestones.
  - Clear MCTS pseudocode and interfaces; single-fidelity evaluator for both planning and commit is consistent with the ask.
  - Broader cache strategy and replay buffer plumbing; thorough testing plan and performance budgets.
- Weaknesses
  - Rate_finder includes coarse-to-fine, surrogate scoring, and micro-annealing—more moving parts and tuning risk for v1.
  - Mentions rate templates and multi-bin knobs; conflicts with the “one unique rate per flow per regulation” constraint.
  - Larger milestone scope than needed for earliest working prototype.

Final plan: focused, complete, and achievable

Goals and constraints
- Deliver a working prototype that:
  - Uses MCTS (PUCT + progressive widening + transpositions) as the decision core.
  - Commits regulations where each selected flow in the regulation window has one unique, constant rate (no multi-bin schedules in v1). Optional blanket-cap mode supported as a special case.
  - Evaluates J only at commit via NetworkEvaluator; uses potential ϕ for non-commit shaping.
  - Produces measurable ΔJ improvement within strict time budgets.
- Out of scope for v1: multi-bin schedules, micro-annealing, complex fairness penalties, heavy ML for priors, parallel MCTS.

Architecture and modules
- Core
  - src/parrhesia/flow_agent/state.py
    - PlanState: current plan (list of RegulationSpec), hotspot context, cheap residual cache, canonical_key().
    - RegulationSpec: control_volume_id, window [t0, t1), flow_ids (set), mode (per-flow or blanket), and (for commit-time) inferred rates per flow.
  - src/parrhesia/flow_agent/actions.py
    - Enums/constructors for actions; guards; signatures for canonicalization.
  - src/parrhesia/flow_agent/transition.py
    - CheapTransition: apply actions symbolically, maintain cheap residuals; returns (s′, isCommit, isTerminal).
  - src/parrhesia/flow_agent/rate_finder.py
    - Deterministic coordinate-descent over discrete rate grid; queue-cap simulation; integrates evaluator delta J.
  - src/parrhesia/flow_agent/mcts.py
    - PUCT + progressive widening + transposition table; shaped backup; single-fidelity commit integration.
  - src/parrhesia/flow_agent/priors.py
    - Heuristic priors from flow/hotspot features; optional replay buffer-based update.
  - src/parrhesia/flow_agent/replay.py
    - Minimal buffer for priors training later (can be stubbed for v1).
- Reuse
  - NetworkEvaluator and existing caches from automatic_rate_adjustment.
  - Flow/Hotspot features from FlowFeaturesExtractor for priors and cheap residuals.

Action/state model (complete, simplified for v1)
- States
  - ChooseRegulationMode: NewRegulation | STOP (AdjustExisting deferred; see backlog).
  - SelectHotspot: PickHotspot(h) → SelectFlows
  - SelectFlows: AddFlow(f) | RemoveFlow(f) | Continue → CommitRegulation | Back
  - CommitRegulation: run rate_finder + evaluator, apply regulation to plan, then back to ChooseRegulationMode
- Terminal conditions
  - STOP, or N_MAX_COMMITS reached.
- Notes
  - Window selection v1: given by picked hotspot (its natural [t0, t1)); allow ChangeWindow later (backlog).
  - Regulation modes: per-flow rates (default); blanket cap optional toggle (single rate shared by all selected flows).

Potential shaping ϕ and rewards
- Base reward
  - Commit: r_base = −ΔJ
  - Non-commit: r_base = 0
- Potential ϕ
  - ϕ(s) = −θ1 Σ z − θ2 Σ z^2 with small, fixed θ for v1 (e.g., θ1=0, θ2>0) to avoid calibration drift; clip ϕ and Δϕ.
  - Later: optionally learn θ online from observed (ΔΣz, ΔΣz^2) → ΔJ via robust regression.
- Shaped reward
  - r′ = r_base + ϕ(s′) − ϕ(s); γ=1; ϕ(terminal)=0.
- Cheap residuals
  - Maintain ẑ using precomputed per-flow contribution histograms into (h, t); for add/remove flows, update ẑ by adding/removing the flow’s projected demand in the regulation window; for v1 this is a first-order proxy.

Rate_finder (single regulation commit; one unique rate per flow)
- Inputs: flows, control_volume, window [t0, t1), mode (per-flow or blanket), evaluator, caches.
- Rate grid
  - Discrete ladder shared by all flows (configurable), e.g., {∞, 60/h, 48/h, 36/h, 24/h, 18/h, 12/h, 6/h}.
  - If blanket mode: one scalar chosen from same grid.
- Queue-cap simulation (deterministic, FIFO)
  - For each flow, precompute its flights’ entry times to the control volume within [t0, t1).
  - Given a candidate rate r_f for flow f, simulate per-minute release counts using a fractional accumulator (token-bucket style), assign delays FIFO to respect the cap; produce a delta schedule (only flights in window).
  - Apply deltas; compute ΔJ via evaluator’s delta evaluation.
- Search procedure (coordinate descent)
  - Initialize all r_f=∞ (no cap).
  - Repeat for 2–3 passes:
    - For each flow f (descending heuristic priority, e.g., overlap with hotspot), line search r_f over the rate grid with others fixed:
      - Evaluate ΔJ using queue-cap + evaluator; pick best r_f.
    - Early stop if improvement < ε.
  - Optional one pairwise adjust pass for top-2 flows if residual overload remains.
- Complexity target
  - Evaluator calls per commit: O(passes × |flows| × |grid|), typically < 50–150; wall time < 1 s.
- Output: rate vector {f → r_f} (or single r for blanket), ΔJ, diagnostics.
- Caching
  - Memoize (plan_key, regulation_signature, rate_vector) → (ΔJ, diagnostics); LRU bounded.

MCTS details (not simplified)
- Selection: PUCT with priors
  - U(s,a) = Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a)); c_puct ~ 1.5–3.0.
- Progressive widening for action explosion
  - Child budget m(s) = k0 + k1 * N(s)^α with α in [0.6, 0.8]; expand by descending prior among not-yet-expanded candidates (flows, STOP).
- Expansion and simulation depth
  - Depth counted in commits; D=2 for v1 (configurable to 3).
  - Non-commit transitions are CheapTransition only; CommitRegulation triggers rate_finder + evaluator once.
- Backup
  - Use shaped return; bootstrap leaf with −ϕ(s) if no v_ψ.
- Transpositions
  - Canonical key = sorted regulations with normalized windows and sorted flow IDs; merge nodes across action orders to reuse stats.

Priors and replay (minimal for v1)
- Priors g_θ (v1)
  - Handcrafted from features: flow’s share of hotspot load in [t0, t1), historical congestion impact, distance/overlap to control TV, number of flights in window.
  - Rank actions and convert to probabilities via softmax with temperature.
- Replay buffer (post-MVP)
  - Store (x(s,a), −ΔJ_true, Δϕ) at commits; optional later training of g_θ via pairwise ranking. Not required for first working prototype.

Caching and data flow
- Single evaluator state reused at root; delta evaluation for commits.
- Transposition table stores N, W, Q, P, widen counters, cached ϕ inputs.
- LRU caches for rate_finder memo and evaluator deltas; persist within a planning call.

Performance budgets and targets
- Rate_finder per commit: < 1 s on typical 1–4 flow selections, 60–120 min windows.
- MCTS simulations per decision: 300–1000 (config); average ≤ 1 commit per simulation.
- End-to-end per decision: 2–10 s on 8 cores; prototype acceptable at 5–20 s if ΔJ improves reliably.

Testing and benchmarks
- Unit tests
  - Queue-cap simulator: capacity conformance, FIFO, deterministic reproducibility.
  - Evaluator: delta vs full evaluation consistency on small plans.
  - Canonicalization invariants for plans and actions.
- Integration
  - Single hotspot scenario: MCTS selects 1–2 flows, commits regulation with ΔJ<0 (improvement).
  - Reproducibility with fixed seeds.
- Metrics
  - ΔJ, Σz, Σz^2, runtime breakdown, cache hit rates, sim/sec, commits/sim.
- Acceptance for MVP
  - On at least two real scenarios, within 60 s, produce a plan with ≥ X% (define, e.g., 5–15%) ΔJ improvement vs no-regulation baseline; deterministic reruns within ±1% ΔJ.

Implementation steps and timeline (achievable)
- S0 (0.5 day): Scaffolding
  - Package skeleton, dataclasses (PlanState, RegulationSpec), canonical_key(), configs, logging.
- S1 (1.5–2 days): Rate_finder v1
  - Implement queue-cap simulation and discrete coordinate descent; integrate evaluator; microbenchmarks; memo cache.
- S2 (0.5–1 day): CheapTransition + ϕ v1
  - Cheap residual ẑ and ϕ(s) = −θ2 Σ ẑ^2 with clipping; no online calibration yet.
- S3 (0.5 day): Actions and guards
  - NewRegulation flow with SelectHotspot → SelectFlows → CommitRegulation; STOP; hard limits N_MAX_COMMITS.
- S4 (1.5–2 days): MCTS core
  - PUCT, progressive widening, transpositions; shaped backup; run end-to-end on small scenario.
- S5 (0.5–1 day): Heuristic priors + minimal replay plumbing
  - Simple feature-based prior; stub replay buffer API for later learning.
- S6 (1–2 days): E2E hardening, benchmarks, docs
  - CLI entrypoint, configs, deterministic seeds; basic reports; thresholds and budgets.

Optional backlog (post-MVP)
- AdjustExisting regulation mode; ChangeWindow action with guardrails.
- Online calibration of ϕ weights; fairness penalty term if needed.
- Learned priors from replay; parallel candidate evaluation inside rate_finder.
- Deeper commit depth D=3 with budget tuning; parallel MCTS sims with virtual loss.
- Blanket cap mode UI and priors.

Interfaces (concrete contracts)
- NetworkEvaluator
  - evaluate(plan) -> {J, z, components}
  - deltaJ_commit(s, a) -> float  (invokes rate_finder internally and applies resulting deltas in a single pass)
- RateFinder
  - find_rates(plan, control_volume, window, flows, mode, evaluator, cache, cfg) -> (rate_map, delta_J, info)
- CheapTransition
  - step(s, a) -> (s′, isCommit, isTerminal), updates ẑ and ϕ incrementally; no full evaluation.
- MCTS
  - run(s0, budget_sims, commit_depth) -> best_action; returns action to execute now.
- Priors
  - score_prior(s, candidate_actions) -> dict[action → P]

Key defaults to speed up v1
- Commit depth D=2; N_MAX_COMMITS=2 per plan call.
- Rate grid as above; 2 passes of coordinate descent; ε improvement threshold small but nonzero.
- Progressive widening α=0.7, k0 small (e.g., 4), k1 tuned to budget.
- ϕ uses only Σ ẑ^2; θ2=1.0 (scale to metric); clipping to guard against misguidance.

Assumptions to confirm
- Single unique rate per flow across the regulation window is acceptable for v1; blanket cap allowed as a special case.
- Hotspot-derived [t0, t1) window is acceptable for v1; window editing deferred.
- Evaluator delta application supports “apply only flights in window” schedule updates with caches.
- Accept initial heuristic priors; learned priors can follow after MVP.

Why this plan is achievable
- It retains the full MCTS machinery (PUCT, widening, transpositions, shaped backup) per your constraint not to simplify MCTS.
- It adopts Plan A’s deterministic, fast rate_finder to keep inner-loop costs predictable and low.
- It trims optional complexities (templates, multi-bin, annealing, heavy learning) that commonly prolong tuning cycles.
- It preserves Plan B’s strengths in packaging, testing, caching, and deliverables so we can ship a usable prototype quickly.

If you confirm the assumptions, I can produce the initial module skeletons and the rate_finder interface/stubs next, aligned with the file layout above.