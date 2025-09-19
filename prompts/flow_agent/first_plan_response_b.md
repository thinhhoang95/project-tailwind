Goal: Given selected flows (with control TV, window), find good constant caps (or a short cap schedule) that minimizes J under the NetworkEvaluator.
Core idea: Convert rates→delays via a deterministic FIFO queue-cap simulation, avoiding stochastic SA.
Algorithm (single regulation window)
Precompute per‑flow ordered entries into the control TV over [start_bin, end_bin).
Choose a small discrete grid of rate levels per flow (e.g., {∞, 60/h, 48/h, 36/h, 24/h, 18/h, 12/h, 6/h} in flights/hour) derived from gH and xGH.
Coordinate search:
Initialize all flows at ∞ (no cap).
Repeat for K passes (e.g., K=2–3):
For each flow, 1‑D line search over its rate grid with others fixed:
Run queue‑cap simulation to assign per‑flight delays deterministically (FIFO across the window with per‑minute capacity = rate/60).
Build delayed FlightList (delta-only) and compute ΔJ via evaluator (single call).
Accept the best rate per flow; early stop if total ΔJ improvement < ε.
Optionally add a “pairwise adjust” pass for top-2 flows if residual overload persists.
Performance tactics
Cache per‑flow control entries (min arrays of int bins/seconds).
Vectorize per‑minute release and cumulative queues; preallocate.
Reuse evaluator caches; apply “delta scheduling” only for affected flights.
API
find_rates(flows, control_tv, window, evaluator, rate_grid=None, max_passes=3) -> RateFinderResult
apply_rates_to_flights(result) -> FlightList (deterministic schedule)
Why this is fast
No acceptance test/temperature. Pure best‑improvement 1‑D steps.
Minimal evaluator invocations: O(K × |flows| × |grid|).
Deterministic, reproducible, and cache‑friendly.
Fallback
If needed, add “SA‑lite” mode (few iterations, rapid cooling) mirroring run_sa but constrained to top candidates for parity with API docs.
Complete Action Set

Game states
S0 ChooseRegulation (new or adjust existing)
S1 ChooseHotspot
S2 ChooseFlows (add/remove/edit selection)
S3 TuneTemplate (rate template/schedule shape)
S4 Commit (run rate_finder, evaluate, add regulation)
S5 AdjustRegulation (select existing regulation to revise)
S6 STOP
Actions and guards
From S0: {NewRegulation → S1, AdjustExisting → S5, STOP → S6}
From S1: {PickHotspot → S2, Back → S0}
From S2: {AddFlow, RemoveFlow, Next → S3, Back → S1}
From S3: {PickRateTemplate, SetParams, Back → S2, Commit → S4}
From S4: {CommitRegulation (runs rate_finder + J), then → S0}
From S5: {SelectRegulation, AddFlow, RemoveFlow, ChangeWindow, Retune → S3, Back → S0}
From any: STOP
Notes
Commit is the only place where J is evaluated.
Non‑commit steps reward 0 base with shaping from ϕ.
Hard limits: N_MAX_COMMITS, N_MAX_REVISIONS.
MCTS Core

Selection: PUCT with learned priors
U = Q + c_puct * P * sqrt(N) / (1 + N_a).
Expansion: progressive widening for combinatorial actions (flows)
Allow at most k0 + k1 * N^α children; sample by descending prior.
Simulation: depth measured in commits (e.g., D=2–5).
Call CheapTransition for non‑commit steps; Commit runs rate_finder + evaluator once.
Backup: shaped return r′ per your doc, with γ=1 and ϕ(terminal)=0.
Transpositions
Key: canonical (sorted regulations with normalized windows, sorted flow memberships).
Merge nodes across different action orders to reuse statistics.
Potential ϕ (single‑fidelity shaping)

Definition
ϕ(s) = −θ1 Σ z − θ2 Σ z^2 − θ3 FairnessPenalty(s)
z_{h,t} = max(0, demand_{h,t} − c_{h,t})
Implementation
Incremental update using cached slack/excess matrices for current partial plan.
Clip ϕ and Δϕ; update θ online via robust regression against observed ΔJ at commits.
No multi‑fidelity
Only one evaluator for J at commit; non‑commit uses ϕ for dense signal.
No “coarse vs fine” calibration passes.
Priors and Optional Leaf Value

Priors g_θ
Inputs x(s,a): per‑flow features (gH, xGH, ρ, phase bounds, bins_count) from FlowFeaturesExtractor and group-level summaries.
Initial heuristic scorer: linear ranker with hand‑tuned weights; switch to trained model from replay buffer with pairwise ranking loss.
Leaf value v_ψ (optional)
Simple regressor on global summaries (histograms of z) as a bootstrapped baseline; use only when commits are scarce.
Caching and Data Flow

Single evaluator state
Maintain baseline caches at root; apply delta scheduling for candidate commits.
Transposition table
Node stats (N, W, Q, P) keyed by canonical plan; store ϕ and cheap summaries for quick reuse.
Replay buffer
At each commit: store (x(s,a), −ΔJ_true, Δϕ); periodically update g_θ.
Testing and Benchmarks

Correctness
Unit tests for queue‑cap simulator: FIFO, capacity conformance, reproducibility.
Invariance: potential shaping telescopes to −(J(final) − J(baseline)) − ϕ(s0).
Transposition: different action orders share keys.
Performance targets
rate_finder on 1–4 flows, 60–120 min window: < 50–150 ms per 1‑D evaluation; < 1 s per commit.
MCTS (D=3, B=100–300): 1–3 s per planning call on typical hotspots.
Integration smoke
Baseline counts/evaluator parity with automatic_rate_adjustment inputs/outputs.
End‑to‑end: pick 1 hotspot → plan 1–2 regulations → J improvement observed.
Implementation Plan (sequenced)

P0 Scaffolding (0.5d)
Add flow_agent package skeleton and dataclasses for state and regulations.
P1 Rate Finder (1.5–2d)
Implement queue‑cap simulator and coordinate line search; integrate evaluator; microbenchmarks.
P2 Potential ϕ + CheapTransition (1d)
Implement ϕ with incremental updates; clip; simple θ fit.
P3 Actions + Game State (0.5d)
Encode states, guards, canonical key; unit tests.
P4 MCTS Core (1.5–2d)
PUCT, progressive widening, transpositions; single‑fidelity integration; smoke tests.
P5 Priors + Replay (1–2d)
Wire FlowFeaturesExtractor features; heuristic g_θ then trainable pipeline; minimal persistence.
P6 End‑to‑End + Benchmarks (1–2d)
Scenario scripts; acceptance metrics; tuning.
Interfaces

PlanState: regulations, memberships, cached ϕ inputs; canonical_key().
RegulationSpec: control_tv_id, window, flows, rate_template/levels.
RateFinder.find_rates(...) -> RateFinderResult
CheapTransition.step(s, a) -> (s′, isCommit, isTerminal)
MCTS.run(s0, B, D) -> best_action
Assumptions to Confirm

Rate semantics: constant cap per minute across the regulation window is acceptable for v1.
Commit limits: values for N_MAX_COMMITS and N_MAX_REVISIONS.
Objective J and fairness weights: initial θ and β/α in shaping consistent with ops preferences.
Would you like me to draft the module skeletons and stub the rate_finder interface next, or adjust any of the action/state definitions before scaffolding?