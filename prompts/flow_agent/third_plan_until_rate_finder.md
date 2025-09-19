Here’s a focused implementation plan for second_plan.md up to and including rate_finder, stopping before MCTS.

Scope

Implement core state, actions, cheap transition, and rate_finder.
Reuse existing evaluator, FCFS queue simulation, and delta flight overlay.
Exclude MCTS internals; only define the needed interfaces so it can plug in later.
Key Dependencies

NetworkEvaluator: src/project_tailwind/optimize/eval/network_evaluator.py
DeltaFlightList: src/project_tailwind/optimize/eval/delta_flight_list.py
FCFS queue: src/project_tailwind/optimize/fcfs/scheduler.py
TV indexer: src/project_tailwind/impact_eval/tvtw_indexer.py
File Scaffolding

src/parrhesia/flow_agent/__init__.py (package)
src/parrhesia/flow_agent/state.py (PlanState, RegulationSpec)
src/parrhesia/flow_agent/actions.py (action types/constructors/guards)
src/parrhesia/flow_agent/transition.py (CheapTransition)
src/parrhesia/flow_agent/rate_finder.py (RateFinder)
Optional tests: tests/flow_agent/test_rate_finder.py, test_transition.py, test_state_actions.py
Interfaces And Data

RegulationSpec (one committed regulation)
control_volume_id: str
window_bins: tuple[int, int] // half‑open [t0, t1)
flow_ids: list[str] // disjoint flight sets across flows
mode: Literal['per_flow','blanket']
committed_rates: Optional[dict[str,int]] | Optional[int] // filled at commit
PlanState
plan: list[RegulationSpec]
hotspot_context: Optional[dict] // current selection: cv_id, [t0,t1), candidate flows
z_hat: Optional[np.ndarray] // cheap residuals proxy
canonical_key() -> str // json-stable hash of (plan, context)
Actions
NewRegulation | STOP
PickHotspot(control_volume_id) -> sets hotspot_context.window_bins via hotspot
AddFlow(flow_id) | RemoveFlow(flow_id) | Continue | Back
CommitRegulation // triggers rate_finder, appends committed RegulationSpec
CheapTransition.step(s,a) -> (s’, isCommit, isTerminal)
Applies symbolic updates to state; keeps z_hat cheap proxy in sync
On CommitRegulation marks isCommit=True, but evaluation is delegated to RateFinder
RateFinder.find_rates(plan, control_volume, window_bins, flows, mode, evaluator, cache, cfg) -> (rate_map_or_scalar, delta_J, info)
Deterministic coordinate descent on discrete rate grid; uses FCFS + evaluator
Phase Plan

S0: Types, config, and wiring
Define dataclasses and enums; include mode parsing and config defaults:
rate grid (e.g., [inf, 60, 48, 36, 24, 18, 12, 6] flights/hour)
passes=2, eps_improve, max_eval_calls, diag flags
Add stable canonicalization (sorted keys, tuples) for plan/regs.
S1: State and actions
Implement RegulationSpec, PlanState with canonical_key().
Implement action constructors and guards (valid windows, flows exist, Back/Continue semantics).
Define “hotspot window” inference: derive [t0,t1) from chosen hotspot (v1: its natural window slice).
S2: CheapTransition
Implement step(s,a):
NewRegulation/PickHotspot mutate hotspot_context.
Add/RemoveFlow mutates flow set and updates z_hat via a cheap additive proxy (see below).
Continue moves to CommitRegulation; Back returns to previous state level.
CommitRegulation marks isCommit=True; terminal if STOP or commit count cap reached.
Cheap residual proxy z_hat:
Maintain a vector proxy updated by add/remove of flows using precomputed per-flow “earliest hotspot crossing mass” within [t0,t1). For v1, sum of earliest‑bin hits across selected flows provides ẑ shaping; clip and decay as needed.
S3: RateFinder
Inputs: control_volume_id, window_bins [t0,t1), flows -> list of flight_ids per flow, mode, evaluator, indexer, flight_list, cache, cfg.
Precompute entrants:
Build active_windows = range(t0, t1)
For each flow, determine identifier_list as the flow’s flights that cross the control_volume_id within active_windows (use FlightList.iter_hotspot_crossings or metadata scan).
Queue-cap simulation:
Per‑flow mode: run FCFS independently per flow using assign_delays(flight_list, identifier_list, control_volume_id, indexer, hourly_rate=r_f, active_time_windows=active_windows).
Blanket mode: one FCFS call with union of all identifiers and a single rate r.
Evaluation and ΔJ:
Use DeltaFlightList(base_flight_list, delays_by_flight) to overlay current candidate delays.
Construct a NetworkEvaluator (or reuse with update_flight_list) and compute objective J; compute ΔJ vs baseline (baseline can be precomputed once per PlanState key).
Sign convention: r_base = −ΔJ (docs); return ΔJ and diagnostics.
Coordinate descent search:
Per‑flow: initialize all r_f = inf (skip FCFS = no delays).
For 2 passes: for each flow in priority order (e.g., most entrants or overlap with hotspot), line search over the discrete grid holding others fixed:
For each candidate r, compute per‑flow delays_f and union with fixed delays for other flows; evaluate ΔJ; pick best r for the flow.
Early stop if pass improvement < ε.
Blanket: 1D grid search over r; pick argmin ΔJ.
Caching:
Key: (plan.canonical_key(), regulation_signature(control_volume_id,[t0,t1), sorted(flow_ids), mode), rate_vector_or_scalar)
Value: (ΔJ, info) with an LRU cap (size configurable).
Diagnostics in info:
Evaluator call count, chosen rates, per‑flow entrant counts, ΔJ improvement per pass, top contributing TVs/hours, timing.
Performance target:
Evaluator calls O(passes × |flows| × |grid|), typical 50–150; wall time < 1s.
S4: Validation and tests
Unit tests for state/actions canonicalization and CheapTransition pathing.
FCFS integration tests:
Per‑flow: synthetic flows with overlapping entrants; verify delay monotonicity vs rate; ΔJ improves or no worse at r=inf.
Blanket: union scheduling correctness.
Caching tests: repeated calls hit cache; keys stable with flow order permutations.
Microbench harness to assert max evaluator calls and rough timing on a small dataset.
Detailed Algorithm Notes

Entrant filtering
Use FlightList.iter_hotspot_crossings([control_volume_id], active_windows) to gather flights per flow within window; stable sorting by (entry_dt, flight_id).
Delta overlay and evaluation
Maintain delays_by_flight as union of per‑flow FCFS results (flows disjoint; if not, later: resolve by max delay per flight).
Build DeltaFlightList(base, delays_by_flight); either create a fresh NetworkEvaluator or call evaluator.update_flight_list(delta_view) then compute J with existing methods (e.g., recompute excess vector and objective aggregation consistent with project conventions).
Cheap residuals ẑ for shaping
Per flow, precompute “earliest hotspot hit per flight” counts into [t0,t1); ẑ += sum over flows when adding; subtract on removal; clip to a small range; use Σ ẑ^2 to compute ϕ as in docs.
Configs

Rate grid: default [inf, 60, 48, 36, 24, 18, 12, 6]
Passes: 2; epsilon: small positive (e.g., 1e-3 of baseline J)
Limits: max_eval_calls, LRU size, diag on/off
Mode: per_flow (default) or blanket
Acceptance Criteria

Deterministic outputs for fixed seeds and same inputs.
Correct ΔJ sign and monotonicity checks on toy cases.
Per‑flow and blanket modes both supported.
Cache hits for repeated queries with identical signatures.
Typical commit within configured budget (<= 1s on sample size).