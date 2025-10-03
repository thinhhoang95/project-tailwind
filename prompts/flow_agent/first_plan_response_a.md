## Implementation Plan (single-fidelity; no multi-fidelity)

This plan instantiates the framework in `docs/flow_agent/first_plan.md` using a single `NetworkEvaluator` for both planning and execution. Multi-fidelity variants are intentionally omitted.

### 1) Architecture and modules

- **Core data**
  - `Flow`, `Hotspot(h, t)`, `Regulation(id, control_volume, time_window, rate_template, member_flows)`
  - `Plan(regulations: List[Regulation])` with canonicalization for hashing/transpositions
  - `State`: partial `Plan` + cheap summaries (residuals, fairness)
- **Evaluator**
  - `NetworkEvaluator`: fast, incremental; computes `J(plan)` and residuals `z_{h,t}`
- **Rate Finder**
  - `rate_finder`: fast local optimizer for regulation rates given selected flows
- **Search**
  - `MCTS`: PUCT with priors, progressive widening, potential shaping
- **Learning**
  - `Priors (g_θ)`, optional leaf value `v_ψ`, replay buffer
- **Caches**
  - Transposition table by canonical plan
  - Delta-eval cache by `(plan_key, op_key)` and rate-finder memo
- **I/O and config**
  - Scenario loader, config, logging/telemetry, deterministic seeding

### 2) State, actions, phases (complete action set)

- **Phases**
  - 0: `ChooseRegulationMode` (CreateNew, AdjustExisting, STOP)
  - 1: `SelectHotspot` (when CreateNew)
  - 2: `SelectFlows` (AddFlow(f), RemoveFlow(f), Continue, Cancel)
  - 3: `ConfigureTemplate` (PickRateTemplate, PickTimeWindow if relevant)
  - 4: `CommitRegulation` (run `rate_finder`, apply to plan, observe ΔJ)
  - 5: `AdjustRegulation` (SelectRegulation(id), then go to 2 or 3 to modify)
- **STOP**
  - Ends episode; shaped reward only; final value from the same evaluator.
- **Terminal conditions**
  - STOP; or `N_MAX_COMMITS` reached; or `N_MAX_REVISIONS` reached.
- **CheapTransition(s, a)**
  - Applies symbolic changes without full evaluation except at commits.
  - Maintains incremental residual estimates for shaping and priors.

### 3) Objective, rewards, shaping

- **Base reward**
  - Commit only: `r_base = −ΔJ = −(J(plan') − J(plan))`
  - Non-commit: `r_base = 0`
- **Potential shaping**
  - `ϕ(s) = −θ1 Σ z_{h,t} − θ2 Σ z_{h,t}^2 − θ3 FairnessPenalty(s)`
  - `r′ = r_base + γ ϕ(s′) − ϕ(s)` with `γ = 1`
  - Fit `(θ1, θ2, θ3)` online via robust regression from `(ΔΣz, ΔΣz^2, ΔFairness) → ΔJ`
  - Clip `ϕ` and `Δϕ`

### 4) Fast rate_finder (single regulation per commit)

- **Goal**
  - Given: control volume, time window, selected flows, rate template
  - Find rate(s) minimizing `J(plan ⊕ regulation)` quickly
- **Key ideas**
  - Coarse-to-fine, deterministic local search with micro-annealing
  - Progressive candidate generation with surrogate ranking
  - Incremental evaluator: reuse residual deltas; avoid recomputing full network
- **Inputs/Outputs**
  - Input: `(selected_flows, control_volume, time_window, template, evaluator, cache)`
  - Output: `best_rates` (per template bins), `ΔJ`, diagnostics
- **Candidate generation**
  - Discrete ladder per bin, e.g. rates in {6, 8, 10, 12, 15, 18, 20} (configurable)
  - Start from heuristic seed:
    - Estimated needed reduction ≈ `Σ z_{h,t} / window_length`
    - Round to nearest ladder point, respect min/max and monotonicity if required
- **Search loop**
  - Stage A: Neighborhood screening (K-best)
    - Enumerate or sample top-N 1- and 2-bin moves around seed
    - Score with surrogate `ŝ = α1 ΣΔz + α2 ΣΔz^2 + α3 fairness`, coefficients tied to `θ`
    - Keep top-K; evaluate exactly with incremental evaluator
  - Stage B: Local refinement
    - Coordinate descent on bins; optional 1D bracketed search if continuous
    - Early stop when ΔJ improvements < ε for L consecutive passes
  - Stage C: Micro-annealing (optional)
    - Small random tweaks with cooling schedule; accept if improves ΔJ or with small prob
- **Complexity**
  - Keep evaluations per commit small: 50–200 evaluator calls
  - Parallelize candidate evaluation (thread pool) when safe
- **Caching**
  - Memoize `(plan_key, regulation_signature, rate_vector) → (ΔJ, ΔΣz, ...)`
  - Keep last-M entries; reuse across MCTS rollouts

- **API**

```python
def rate_finder(plan, control_volume, time_window, flows, template, evaluator, cache, cfg) -> tuple[RateVector, float, dict]:
    # returns (best_rates, delta_J, info)
```

### 5) Cheap residuals and fairness for shaping/priors

- Maintain per-flow contribution histograms to `(h, t)` from precomputed trajectories.
- On add/remove/retime/rate-template changes, update a lightweight residual estimate:
  - `ẑ'_{h,t} = max(0, ẑ_{h,t} + Δdemand_{h,t})` with cached `Δdemand` tensors per action.
- Fairness penalty options (choose one to start):
  - Gini of per-flow delays, or max-min discrepancy across airports/superflows.
- Keep a calibration thread that periodically regresses `(ΔΣz, ΔΣz^2, ΔFairness) → ΔJ`.

### 6) Priors `g_θ` and optional leaf value `v_ψ`

- **V1 (no learning)**: Handcrafted priors from features (see `docs/flow_features_extractor.md`):
  - Hotspot severity, overlap of candidate flows with hotspot, sensitivity of `ẑ`
  - Rank actions; convert to softmax with temperature τ
- **V2 (learned priors)**
  - Replay buffer stores `(x(s,a), y)` with `y = −ΔJ_true + β Δϕ`
  - Train pairwise ranking or L2 regression with regularization, early stopping
- **Leaf value (optional)**
  - `v_ψ(s)` from global state features; bootstrap from deeper nodes as data appears

### 7) MCTS details (single-fidelity)

- **PUCT**
  - `U(s,a) = Q + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))`
  - Typical: `c_puct ≈ 1.5–3.0`
- **Progressive widening**
  - `m(s) = k0 + k1 * N(s)^α` with `α ∈ [0.5, 0.8]`
- **Depth**
  - Limit by commits `D` (e.g., 2–5)
- **Backup**
  - Store shaped returns; `γ=1`; leaf bootstrap with `v_ψ` or `−ϕ(s)`
- **Default policy**
  - On STOP at root, finalize plan and evaluate once with the same evaluator.

- **Pseudocode (single-fidelity)**

```text
function MCTS(s0, B, D)
  ϕ0 ← Φ(s0)
  root ← get_or_create_node(s0)
  for b in 1..B:
    path ← []; s ← s0; dcommits ← 0; G ← 0; ϕprev ← Φ(s)
    while True:
      node ← get_or_create_node(s)
      a ← (expand by prior if allowed) else argmax U(node, ·)
      path.append((node, a))
      (s', isCommit, isTerminal) ← CheapTransition(s, a)
      ϕnext ← Φ(s')
      if isCommit:
        # Run single-fidelity evaluator inside CommitRegulation
        ΔJ ← evaluator.deltaJ_commit(s, a)  # includes rate_finder() call
        rbase ← −ΔJ
        dcommits ← dcommits + 1
      else:
        rbase ← 0
      rstep ← rbase + ϕnext − ϕprev
      G ← G + rstep
      ϕprev ← ϕnext
      s ← s'
      if isTerminal or dcommits ≥ D: break
    if not isTerminal:
      Vleaf ← v_ψ(s) if available else −ϕprev
      G ← G + Vleaf
    backup(G, path)
  return argmax_a root.child[a].N
```

### 8) Interfaces and contracts

```python
class NetworkEvaluator:
    def evaluate(self, plan) -> dict:  # {J, z, fairness, components}
        ...
    def deltaJ_commit(self, s, a) -> float:  # run rate_finder + eval once
        ...
    def apply_regulation(self, plan, regulation_with_rates) -> Plan:
        ...

def CheapTransition(s, a) -> tuple[State, bool, bool]:
    # Returns (s', isCommit, isTerminal). No full evaluation here.

def Φ(s) -> float:  # potential
    # Uses cheap residual ẑ and fairness estimates.

def x_features(s, a) -> np.ndarray:
    # Uses feature spec from flow_features_extractor.md
```

### 9) Caching and transpositions

- **Plan key**: sorted regulations + sorted flow IDs per regulation (+ template/time window normalization)
- **Delta key**: `(plan_key, op_signature)` where `op_signature` encodes change intent
- **Transposition table**: `{Q, N, P, children_count, widen_limit, last_seen}`
- **Eviction**: LRU for TT and evaluator caches
- **Warm reuse**: Persist caches across replans in a single scenario run

### 10) Performance targets and budgets

- Per commit: ≤ 200 evaluator calls in `rate_finder`
- Per MCTS simulation: ≤ 1 commit on average
- Root: 512–2048 simulations per decision (config-based)
- End-to-end: initial usable plan in ≤ 60 seconds on 8 cores (target)

- **Parallelism**
  - Parallel candidate evals in `rate_finder`
  - Batch Φ(s) computations
  - Optional: multiple simulations in parallel with virtual loss

### 11) Testing and benchmarking

- **Unit**
  - Evaluator incremental vs full consistency
  - Φ(s) monotonicity wrt residuals; clipping behavior
  - Canonicalization/hashing invariants for plans
- **Rate_finder**
  - Known synthetic cases with analytic minima
  - Regression tests on ΔJ with frozen seeds
- **Integration**
  - End-to-end small scenarios; reproducible with fixed seeds
  - Compare against `automatic_rate_adjustment.py` solutions on time/quality
- **Metrics**
  - J, Σz, Σz^2, fairness; runtime breakdown; cache hit rates; sim/sec
- **Ablations**
  - Priors on/off; micro-annealing on/off; widening exponents; shaping weights

### 12) Milestones

- **M0 (Days 1–3)**: Scaffolding, data classes, canonical plan key, config/logging
- **M1 (Days 4–7)**: NetworkEvaluator incremental deltas; Φ(s) with cheap residuals
- **M2 (Days 8–12)**: `rate_finder` v1 (seed + neighborhood + refine + cache)
- **M3 (Days 13–17)**: MCTS loop with widening; STOP/default policy; deterministic runs
- **M4 (Days 18–22)**: Priors v1 (heuristics), replay buffer plumbing
- **M5 (Days 23–28)**: Benchmarks, tuning, stabilization; documentation and examples

### 13) Risks and mitigations

- **Rate_finder stalls or is slow**
  - Limit eval budget; enforce early stopping; fallback to seed; reduce ladder
- **Shaping misguides search**
  - Clip Φ; recalibrate θ periodically; introduce β schedule on Δϕ target
- **Cache blow-up**
  - Aggressive LRU; hash compaction; cap memo size
- **Fairness penalty knocks optima**
  - Start with small θ3; sweep sensitivity; support opt-out

### 14) Deliverables and Definition of Done

- Executable CLI: `python -m flow_agent.plan --scenario X --budget B`
- JSON plan export with full regulation list and rates
- Reproducible benchmark report vs baseline SA
- Docs: module READMEs + example notebooks

### 15) CLI and config (proposed defaults)

- **Search**: `cpuct=2.0`, `widen_k0=2`, `widen_k1=6`, `widen_alpha=0.65`, `D=3`, `B=1024`
- **Rate finder**: rate ladder per 5-min bin `{6,8,10,12,15,18,20}`, `ε=0.001*J`, `max_refine_passes=3`
- **General**: seeds, logging level, cache sizes, parallel threads


