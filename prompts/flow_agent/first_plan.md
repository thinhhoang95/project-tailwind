# Goal

To implement an agent to perform Air Traffic Flow Management (ATFM) using Monte Carlo Tree Search (MCTS). 

# Context

Dynamic Capacity Balancing (DCB) is the task where you define a set of *regulations* in order to get the *rolling-hour occupancy count* to stay below the *rolling-hour capacity*.

A *regulation* comprises of four components:

- The control traffic volume (sometimes referred to as the reference location).

- The regulation time period (for example: from 10:00 to 11:15). 

- The flow(s) to be targeted, for example: flights departing from London.

- The allowed entry rate.

# Sample workflow

We start with an empty set of regulations. First, we look at the hotspots (each hotspot is a cell - a traffic volume/sector, localized in one or several time bins). Probably we will want to start with the heaviest hotspot (the traffic volumes whose exceedances are largest). Then, we extract the *flows* from the flight list that contributed to the hotspot. For instance, we may see most of the traffic are coming from London or departing for the Canary Islands. We then *select* one or several flows to be targeted with regulations. From there we have two options: either we can impose a blanket cap for all flows included, or we can impose a cap for each flow. Then the whole process restarts perhaps with a new hotspot. 

Occasionally, we may need to revisit the previous regulations and make adjustments, either tuning the rate, or even adjusting the component flows.

# MCTS Framework


## Inputs, outputs, and maintained data

### Inputs
- Network: hotspots `h` with time buckets `t`, capacities `c_{h,t}`
- Flows `f`: trajectories through `(h, t)`; feasible regulation/reroute options
- Cost model: delay/reroute/inequity penalties → `J(·)`
- Baseline plan for normalization `J_baseline`

### Outputs
- Action sequence (plan): regulations, rates, flow assignments
- Trace per step: shaped reward `r_t`, potential `ϕ`, priors/values used by MCTS

### Caches / Tables
- Transposition table keyed by canonical partial plan
- Replay buffer of `(x(s,a), Δϕ, ΔJ)` to train the action scorer

---

## State, action, and transpositions

### State `s`
A partial plan, a "game state" + cheap summaries:
1. Chosen regulations and flow memberships
2. The objective components for overload, delay cost 

### Hierarchical action space (example - to be planned)
1. Pick a hotspot to regulate
2. Pick a template/rate (e.g., ground-delay bin or reroute pattern)
3. Assign one or a small group of flows to that regulation
4. Run a quick rate finder (it is a lightweight optimizer, fast, but not instantly fast). "Committing" a regulation implies running this rate finder and evaluate the network.
5. Optional merge/split or “resolve jointly” actions
6. A STOP action that can be called at any time.

The "game" ends when the agent calls "STOP" or the number of commits reach `N_MAX_COMMITS`, or regulation revision count hits `N_MAX_REVISIONS`.

### Transpositions
- Key: `key(s) = (sorted regulations, sorted memberships)` so different orders leading to the same partial plan share value estimates

---

## Reward appearance, shaping, and planning details

### 4.1 Termination, horizon, and rewards
- “True” objective appears only at commit steps (when a regulation is finalized)
- Non-commit choices get base reward 0 but receive potential-based shaping
- A STOP action finalizes the plan; default completion may run after

Base objective and immediate rewards:
- At commit: `r = −ΔJ`
- More explicitly:
  - `r_base_k = −( J(plan_k) − J(plan_{k−1}) )` at commit steps
  - `r_base_k = 0` at non-commit steps

Potential-based shaping (choose discount `γ`, typically `γ=1`):
- Shaped reward: `r′(s,a,s′) = r_base(s,a,s′) + γ ϕ(s′) − ϕ(s)`
- With `γ=1` and `ϕ(terminal)=0`:
  - `Σ_t r′_t = −( J(final) − J(baseline) ) − ϕ(s0)` (optimal plan unchanged; dense signal at all steps)

Termination and horizon:
- Each MCTS call uses a depth limit `D` measured in commits (e.g., 2–5)
- STOP contributes only shaping (base reward 0) unless a final high-fidelity evaluation calibrates the root value
- Execute the best root action (typically highest visit count), run high-fidelity evaluator for `ΔJ`, store experience, and replan

### 4.2 Designing a good potential `ϕ`
Define residual demand after a partial plan:
- `z_{h,t} = max(0, demand_{h,t} − c_{h,t})`

Convenient form:
- `ϕ(s) = −θ1 Σ_{h,t} z_{h,t} − θ2 Σ_{h,t} z_{h,t}^2 − θ3 FairnessPenalty(s)`

Fit coefficients online via robust regression:
- `ΔJ ≈ θ1 ΔΣ z_{h,t} + θ2 ΔΣ z_{h,t}^2 + θ3 ΔFairness`

Clip `ϕ` and `Δϕ` to stabilize learning and backup.

### 4.3 PUCT with learned action priors and progressive widening

Action prior:
- For each candidate action `a` in state `s`, compute features `x(s,a)`
- Score with `g_θ(x)`, then:
  - `P(a | s) = exp(g_θ(x(s,a))/τ) / Σ_{a′} exp(g_θ(x(s,a′))/τ)`

Selection rule (PUCT):
- `U(s,a) = Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))`
- `Q(s,a)` stores mean shaped returns `r′`

Progressive widening:
- Allow at most `m(s) = k0 + k1 * N(s)^α` children, `α ∈ [0.5, 0.8]`
- When `|C(s)| < m(s)`, add new child sampled from prior-ranked list

### 4.5 Training the action scorer `g_θ` (priors) and optional leaf value `v_ψ`

Replay buffer and targets (after a commit `a` in `s` with high-fidelity `ΔJ_true` and observed `Δϕ`):
- Store `(x(s,a), y)`, where `y ∈ { −ΔJ_true − median_peer,  −ΔJ_true + β Δϕ }`
- Train `g_θ` with pairwise hinge/logistic loss (robust to scale) or L2 regression with early stopping and L2 regularization

Optional leaf value `v_ψ(s) ≈ E[future shaped return | s]`:
- Use only state features (global summaries, histograms of `z_{h,t}`, etc.)
- Use bootstrapped targets from deeper nodes once some L2 evaluations exist

### 4.6 Full MCTS pseudocode (single planning call)
```
Algorithm 1: MCTS with Shaping, Priors

function MCTS(s0, B, D)
  ϕ0 ← Φ(s0)
  root ← get_or_create_node(s0)
  for b = 1..B do
    path ← []; s ← s0; dcommits ← 0
    G ← 0; ϕprev ← Φ(s)
    while True do
      node ← get_or_create_node(s)
      if node.unexpanded_count() > 0 and node.child_count() < widen_limit(node) then
        a ← sample_new_action_by_prior(node)
      else
        a ← argmax_a' U(node, a')
      end if
      path.append((node, a, ϕprev))
      (s', isCommit, isTerminal) ← CheapTransition(s, a)
      ϕnext ← Φ(s')
      if isCommit then
        dcommits ← dcommits + 1
        
        ΔJ ← HighFidelityDeltaJ(s, a)
        
        rbase ← −ΔJ
      else
        rbase ← 0
      end if
      rstep ← rbase + γ * ϕnext − ϕprev
      G ← G + rstep
      ϕprev ← ϕnext
      s ← s'
      if isTerminal or dcommits ≥ D then
        break
      end if
    end while
    if not isTerminal then
      Vleaf ← v_ψ(s) if available else −ϕprev
      G ← G + γ * Vleaf
    end if
    for (node_i, a_i, ϕ_i) in reverse(path) do
      node_i.N ← node_i.N + 1
      child ← node_i.child[a_i]
      child.N ← child.N + 1
      child.W ← child.W + G
      child.Q ← child.W / child.N
    end for
  end for
  return argmax_a root.child[a].N
end function
```

> The features for model can be found in the find `flow_features_extractor.md` which resulted in the features that deemed useful by mathematical analysis.

# Instructions

The hotspot listing, flow extraction are already available. Have a look at `automatic_rate_adjustment.py` (though that code uses simulated annealing, you may see great caching examples and solution evaluation techniques - which is very fast in evaluation).

Please plan for the followings:

1. The quick `rate_finder` which, given a list of flows (each flow associated with a control volume), the `NetworkEvaluator`, quickly find the "good rates" that minimizes the objective function found by the `NetworkEvaluator`. I am aware that one using Simulated Annealing exists (`automatic_rate_adjustment`), but we need something much, much faster because it is part of a larger Monte Carlo Tree Search process. One idea is to use the same Simulated Annealing loop but with good engineering, less iterations, more rapid cooling. 

2. Propose the complete action set. For example: the game state could be 0 - choosing regulation, 1 - choosing hotspots, 2 - choosing flows,... 5 - choosing regulation to adjust.

    The idea is that this game state will control which action becomes available: if there is no regulations set yet then we go straight to choosing hotspot, when choosing flows, we can continue to add flows, remove previous ones or we "commit" which means that we launch the rate finder and retrieve the objectives. Then the game state goes back to choosing regulation, or we can STOP.

    In choosing regulation, we can choose an existing flow to adjust (which will automatically select the corresponding regulation). We can then either remove the flow, or add a new one. 