# General Requirements for Operator Design

Here’s a compact, practical “requirements spec” you can use to design a strong set of ALNS operators. It’s domain-agnostic (VRP/TSP/etc.) with concrete checks you can literally turn into unit tests and dashboards.

# 1) Operator interface (contract)

* **Inputs:** solution $S$, destroy degree $q$ (or target size), RNG/state, time budget, penalty weights, candidate structures.
* **Outputs:** a (feasible or penalized) solution $S'$, plus metadata: run time, #items removed/inserted, Δcost, flags (improved/global best/accepted).
* **Pre/Post conditions:** no corruption of shared data; if infeasible is allowed, all violations must be computable and penalizable; terminate within budget; no infinite loops.
* **Determinism toggle:** same seed ⇒ same result (for reproducibility testing).

# 2) Portfolio composition (coverage & complementarity)

* **Diversity:** include both *diversification* (random/relatedness/segment removals, large $q$) and *intensification* (cheapest/regret insertions, local search).
* **Move scales:** small, medium, and large neighborhoods (vary $q$; include single-item tweaks and block/path moves).
* **Topology coverage:** operators that change *where* items go (global) and *how* they connect (local edges/adjacencies).
* **Orthogonality:** each operator should create solutions others rarely produce (low overlap). Verify via pairwise similarity of outcomes.

# 3) Destruction (ruin) operator requirements

* **Controllability:** accepts $q$ (or a policy) and achieves it within ±1 item.
* **Selection logic:** support at least three styles:

  * Random/uniform removal (baseline).
  * Cost/utility/worst-contribution removal (breaks “bad” structure).
  * Relatedness/cluster/segment removal (exploits structure).
* **No triviality:** must not repeatedly remove the same tiny set unless guided by adaptivity.
* **Scalability:** $O(q \log n)$ to $O(q k)$ using candidate sets/indices; avoid full $O(n^2)$ scans.
* **Feasibility awareness:** avoid removals that make the instance unrecoverable (or mark as “risky” so the controller can throttle their frequency).

# 4) Repair (recreate) operator requirements

* **Completeness:** if a feasible completion exists and penalties permit, operator should typically find one (use regret-k and tie-breaking).
* **Lookahead:** provide a regret mechanism (k≥2) or a small beam/RCL to reduce myopia.
* **Incremental evaluation:** cost deltas in $O(1)$–$O(k)$; maintain caches for feasibility checks.
* **Batch capability:** can reinsert blocks/paths efficiently (try both orientations if sequence matters).
* **Constraint handling:** hard constraints enforced inline; soft ones penalized consistently with global penalty model.

# 5) Local improvement hooks (post-repair)

* **Mandatory:** at least one fast local search (e.g., 2-opt/relocate/swap).
* **Occasional deepening:** heavier move (e.g., 3-opt/LK/ejection chains) on a schedule or on promising candidates.
* **Don’t-look bits / candidate sets:** limit neighborhoods to speed up.

# 6) Adaptive layer (the “A” in ALNS)

* **Credit scheme:** reward chosen operators by outcome tier (global best > improving > accepted > rejected). Use recency-weighted averages.
* **Selection:** roulette-wheel by weights; include ε-greedy/softmax options to keep exploring.
* **Stability:** reaction factor ρ tuned to avoid thrashing; minimum selection probability floor.
* **Cooling/acceptance:** SA-style or threshold acceptance defined outside operators; operators must be neutral to acceptance policy.

# 7) Complexity & performance

* **Time budget awareness:** each operator respects per-call caps; supports early-exit when no improving insertion remains.
* **Candidate structures:** k-nearest lists, α-nearness, feasibility indices; precompute once and update lazily.
* **Caching:** reuse deltas; store per-node best positions; invalidate incrementally.

# 8) Robustness & safety

* Handles degenerate cases (tiny $n$, identical costs, disconnected candidates).
* Avoids numerical issues (tie handling, tolerance ε).
* Graceful degradation if no feasible insertion found (fallback: different operator or penalty push).

# 9) Randomization & reproducibility

* Single, centrally managed RNG with streams per operator.
* Seed in/out logged; ability to re-run a trajectory exactly.

# 10) Penalty & feasibility model (global)

* **Unified penalties:** operators read the *same* penalty weights; no hidden operator-specific “fixups”.
* **Adaptive penalties:** optional global controller adjusts weights based on violation rates.
* **Fast feasibility checks:** incremental resource/time-window tests; early reject infeasible placements.

# 14) Portfolio “minimum viable set”

* **Destruction:** Random, Worst-cost, Relatedness/cluster, Segment/block.
* **Repair:** Cheapest insertion, Regret-2 (or 3), Randomized-cheapest (RCL).
* **Local search:** At least one fast (2-opt/relocate) + one periodic heavy.

# 15) Controller policies that tie it together

* **q policy:** sample from $[q_{\min}, q_{\max}]$ (e.g., 5–30% of items) with occasional large shocks.
* **Restart/diversification:** trigger on stall (e.g., no improvement in X iters); bias destruction away from elite structure (path-relinking flavor).
* **Anytime behavior:** maintain incumbent and elite pool; return best-so-far at any moment.

