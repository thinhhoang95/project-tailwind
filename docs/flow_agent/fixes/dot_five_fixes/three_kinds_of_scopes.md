This document clarifies the variants of objective functions used throughout the `src/parrhesia/flow_agent` code base, drawing from `ABOUT_OBJECTIVEs.md`.

# Three Types of Objectives

### Type 1: The Network Objective (System-wide)

This is the largest scoped objective, and is almost always huge in magnitude compared to other objective functions. It measures exceedances, delays, and other costs across all traffic volumes (TVs) in the network over all timeslots.

-   **Details**: Objective over ALL TVs known to the indexer (global, not scoped to candidates or plan). Scheduling is applied only to flows in the plan’s regulated TVs/windows (if any); other TVs contribute via baseline occupancy.
-   **Where Logged**: Attached to the `run_end` payload as `system_objective` and optionally to `run_start` (both labeled with `system_objective_scope = "all_tvs"`). The runner prints this as “System objective (all TVs)”.
-   **Purpose**: Provides an apples-to-apples system metric across the entire network, regardless of which TVs/windows were considered or regulated.
-   **Code Pointer**: Computed in `src/parrhesia/flow_agent/agent.py` by `_compute_system_objective(...)`.

### Type 2: The Hotspot-Scoped Objectives

This is smaller in magnitude compared to the Network Objective. It calculates the same objective function, but only over a designated set of traffic volumes (typically the hotspots).

> It is extremely important to realize that the scope may change throughout the search trajectory. For example, the set of hotspots at the beginning of an episode could be different from the set at the end. Comparing these objectives requires caution.

There are two main instances of this objective type:

1.  **System Baseline (at `run_start`)**
    -   **What**: Objective over the union of all hotspot candidate TVs/windows discovered at the beginning of the run, with no regulations applied.
    -   **Where Logged**: `run_start` event.
    -   **Keys**: `objective` with `objective_scope = "system_candidates_union"`.
    -   **Purpose**: To establish a broad, system-level baseline for the session’s candidate set.

2.  **Final Plan (at `run_end`)**
    -   **What**: Objective computed from the current plan’s committed regulations. The domain is limited to only those TVs/windows that actually received a regulation.
    -   **Where Logged**: `run_end` event.
    -   **Keys**: `objective` with `objective_scope = "plan_domain"`.
    -   **Purpose**: To select the best inner run at the end of the process and serve as the authoritative cost for the delivered plan.
    -   **Code Pointer**: Produced in `src/parrhesia/flow_agent/agent.py` by `_compute_final_objective(...)`.

### Type 3: The Local Objective (Candidate-Scoped)

This is the objective associated with only one hotspot (or traffic volume). The reward signal for the agent is derived from this objective. It does not consider any downstream or secondary network effects.

-   **Details**: Objective and ΔJ evaluated by `RateFinder` for a single TV/window under consideration. The reward used in MCTS for a commit is `r_base = −ΔJ`.
-   **Where Logged/Shown**: Appears in the live panel titled “Best Candidate (Local)” and is logged via `global_best_update` snapshots.
-   **Keys**: `best_objective_value`, `best_delta_j`, with `best_scope = "local_candidate"`.
-   **Purpose**: Guides MCTS during search. This is the learning signal; it is local by design and faster to compute.

