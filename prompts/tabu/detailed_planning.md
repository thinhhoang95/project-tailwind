### Goal
Design and implement a Tabu search–based planner that produces a compact network plan (set of regulations) to reduce overloads while minimizing delay cost, using a reimplemented FCFS queue and incremental network evaluation.

### Scope and high-level architecture
- **Core data**: reuse `FlightList`/`DeltaFlightList` and `NetworkEvaluator` for occupancy and overload metrics; reuse `Regulation`/`NetworkPlan`/`RegulationParser` for plan representation and flight selection; reuse `TVTWIndexer` for index mapping.
- **New modules**:
  - `optimize/fcfs/` for a clean FCFS scheduler (do not rely on CASA).
  - `optimize/tabu/` for the Tabu engine plus solution representation and neighborhood moves.
  - `optimize/features/` for heuristics/scoring (multiplicity, footprint similarity, slack valleys).
  - `optimize/eval/plan_evaluator.py` to apply plans incrementally and compute the objective.
- **Integration points**:
  - Replace `run_readapted_casa` in `NetworkPlanMove` with the new FCFS scheduler.
  - Drive `NetworkEvaluator` off `DeltaFlightList` to avoid rebuilding matrices.
  - Maintain per-regulation and per-flight delay caches to accelerate neighborhood evaluations.

### Detailed implementation plan

#### 1) FCFS reimplementation (no CASA dependency)
- Create `src/project_tailwind/optimize/fcfs/scheduler.py`:
  - Inputs: `flight_list: FlightList`, `identifier_list: List[str]`, `reference_location: str`, `tvtw_indexer: TVTWIndexer`, `hourly_rate: int`, `active_time_windows: List[int]`.
  - Output: `Dict[str, int]` mapping `flight_id -> delay_minutes`.
  - Steps:
    - For each `flight_id` in `identifier_list`:
      - From `flight_list.flight_metadata[flight_id]['occupancy_intervals']`, find the earliest interval whose `tvtw_index` decodes to `(reference_location, time_bin)` with `time_bin in active_time_windows`. If none, skip flight.
      - Compute scheduled crossing timestamp: `takeoff_time + entry_time_s`.
    - Build entrant list `[(flight_id, scheduled_dt)]`, sorted by `scheduled_dt`.
    - FCFS continuous server model:
      - Service time per flight `s = 3600 / hourly_rate` seconds.
      - Maintain `server_available_at = -inf`.
      - For each entrant in order:
        - `service_start = max(scheduled_dt, server_available_at)`.
        - `delay_s = (service_start - scheduled_dt).total_seconds()`.
        - `server_available_at = service_start + s`.
      - Record `delay_minutes = ceil(delay_s / 60)` or round as specified.
    - Return flight->delay mapping.
  - Edge cases:
    - Flights with multiple entries to `reference_location`: use first match in `active_time_windows`.
    - Maintain determinism ties by `(scheduled_dt, flight_id)`.
    - If `hourly_rate <= 0`, treat as infinite delay or skip regulation with warning.
- Unit tests (new `tests/test_fcfs.py`):
  - Deterministic order, no-overload case, overload cascades across multiple hours, partial active windows (flights outside active windows unaffected).

#### 2) Regulation application and incremental evaluation
- Update `src/project_tailwind/optimize/moves/network_plan_move.py`:
  - Replace the call to `run_readapted_casa` with `optimize.fcfs.scheduler.assign_delays(...)`.
  - Keep the existing behavior of aggregating per-regulation delays by taking the highest delay per flight across all regulations.
  - Return a `DeltaFlightList` overlay with those delays, not a deep copy of `FlightList`.
- Create `src/project_tailwind/optimize/eval/plan_evaluator.py`:
  - Responsibilities:
    - Given a `NetworkPlan`, produce combined per-flight delays by applying FCFS per regulation and aggregating via max-per-flight.
    - Construct a `DeltaFlightList` with the delay overlay (leveraging `DeltaFlightList` already in repo).
    - Build a `NetworkEvaluator` from traffic volumes GDF and the `DeltaFlightList`.
    - Compute overload metrics using `NetworkEvaluator.compute_excess_traffic_vector()` and `compute_delay_stats()`.
    - Objective aggregation (see next section).
  - Caching:
    - Cache FCFS results per regulation fingerprint `{location, rate, sorted(active_time_windows), sorted(flights)}` with a stable hash.
    - Cache per-flight max delay overlay across the plan for rapid delta updates when a single regulation changes.

#### 3) Objective function
- Single scalar for Tabu search, tunable via config:
  - Let `excess = evaluator.compute_excess_traffic_vector()` and `delay_stats = evaluator.compute_delay_stats()`.
  - Propose default:
    - `z_sum = excess.sum()`
    - `z_max = excess.max()`
    - `delay_min = delay_stats['total_delay_seconds'] / 60`
    - `num_regs = len(plan.regulations)`
  - Objective: `alpha * z_sum + beta * z_max + gamma * delay_min + delta * num_regs`
  - Defaults: `alpha=1.0`, `beta=2.0`, `gamma=0.1`, `delta=25.0` (penalize fragmentation). Tune later.
  - Constraints/guards:
    - Hard cap on `num_regs` (config), or soft via high `delta`.
    - Optional lexicographic fallback (if two objectives equal within tolerance, prefer fewer regs, then less delay).

#### 4) Heuristic feature computation to guide candidate selection
- Create `src/project_tailwind/optimize/features/flight_features.py`:
  - Inputs: `FlightList`, `NetworkEvaluator`.
  - Outputs, precomputed per flight:
    - **Multiplicity score**: count of overloaded TVTWs (or overloaded TVs by hour) the flight intersects. Use baseline initially; optionally refresh every N iterations using current solution.
    - **Footprint set**: set of traffic volumes visited by the flight; store as bitsets or Python `set[str]`.
    - **Footprint similarity**: Jaccard similarity to a seed set; compute on demand for candidate enrichment.
    - **Slack valley score**: for each TV in flight’s footprint, compute slack time series `hourly_capacity - hourly_occupancy`; take 5th percentile and mean around the flight’s scheduled times at that TV; aggregate min/mean to yield a flight-level score.
  - Provide helper: `rank_candidates(seed_set, pool, weights)` to combine multiplicity (high), similarity (high), slack valley (high) with configurable weights.

#### 5) Initial solution construction
- Create `src/project_tailwind/optimize/tabu/initializer.py`:
  - Steps:
    - Compute baseline overloads via `NetworkEvaluator`.
    - Select the top hotspot (by `excess` or `utilization_ratio`) as `reference_location`.
    - Initialize a regulation with `{location=TV, rate=guess, active_time_windows=worst-hour windows}`; start with empty targeted flights.
      - Rate guess: start at the current hourly occupancy minus capacity + ε, capped; or use a small set {10, 20, 30} to test quickly and pick lowest objective.
    - Grow the regulation flight list iteratively:
      - Candidate pool: flights crossing the hotspot during active windows, prioritized by multiplicity; enrich by adding high Jaccard to current set; filter by slack valleys.
      - For each batch add (e.g., add K flights per step), run FCFS for this regulation only, update aggregate delays, evaluate objective via `DeltaFlightList` without full rebuilds.
      - Stop when objective stops improving, or marginal benefit falls below threshold.
    - Optionally add 1–2 more regulations for next hotspots if objective improves and `num_regs` cap not exceeded.

#### 6) Tabu search engine
- Create `src/project_tailwind/optimize/tabu/engine.py`:
  - Solution representation:
    - `TabuSolution`: holds a `NetworkPlan`, aggregated per-flight delays (dict), cached `objective`, and any per-regulation FCFS cache keys.
  - Neighborhood moves (keep branching factor small with beam width):
    - **AddFlight(reg_i, flight_id)**
    - **RemoveFlight(reg_i, flight_id)**
    - **AdjustRate(reg_i, +Δ/-Δ)**
    - **ShiftWindows(reg_i, +/- one bin or re-center)**
    - **AddRegulation(tv_id at overloaded hotspot, seed flights from candidate pool)**
    - **RemoveRegulation(reg_i)**
    - **MergeRegulations(reg_i, reg_j)** (if same `location` or overlapping windows)
  - Move application:
    - Update only the affected regulation’s FCFS result; recompute the aggregated per-flight max delays; refresh `DeltaFlightList` view; recompute objective.
  - Tabu list:
    - Record attributes of reverse moves (e.g., a tuple key: `(op, reg_id, flight_id or rate-change or window-change)`).
    - Tenure: 10–30 iterations (config).
    - Aspiration: allow tabu if it yields best-so-far objective.
  - Intensification & diversification:
    - Intensify: revisit top-2 hotspots; limit candidate pool to top-multiplicity flights; smaller beam width.
    - Diversify: randomize seed hotspot among top-K overloaded; reset candidate pools; allow larger beam width briefly.
  - Stopping criteria:
    - Max iterations or wall-clock.
    - No improvement for G iterations.
    - Objective below threshold (near-feasible).

#### 7) Incremental evaluation mechanics and caches
- Per-regulation FCFS cache keyed by hash of `{location, rate, windows, sorted(flight_ids)}`.
- Aggregated overlay:
  - Combine regulation delays via `per_flight_delay[fid] = max(delays_from_all_regs)`.
  - Wrap base `FlightList` with `DeltaFlightList(per_flight_delay)` to get updated occupancy quickly.
- `NetworkEvaluator` reuse:
  - Instantiate once per evaluation, or reuse with new `DeltaFlightList` (keep consistent with its expectations about original baseline).
  - Read overload vector and delay stats (it already compares to the original baseline via `original_flight_list` snapshot created in `NetworkEvaluator.__init__`).
- Optional: maintain “local delta” of total occupancy by subtracting and adding shifts for only changed flights (advanced optimization if needed; start with `DeltaFlightList`).

#### 8) Configuration and parameters
- Add `src/project_tailwind/optimize/tabu/config.py`:
  - `alpha, beta, gamma, delta` objective weights.
  - Tabu tenure, max iterations, no-improvement patience.
  - Beam width for neighborhoods, candidate pool sizes.
  - Rate step sizes {+/- 5 flights/hour}.
  - Max regulations allowed.
  - Random seed.

#### 9) Developer workflow and tests
- Unit tests:
  - `tests/test_fcfs.py`: correctness under various rates/time windows; stability; edge cases.
  - `tests/test_plan_application.py`: applying a single regulation via `NetworkPlanMove` yields expected per-flight delays and occupancy shifts versus a small handcrafted scenario.
  - `tests/test_tabu_smoke.py`: tiny synthetic dataset; confirm Tabu reduces `z_sum` over iterations; ensure tabu list blocks immediate reversals; aspiration works.
- Integration tests:
  - Repurpose `tests/test_network_eval.py`-style flow to:
    - Load a small sample `FlightList` and traffic volumes.
    - Run initializer and Tabu engine for a small iteration budget.
    - Assert objective improves from baseline; ensure `num_regs` ≤ cap.

#### 10) Observability and debugging
- Add structured logging with a per-iteration record:
  - Iteration, move type, delta objective, `z_sum`, `z_max`, total delay, regs count.
  - Top-5 overloaded TVs/hours snapshot periodically for sanity.
- Add `debug` printing hooks similar to `optimize/debug.py`, controllable via env var.

#### 11) Performance plan
- Target sub-second evaluation per move on moderate datasets by:
  - FCFS on limited flight sets (only those matching a regulation).
  - Per-regulation FCFS result caching.
  - `DeltaFlightList` overlay instead of rebuilding matrices.
  - Candidate pool throttling (top-N multiplicity, top-Jaccard within M).
- If needed, add parallel neighborhood evaluation (threaded per candidate) guarded by GIL-safe sections (most is numpy/pure Python; keep pools small).

#### 12) Risks and mitigations
- Risk: using baseline overloads to compute multiplicity/slack may diverge as plan changes.
  - Mitigation: refresh features every X iterations with current solution and throttle CPU cost.
- Risk: FCFS modeling differences vs legacy CASA.
  - Mitigation: define FCFS spec (continuous server at rate r) and validate with unit tests; keep epsilon handling optional.
- Risk: Large candidate branching.
  - Mitigation: beam search caps and heuristic ranking.

### File and edit plan
- New:
  - `src/project_tailwind/optimize/fcfs/scheduler.py` (FCFS).
  - `src/project_tailwind/optimize/eval/plan_evaluator.py`.
  - `src/project_tailwind/optimize/features/flight_features.py`.
  - `src/project_tailwind/optimize/tabu/engine.py`, `initializer.py`, `config.py`.
  - Tests: `tests/test_fcfs.py`, `tests/test_plan_application.py`, `tests/test_tabu_smoke.py`.
- Edits:
  - `src/project_tailwind/optimize/moves/network_plan_move.py`: replace CASA invocation with FCFS; keep max-per-flight delay aggregation.
  - Optional small helpers in `RegulationParser` if needed to expose “flights crossing reference_location during windows” efficiently.

### Milestones and acceptance criteria
- M1: FCFS implemented and tested; `NetworkPlanMove` uses FCFS for one regulation; unit tests pass.
- M2: `PlanEvaluator` and objective function complete; can evaluate a static `NetworkPlan`.
- M3: Feature computation ready; initializer builds a non-empty plan for a top hotspot and improves baseline objective.
- M4: Tabu engine runs on a toy dataset; objective decreases by ≥20% in `z_sum` with ≤K regulations; tests pass.
- M5: Run on a medium dataset within time budget; log stability and improvement; document configs.

### Runbook (once implemented)
- Load `FlightList` and traffic volumes GDF.
- Build features; run initializer to seed plan.
- Start Tabu engine with configured iteration budget.
- Persist best plan (list of regulation strings), logs, and summary metrics.

- - -

- Swapped to a continuous-rate FCFS model, designed around `FlightList`’s `occupancy_intervals` and `takeoff_time`, with per-regulation caching.
- Planned incremental evaluation via `DeltaFlightList` and `NetworkEvaluator`, with a clear scalar objective and penalties to avoid fragmented plans.
- Defined concrete modules, functions, neighborhood moves, tests, configs, and milestones to implement Tabu search end-to-end without ALNS or CASA.