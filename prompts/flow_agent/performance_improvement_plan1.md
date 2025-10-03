## Performance Improvement Plan: RateFinder.find_rates and MCTS

### Goal and SLA
- **Target**: Reduce `RateFinder.find_rates` from ~25s to **< 3s** per call on the benchmark in `tests/flow_agent/mcts_benchmarking/test_mcts_1.py`.
- **Constraints**: Preserve functional correctness vs `parrhesia.optim.objective` scoring. Keep MCTS behavior semantically unchanged except for speed.

### Current Flow and Hotspots
The core runtime comes from occupancy recomputation during scoring.

- `src/parrhesia/flow_agent/rate_finder.py`
  - `find_rates` → `_ensure_context_and_baseline` → `build_score_context`
    - builds `d_by_flow`, `beta_gamma_by_flow`, and baseline occupancy caches via `compute_occupancy`
  - For each candidate (rate grid × flows): `_evaluate_candidate` → `score_with_context(...)`
    - Calls `assign_delays_flowful_preparsed` (fast)
    - Calls `compute_occupancy` for the control TV and scheduled flights (slow)
    - Computes objective components (fast, vectorized)
  - `_compute_entrants` scans all flights via `FlightList.iter_hotspot_crossings` and filters by flows (wasteful)

- `src/parrhesia/optim/objective.py`
  - `score_with_context(...)` combines occupancy as:
    - `occ_by_tv[tv] = base_all - base_sched_zero + sched_cur`
    - where `sched_cur` is computed by an expensive `compute_occupancy` pass for scheduled flights only.

### Big Wins (Overview)
1) **Fast scorer for control TV**: Replace per-candidate `compute_occupancy` with `sched_sum = sum_f n_f(t)` and use the identity:
   - `occ_by_tv[control_tv] = base_all - base_sched_zero + sched_sum`.
   - This is equivalent to the reference scorer for the control TV (proof below), but avoids footprint traversal.
2) **Flow‑restricted entrants**: Only scan occupancy intervals for selected flights when computing entrants to the control TV and active windows.
3) **Cache tuning**: Reuse/base occupancy by `(plan_key, control_tv)` across contexts; narrow `tv_filter`.
4) **Budget tuning**: Reduce search passes and candidate grid size without quality loss; cap eval calls.
5) **I/O silencing and micro‑opts**.

---

## Phase 1 — Fast Scorer + Entrants Restriction (Primary speedup)

### A. Add a fast scorer that takes precomputed occupancy
- File: `src/parrhesia/optim/objective.py`
- New function (adjacent to `score_with_context`): `score_with_context_precomputed_occ(...)`.

Responsibilities:
- Input: `n_f_t`, `flights_by_flow`, `capacities_by_tv`, `flight_list`, `context: ScoreContext`, and `occ_by_tv` (already assembled).
- Steps (same as `score_with_context` except it skips `compute_occupancy`):
  - Normalize `n_f_t` → `n_by_flow`.
  - Compute delays via `assign_delays_flowful_preparsed(context.flights_sorted_by_flow, n_by_flow, indexer)`.
  - Compute `J_cap` via `_compute_J_cap(occ_by_tv, capacities_by_tv, context.alpha_by_tv, K)`.
  - Compute `J_delay`, `J_reg`, `J_tv`, optional `J_share`, `J_spill`.
  - Return `(J_total, components, artifacts)`; include `delays_min`, `realised_start`, `occupancy` (use provided `occ_by_tv`).

Correctness rationale:
- `score_with_context` computes `occ_by_tv[tv] = base_all - base_sched_zero + sched_cur` (see lines 283–296 in `objective.py`).
- For the control TV in `RateFinder`, `sched_cur[t]` equals the aggregate schedule `S[t] = sum_f n_f(t)` since each scheduled flight contributes exactly one count in its realized bin at the controlled volume.
- Therefore `occ_fast = base_all - base_sched_zero + S == occ_ref`.

Optional debug assertion (dev-only):
```python
occ_slow = compute_occupancy(flight_list, delays_min, indexer, tv_filter=[control_tv], flight_filter=context.sched_fids)
S = np.sum([_to_len_T_plus_1_array(n_by_flow[f], T)[:T] for f in n_by_flow], axis=0)
occ_fast_tv = context.base_occ_all_by_tv[control_tv] - context.base_occ_sched_zero_by_tv[control_tv] + S
assert np.array_equal(occ_fast_tv.astype(np.int64), occ_slow[control_tv].astype(np.int64))
```

### B. Use the fast scorer in RateFinder candidate evaluation
- File: `src/parrhesia/flow_agent/rate_finder.py`

Changes:
- In `_evaluate_candidate(...)` replace:
  - `compute_occupancy` + `score_with_context` with:
    1) Build `sched_sum` from the candidate `schedule`:
       - `T = indexer.num_time_bins`
       - `sched_sum = np.zeros(T, dtype=np.int64)`
       - For each flow `f` in `schedule`, `sched_sum += np.asarray(schedule[f], int)[:T]`.
    2) Build `occ_by_tv` for the single control TV:
       - `base_all = context.base_occ_all_by_tv[control_tv]`
       - `base_zero = context.base_occ_sched_zero_by_tv[control_tv]`
       - `occ_by_tv = {control_tv: base_all - base_zero + sched_sum}`
    3) Call `score_with_context_precomputed_occ(..., occ_by_tv=occ_by_tv)`.

Notes:
- Keep all other inputs (weights, `beta_gamma_by_flow`, `d_by_flow`, `capacities_by_tv`) identical.
- Artifacts should mirror `score_with_context` except for `occupancy`, which we set to `occ_by_tv`.

### C. Restrict entrant computation to selected flights only
- File: `src/parrhesia/flow_agent/rate_finder.py`

Replace the global `FlightList.iter_hotspot_crossings([...])` scan with a flow‑restricted pass:
- Build `allowed_fids = ⋃ flows.values()`.
- Decode intervals arithmetically to avoid indexer lookups per record:
  - `T = indexer.num_time_bins`
  - `target_row = indexer.tv_id_to_idx[control_tv]`
  - For each interval of flights in `allowed_fids`:
    - `row = tvtw_idx // T`, `bin = tvtw_idx - row * T`
    - Keep if `row == target_row` and `bin ∈ active_windows`.
    - `entry_dt = takeoff_time + entry_time_s` (only for kept)
- Return entrants per flow.

Caching:
- Augment entrants cache key with the set (or stable hash) of `allowed_fids` to reuse across candidate passes within the same `find_rates` call.

Expected impact:
- Eliminates per‑candidate occupancy recomputation (dominant cost). Candidate evaluation largely becomes vector math + FIFO delays.
- Entrants scan cost scales with selected flows, not all flights.

---

## Phase 2 — Reuse and Narrowing of Context

### A. Narrow `tv_filter` strictly to the control TV
- Already in `RateFinder._ensure_context_and_baseline` pass `tv_filter=[control_volume_id]` to `build_score_context`.
- This ensures base occupancy caches (`base_occ_all_by_tv`, `base_occ_sched_zero_by_tv`) are built only for the single TV.

### B. Cache base occupancy across flow subsets
- Observation: `base_occ_all_by_tv[control_tv]` does not depend on flow IDs.
- Add an LRU cache in `RateFinder` keyed by `(plan_key, control_tv)` storing `base_occ_all_by_tv[control_tv]` and `base_occ_sched_zero_by_tv[control_tv]`.
- When building new contexts for different `flow_ids` but the same `(plan_key, control_tv)`, reuse these base arrays rather than recomputing.

Implementation sketch:
- Extend `_score_context_cache` entry to optionally reference arrays from the per‑TV cache; or inject them after `build_score_context` returns if lengths match.
- Respect existing `cache_size` and LRU trimming.

Expected impact:
- Saves ~100–500ms per new flow subset context, especially when MCTS tries different subsets for the same hotspot.

---

## Phase 3 — Budget/Heuristic Tuning

### A. Reduce passes with adaptive grid
- Set `RateFinderConfig.passes = 1` when `use_adaptive_grid=True`.
- Rationale: adaptive anchors typically locate a good rate in one sweep.

### B. Limit grid breadth
- Set `max_adaptive_candidates = 8` (finite portion; infinity is always included).
- Merge with configured `rate_grid` anchors as implemented.

### C. Cap eval calls per commit
- Guard: `eval_calls >= min(config.max_eval_calls, len(flow_ids) * (len(rate_grid) + 3))` then early stop.

### D. Silence debug prints
- Behind a config flag or logger level, suppress:
  - `[RateFinder] candidate rates ...` per flow
  - Blanket mode prints

Expected impact:
- Avoids unnecessary exploration and I/O overhead, ensuring sub‑3s target even on larger flow sets.

---

## Phase 4 — Micro‑Optimizations

### A. `assign_delays_flowful_preparsed`
- Already efficient; ensure minimal overhead by avoiding extra copying; reuse local variables.

### B. Occupancy numba threshold
- `parrhesia/optim/occupancy.py` switches to numba kernel for ≥100k contributions; keep as is.

### C. Typing/array dtypes
- Keep `int64` for occupancy vectors and schedules to match scorer expectations; avoid hidden casts.

---

## Correctness: Equivalence to Reference Scorer

Given:
- Reference combines occupancy as `occ_ref = base_all - base_sched_zero + sched_cur`.
- On the control TV, `sched_cur[t] == sum_f n_f(t)` (each scheduled flight contributes one to its realized bin at the control volume).

Therefore the fast path:
- `occ_fast = base_all - base_sched_zero + sum_f n_f(t) == occ_ref`.
- Other components (J_delay via FIFO, J_reg/J_tv vs `d_f(t)`, optional terms) are identical.

Optional dev-only verification is outlined in Phase 1A (assert array equality for the control TV).

---

## Test & Benchmark Plan

### A. Unit parity tests
- File: `tests/flow_agent/test_rate_finder.py`
  - For several seeds and random small flow sets:
    - Evaluate a fixed candidate with original scorer and with fast scorer; assert identical objective and components for control TV.
    - Enable the debug equality check in one test case.

### B. Integration
- `tests/flow_agent/mcts_benchmarking/test_mcts_1.py` should pass unchanged.
- Confirm `agent.run()` produces ≥1 commit and a finite final objective.

### C. Performance
- Use the `timed(...)` reporters already present in the test to capture:
  - `rate_finder.evaluate_candidate` per call
  - `rate_finder.score_with_context.candidate` vs fast scorer path
  - Total `agent.run`
- Acceptance: median `find_rates` wall time < 3.0s on the benchmark machine.

---

## Rollback & Flags
- Add a runtime toggle to disable the fast scorer (e.g., env `RATE_FINDER_FAST_SCORER=0` or config flag) that reverts to the original `score_with_context` path for diagnostics.
- Keep verbose timing under a debug flag; no prints in hot path by default.

---

## Risks & Mitigations
- Risk: Multiple entries per flight at the control TV within the window could overcount when aggregating `sum_f n_f(t)`.
  - Mitigation: Build entrants per flight with a single spec at the control TV (earliest bin per active window) so schedules count flights, not intervals.
- Risk: Context cache inconsistencies across flows.
  - Mitigation: Key caches by `(plan_key, control_tv, T, bin_minutes)` and enforce LRU trimming.
- Risk: Silent mismatches.
  - Mitigation: Temporary assertion (Phase 1A) behind a debug flag; parity tests.

---

## Work Items (Implementation Checklist)

1) `objective.py`
   - [ ] Add `score_with_context_precomputed_occ(...)` (fast scorer).

2) `rate_finder.py`
   - [ ] Replace candidate evaluation with fast scorer using `sched_sum` composition.
   - [ ] Restrict `_compute_entrants` to selected flights; arithmetic decode per interval.
   - [ ] Silence prints unless debug.
   - [ ] Narrow `tv_filter` to the control TV (verify already in place).
   - [ ] Add optional LRU cache for `base_occ_*` by `(plan_key, control_tv)`.

3) Config tuning
   - [ ] Set `RateFinderConfig(passes=1, max_adaptive_candidates=8)` defaults for MCTS agent.
   - [ ] Add eval call cap guard.

4) Tests/benchmarks
   - [ ] Parity unit tests for fast scorer.
   - [ ] Rerun `test_mcts_1.py` and record timing; target < 3s per `find_rates`.

---

## Acceptance Criteria
- Functional parity: objectives and components match the reference scorer for the control TV within floating‑point equality for tested cases.
- Performance: `find_rates` median wall time < 3s on the provided benchmark; total `agent.run` reduced accordingly.
- No regressions in existing test suite.


