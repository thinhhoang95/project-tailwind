### Goals
- Maximize throughput on CPU for the expensive parts (commit evaluation and candidate scoring).
- Keep determinism and reproducibility.
- Preserve current user experience (progress UI, logs), with safe, process-aware aggregation.

### Phase 0 — Profiling and baselines
- Measure wall-time shares with the existing timers already present:
  - `mcts.rate_finder.find_rates`, `rate_finder.*`, `agent.mcts.run`, total runtime.
- Establish single-core baselines on the same dataset to quantify speedup.

### Phase 1 — Parallelize inside commit evaluation (intra-commit)
- **Target**: `RateFinder.find_rates` inner loops over candidate rates and flows.
- **Approach**:
  - Add an optional worker pool (process-based by default; thread-based fallback) to evaluate independent candidate schedules concurrently during:
    - per-flow rate sweeps in per-flow mode,
    - global rate sweep in blanket mode,
    - final confirmation evaluation.
  - Use a pool initializer to hydrate immutable, read-only globals (score context, `flights_by_flow`, `capacities_by_tv`, `indexer`-derived constants) to minimize serialization overhead per task.
- **Determinism**: Fix iteration order and reduce with stable argmin; seed not relevant here, as scoring is deterministic.
- **Config**: `RateFinderConfig(num_workers_rf, backend={'process','thread'}, chunk_size, max_tasks_in_flight)`.
- **Expected**: 2-6x speedups for large candidate grids; retains exact search semantics.

### Phase 2 — Root-parallel MCTS (multi-instance search for one commit)
- **Target**: Top-level `MCTSAgent.run` loop calling `mcts.run(state)`.
- **Approach**:
  - Spin up W independent MCTS instances (separate processes), each with:
    - Same `PlanState` snapshot, same `RateFinderConfig`, but different RNG seeds (e.g., base_seed + worker_id).
    - Per-process `RateFinder` (and associated caches).
  - Give each instance a fraction of the budget:
    - Either sims-per-worker = `max_sims / W`, or time cap per worker = `max_time_s / W`, or keep the same max_time per worker for faster overall convergence; decide via small benchmark.
  - Aggregate results in parent:
    - Collect each process’s best commit and pick the best by lowest ΔJ.
    - Merge `action_counts` for reporting.
- **Config**: `MCTSConfig(num_workers_root, root_parallel_mode={'time','sims'})`.
- **UI/logging**:
  - Each worker reports progress payloads through a multiprocessing queue with `worker_id`; the main process aggregates into the Rich progress bar (sum sims_done, max best_delta_j, etc.) and writes a unified log.
  - Worker logs can optionally go to per-worker files; main log references them.
- **Expected**: Near-linear speedup up to memory/CPU limits; zero shared-state complexity.

### Phase 3 — Asynchronous commit evaluation (inter-simulation pipelining)
- **Target**: `MCTS._evaluate_commit` inside `_simulate`.
- **Approach**:
  - Introduce a shared process pool for commit evaluations; on encountering a commit:
    - Submit the evaluation and continue simulations that don’t require its immediate value.
    - Use a temporary optimistic/neutral prior for that edge; on completion, patch edge statistics and re-backup if needed.
  - Maintain a deduplicating future map keyed by the existing `base_key` to prevent identical in-flight evaluations.
- **Tradeoff**: More invasive; improves utilization when commit evaluations dominate and tree search is otherwise idle.
- **Guardrails**: Preserve `commit_eval_limit`; ensure consistency of backups on late-arriving results; carefully measure impact on policy.

### Phase 4 — Multi-regulation orchestration
- **Keep sequential plan application** (regulations change the state).
- Optional speculative mode (advanced): in parallel, run short-horizon MCTS instances focusing on different hotspot candidates to pre-score promising commits; pick one to apply. Only if Phase 2 is insufficient.

### Phase 5 — Data, memory, and process startup
- **Worker initialization**:
  - Avoid passing heavy objects; build `NetworkEvaluator`, `FlightList`, `TVTWIndexer`, and `RateFinder` inside worker `initializer` using the artifact paths already located in `run_agent.py`.
- **Memory reuse**:
  - Consider memory-mapped arrays for large read-only artifacts if serialization costs are notable.
- **Start method**:
  - Use `'spawn'` for portability on macOS; gate via config.

### Phase 6 — Caching strategy across processes
- Start with per-process caches (simplest).
- Optional shared on-disk caches for:
  - `RateFinder` baseline/context and candidate caches (SQLite or file-backed LRU), keyed by signatures already implemented.
- Only add if re-computation becomes a bottleneck across workers.

### Phase 7 — Configuration knobs and CLI integration
- Add knobs surfaced in `examples/flow_agent/run_agent.py`:
  - `--num-workers-root`, `--num-workers-rf`, `--parallel-backend`, `--parallel-mode`, `--max-tasks-in-flight`, `--progress-interval`.
- Respect environment overrides for easy benchmarking on different machines.

### Phase 8 — Progress, logging, and failure handling
- **Progress**: Single aggregator in the main process; throttle updates to avoid UI contention.
- **Logging**: Keep main log authoritative; include per-worker summaries and links to worker logs when enabled.
- **Failures/timeouts**:
  - Timebox worker runs; on failure, drop that worker’s contribution and proceed with remaining results.
  - Graceful shutdown of pools on early termination.

### Phase 9 — Testing and benchmarking
- Extend `tests/flow_agent/mcts_benchmarking`:
  - Determinism tests (same seed + 1 worker matches baseline).
  - Correctness (best ΔJ no worse than baseline).
  - Speedup curves vs workers for both Phase 1 and Phase 2.
- Add an example benchmark script to emit a short CSV (workers vs time vs ΔJ).

### Rollout order (safe → advanced)
- Phase 1 (intra-commit) → Phase 2 (root-parallel) → Phase 8 (robustness) → Phase 9 (benchmarks) → Phase 3 (async commit) if needed.

- Use modest defaults (e.g., root workers = min(4, cpu_count//2); rf workers = 0/off by default). Enable via flags to compare cleanly with baseline.

- Keep seeds reproducible: base seed from `MCTSConfig.seed`; derive worker seeds deterministically (e.g., `seed + 10_000 * worker_id`).

- Resource caps: set pools to at most `os.cpu_count()` total cores; if both root and RF parallelism are on, auto-scale RF workers per process to avoid oversubscription.

- UI: maintain current Rich progress display; in root-parallel mode, show aggregate sims, best ΔJ, and per-worker ETA in a compact second table.

- Deliverables
  - Config/API changes to `RateFinderConfig` and `MCTSConfig` for worker counts.
  - Orchestrator in `run_agent.py` to run root-parallel mode and aggregate results.
  - Optional shared utilities: process pool manager, progress aggregator, safe logger wrapper.

- Key risks
  - Oversubscription causing slowdown: mitigate with auto-scaling.
  - Added complexity in async commit updates: feature-flag behind an experimental mode.
  - Serialization overhead: address with per-process initialization and limited task payloads.

- Validation
  - Wall-clock speedup on your current dataset.
  - No regression in final objective vs baseline.
  - Stable behavior across macOS and Linux.

- If you want, I can now translate this into concrete edits behind flags in `run_agent.py`, `MCTSAgent`, `MCTS`, and `RateFinder`, starting with Phase 1 and Phase 2.

- We’ll start with root-parallel (fastest ROI) and then add RF intra-commit parallelism to compound gains.

- Benchmarks and toggles will make it easy to measure and tune.

- This plan keeps semantics unchanged until you enable the flags; advanced async commit mode is optional.

- Minimal impact on existing tests; we’ll add separate tests for parallel modes.

- Clear migration path: all changes are opt-in.