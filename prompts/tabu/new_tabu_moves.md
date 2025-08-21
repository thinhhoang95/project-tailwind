I'll scan the existing `Regulation` and related classes to align the move definitions with actual fields and data flow. Then I’ll outline detailed implementation steps for each move, including candidate generation, validation, incremental evaluation, caching, and tabu keys.
Read file: /Volumes/CrucialX/project-tailwind/src/project_tailwind/optimize/regulation.py

I'll look at the current Tabu module files to align the planning with how moves are represented and applied today, then outline precise steps for each new move including candidate generation, validity checks, caching, and tabu keys.
Ran tool
Read file: /Volumes/CrucialX/project-tailwind/src/project_tailwind/optimize/tabu/engine.py

I’m going to scan the core classes used by the Tabu engine—`NetworkPlan`, `RegulationParser`, `PlanEvaluator`, `FlightList`, and `FlightFeatures`—so the move plans align with current data flow and caching. Then I’ll detail implementation steps for AddRegulation, RemoveRegulation, ShiftWindow, and ChangeWindowLength.
Read file: /Volumes/CrucialX/project-tailwind/src/project_tailwind/optimize/network_plan.py
Read file: /Volumes/CrucialX/project-tailwind/src/project_tailwind/optimize/parser/regulation_parser.py
Read file: /Volumes/CrucialX/project-tailwind/src/project_tailwind/optimize/eval/plan_evaluator.py
Read file: /Volumes/CrucialX/project-tailwind/src/project_tailwind/optimize/eval/flight_list.py
Read file: /Volumes/CrucialX/project-tailwind/src/project_tailwind/optimize/features/flight_features.py

I reviewed `Regulation`, `NetworkPlan`, `RegulationParser`, `PlanEvaluator`, `FlightFeatures`, and the current Tabu engine. I’ll now lay out a detailed, code-free plan to add AddRegulation, RemoveRegulation, ShiftWindow, and ChangeWindowLength that fits the current implementation and caches.

### Context alignment (what exists today)
- **Regulation**: fields `location`, `rate`, `time_windows`, `filter_type`, `filter_value`, optional `target_flight_ids`.
- **NetworkPlan**: ordered list `regulations` with add/remove by index.
- **RegulationParser.parse(reg)**: returns flight IDs crossing `reg.location` during `reg.time_windows` and matching filter.
- **PlanEvaluator**:
  - Aggregates per-regulation FCFS delays via max-per-flight.
  - FCFS cache key: `(location, rate, sorted(time_windows), sorted(matched_flights))`.
  - Objective includes penalty `delta * num_regs`.
- **TabuEngine**: Implements neighbors for AddFlight, RemoveFlight, AdjustRate; single global beam; simple tabu with reverse-key.
- **FlightFeatures**: Baseline-derived heuristics; provides candidate ranking by multiplicity/similarity/slack. Has internal overloaded hotspot info by (tv_id, hour).

### Design decisions for new moves
- Keep all four moves; unify their ergonomics and avoid overlap:
  - ShiftWindow: translate existing `time_windows` without changing length.
  - ChangeWindowLength: grow/shrink `time_windows` length while keeping center/anchor consistent.
  - AddRegulation and RemoveRegulation: manage plan cardinality.
- Add per-move beam quotas to avoid starvation (fix noted trade-off).
- Use PlanEvaluator’s cache via stable regulation “fingerprints” so most neighbor evals are incremental.
- Keep `target_flight_ids` when windows change; FCFS will naturally skip flights not entering the new windows.

### Common helpers these moves will rely on
- **Fingerprint for a regulation (for tabu keys and dedupe)**:
  - `(location, rate, tuple(sorted(time_windows)), filter_type, filter_value, ("seeded" or "blanket"))`
  - Note: PlanEvaluator caching already hinges on matched flights; tabu keys should include filter fields to avoid ambiguity.
- **Window normalization**:
  - Always store `time_windows` as a sorted list of unique ints within valid bounds from `TVTWIndexer`.
  - Clamp out-of-range windows; drop candidate if empty after clamp.
- **Contiguity policy**:
  - Treat windows as the full sorted set; contiguity is not required by the evaluator, but candidates we create for Shift/Change will start from contiguous blocks to control search branching.
- **Beam allocation**:
  - Reserve shares per move, e.g. `{AddRegulation: q1, RemoveRegulation: q2, ShiftWindow: q3, ChangeWindowLength: q4, AddFlight/RemoveFlight/AdjustRate: rest}` summing to `beam_width`.
  - Interleave picks across move pools until beam is filled.

### Move 1: AddRegulation
- Purpose: introduce a new regulation at a hotspot (new `location` and `time_windows`) with an initial `rate` and optional seed targets.
- Preconditions:
  - `len(plan.regulations) < config.max_regulations`.
- Candidate generation (ranked):
  - **Hotspot selection**:
    - Use `FlightFeatures`’s overloaded-by-hour signal to form candidate `(tv_id, contiguous_hour_block)` pairs.
    - Build hour-blocks by grouping consecutive hours among overloaded ones for each tv_id; score blocks by sum of overload or count of overloaded hours.
    - Keep top `config.addreg_hotspot_blocks_topk`.
  - **Window instantiation**:
    - Convert hour blocks to `time_windows` indices using `TVTWIndexer` mapping (hours → bins; bins per hour known from `FlightFeatures`).
    - Optionally generate variants per block: min-len, mid-len, max-len windows bounded by `config.window_min_len` and `config.window_max_len`.
  - **Rate guess set**:
    - Discrete candidates: `config.addreg_rate_candidates` (e.g., [10, 20, 30]) or estimate from baseline overload: `cap - (occ - ε)` then clamp to positive ints; keep 2–3 candidates per block.
  - **Target seeding strategy** (two modes; choose via config):
    - Blanket: `target_flight_ids=None` (parser selects all flights crossing in those windows).
    - Seeded: pick top `config.addreg_seed_targets_k` flights from the block’s blanket pool ranked by `FlightFeatures.rank_candidates` with a seed footprint from the block’s busiest hour.
  - Create up to `quota(AddRegulation)` candidate regs combining top blocks × rate guesses × seeding mode (prune with a simple product cap).
- Plan assembly and validation:
  - For each candidate reg, create `new_plan = NetworkPlan([...old regs..., new_reg])`.
  - Validate `time_windows` non-empty; dedupe against already-present regs by fingerprint.
- Tabu key and reverse:
  - Key: `("AddRegulation", new_index, fingerprint)` where `new_index = len(plan.regulations)`.
  - Reverse key: `("RemoveRegulation", new_index)`.
  - Additionally, maintain a short-term tabu set by fingerprint to avoid re-adding identical regs immediately after removal.
- Beam share:
  - Use `quota(AddRegulation)` to push diverse locations and window lengths; avoid more than 1–2 per unique tv_id per iteration.
- Interactions:
  - Subsequent iterations can refine via AdjustRate, ShiftWindow, ChangeWindowLength, and Add/RemoveFlight.

### Move 2: RemoveRegulation
- Purpose: reduce plan size and eliminate low-impact or redundant regulations.
- Candidate generation (ranked):
  - For each `reg_i`:
    - Fast impact proxy: use PlanEvaluator’s FCFS cache for this reg’s current fingerprint to get its per-flight delays; estimate contribution as sum or max of those delays and the unique flight count affected during its windows.
    - Also compute a simplicity proxy: `len(time_windows)` and rate (higher rate with larger windows typically contributes more).
    - Rank regs by “remove-score” that prefers removal of low impact and complex/overlapping ones.
  - Propose up to `quota(RemoveRegulation)` candidates by removing the top-ranked-for-removal regs.
- Plan assembly:
  - `new_plan = NetworkPlan([...])` without `reg_i`.
- Tabu key and reverse:
  - Key: `("RemoveRegulation", reg_idx)`.
  - Reverse: `("AddRegulation", reg_idx, fingerprint_of_removed)`; allow aspiration if re-adding improves best objective.
- Beam share:
  - Small but persistent share to avoid bloating the plan under the `delta * num_regs` penalty.

### Move 3: ShiftWindow
- Purpose: translate a regulation’s `time_windows` earlier or later without changing length.
- Step size and domain:
  - Shift by ±`config.window_shift_step` bins; also allow multi-step variants ±2 if beam budget allows.
  - Clamp to valid time-window indices; drop if empty after clamp.
- Candidate generation per regulation:
  - Compute sorted `wins = sorted(reg.time_windows)`.
  - Propose:
    - Left shift: `wins - step` (all entries reduced by step).
    - Right shift: `wins + step`.
  - Optionally restrict to only regs whose current windows intersect overloaded hours to avoid noise.
- Targets handling:
  - Keep `reg.target_flight_ids` as-is; FCFS will skip flights that no longer cross the shifted windows.
- Plan assembly:
  - Replace `reg_i` with shifted windows.
- Tabu key and reverse:
  - Key: `("ShiftWindow", reg_idx, +step)` or `(..., -step)`.
  - Reverse: invert the sign.
- Beam share:
  - Modest share per iteration; shifting is often a cheap improvement before resizing.

### Move 4: ChangeWindowLength
- Purpose: grow or shrink the time coverage of a regulation to better match overload span.
- Resize model:
  - Maintain the block’s “center” (median of current windows) or anchor to the busiest hour within current windows (pick via `config.window_resize_anchor = center | busiest_hour`).
  - Expand or shrink symmetrically by `config.window_resize_step` bins per side when possible; if at bounds, expand on the feasible side only.
  - Enforce `config.window_min_len <= len(wins) <= config.window_max_len`.
- Candidate generation per regulation:
  - Propose at most 3–4 variants:
    - Expand both sides (+step each side)
    - Shrink both sides (-step each side)
    - Expand left only / right only (if asymmetric is allowed)
  - Skip if the operation violates min/max length or goes out of bounds; normalize and dedupe.
- Targets handling:
  - Keep current `target_flight_ids`; FCFS will include only crossings in new windows.
- Plan assembly:
  - Replace `reg_i` with resized windows.
- Tabu key and reverse:
  - Key: `("ChangeWindowLength", reg_idx, delta_left, delta_right)` where deltas are signed bin counts; symmetrical case is `(step, step)` or `(-step, -step)`.
  - Reverse: `("ChangeWindowLength", reg_idx, -delta_left, -delta_right)`.
- Beam share:
  - Similar to ShiftWindow; resizing can precede fine-grained Add/RemoveFlight.

### Integration into the neighbor generator
- Replace the strictly sequential fill with a quota-based interleaving to avoid starvation:
  - For each move type in a fixed round-robin order, generate up to its quota and push into a common pool.
  - If pool > beam width, downselect by quick heuristics (e.g., keep top-N per move by a cheap proxy like estimated overload covered or rate×window_len) before evaluation.
- Maintain dedupe set by candidate-plan fingerprint (sorted list of per-reg fingerprints) to avoid duplicate neighbors.

### Tabu list updates
- Extend `_reverse_key`:
  - Map AddRegulation ↔ RemoveRegulation using indices.
  - Map ShiftWindow and ChangeWindowLength to their inverse deltas.
- Also keep a short-lived tabu set keyed by regulation fingerprint for Add/Remove to reduce immediate re-creation after deletion, even if index changes.

### Config additions
- AddRegulation:
  - `addreg_hotspot_blocks_topk`, `addreg_rate_candidates`, `addreg_seed_targets_k`, `addreg_mode = "blanket"|"seeded"`, `addreg_quota`.
- RemoveRegulation:
  - `removereg_quota`.
- Windows:
  - `window_shift_step` (bins), `window_resize_step` (bins per side), `window_min_len`, `window_max_len`, `window_resize_anchor`, `shift_quota`, `resize_quota`.
- Global:
  - Per-move beam shares; `max_regulations`.

### Validation and safeguards
- Ensure `time_windows` never empty after operations; skip invalid candidates early.
- Clamp rates to positive ints; respect `max_regulations`.
- Use PlanEvaluator’s cache keys to avoid redundant FCFS runs across similar candidates.
- Prefer candidates whose windows intersect overloaded hours; apply light heuristics to prune.

### Optional simplification (if you want to reduce move surface)
- You can unify ShiftWindow and ChangeWindowLength into a single “AdjustWindows” move with two sub-ops: shift and resize. However, keeping them separate makes tabu keys and quotas clearer and easier to tune.

### Test plan (high-level, still plan-only)
- Unit smoke for each move:
  - AddRegulation: adds 1 regulation; objective not worse than baseline in at least one candidate on toy data.
  - RemoveRegulation: reduces `num_regs`; on plans with low-impact regs, objective improves because of the `delta` penalty.
  - ShiftWindow: windows translated by exactly ±step; empty-window candidates pruned.
  - ChangeWindowLength: length changes by exactly ±2×step in symmetric mode; min/max respected.
- Integration smoke:
  - With quotas enabled, confirm all move types are represented in neighbor pool.
  - Confirm tabu prevents immediate reversals: e.g., AddReg → RemoveReg next iter (unless aspiration).

Blocking risks
- None; the plan aligns with existing APIs and caching. The only “nice-to-have” would be a public accessor on `FlightFeatures` for overloaded-by-hour; you can either use its internal field or add a small getter later.

- Added a detailed, code-free plan to implement AddRegulation, RemoveRegulation, ShiftWindow, and ChangeWindowLength.
- Defined candidate generation, validation, tabu keys/reversals, beam quotas, config knobs, and cache interactions for each move.
- Proposed neighbor generator changes to avoid move starvation; no code was written.