### Chronological logic walk (from “run” to objective evaluation)

1) Orchestrator boot
- `AlnsOrchestrator.setup()` loads:
  - TVTW indexer, base `FlightList`, `RegulationParser`, `traffic_volumes` GeoJSON, and builds `OptimizationProblem`.
- `AlnsOrchestrator.run()` creates `initial_state` and starts ALNS:
  - Destroy operator: removes a random regulation
  - Repair operator: adds a random regulation
  - Acceptance: HillClimbing; Selection: RouletteWheel; Stop: max iterations

2) Initial state
- `OptimizationProblem.create_initial_state()` returns a `ProblemState` with:
  - An empty `NetworkPlan`
  - The shared base `FlightList`
  - Optimization context (weights, indexer, parser, TVs, horizon)

3) Cheap state copies for ALNS moves
- `ProblemState.copy()` now avoids deep-copying `FlightList`. It copies only the `NetworkPlan` (lightweight) and `aux`.
- This keeps ALNS move semantics correct while making state copying fast.

```49:66:src/project_tailwind/optimize/alns/pstate.py
def copy(self) -> "ProblemState":
    return ProblemState(
        network_plan=self.network_plan.copy() if hasattr(self.network_plan, 'copy') else self.network_plan,
        flight_list=self.flight_list,
        optimization_problem=self.optimization_problem,
        aux=self.aux.copy(),
    )
```

4) ALNS iteration and when objective() is called
- Every time ALNS needs a score, it calls `ProblemState.objective()` on a candidate state.

5) Build the per-flight delays implied by the current plan
- We compute but do not apply delays:
  - `NetworkPlanMove.compute_final_delays(state: FlightList)`:
    - For each regulation in the `NetworkPlan`:
      - `RegulationParser.parse()` finds matching flights.
      - `run_readapted_casa(...)` computes per-flight delay minutes for that regulation.
    - For flights affected by multiple regs, take the maximum delay per flight.
  - This returns a `Dict[flight_id -> delay_min]`.

```138:173:src/project_tailwind/optimize/moves/network_plan_move.py
def compute_final_delays(self, state: FlightList) -> Dict[str, int]:
    if not self.network_plan.regulations:
        return {}
    flight_delays_by_regulation: Dict[str, List[float]] = defaultdict(list)
    for regulation in self.network_plan.regulations:
        matched_flights = self.parser.parse(regulation)
        if not matched_flights:
            continue
        delays = run_readapted_casa(
            flight_list=state,
            identifier_list=matched_flights,
            reference_location=regulation.location,
            tvtw_indexer=self.tvtw_indexer,
            hourly_rate=regulation.rate,
            active_time_windows=regulation.time_windows,
        )
        for flight_id, delay_minutes in delays.items():
            if delay_minutes > 0:
                flight_delays_by_regulation[flight_id].append(delay_minutes)
    final_flight_delays = {fid: int(max(dlist)) for fid, dlist in flight_delays_by_regulation.items()}
    return final_flight_delays
```

6) Overlay delays without mutating the base flight list
- `NetworkPlanMove.build_delta_view(state)` creates a `DeltaFlightList` view and returns total delay (sum of mins):
  - The view wraps the base `FlightList` and a `delays` dict.

```174:185:src/project_tailwind/optimize/moves/network_plan_move.py
def build_delta_view(self, state: FlightList) -> tuple[DeltaFlightList, int]:
    final_delays = self.compute_final_delays(state)
    total_delay = sum(final_delays.values())
    return DeltaFlightList(state, final_delays), total_delay
```

7) What `DeltaFlightList` does
- It is read-only and implements only what evaluators read:
  - `get_total_occupancy_by_tvtw()` = base totals + sum over delayed flights of (shifted - original) vectors.
  - `get_occupancy_vector(flight_id)` returns shifted vector if delayed; else base vector.
  - It also overlays `flight_metadata.takeoff_time` by adding the delay in minutes, so delay statistics work.
- No sparse matrix rebuilds, no in-place mutations.

```20:63:src/project_tailwind/optimize/eval/delta_flight_list.py
def get_total_occupancy_by_tvtw(self) -> np.ndarray:
    if self._cached_total_occupancy is not None:
        return self._cached_total_occupancy
    total = self._base.get_total_occupancy_by_tvtw().astype(np.float32, copy=True)
    if not self._delays:
        self._cached_total_occupancy = total
        return total
    delta_accumulator = np.zeros_like(total)
    for flight_id, delay_min in self._delays.items():
        if delay_min == 0:
            continue
        orig = self._base.get_occupancy_vector(flight_id)
        shifted = self._base.shift_flight_occupancy(flight_id, delay_min)
        delta_accumulator += (shifted - orig)
    total += delta_accumulator
    self._cached_total_occupancy = total
    return total
```

8) Objective evaluation with delta view
- `ProblemState.objective()`:
  - Creates `NetworkPlanMove` and calls `build_delta_view(self.flight_list)` to get the `DeltaFlightList` and total delay.
  - Constructs `NetworkEvaluator(traffic_volumes_gdf, delta_view)`.
  - Gets metrics over the horizon and computes weighted score.

```64:75:src/project_tailwind/optimize/alns/pstate.py
move = NetworkPlanMove(
    network_plan=self.network_plan,
    parser=self.optimization_problem.regulation_parser,
    tvtw_indexer=self.optimization_problem.tvtw_indexer,
)
delta_view, total_delay = move.build_delta_view(self.flight_list)
network_evaluator = NetworkEvaluator(
    traffic_volumes_gdf=self.optimization_problem.base_traffic_volumes,
    flight_list=delta_view,
)
metrics = network_evaluator.compute_horizon_metrics(self.optimization_problem.horizon_time_windows)
score = (
    self.optimization_problem.objective_weights["z_95"] * metrics["z_95"]
    + self.optimization_problem.objective_weights["z_sum"] * metrics["z_sum"]
    + self.optimization_problem.objective_weights["delay"] * total_delay
)
```

9) How `NetworkEvaluator` interprets the view
- On init, if given a `DeltaFlightList`, it uses the underlying base as the “original” baseline for delay stats.
- `compute_excess_traffic_vector()` uses `delta_view.get_total_occupancy_by_tvtw()`, so occupancies reflect delays.
- `compute_delay_stats()` compares `delta_view.flight_metadata.takeoff_time` vs the baseline’s for seconds of delay.

```31:38:src/project_tailwind/optimize/eval/network_evaluator.py
self.flight_list = flight_list
if hasattr(flight_list, "_base"):
    self.original_flight_list = flight_list._base
else:
    self.original_flight_list = flight_list.copy()
```

10) Mutation path (outside objective)
- If you actually want to apply a plan to mutate a `FlightList`, the existing path remains:
  - `NetworkPlanMove._apply_to_flight_list()` collects max delays and calls `batch_delay_operator(...)` to modify the sparse matrix and metadata in bulk.
- This is only used in mutation scenarios; objective evaluation now uses the non-mutating delta path.

11) ALNS loop control
- With cheap state copies and fast objective eval, ALNS iterates:
  - Applies destroy/repair on `NetworkPlan` inside the copied `ProblemState`.
  - Compares objective scores of candidates; HillClimbing decides acceptance; RouletteWheel controls selection; stop after 1000 iterations (configurable).

12) Final outputs
- After iterations, `orchestrator.run()` prints the best objective and the best network plan found.

---

- Implemented a delta-based evaluation layer to avoid `FlightList` deep copies:
  - Added `DeltaFlightList` view and adjusted `NetworkEvaluator` to use the base as baseline for delay stats.
  - Extended `NetworkPlanMove` with `compute_final_delays()` and `build_delta_view()` to compute per-flight delays without applying them.
  - Updated `ProblemState.copy()` to be cheap; `objective()` now evaluates via the delta view.
  - Added `NetworkPlan.copy()` for safe, independent state copies.

- Impact: ALNS state copying is now light, objective evaluation avoids heavy sparse mutations, and overall iteration speed should improve while keeping correctness of z_95, z_sum, and delay terms.