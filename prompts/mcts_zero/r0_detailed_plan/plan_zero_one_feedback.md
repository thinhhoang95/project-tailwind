### Critical issues to address

- Global resource leakage during sandboxed search
  - Flows use a different global resource hook than DF plan evaluation. Flows load `indexer`/`flight_list` from `parrhesia.api.resources` whereas DF plan evaluation always reads from `server_tailwind.core.resources.get_resources()` and ignores the paths you pass.
  - If your sandbox calls `refresh_after_state_update(res)` it will reset the process-wide flows globals to the sandbox’s `flight_list`. In a server context this can bleed the simulation state into live endpoints.
  - Evidence:
    ```208:213:src/server_tailwind/core/resources.py
    def get_resources() -> AppResources:
        global _GLOBAL_RESOURCES
        with _GLOBAL_LOCK:
            if _GLOBAL_RESOURCES is None:
                _GLOBAL_RESOURCES = AppResources()
            return _GLOBAL_RESOURCES
    ```
    ```7:14:src/parrhesia/api/resources.py
    def set_global_resources(indexer: Any, flight_list: Any) -> None:
        global _GLOBAL_INDEXER, _GLOBAL_FL
        _GLOBAL_INDEXER = indexer
        _GLOBAL_FL = flight_list

    def get_global_resources() -> Tuple[Optional[Any], Optional[Any]]:
        return _GLOBAL_INDEXER, _GLOBAL_FL
    ```
    ```46:63:src/parrhesia/api/flows.py
    def _load_indexer_and_flights(
        *,
        indexer_path: Optional[Path] = None,
        flights_path: Optional[Path] = None,
    ) -> Tuple[TVTWIndexer, FlightList]:
        # 1) Try shared resources first
        g_idx, g_fl = get_global_resources()
        if g_idx is not None and g_fl is not None:
            return g_idx, g_fl  # type: ignore[return-value]
    ```
    ```13:31:src/server_tailwind/core/cache_refresh.py
    def refresh_after_state_update(
        resources: Any,
        *,
        airspace_wrapper: Optional[Any] = None,
        count_wrapper: Optional[Any] = None,
        query_wrapper: Optional[Any] = None,
    ) -> None:
        """Refresh global caches so subsequent queries see the latest flight list state."""
        ...
        if _parrhesia_resources is not None:
            try:
                _parrhesia_resources.set_global_resources(resources.indexer, resources.flight_list)
    ```
  - Fix:
    - In sandboxed MCTS, don’t call `refresh_after_state_update(res)`. Instead, call `set_global_resources(res.indexer, res.flight_list)` immediately before `compute_flows(...)` and restore the previous pair after the call (tiny context manager). Keep the DF evaluator’s temporary `get_resources()` override (already correctly wrapped and restored).
    - Only use `refresh_after_state_update(...)` when mutating the real process-wide `AppResources` (e.g., in server handlers), and pass wrappers to invalidate their caches.

- DF plan evaluator reads from the global `AppResources` you temporarily bind
  - It purposely ignores the `indexer_path`/`flights_path` arguments and validates the time-axis alignment of `indexer` vs `flight_list`. This is correct, but you must keep the temporary override/finally block or you’ll evaluate against the wrong state.
  - Evidence:
    ```233:246:src/parrhesia/actions/dfplan_evaluator.py
    def evaluate_df_regulation_plan(
        plan: DFRegulationPlan,
        *,
        indexer_path: str,
        flights_path: str,
        capacities_path: Optional[str] = None,
        weights: Optional[Mapping[str, float]] = None,
        include_excess_vector: bool = False,  # reserved; not returned by result
    ) -> DFPlanEvalResult:
        """Evaluate a DFRegulationPlan against the flow objective.
        Uses process-wide AppResources for indexer/flight list. `indexer_path` and
        `flights_path` are ignored and retained only for backward compatibility.
    ```
    ```254:264:src/parrhesia/actions/dfplan_evaluator.py
    res = get_resources()
    tvtw_indexer = res.indexer
    flight_list = res.flight_list
    ...
    if T_idx != T_fl:
        raise RuntimeError(
            f"Resource mismatch: indexer.num_time_bins ({T_idx}) != flight_list.num_time_bins_per_tv ({T_fl})"
        )
    ```

- Forking the `FlightList` by `deepcopy` will be unnecessarily heavy
  - You referenced the evaluator’s lightweight clone; use the same structural copy strategy rather than a full deep copy to keep per-simulation memory down.
  - Evidence:
    ```111:173:src/server_tailwind/airspace/network_evaluator_for_api.py
    def _clone_flight_list_for_baseline(self, flight_list: FlightList) -> FlightList:
        """Create a detached snapshot of ``flight_list`` without reloading from disk."""
        ...
        clone.occupancy_matrix = occupancy.copy()
        ...
        clone._occupancy_matrix_lil = lil_matrix.copy()
    ```

- Hotspots/flows must be recomputed from the current sandbox, not from stale caches
  - Hotspots are correctly recomputed off the current `res.flight_list`/`capacity_per_bin_matrix`; good.
  - Flows must read the sandbox’s `flight_list`. They do if you set the parrhesia globals (see “Global leakage” above). That’s necessary because there isn’t a pure in‑memory override API for `compute_flows`.
  - Evidence:
    ```55:66:src/parrhesia/flow_agent35/regen/hotspot_segment_extractor.py
    res = (resources or get_resources()).preload_all()
    fl = res.flight_list
    ...
    total_occ = fl.get_total_occupancy_by_tvtw().astype(np.float32, copy=False)
    ```
    ```80:87:src/parrhesia/flow_agent35/regen/hotspot_segment_extractor.py
    cap = res.capacity_per_bin_matrix
    if cap is None or cap.shape != (num_tvs, bins_per_tv):
        raise RuntimeError("capacity_per_bin_matrix is missing or has unexpected shape")
    ```
    ```56:69:src/project_tailwind/stateman/flight_list_with_delta.py
    def step_by_delay(self, *views: DeltaOccupancyView, finalize: bool = True) -> None:
        ...
        for view in views:
            ...
            self._apply_single_view(view)
        if finalize:
            self.finalize_occupancy_updates()
    ```

### Design/behavior caveats (not bugs, but important)

- Additivity of rewards
  - `predicted_improvement.delta_objective_score` is an immediate reward under the current state. Summing these along a path is an approximation; the joint effect is not guaranteed additive. Your zero-rollout backup is consistent with that approximation, but you should re-evaluate the final chosen sequence with the DF evaluator to report the true cumulative improvement (your runner mentions this).

- Proposal caching scope
  - Caching proposals by `(state, hotspot_key)` is safe if (a) the path uniquely determines the mutated `flight_list`, and (b) you never reuse entries across different root baselines. Your `RZPathKey` scheme satisfies (a). Be careful not to reuse the `ProposalsCache` across different runs or different baseline `AppResources`.

- Flow extraction parameters
  - In `env.proposals_for_hotspot`, you don’t pass `threshold`/`resolution`/`seed`. Defaults may diverge from your example script and make priors non‑reproducible. Either pass them or surface them in `RZConfig`.

- Performance of `fork()`
  - Prefer the structural clone to reduce per-simulation memory churn. A deep copy of CSR/LIL and metadata for thousands of flights will cost seconds and hundreds of MB per simulation.

### Concrete tweaks to make now

- Avoid global leakage for flows in sandboxes; use a tiny context manager and remove `refresh_after_state_update` from the sandbox path:
```python
# proposed helper
from contextlib import contextmanager
import parrhesia.api.resources as parr_res

@contextmanager
def bind_parrhesia_resources(indexer, flight_list):
    prev = parr_res.get_global_resources()
    parr_res.set_global_resources(indexer, flight_list)
    try:
        yield
    finally:
        # restore previous (even if None/None)
        try:
            parr_res.set_global_resources(prev[0], prev[1])
        except Exception:
            pass

# in sandbox proposals_for_hotspot
with bind_parrhesia_resources(res.indexer, res.flight_list):
    flows_payload = compute_flows(
        tvs=[control_tv],
        timebins=timebins_h,
        direction_opts={"mode": "coord_cosine", "tv_centroids": res.tv_centroids},
        # optionally thread through threshold/resolution/seed here
    )
# do NOT call refresh_after_state_update(res) inside the sandbox
```

- Keep the DF evaluator’s temporary override exactly as you drafted (set, evaluate, restore). That matches its contract and avoids mismatched resources.

- Implement a lightweight `fork()` by porting `_clone_flight_list_for_baseline` logic so you copy only CSR/LIL and small dicts rather than `deepcopy` the full object.

- Thread flows config into `compute_flows` (e.g., from `RZConfig`) to match the example runner’s parameters for determinism.

### Sanity checks to keep

- Guard that indexer/time-bins match before each simulation step; the evaluator already enforces this.
- After each apply in the sandbox, verify there is a non‑zero occupancy delta to catch no‑op plans (you already do in the example with L1 diffs).

### What’s already safe in your plan

- Hotspots are recomputed from the sandboxed `flight_list` and capacity matrix each time.
- Proposals are generated against the sandbox’s `indexer`/`flight_list` with normalized capacities and travel minutes, not relying on stale singletons.
- DF plan application mutates only the sandbox’s `flight_list`, then you restore the global `AppResources` after evaluation.

- Key references:
  ```233:246:src/parrhesia/actions/dfplan_evaluator.py
  def evaluate_df_regulation_plan(...):
      """Evaluate a DFRegulationPlan against the flow objective.
      Uses process-wide AppResources for indexer/flight list. `indexer_path` and
      `flights_path` are ignored ...
  ```
  ```13:31:src/server_tailwind/core/cache_refresh.py
  ... set_global_resources(resources.indexer, resources.flight_list)
  ```
  ```111:173:src/server_tailwind/airspace/network_evaluator_for_api.py
  def _clone_flight_list_for_baseline(self, flight_list: FlightList) -> FlightList:
      """Create a detached snapshot ..."""
  ```
  ```55:66:src/parrhesia/flow_agent35/regen/hotspot_segment_extractor.py
  res = (resources or get_resources()).preload_all()
  fl = res.flight_list
  ```

- I can wire the context manager and a lighter `fork()` if you want me to proceed.

- I validated where globals are read/written and how hotspots/flows/evaluation bind to current state. The main correctness risk is global leakage from `refresh_after_state_update` inside sandboxes; avoid it and bind flows globals only around `compute_flows`.

- Add reproducibility knobs to flows in `env.proposals_for_hotspot` and replace `deepcopy` with a structural clone to keep MCTS runs fast and contained.