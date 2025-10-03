I reviewed the plan end-to-end against the codebase APIs and data flow. Here are the hard findings and required fixes before you implement.

### Critical issues to fix

- Strong mismatch: evaluation uses a global resources singleton, not your sandbox
  - `evaluate_df_regulation_plan(...)` pulls `flight_list/indexer` from a process-wide singleton, so it will ignore your per-node `RZSandbox` state unless you swap the singleton to your sandbox before calling it:
  ```233:258:/mnt/d/project-tailwind/src/parrhesia/actions/dfplan_evaluator.py
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
      """
      # Step 1: Load essential artifacts for evaluation from AppResources.
      if get_resources is None:
          raise RuntimeError(
              f"AppResources are unavailable (import error: {_RES_IMPORT_ERR}); cannot evaluate DF plan"
          )
      res = get_resources()
      tvtw_indexer = res.indexer
      flight_list = res.flight_list  # FlightListWithDelta (subclass of FlightList)
  ```
  ```208:214:/mnt/d/project-tailwind/src/server_tailwind/core/resources.py
  def get_resources() -> AppResources:
      global _GLOBAL_RESOURCES
      with _GLOBAL_LOCK:
          if _GLOBAL_RESOURCES is None:
              _GLOBAL_RESOURCES = AppResources()
          return _GLOBAL_RESOURCES
  ```
  - Fix: In `RZSandbox.apply_proposal` (or just before invoking the evaluator), temporarily bind the singleton to this sandbox, then restore:
  ```python
  import server_tailwind.core.resources as core_res

  prev = core_res.get_resources()
  core_res._GLOBAL_RESOURCES = self._res
  try:
      eval_res = evaluate_df_regulation_plan(...)
  finally:
      core_res._GLOBAL_RESOURCES = prev
  ```
  - Without this, multi-step MCTS playouts won’t reflect the mutated sandbox state and the computed delays will be inconsistent.

- Normalize capacities before passing to regen engine
  - You’re building `capacities_by_tv` straight from `res.capacity_per_bin_matrix` rows. That matrix uses -1.0 for “unknown” bins; regen’s caches treat those as capacity, corrupting exceedance and scoring. Normalize to unconstrained with `normalize_capacities(...)` first (the wrappers/examples do this).
  ```110:150:/mnt/d/project-tailwind/src/parrhesia/optim/capacity.py
  def normalize_capacities(
      capacities_by_tv: Dict[str, np.ndarray],
      unconstrained_value: float = 9999.0,
  ) -> Dict[str, np.ndarray]:
      ...
      for tv_id, arr in capacities_by_tv.items():
          arr_copy = np.asarray(arr, dtype=np.float64).copy()
          # Replace non-positive capacities with unconstrained value
          arr_copy[arr_copy <= 0.0] = float(unconstrained_value)
          result[str(tv_id)] = arr_copy
  ```
  - Example of the normalized build in your wrapper:
  ```120:141:/mnt/d/project-tailwind/src/server_tailwind/regen/regen_api_wrapper.py
  self._capacities_by_tv = normalize_capacities(raw_capacities)
  ...
  return self._capacities_by_tv
  ```

### Important integration notes

- Flows API uses a different global resource hook than dfplan evaluator
  - You already set `parrhesia.api.resources.set_global_resources(indexer, flight_list)` before calling `compute_flows(...)`, which is correct for flows:
  ```159:187:/mnt/d/project-tailwind/src/parrhesia/api/flows.py
  def compute_flows(...):
      ...
      idx, fl = _load_indexer_and_flights(indexer_path=idx_path, flights_path=fl_path)
  ```
  ```46:63:/mnt/d/project-tailwind/src/parrhesia/api/flows.py
  def _load_indexer_and_flights(...):
      # 1) Try shared resources first
      g_idx, g_fl = get_global_resources()
      if g_idx is not None and g_fl is not None:
          return g_idx, g_fl
  ```
  - But the evaluator uses `server_tailwind.core.resources.get_resources()` (see critical fix above). Treat them separately.

- Use the same AppResources instance as the global when convenient
  - In your `runner`, prefer `get_resources().preload_all()` over `AppResources().preload_all()` so the root matches the global used by the evaluator:
  ```208:214:/mnt/d/project-tailwind/src/server_tailwind/core/resources.py
  def get_resources() -> AppResources: ...
  ```

- Ensure you import `segment_to_hotspot_payload` in `mcts.py`
  - It exists and has the right shape:
  ```147:154:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/hotspot_segment_extractor.py
  def segment_to_hotspot_payload(seg: Mapping[str, Any]) -> Dict[str, Any]:
      # regen uses [start, end_exclusive]; segments are inclusive
      return {
          "control_volume_id": str(seg["traffic_volume_id"]),
          "window_bins": [int(seg["start_bin"]), int(seg["end_bin"]) + 1],
          "metadata": {},
          "mode": "inventory",
      }
  ```

### Smaller correctness notes

- DF plan construction fits your mapping of `flights_by_flow` (int or str keys)
  ```651:659:/mnt/d/project-tailwind/src/parrhesia/actions/regulations.py
  @classmethod
  def from_proposal(...):
  ```
  ```709:713:/mnt/d/project-tailwind/src/parrhesia/actions/regulations.py
  flights_spec = (
      flights_by_flow.get(flow_id)
      or flights_by_flow.get(str(flow_id))
      or []
  )
  ```

- Flows inputs and priors/rewards from regen align
  ```195:207:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/engine.py
  def propose_regulations_for_hotspot(... ) -> List[Proposal]:
  ```
  ```138:151:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/types.py
  @dataclass(frozen=True)
  class Proposal:
      hotspot_id: str
      controlled_volume: str
      window: Window
      flows_info: List[Mapping[str, Any]]
      predicted_improvement: PredictedImprovement
      diagnostics: Mapping[str, Any]
  ```

- State updates path is correct
  ```56:69:/mnt/d/project-tailwind/src/project_tailwind/stateman/flight_list_with_delta.py
  def step_by_delay(self, *views: DeltaOccupancyView, finalize: bool = True) -> None:
      ...
      for view in views:
          ...
          self._apply_single_view(view)
      if finalize:
          self.finalize_occupancy_updates()
  ```
  ```55:69:/mnt/d/project-tailwind/src/project_tailwind/stateman/delta_view.py
  @classmethod
  def from_delay_table(... ) -> "DeltaOccupancyView":
      ...
  ```

- Cache refresh semantics
  - `refresh_after_state_update` re-registers parrhesia globals and optionally invalidates server wrappers if passed; with just `refresh_after_state_update(res)` you only re-register parrhesia globals:
  ```13:32:/mnt/d/project-tailwind/src/server_tailwind/core/cache_refresh.py
  def refresh_after_state_update(resources: Any, *, airspace_wrapper: Optional[Any] = None, ...):
      ...
      if _parrhesia_resources is not None:
          try:
              _parrhesia_resources.set_global_resources(resources.indexer, resources.flight_list)
  ```
  - If you ever integrate with HTTP wrappers, pass them in to avoid stale caches.

### Concrete edits to make in your implementation

- In `RZSandbox.proposals_for_hotspot`, normalize capacities:
```python
from parrhesia.optim.capacity import normalize_capacities

caps = res.capacity_per_bin_matrix
cap_map_raw = {str(tv): caps[int(row), :] for tv, row in res.flight_list.tv_id_to_idx.items()}
capacities_by_tv = normalize_capacities(cap_map_raw)
...
proposals = propose_regulations_for_hotspot(
    indexer=res.indexer,
    flight_list=res.flight_list,
    capacities_by_tv=capacities_by_tv,
    ...
)
```

- In `RZSandbox.apply_proposal`, temporarily bind evaluator resources to this sandbox:
```python
import server_tailwind.core.resources as core_res

prev = core_res.get_resources()
core_res._GLOBAL_RESOURCES = res
try:
    eval_res = evaluate_df_regulation_plan(
        plan,
        indexer_path=str(res.paths.tvtw_indexer_path),
        flights_path=str(res.paths.occupancy_file_path),
    )
finally:
    core_res._GLOBAL_RESOURCES = prev
```

- In `runner`, build root from the global resources:
```python
from server_tailwind.core.resources import get_resources
res = get_resources().preload_all()
```

### Optional but helpful

- Seed `np.random.default_rng` if you need reproducible root noise across runs.
- Consider caching `capacities_by_tv` on the sandbox to avoid rebuilding per hotspot.
- Document that MCTS finalization replays the chosen `(hotspot, proposal_rank)` sequence by recomputing proposals for each state to materialize `DFRegulationPlan`s, using the same deterministic flow seed.

### Quick confirmations

- Flows API inputs and direction options are correct:
```159:187:/mnt/d/project-tailwind/src/parrhesia/api/flows.py
def compute_flows(..., direction_opts: Optional[Dict[str, Any]] = None, ...):
```
- `delta_objective_score` is present in regen outputs and used consistently for priors and rewards:
```402:440:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/engine.py
...
predicted_improvement=improvement
...
final_score = improvement.delta_objective_score - penalty
```

If you apply the three concrete edits above (normalize capacities, bind evaluator resources to the sandbox for each apply, and root off the global resources in the runner), the plan is solid and should execute correctly.

- I verified the regen, flows, DF plan, evaluator, and state-transition APIs you reference align with current signatures.
- I’m about to proceed with the edits outlined if you’d like me to scaffold `src/regulation_zero/` with the fixes baked in.