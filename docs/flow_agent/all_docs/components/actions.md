**Actions**

- Purpose: Define the move set for search and guard rules for valid transitions.
- Location: `src/parrhesia/flow_agent/actions.py`

Action types
- `NewRegulation(context_hint=None)`: Start building a new regulation; moves stage to `select_hotspot`.
- `PickHotspot(control_volume_id, window_bins, candidate_flow_ids, mode='per_flow', metadata={})`: Choose the hotspot to regulate and loads its candidate flows.
- `AddFlow(flow_id)`: Add a flow to the current selection.
- `RemoveFlow(flow_id)`: Remove a previously selected flow.
- `Continue()`: Confirm selected flows; advances to `confirm` stage.
- `Back()`: Go back one stage (`confirm` → `select_flows`, or drop hotspot back to `select_hotspot`).
- `CommitRegulation(committed_rates=None, diagnostics={})`: Finalize a regulation; appended to plan by the transition.
- `Stop()`: Abort current regulation and mark plan as `stopped`.

Guards (validation helpers)
- `guard_can_start_new_regulation(state)`: Disallows starting after `stopped`.
- `guard_can_pick_hotspot(state, action)`: Requires non-empty `candidate_flow_ids` and valid stage.
- `guard_can_add_flow(state, flow_id)`: Valid only in `select_flows` and for known candidates.
- `guard_can_remove_flow(state, flow_id)`: Valid only when selected.
- `guard_can_continue(state)`: Requires at least one selected flow while in `select_flows`.
- `guard_can_back(state)`: Disallows from `idle` or early `select_hotspot` with no context.
- `guard_can_commit(state)`: Requires `confirm` stage and at least one selected flow.

Flow of a typical episode
1) `NewRegulation()` → stage `select_hotspot`.
2) `PickHotspot(...)` → stage `select_flows` with candidate flows.
3) `AddFlow(...)` one or more times.
4) `Continue()` → stage `confirm`.
5) `CommitRegulation(...)` → appends a `RegulationSpec` and returns to `idle`.

Example
```python
from parrhesia.flow_agent.actions import NewRegulation, PickHotspot, AddFlow, Continue, CommitRegulation
from parrhesia.flow_agent.state import PlanState
from parrhesia.flow_agent.transition import CheapTransition

state = PlanState()
trans = CheapTransition()

state, *_ = trans.step(state, NewRegulation())
state, *_ = trans.step(state, PickHotspot("TV123", (20, 24), ("F1", "F2")))
state, *_ = trans.step(state, AddFlow("F1"))
state, *_ = trans.step(state, Continue())
state, is_commit, _ = trans.step(state, CommitRegulation({"F1": 24}))
assert is_commit and len(state.plan) == 1
```

