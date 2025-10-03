**Transition Model**

- Purpose: Applies actions to `PlanState` cheaply and maintains a residual overload proxy `z_hat` to guide MCTS.
- Location: `src/parrhesia/flow_agent/transition.py`

`CheapTransition`
- Stage changes: Implements the state machine over `idle → select_hotspot → select_flows → confirm → idle/stopped`.
- Hotspot lifecycle:
  - `NewRegulation`: resets context and moves to `select_hotspot`.
  - `PickHotspot`: creates `HotspotContext` and allocates `z_hat` with length equal to the window length.
- Flow selection updates:
  - Maintains `z_hat` by adding/removing a “proxy” vector per flow that estimates its contribution to overload within the hotspot window.
  - Proxies come from `context.metadata['flow_proxies']` if present (provided by discovery), otherwise from transition’s `flow_proxies` map, else fall back to ones.
  - Optional `decay` and `clip_value` keep `z_hat` stable as flows are toggled.
- Confirmation & commit:
  - `Continue`: marks `awaiting_commit` and changes stage to `confirm`.
  - `CommitRegulation`: appends a new `RegulationSpec` populated with current context and the action’s `committed_rates`/`diagnostics`, then resets hotspot back to `idle`.
- Stop:
  - `Stop`: clears hotspot and sets stage `stopped`.

Examples
- Updating `z_hat` when flows change
```python
from parrhesia.flow_agent.state import PlanState
from parrhesia.flow_agent.transition import CheapTransition
from parrhesia.flow_agent.actions import PickHotspot, AddFlow, RemoveFlow

state = PlanState()
trans = CheapTransition()
state, *_ = trans.step(state, PickHotspot("TV1", (5, 9), ("A", "B")))

z0 = state.z_hat.copy()
state, *_ = trans.step(state, AddFlow("A"))  # z_hat += proxy(A)
state, *_ = trans.step(state, AddFlow("B"))  # z_hat += proxy(B)
state, *_ = trans.step(state, RemoveFlow("A"))  # z_hat -= proxy(A)

assert state.z_hat.shape[0] == (9 - 5)
```

Tips
- When writing tests, supply deterministic `flow_proxies` so `z_hat` and priors are stable.
- Proxies are sliced/padded to the hotspot window; discovery emits aligned histograms, so no extra work is needed in common cases.

