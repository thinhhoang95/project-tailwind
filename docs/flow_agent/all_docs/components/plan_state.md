**Plan State**

- Purpose: Captures the current plan, in-flight hotspot context, and a shaping proxy for MCTS.
- Location: `src/parrhesia/flow_agent/state.py`

Data classes
- `RegulationSpec` (immutable): Finalized regulation appended to the plan.
  - `control_volume_id: str`, `window_bins: (t0, t1)` [half-open).
  - `flow_ids: tuple[str, ...]` sorted stable.
  - `mode: 'per_flow' | 'blanket'`.
  - `committed_rates: dict[str,int] | int | None`.
  - `to_canonical_dict()`: Stable serialization used in `PlanState.canonical_key()`.
- `HotspotContext` (immutable): Editing state while constructing a regulation.
  - Holds `control_volume_id`, `window_bins`, candidate vs. selected flows, `mode`, and `metadata`.
  - `add_flow(flow_id)`/`remove_flow(flow_id)`: Return copies with updated selection.
- `PlanState` (mutable): Search root/state snapshot.
  - `plan: list[RegulationSpec]`, `hotspot_context`, `z_hat: np.ndarray | None` (potential proxy), `stage`.
  - `metadata: dict` for transient info like `hotspot_candidates` and `awaiting_commit`.
  - `copy()`: Deep-ish copy for safe mutation by transitions.
  - `canonical_key()`: Returns a stable JSON string capturing plan, context, stage, and `z_hat` signature.
  - Stage API: `reset_hotspot(next_stage)`, `set_hotspot(context)`, `ensure_stage(expected)`.

Stages
- `idle`: No active hotspot; begin with `NewRegulation`.
- `select_hotspot`: Choose a hotspot via `PickHotspot` from `metadata['hotspot_candidates']`.
- `select_flows`: Add/remove flows; `Continue` moves to confirmation.
- `confirm`: `CommitRegulation` to finalize; `Back` returns to selecting flows.
- `stopped`: Terminal state after `Stop`.

Examples
- Create a state and add a simple regulation
```python
from parrhesia.flow_agent.state import PlanState, HotspotContext, RegulationSpec

state = PlanState()
ctx = HotspotContext(
    control_volume_id="TV123",
    window_bins=(10, 14),
    candidate_flow_ids=("F1", "F2"),
)
state.set_hotspot(ctx, z_hat_shape=4)

# Choose flows and finalize a specification
ctx2 = state.hotspot_context.add_flow("F1")  # immutable update
state.hotspot_context = ctx2
spec = RegulationSpec(
    control_volume_id=ctx2.control_volume_id,
    window_bins=ctx2.window_bins,
    flow_ids=ctx2.selected_flow_ids,
    mode="per_flow",
    committed_rates={"F1": 36},
)
state.plan.append(spec)
state.reset_hotspot(next_stage="idle")
```

Tips
- `window_bins` are validated to be non-empty and non-negative; errors are raised early for invalid ranges.
- `z_hat` is only a local, per-hotspot proxy. It is reset when changing hotspot or after commit/stop.

