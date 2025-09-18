from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .state import HotspotContext, PlanState, RegulationMode


@dataclass(frozen=True)
class Action:
    """Base action type for static typing."""

    kind: str = field(init=False, default="action")


@dataclass(frozen=True)
class NewRegulation(Action):
    kind: str = field(init=False, default="new_regulation")
    context_hint: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class PickHotspot(Action):
    control_volume_id: str
    window_bins: Tuple[int, int]
    candidate_flow_ids: Sequence[str]
    mode: RegulationMode = "per_flow"
    metadata: Dict[str, object] = field(default_factory=dict)
    kind: str = field(init=False, default="pick_hotspot")


@dataclass(frozen=True)
class AddFlow(Action):
    flow_id: str
    kind: str = field(init=False, default="add_flow")


@dataclass(frozen=True)
class RemoveFlow(Action):
    flow_id: str
    kind: str = field(init=False, default="remove_flow")


@dataclass(frozen=True)
class Continue(Action):
    kind: str = field(init=False, default="continue")


@dataclass(frozen=True)
class Back(Action):
    kind: str = field(init=False, default="back")


@dataclass(frozen=True)
class CommitRegulation(Action):
    committed_rates: Optional[object] = None
    diagnostics: Dict[str, object] = field(default_factory=dict)
    kind: str = field(init=False, default="commit_regulation")


@dataclass(frozen=True)
class Stop(Action):
    kind: str = field(init=False, default="stop")


# --- Guard helpers -------------------------------------------------------------

def _validate_window(window_bins: Tuple[int, int]) -> None:
    t0, t1 = window_bins
    if t0 >= t1:
        raise ValueError("window_bins must be a non-empty half-open interval")
    if t0 < 0 or t1 < 0:
        raise ValueError("window_bins must be non-negative")


def guard_can_start_new_regulation(state: PlanState) -> None:
    if state.stage == "stopped":
        raise RuntimeError("cannot start a new regulation after stop")


def guard_can_pick_hotspot(state: PlanState, action: PickHotspot) -> None:
    _validate_window(action.window_bins)
    if not action.candidate_flow_ids:
        raise ValueError("candidate_flow_ids must not be empty")
    if state.stage not in {"idle", "select_hotspot"}:
        raise RuntimeError(f"cannot pick hotspot while in stage {state.stage}")


def guard_can_add_flow(state: PlanState, flow_id: str) -> None:
    ctx = state.hotspot_context
    if ctx is None:
        raise RuntimeError("no active hotspot to add flows to")
    if state.stage != "select_flows":
        raise RuntimeError("can only add flows while selecting flows")
    if flow_id not in ctx.candidate_flow_ids:
        raise KeyError(f"flow {flow_id} is not a candidate")
    if flow_id in ctx.selected_flow_ids:
        raise ValueError(f"flow {flow_id} already selected")


def guard_can_remove_flow(state: PlanState, flow_id: str) -> None:
    ctx = state.hotspot_context
    if ctx is None:
        raise RuntimeError("no active hotspot to remove flows from")
    if flow_id not in ctx.selected_flow_ids:
        raise KeyError(f"flow {flow_id} not currently selected")


def guard_can_continue(state: PlanState) -> None:
    ctx = state.hotspot_context
    if ctx is None:
        raise RuntimeError("cannot continue without an active hotspot")
    if not ctx.selected_flow_ids:
        raise RuntimeError("cannot continue without selecting at least one flow")
    if state.stage != "select_flows":
        raise RuntimeError("continue is only valid while selecting flows")


def guard_can_back(state: PlanState) -> None:
    if state.stage == "select_hotspot":
        raise RuntimeError("cannot back from hotspot selection without context")
    if state.stage == "idle":
        raise RuntimeError("cannot back from idle state")


def guard_can_commit(state: PlanState) -> None:
    ctx = state.hotspot_context
    if ctx is None:
        raise RuntimeError("no active hotspot to commit")
    if not ctx.selected_flow_ids:
        raise RuntimeError("no flows selected to commit")
    if state.stage != "confirm":
        raise RuntimeError("commit must follow continue to confirmation stage")


__all__ = [
    "Action",
    "NewRegulation",
    "PickHotspot",
    "AddFlow",
    "RemoveFlow",
    "Continue",
    "Back",
    "CommitRegulation",
    "Stop",
    "guard_can_start_new_regulation",
    "guard_can_pick_hotspot",
    "guard_can_add_flow",
    "guard_can_remove_flow",
    "guard_can_continue",
    "guard_can_back",
    "guard_can_commit",
]
