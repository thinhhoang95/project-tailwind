from __future__ import annotations

import json
import numpy as np
import pytest

from parrhesia.flow_agent.state import HotspotContext, PlanState, RegulationSpec
from parrhesia.flow_agent.actions import (
    PickHotspot,
    guard_can_add_flow,
    guard_can_continue,
    guard_can_pick_hotspot,
)


def test_regulation_spec_canonicalization_and_rates():
    spec = RegulationSpec(
        control_volume_id="TV1",
        window_bins=(5, 8),
        flow_ids=("flow_b", "flow_a"),
        committed_rates={"flow_b": 20, "flow_a": 15},
    )
    assert spec.flow_ids == ("flow_a", "flow_b")
    canonical = spec.to_canonical_dict()
    assert canonical["flow_ids"] == ["flow_a", "flow_b"]
    assert canonical["committed_rates"] == {"flow_a": 15, "flow_b": 20}


def test_plan_state_canonical_key_stable_round_trip():
    state = PlanState()
    empty_key = state.canonical_key()
    assert json.loads(empty_key)["plan"] == []

    ctx = HotspotContext(
        control_volume_id="TV1",
        window_bins=(10, 12),
        candidate_flow_ids=("f1", "f2"),
        selected_flow_ids=("f2",),
    )
    state.set_hotspot(ctx, z_hat_shape=2)
    state.z_hat = np.array([1.0, 2.0])
    key1 = state.canonical_key()

    clone = PlanState()
    clone.set_hotspot(
        HotspotContext(
            control_volume_id="TV1",
            window_bins=(10, 12),
            candidate_flow_ids=("f2", "f1"),
            selected_flow_ids=("f2",),
        ),
        z_hat_shape=2,
    )
    clone.z_hat = np.array([1.0, 2.0])
    assert clone.canonical_key() == key1


def test_action_guards_require_valid_state():
    state = PlanState()
    pick = PickHotspot(
        control_volume_id="TV1",
        window_bins=(0, 4),
        candidate_flow_ids=("f1", "f2"),
    )
    guard_can_pick_hotspot(state, pick)

    state.set_hotspot(
        HotspotContext(
            control_volume_id="TV1",
            window_bins=(0, 4),
            candidate_flow_ids=("f1", "f2"),
        ),
        z_hat_shape=4,
    )

    with pytest.raises(RuntimeError):
        guard_can_continue(state)

    guard_can_add_flow(state, "f1")
    state.hotspot_context = state.hotspot_context.add_flow("f1")
    with pytest.raises(ValueError):
        guard_can_add_flow(state, "f1")

    guard_can_continue(state)
