from __future__ import annotations

import numpy as np

from parrhesia.flow_agent.actions import (
    AddFlow,
    Back,
    CommitRegulation,
    Continue,
    NewRegulation,
    PickHotspot,
    RemoveFlow,
    Stop,
)
from parrhesia.flow_agent.state import PlanState
from parrhesia.flow_agent.transition import CheapTransition


def test_transition_pipeline_updates_z_hat_and_plan():
    proxies = {
        "flow_a": np.arange(10, dtype=float),
        "flow_b": np.linspace(0.5, 1.5, num=10, dtype=float),
    }
    transition = CheapTransition(flow_proxies=proxies, clip_value=100.0)
    state = PlanState()

    state, commit, terminal = transition.step(state, NewRegulation())
    assert not commit and not terminal
    assert state.stage == "select_hotspot"

    state, _, _ = transition.step(
        state,
        PickHotspot(
            control_volume_id="TV1",
            window_bins=(2, 5),
            candidate_flow_ids=("flow_a", "flow_b"),
        ),
    )
    assert state.stage == "select_flows"
    assert state.z_hat is not None and state.z_hat.shape == (3,)
    assert np.allclose(state.z_hat, np.zeros(3))

    state, _, _ = transition.step(state, AddFlow("flow_a"))
    assert np.allclose(state.z_hat, proxies["flow_a"][2:5])

    state, _, _ = transition.step(state, AddFlow("flow_b"))
    expected = proxies["flow_a"][2:5] + proxies["flow_b"][2:5]
    assert np.allclose(state.z_hat, expected)

    state, _, _ = transition.step(state, RemoveFlow("flow_b"))
    assert np.allclose(state.z_hat, proxies["flow_a"][2:5])

    state, _, _ = transition.step(state, Continue())
    assert state.stage == "confirm"

    state, commit, terminal = transition.step(
        state,
        CommitRegulation(committed_rates={"flow_a": 24}),
    )
    assert commit and not terminal
    assert state.stage == "idle"
    assert len(state.plan) == 1
    spec = state.plan[0]
    assert spec.control_volume_id == "TV1"
    assert spec.committed_rates == {"flow_a": 24}


def test_back_and_stop_transitions():
    transition = CheapTransition()
    state = PlanState()

    state, _, _ = transition.step(state, NewRegulation())
    state, _, _ = transition.step(
        state,
        PickHotspot(
            control_volume_id="TV1",
            window_bins=(0, 2),
            candidate_flow_ids=("flow_a",),
        ),
    )
    state, _, _ = transition.step(state, Back())
    assert state.stage == "select_hotspot"
    assert state.hotspot_context is None

    state, commit, terminal = transition.step(state, Stop())
    assert terminal and not commit
    assert state.stage == "stopped"
