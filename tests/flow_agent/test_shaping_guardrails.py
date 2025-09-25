import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from parrhesia.flow_agent.actions import CommitRegulation
from parrhesia.flow_agent.mcts import MCTS, MCTSConfig
from parrhesia.flow_agent.state import HotspotContext, PlanState
from parrhesia.flow_agent.transition import CheapTransition


class _StubRateFinder:
    def find_rates(self, **kwargs):
        flows = kwargs.get("flows", {}) or {}
        rates = {fid: 10 for fid in flows.keys()}
        return rates, -5.0, {"reason": "ok"}


class _AsyncRateFinder:
    def __init__(self):
        self.calls = 0

    def find_rates(self, **kwargs):
        self.calls += 1
        time.sleep(0.01)
        flows = kwargs.get("flows", {}) or {}
        rates = {fid: 7 for fid in flows.keys()}
        return rates, -4.0, {"reason": "ok"}


def _make_state(z_hat):
    state = PlanState()
    if z_hat is not None:
        state.z_hat = np.asarray(z_hat, dtype=float)
    return state


def _compute_shaped_total(mcts: MCTS, states, base_rewards):
    total = 0.0
    phi_prev = mcts._phi(states[0])
    for idx, r_base in enumerate(base_rewards):
        phi_s = phi_prev
        phi_sp = mcts._phi(states[idx + 1])
        total += float(r_base) + (phi_sp - phi_s)
        phi_prev = phi_sp
    total += -mcts._phi(states[-1])
    return total


def test_shaping_telescopes_to_base_rewards():
    cfg = MCTSConfig(phi_scale=0.75)
    mcts = MCTS(transition=CheapTransition(), rate_finder=_StubRateFinder(), config=cfg)
    rng = np.random.default_rng(0)

    for _ in range(16):
        length = rng.integers(2, 6)
        states = [_make_state(rng.uniform(-0.4, 0.4, size=4)) for _ in range(length)]
        base_rewards = rng.uniform(-3.0, 5.0, size=length - 1)
        shaped = _compute_shaped_total(mcts, states, base_rewards)
        expected = float(np.sum(base_rewards)) - mcts._phi(states[0])
        assert np.isclose(shaped, expected, atol=1e-6)


def test_shaping_delta_phi_magnitude_guardrail():
    cfg = MCTSConfig(phi_scale=0.1)
    mcts = MCTS(transition=CheapTransition(), rate_finder=_StubRateFinder(), config=cfg)
    rng = np.random.default_rng(1)
    delta_phis = []
    delta_js = []

    for _ in range(256):
        s = _make_state(rng.uniform(-0.3, 0.5, size=5))
        sp = _make_state(rng.uniform(-0.3, 0.5, size=5))
        delta_phi = mcts._phi(sp) - mcts._phi(s)
        delta_phis.append(abs(delta_phi))
        delta_js.append(rng.uniform(2.0, 6.0))

    median_phi = float(np.median(delta_phis))
    median_j = float(np.median(delta_js))
    assert median_phi <= 0.5 * median_j


def test_async_commit_eval_returns_cached_result():
    rate_finder = _AsyncRateFinder()
    cfg = MCTSConfig(commit_eval_limit=4, min_unique_commit_evals=0)
    transition = CheapTransition()
    with ThreadPoolExecutor(max_workers=2) as executor:
        mcts = MCTS(
            transition=transition,
            rate_finder=rate_finder,
            config=cfg,
            commit_executor=executor,
        )
        ctx = HotspotContext(
            control_volume_id="TV1",
            window_bins=(0, 2),
            candidate_flow_ids=("F1", "F2"),
            selected_flow_ids=("F1",),
            metadata={"flow_to_flights": {"F1": ("flight",)}},
        )
        state = PlanState(
            plan=[],
            hotspot_context=ctx,
            z_hat=np.zeros(2),
            stage="confirm",
            metadata={"awaiting_commit": True},
        )

        action_pending, delta_pending = mcts._evaluate_commit(state)
        assert isinstance(action_pending, CommitRegulation)
        diag = action_pending.diagnostics.get("rate_finder", {})
        assert diag.get("reason") == "async_pending"
        assert pytest.approx(delta_pending, abs=1e-9) == 0.0
        assert rate_finder.calls == 1

        mcts.wait_for_pending_commit_evals(timeout=2.0)
        action_ready, delta_ready = mcts._evaluate_commit(state)
        assert isinstance(action_ready, CommitRegulation)
        assert delta_ready == pytest.approx(-4.0)
        assert action_ready.diagnostics.get("rate_finder", {}).get("reason") == "ok"
        assert rate_finder.calls == 1
        with mcts._commit_lock:
            assert not mcts._pending_commit_jobs
