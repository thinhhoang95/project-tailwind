from __future__ import annotations

"""PUCT-based MCTS for Regulation Zero without rollouts.

Each simulation forks a sandbox and traverses a path of actions. Expansion at a
node enumerates the current top hotspot segments, generates ranked proposals per
hotspot, and sets child priors via a softmax over predicted improvements. We do
not use rollouts; leaf value is 0, and edge rewards are the immediate deltas.
"""

import math
from typing import Dict, List, Tuple

import numpy as np

from parrhesia.flow_agent35.regen.hotspot_segment_extractor import (
    segment_to_hotspot_payload,
)

from .types import ActionKey, RZAction, RZConfig, RZPathKey
from .cache import ChildStats, NodeStats, ProposalsCache, TranspositionTable
from .env import RZSandbox


def softmax(xs: List[float], tau: float = 1.0) -> List[float]:
    """Numerically stable softmax with temperature tau."""
    if not xs:
        return []
    arr = np.asarray(xs, dtype=np.float64) / max(float(tau), 1e-6)
    arr -= arr.max()
    ex = np.exp(arr)
    Z = ex.sum()
    if Z <= 0.0:
        return [1.0 / float(len(xs)) for _ in xs]
    return [(float(x) / float(Z)) for x in ex]


class MCTS:
    """AlphaZero-style MCTS using PUCT selection and no rollouts."""

    def __init__(self, env_factory, cfg: RZConfig):
        # env_factory must return a fresh RZSandbox fork at root state per simulation
        self.env_factory = env_factory
        self.cfg = cfg
        self.tt: TranspositionTable = {}
        # (state, hotspot_key) -> list of (proposal_obj, flights_by_flow, delta)
        self.cache: ProposalsCache = {}

    def run(self) -> List[RZAction]:
        """Run MCTS for cfg.num_simulations; return greedy action sequence by visits."""
        sims = int(self.cfg.num_simulations)
        for _ in range(sims):
            env = self.env_factory()
            self._simulate((), env)

        # Greedy extraction by max visit count from root downward
        actions: List[RZAction] = []
        state: RZPathKey = ()
        depth = 0
        while depth < self.cfg.max_depth and state in self.tt and self.tt[state].children:
            best = max(self.tt[state].children.items(), key=lambda kv: kv[1].N)
            ak = best[0]
            actions.append(RZAction(hotspot_key=ak[0], proposal_rank=ak[1], delta_obj=0.0))
            state = tuple(list(state) + [ak])
            depth += 1
        return actions

    def _simulate(self, state: RZPathKey, env: RZSandbox) -> float:
        depth = len(state)
        if depth >= int(self.cfg.max_depth):
            return 0.0

        node = self.tt.setdefault(state, NodeStats())
        if not node.expanded:
            # Expand using the env which already represents current state
            segments = env.extract_hotspots(int(self.cfg.max_hotspots_per_node))
            actions: List[Tuple[ActionKey, float]] = []
            for seg in segments:
                # Canonical hotspot key: "TV:start-end"
                hk = f"{seg['traffic_volume_id']}:{int(seg['start_bin'])}-{int(seg['end_bin'])}"
                hot_payload = segment_to_hotspot_payload(seg)
                key = (state, hk)
                if key not in self.cache:
                    proposals, f2f = env.proposals_for_hotspot(
                        hot_payload, int(self.cfg.k_proposals_per_hotspot)
                    )
                    ranked = [
                        (p, f2f, float(p.predicted_improvement.delta_objective_score)) for p in proposals
                    ]
                    self.cache[key] = ranked
                ranked = self.cache[key]
                for rank, (_p, _f2f, delta) in enumerate(
                    ranked[: int(self.cfg.k_proposals_per_hotspot)]
                ):
                    if not np.isfinite(float(delta)):
                        raise RuntimeError("Non-finite delta_objective_score in proposals cache")
                    actions.append(((hk, int(rank)), float(delta)))

            # No actions → terminal leaf with zero value
            if not actions:
                node.expanded = True
                return 0.0

            # Set priors via softmax of immediate deltas
            priors = softmax([d for (_ak, d) in actions])
            for (ak, _d), p in zip(actions, priors):
                node.children.setdefault(ak, ChildStats(P=float(p)))
            node.expanded = True

            # Root Dirichlet noise
            if depth == 0 and node.children:
                eps = float(self.cfg.dirichlet_epsilon)
                alpha = float(self.cfg.dirichlet_alpha)
                noise = np.random.default_rng().dirichlet([alpha] * len(node.children))
                for (cs, n) in zip(node.children.values(), noise):
                    cs.P = float((1.0 - eps) * cs.P + eps * float(n))

        # Selection: maximize Q + U
        total_N = sum(cs.N for cs in self.tt[state].children.values()) + 1
        best_ak, best_cs = max(
            self.tt[state].children.items(),
            key=lambda kv: kv[1].Q
            + float(self.cfg.puct_c) * kv[1].P * math.sqrt(total_N) / (1 + kv[1].N),
        )

        # One-step roll on selected edge
        r = self._roll(state, best_ak, env)
        best_cs.N += 1
        best_cs.W += float(r)
        return float(r)

    def _roll(self, state: RZPathKey, ak: ActionKey, env: RZSandbox) -> float:
        """Apply selected action to env and recurse; return accumulated reward."""
        hk, rank = ak
        key = (state, hk)
        if key not in self.cache:
            raise RuntimeError("Cache miss for selected action; expansion logic broken")
        ranked = self.cache[key]
        if int(rank) < 0 or int(rank) >= len(ranked):
            raise IndexError("Selected proposal rank out of range")
        prop, f2f, delta = ranked[int(rank)]
        if not np.isfinite(float(delta)):
            raise RuntimeError("Selected action has non-finite immediate reward")

        # Mutate sandbox by applying regulation derived from the proposal
        env.apply_proposal(prop, f2f)

        # Recurse into child state
        child_state: RZPathKey = tuple(list(state) + [ak])
        v = self._simulate(child_state, env)
        return float(delta) + float(v)


__all__ = ["MCTS", "softmax"]

