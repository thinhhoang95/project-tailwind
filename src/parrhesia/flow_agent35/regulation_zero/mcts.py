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
    """Computes the softmax of a list of numbers with a temperature parameter.

    Softmax converts a vector of values into a probability distribution. The
    temperature parameter `tau` controls the level of concentration of the
    distribution. A higher temperature results in a softer probability
    distribution (more uniform), while a lower temperature results in a harder
    distribution (closer to a one-hot encoding of the max value).

    Args:
        xs: A list of floats.
        tau: The temperature parameter. Must be positive. Defaults to 1.0.

    Returns:
        A list of floats representing the probability distribution.
    """
    if not xs:
        return []
    # Divide by temperature. Use a small epsilon to avoid division by zero.
    arr = np.asarray(xs, dtype=np.float64) / max(float(tau), 1e-6)
    # Subtract the maximum value for numerical stability to prevent overflow.
    arr -= arr.max()
    ex = np.exp(arr)
    Z = ex.sum()
    # Handle cases where sum of exponentials is zero or negative.
    if Z <= 0.0:
        # Return a uniform distribution if softmax is ill-defined.
        return [1.0 / float(len(xs)) for _ in xs]
    return [(float(x) / float(Z)) for x in ex]


class MCTS:
    """AlphaZero-style MCTS using PUCT selection and no rollouts.

    This MCTS implementation is adapted for the Regulation Zero problem. It uses
    the PUCT (Polynomial Upper Confidence bound applied to Trees) algorithm for
    action selection, which is known for its effectiveness in games like Go, as
    demonstrated by AlphaZero.

    Key characteristics:
    - No random rollouts: The value of a leaf node is considered to be 0. The
      value of an action is based on the immediate reward (delta) and the
      discounted value of the subsequent state.
    - Transposition Table: Stores the MCTS tree, including node and edge statistics.
    - Proposals Cache: Caches generated proposals for hotspots to avoid re-computation.
    - Dirichlet Noise: Added to root node priors to encourage exploration.
    """

    def __init__(self, env_factory, cfg: RZConfig):
        """Initializes the MCTS search.

        Args:
            env_factory: A callable that returns a fresh RZSandbox instance,
                         representing the root state of the environment. Each
                         simulation needs a clean fork.
            cfg: An RZConfig object containing MCTS hyperparameters.
        """
        # env_factory must return a fresh RZSandbox fork at root state per simulation
        self.env_factory = env_factory
        self.cfg = cfg
        # Transposition table to store the search tree (nodes and their stats).
        # Maps a state (RZPathKey) to NodeStats.
        self.tt: TranspositionTable = {}
        # Cache for proposals to avoid re-generating them for the same (state, hotspot) pair.
        # Maps (state, hotspot_key) to a list of ranked proposals.
        self.cache: ProposalsCache = {}

    def run(self) -> List[RZAction]:
        """Runs the MCTS algorithm for a configured number of simulations.

        After running the simulations, it extracts the best sequence of actions
        by greedily following the path with the highest visit counts from the root.

        Returns:
            A list of RZAction objects representing the best action sequence found.
        """
        sims = int(self.cfg.num_simulations)
        for _ in range(sims):
            # Each simulation starts from a fresh environment at the root state.
            env = self.env_factory()
            # A simulation traverses the tree from the root down to a leaf.
            self._simulate((), env)

        # Extract the principal variation: the sequence of actions with the highest visit counts.
        actions: List[RZAction] = []
        state: RZPathKey = ()
        depth = 0
        while depth < self.cfg.max_depth and state in self.tt and self.tt[state].children:
            # Select the child with the maximum visit count (N).
            best = max(self.tt[state].children.items(), key=lambda kv: kv[1].N)
            ak = best[0]
            # Append the chosen action to the list. Delta is not important here.
            actions.append(RZAction(hotspot_key=ak[0], proposal_rank=ak[1], delta_obj=0.0))
            # Move to the next state.
            state = tuple(list(state) + [ak])
            depth += 1
        return actions

    def _simulate(self, state: RZPathKey, env: RZSandbox) -> float:
        """Performs a single simulation from a given state.

        This method recursively traverses the tree using the PUCT selection
        strategy. When an unexpanded node is encountered, it is expanded, and
        the simulation may stop. The value from the simulation is backpropagated
        up the tree.

        Args:
            state: The current state, represented as a path of actions from the root.
            env: The RZSandbox environment, which is at the state corresponding
                 to the provided `state` path.

        Returns:
            The value of the state, which is the sum of immediate rewards along
            the simulation path plus the value of the leaf node.
        """
        depth = len(state)
        # Stop recursion if maximum search depth is reached.
        if depth >= int(self.cfg.max_depth):
            return 0.0

        # Get or create the node for the current state in the transposition table.
        node = self.tt.setdefault(state, NodeStats())

        # ======================================================================
        # Expansion Phase: If the node has not been expanded yet, expand it.
        # ======================================================================
        if not node.expanded:
            # Use the environment, which is already at the current state, for expansion.
            segments = env.extract_hotspots(int(self.cfg.max_hotspots_per_node))
            actions: List[Tuple[ActionKey, float]] = []
            for seg in segments:
                # Create a canonical key for the hotspot.
                hk = f"{seg['traffic_volume_id']}:{int(seg['start_bin'])}-{int(seg['end_bin'])}"
                hot_payload = segment_to_hotspot_payload(seg)
                key = (state, hk)

                # Generate and cache proposals for this hotspot if not already done.
                if key not in self.cache:
                    proposals, f2f = env.proposals_for_hotspot(
                        hot_payload, int(self.cfg.k_proposals_per_hotspot)
                    )
                    # Cache the ranked proposals along with their predicted improvements.
                    ranked = [
                        (p, f2f, float(p.predicted_improvement.delta_objective_score))
                        for p in proposals
                    ]
                    self.cache[key] = ranked
                ranked = self.cache[key]

                # Create actions from the top k proposals.
                for rank, (_p, _f2f, delta) in enumerate(
                    ranked[: int(self.cfg.k_proposals_per_hotspot)]
                ):
                    if not np.isfinite(float(delta)):
                        raise RuntimeError(
                            "Non-finite delta_objective_score in proposals cache"
                        )
                    actions.append(((hk, int(rank)), float(delta)))

            # If no actions can be generated, this is a terminal node. Mark as
            # expanded and return 0 value.
            if not actions:
                node.expanded = True
                return 0.0

            # Set prior probabilities for child nodes using softmax over immediate rewards (deltas).
            priors = softmax([d for (_ak, d) in actions])
            for (ak, _d), p in zip(actions, priors):
                node.children.setdefault(ak, ChildStats(P=float(p)))
            node.expanded = True

            # Add Dirichlet noise to the root node's priors to encourage exploration.
            # This is a key component of AlphaZero-style MCTS.
            if depth == 0 and node.children:
                eps = float(self.cfg.dirichlet_epsilon)
                alpha = float(self.cfg.dirichlet_alpha)
                noise = np.random.default_rng().dirichlet([alpha] * len(node.children))
                for (cs, n) in zip(node.children.values(), noise):
                    # Blend the original prior with the noise.
                    cs.P = float((1.0 - eps) * cs.P + eps * float(n))

        # ======================================================================
        # Selection Phase: Select the best action using the PUCT formula.
        # ======================================================================
        # The total number of visits to the children of the current node. Add 1
        # for the denominator to prevent division by zero and to handle the
        # initial state.
        total_N = sum(cs.N for cs in self.tt[state].children.values()) + 1
        # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        # Q(s,a) is the mean action value (exploitation term).
        # The second part is the exploration term (U), which favors actions with
        # high prior probability (P) and low visit count (N).
        best_ak, best_cs = max(
            self.tt[state].children.items(),
            key=lambda kv: kv[1].Q
            + float(self.cfg.puct_c) * kv[1].P * math.sqrt(total_N) / (1 + kv[1].N),
        )

        # ======================================================================
        # Simulation and Backup Phase (via recursion)
        # ======================================================================
        # Perform a "roll" which applies the action and recurses. The return
        # value `r` will be used to update statistics in the backup phase.
        r = self._roll(state, best_ak, env)
        # Backup: Update the statistics of the selected action.
        best_cs.N += 1
        # W is the total action value. Q is W/N.
        best_cs.W += float(r)
        return float(r)

    def _roll(self, state: RZPathKey, ak: ActionKey, env: RZSandbox) -> float:
        """Applies an action, recurses, and returns the accumulated reward.

        This is not a traditional MCTS rollout (i.e., simulating to the end of
        the episode). Instead, it's a one-step action application followed by a
        recursive call to `_simulate`, which continues the PUCT traversal.

        Args:
            state: The current state path.
            ak: The action key of the selected action.
            env: The sandbox environment to which the action will be applied.

        Returns:
            The accumulated reward from this point in the simulation, which is
            the immediate reward of the action plus the value of the subsequent
            state from the recursive simulation.
        """
        hk, rank = ak
        key = (state, hk)
        if key not in self.cache:
            # This should not happen if expansion logic is correct.
            raise RuntimeError("Cache miss for selected action; expansion logic broken")
        ranked = self.cache[key]
        if int(rank) < 0 or int(rank) >= len(ranked):
            raise IndexError("Selected proposal rank out of range")
        prop, f2f, delta = ranked[int(rank)]
        if not np.isfinite(float(delta)):
            raise RuntimeError("Selected action has non-finite immediate reward")

        # Mutate the sandbox environment by applying the regulation.
        env.apply_proposal(prop, f2f)

        # Recurse into the child state.
        child_state: RZPathKey = tuple(list(state) + [ak])
        v = self._simulate(child_state, env)

        # Return the immediate reward plus the value from the recursive call.
        # This value is then backed up the tree.
        return float(delta) + float(v)


__all__ = ["MCTS", "softmax"]

