from __future__ import annotations

"""Shared types and configuration for Regulation Zero.

This module defines the core action and configuration dataclasses used by the
MCTS search and environment. The configuration includes search hyperparameters
and flow/hotspot extraction knobs to keep behavior consistent across nodes.

Notes
- Keep these models minimal and focused on the search layer. The underlying
  regen/evaluator carry their own configuration.
- We intentionally keep defaults here for convenience, but avoid default values
  in function signatures across other modules to surface errors early.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


# --- Core Search Types ---

# A canonical string representation of a hotspot, typically encoding its type and location.
# Example: "TV:ZID-ZTL" might represent a traffic volume hotspot between two sectors.
HotspotKey = str

# A unique identifier for a potential action. It's a tuple containing the key of the
# hotspot being addressed and the rank of the specific proposal for that hotspot.
# This allows for distinguishing between different regulatory options for the same hotspot.
ActionKey = Tuple[HotspotKey, int]  # (hotspot_key, proposal_rank)


@dataclass(frozen=True)
class RZAction:
    """Represents a concrete action taken in the search: a ranked proposal for a hotspot.

    This is an immutable data structure that captures all the necessary information
    about a single regulatory action considered by the MCTS agent. It links a hotspot
    to a specific mitigation strategy (identified by its rank) and includes the
    predicted outcome of applying that strategy.
    """

    hotspot_key: HotspotKey
    """The identifier for the hotspot this action addresses."""

    proposal_rank: int
    """The rank of this proposal among all generated for the hotspot. A lower rank
    typically signifies a more promising proposal based on initial heuristics."""

    delta_obj: float
    """The immediate change in the objective function predicted by the regeneration
    model if this action is taken. A positive value indicates an improvement."""

    control_tv_id: Optional[str] = None
    """An optional identifier for the controlled Traffic Volume (TV) associated with
    the proposal. Useful for debugging and analyzing the agent's decisions."""

    window_bins: Optional[Tuple[int, int]] = None
    """An optional tuple defining the time window (start_bin, end_bin_exclusive) for
    the regulation. This specifies the temporal scope of the action."""


# A sequence of ActionKeys that defines a complete regulatory plan from the root
# of the search tree to a leaf node. This represents a multi-step solution.
RZPathKey = Tuple[ActionKey, ...]


@dataclass
class RZConfig:
    """Configuration for the Regulation Zero search and environment.

    This class centralizes all hyperparameters and settings for the MCTS search,
    as well as parameters for the underlying flow clustering and hotspot
    extraction algorithms. This ensures that all components of the system
    operate with a consistent configuration.
    """

    # --- MCTS Search Hyperparameters ---

    max_depth: int = 3
    """The maximum depth of the MCTS search tree. Limits how many sequential
    actions the agent can plan ahead."""

    puct_c: float = 1.5
    """The exploration constant used in the PUCT (Polynomial Upper Confidence Trees)
    formula. Higher values encourage broader exploration of different actions."""

    dirichlet_alpha: float = 0.3
    """The concentration parameter (alpha) for the Dirichlet noise added to the
    root node's policy. This encourages exploration at the beginning of the search."""

    dirichlet_epsilon: float = 0.25
    """The weight of the Dirichlet noise. This value determines how much the original
    policy from the neural network is blended with the noise."""

    max_hotspots_per_node: int = 5
    """The maximum number of most significant hotspots to consider for expansion at
    any given node in the search tree."""

    k_proposals_per_hotspot: int = 8
    """The number of top-k action proposals to generate and evaluate for each
    selected hotspot."""

    num_simulations: int = 200
    """The total number of simulations (playouts) to run from the root node to
    build the search tree. More simulations generally lead to better decisions."""

    # --- Flow Clustering and Hotspot Extraction Parameters ---

    flows_threshold: Optional[float] = None
    """The similarity threshold for clustering flows. If None, the default value
    from the `compute_flows` function will be used."""

    flows_resolution: Optional[float] = None
    """The resolution parameter for flow clustering. If None, the default value
    from the `compute_flows` function will be used."""

    hotspot_threshold: float = 0.0
    """The minimum score or significance level for a traffic pattern to be
    considered a hotspot."""

    direction_mode: str = "coord_cosine"
    """The method used to determine the primary direction of a flow.
    'coord_cosine' is one of the possible modes."""

    # --- Guardrails ---

    fail_fast: bool = True
    """If True, the system will raise an exception and terminate immediately upon
    encountering an error. If False, it might attempt to continue, which can be
    useful in some debugging scenarios."""


__all__ = [
    "HotspotKey",
    "ActionKey",
    "RZAction",
    "RZPathKey",
    "RZConfig",
]

