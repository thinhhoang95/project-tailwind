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


# Stable key types
HotspotKey = str
ActionKey = Tuple[HotspotKey, int]  # (hotspot_key, proposal_rank)


@dataclass(frozen=True)
class RZAction:
    """Action representing a ranked proposal for a hotspot.

    - hotspot_key: canonical string key for the hotspot (e.g., "TV:start-end").
    - proposal_rank: integer rank within proposals enumerated for the hotspot
      in the current node expansion.
    - delta_obj: immediate predicted improvement (delta objective) returned by
      regen for this child.
    - control_tv_id: optional controlled TV of the chosen proposal for easier
      debugging/inspection.
    - window_bins: optional (start_bin, end_bin_exclusive) tuple for the window.
    """

    hotspot_key: HotspotKey
    proposal_rank: int
    delta_obj: float
    control_tv_id: Optional[str] = None
    window_bins: Optional[Tuple[int, int]] = None


# Canonical solution path key: tuple of (hotspot_key, rank)
RZPathKey = Tuple[ActionKey, ...]


@dataclass
class RZConfig:
    """Search and extraction configuration.

    Hyperparameters are kept explicit and simple. Values here are sensible
    starting points and may be tuned externally.
    """

    # MCTS search
    max_depth: int = 3
    puct_c: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_hotspots_per_node: int = 5
    k_proposals_per_hotspot: int = 8
    num_simulations: int = 200

    # Flow clustering and hotspot extraction params
    flows_threshold: Optional[float] = None  # if None, compute_flows default
    flows_resolution: Optional[float] = None  # if None, compute_flows default
    hotspot_threshold: float = 0.0
    direction_mode: str = "coord_cosine"

    # Guardrails
    fail_fast: bool = True


__all__ = [
    "HotspotKey",
    "ActionKey",
    "RZAction",
    "RZPathKey",
    "RZConfig",
]

