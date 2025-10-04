from __future__ import annotations

"""Caches for MCTS node statistics and proposals.

This provides small dataclasses for child/node stats and two caches:
- Transposition table keyed by canonical path to store child priors and visit stats.
- Proposals cache keyed by (state_path, hotspot_key) to avoid recomputing regen
  proposals within the same node context.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .types import RZPathKey, ActionKey


@dataclass
class ChildStats:
    """Per-edge statistics for PUCT search.

    P: prior probability
    N: visit count
    W: total accumulated value (sum of rewards from traversals)
    Q: mean value (computed property)
    """

    P: float
    N: int = 0
    W: float = 0.0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


@dataclass
class NodeStats:
    """Aggregated stats for a node keyed by RZPathKey."""

    children: Dict[ActionKey, ChildStats] = field(default_factory=dict)
    expanded: bool = False


# Transposition table: state -> NodeStats
TranspositionTable = Dict[RZPathKey, NodeStats]

# Proposals cache: (state, hotspot_key) -> list of tuples to reconstruct apply
ProposalsCache = Dict[Tuple[RZPathKey, str], List[Any]]


__all__ = [
    "ChildStats",
    "NodeStats",
    "TranspositionTable",
    "ProposalsCache",
]

