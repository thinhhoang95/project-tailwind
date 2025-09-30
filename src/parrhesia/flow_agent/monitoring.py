"""Monitoring hook protocol for optional global statistics aggregators."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence


class GlobalStatsHook(Protocol):
    """Lightweight protocol that consumers can implement to observe agent events."""

    def on_outer_start(self, outer_index: int, limits: Optional[Mapping[str, Any]] = None) -> None:
        """Called once when a new outer iteration begins."""

    def on_outer_end(self, outer_index: int, meta: Optional[Mapping[str, Any]] = None) -> None:
        """Called once when an outer iteration completes."""

    def on_candidate_scored(
        self,
        candidate_id: str,
        objective: float,
        delta_j: float,
        meta: Mapping[str, Any],
    ) -> None:
        """Called whenever a candidate objective is evaluated."""

    def on_action(self, action_type: str, count: int = 1) -> None:
        """Incremental notification for executed actions."""

    def on_plan_committed(
        self,
        plan: Sequence[Any],
        delta_j: float,
        flow_to_flights: Mapping[str, Sequence[str]] | None,
        entrants_by_flow: Mapping[str, Any] | None,
        meta: Mapping[str, Any],
    ) -> None:
        """Called when a plan (regulation) is committed."""

    def on_limit_hit(self, kind: str, meta: Optional[Mapping[str, Any]] = None) -> None:
        """Called when a termination constraint is encountered."""

