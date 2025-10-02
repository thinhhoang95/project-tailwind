"""Placeholder regulation history implementation."""

from __future__ import annotations

from typing import Dict, List, Optional

from .delta_view import DeltaOccupancyView


class RegulationHistory:
    """In-memory mapping between regulation ids and delta views."""

    def __init__(self) -> None:
        self._by_id: Dict[str, DeltaOccupancyView] = {}
        self._order: List[str] = []

    def record(self, regulation_id: str, view: DeltaOccupancyView) -> None:
        regulation_id = str(regulation_id)
        if regulation_id in self._by_id:
            self._by_id[regulation_id] = view
            return
        self._by_id[regulation_id] = view
        self._order.append(regulation_id)

    def get(self, regulation_id: str) -> Optional[DeltaOccupancyView]:
        return self._by_id.get(str(regulation_id))

    def list_ids(self) -> List[str]:
        return list(self._order)

    def __contains__(self, regulation_id: object) -> bool:
        return regulation_id in self._by_id

    def __len__(self) -> int:
        return len(self._order)


__all__ = ["RegulationHistory"]
