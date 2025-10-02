"""Placeholder regulation history implementation."""

from __future__ import annotations

from typing import Dict, List, Optional

from .delta_view import DeltaOccupancyView


class RegulationHistory:
    """In-memory mapping between regulation ids and delta views."""

    def __init__(self) -> None:
        """
        Initialize an empty RegulationHistory.
        
        Creates an internal mapping (`_by_id`) from regulation ID strings to `DeltaOccupancyView`
        and an insertion-order list (`_order`) of regulation IDs.
        """
        self._by_id: Dict[str, DeltaOccupancyView] = {}
        self._order: List[str] = []

    def record(self, regulation_id: str, view: DeltaOccupancyView) -> None:
        """
        Record or update the DeltaOccupancyView for a regulation ID, preserving insertion order.
        
        If the given regulation_id already exists, its associated view is replaced and the original insertion position is kept.
        If it is new, the ID is added to the internal order list and the view is stored.
        
        Parameters:
        	regulation_id (str): Identifier for the regulation; will be normalized to a string.
        	view (DeltaOccupancyView): The occupancy view to associate with the regulation ID.
        """
        regulation_id = str(regulation_id)
        if regulation_id in self._by_id:
            self._by_id[regulation_id] = view
            return
        self._by_id[regulation_id] = view
        self._order.append(regulation_id)

    def get(self, regulation_id: str) -> Optional[DeltaOccupancyView]:
        """
        Retrieve the DeltaOccupancyView associated with a regulation ID.
        
        Returns:
            The DeltaOccupancyView for the given regulation_id, or `None` if not present.
        """
        return self._by_id.get(str(regulation_id))

    def list_ids(self) -> List[str]:
        """
        Return the regulation IDs in insertion order.
        
        Returns:
            A list of regulation ID strings in the order they were recorded; modifying the returned list does not affect the stored order.
        """
        return list(self._order)

    def __contains__(self, regulation_id: object) -> bool:
        """
        Check whether a regulation id is present in the history.
        
        Parameters:
            regulation_id (object): The regulation identifier to test for membership.
        
        Returns:
            bool: `True` if the given regulation id is stored, `False` otherwise.
        """
        return regulation_id in self._by_id

    def __len__(self) -> int:
        """
        Return the number of stored regulation IDs.
        
        Returns:
            int: Number of regulation IDs currently tracked.
        """
        return len(self._order)


__all__ = ["RegulationHistory"]
