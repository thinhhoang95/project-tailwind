"""Utilities to keep shared caches in sync after mutating the FlightList."""

from __future__ import annotations

from typing import Any, Optional

try:
    import parrhesia.api.resources as _parrhesia_resources
except Exception:  # pragma: no cover - optional dependency at runtime
    _parrhesia_resources = None  # type: ignore[assignment]


def refresh_after_state_update(
    resources: Any,
    *,
    airspace_wrapper: Optional[Any] = None,
    count_wrapper: Optional[Any] = None,
    query_wrapper: Optional[Any] = None,
) -> None:
    """
    Refresh shared caches and register global resources so subsequent queries observe the updated flight list state.
    
    Parameters:
        resources (Any): Object exposing `flight_list` and `indexer` representing the current flight data.
        airspace_wrapper (Optional[Any]): Optional wrapper; if it exposes `invalidate_caches`, those caches will be invalidated.
        count_wrapper (Optional[Any]): Optional wrapper; if it exposes `invalidate_caches`, those caches will be invalidated.
        query_wrapper (Optional[Any]): Optional wrapper; if it exposes `refresh_flight_list`, it will be invoked with `resources.flight_list`.
    """
    if airspace_wrapper is not None and hasattr(airspace_wrapper, "invalidate_caches"):
        airspace_wrapper.invalidate_caches()
    if count_wrapper is not None and hasattr(count_wrapper, "invalidate_caches"):
        count_wrapper.invalidate_caches()
    if query_wrapper is not None and hasattr(query_wrapper, "refresh_flight_list"):
        query_wrapper.refresh_flight_list(resources.flight_list)

    if _parrhesia_resources is not None:
        try:
            _parrhesia_resources.set_global_resources(resources.indexer, resources.flight_list)
        except Exception as exc:  # pragma: no cover - defensive logging path
            print(f"Warning: failed to register parrhesia resources: {exc}")
