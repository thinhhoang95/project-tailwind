"""Accelerated occupancy computation via pybind11.

Expose `compute_occupancy` mirroring the Python implementation signature:

    compute_occupancy(flight_list, delays, indexer, tv_filter=None) -> Dict[str, np.ndarray]

Usage:
    from occupance import compute_occupancy
"""

from ._occupancy import compute_occupancy  # type: ignore[attr-defined]

__all__ = ["compute_occupancy"]


