"""Typed helpers for the ``stateman`` package."""

from __future__ import annotations

from typing import Dict, Literal, Sequence, TypedDict


class OccupancyIntervalDict(TypedDict):
    """Dictionary shape used for occupancy intervals in flight metadata."""

    tvtw_index: int
    entry_time_s: float
    exit_time_s: float


DelayMergePolicy = Literal["max", "sum", "overwrite"]
DelayMapping = Dict[str, int]
OccupancyIntervalList = Sequence[OccupancyIntervalDict]
