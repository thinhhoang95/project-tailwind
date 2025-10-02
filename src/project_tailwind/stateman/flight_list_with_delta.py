"""FlightList extension providing incremental delta application."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from project_tailwind.optimize.eval.flight_list import FlightList

from .delta_view import DeltaOccupancyView


class FlightListWithDelta(FlightList):
    """FlightList variant that can ingest :class:`DeltaOccupancyView` instances."""

    def __init__(self, occupancy_file_path: str, tvtw_indexer_path: str):
        super().__init__(occupancy_file_path, tvtw_indexer_path)
        self.applied_regulations: List[str] = []
        self.delay_histogram: Dict[int, int] = {}
        self.total_delay_assigned_min: int = 0
        self.num_delayed_flights: int = 0
        self.num_regulations: int = 0

        self._delta_aggregate = np.zeros(self.num_tvtws, dtype=np.int64)
        self._applied_views: List[DeltaOccupancyView] = []
        self._delay_by_flight: Dict[str, int] = {}

    def step_by_delay(self, *views: DeltaOccupancyView, finalize: bool = True) -> None:
        """Apply one or more delta views to the flight list."""

        if not views:
            if finalize:
                self.finalize_occupancy_updates()
            return

        for view in views:
            if not isinstance(view, DeltaOccupancyView):
                raise TypeError("All arguments to step_by_delay must be DeltaOccupancyView instances")
            self._apply_single_view(view)

        if finalize:
            self.finalize_occupancy_updates()

    def get_delta_aggregate(self) -> np.ndarray:
        """Return the dense aggregate delta vector accumulated so far."""

        return self._delta_aggregate.copy()

    # --- internal helpers --------------------------------------------------------
    def _apply_single_view(self, view: DeltaOccupancyView) -> None:
        dense_delta = view.as_dense_delta(np.int64)
        if dense_delta.size != self._delta_aggregate.size:
            raise ValueError("Delta size does not match flight list occupancy dimensions")
        self._delta_aggregate += dense_delta
        self._applied_views.append(view)
        self.num_regulations = len(self._applied_views)

        if view.regulation_id:
            self.applied_regulations.append(view.regulation_id)

        for flight_id, intervals in view.per_flight_new_intervals.items():
            if flight_id not in self.flight_id_to_row:
                continue
            self.clear_flight_occupancy(flight_id)
            column_indices = np.array([int(iv["tvtw_index"]) for iv in intervals], dtype=np.int64)
            if column_indices.size:
                self.add_flight_occupancy(flight_id, column_indices)
            canonical_intervals = [
                {
                    "tvtw_index": int(iv["tvtw_index"]),
                    "entry_time_s": float(iv["entry_time_s"]),
                    "exit_time_s": float(iv["exit_time_s"]),
                }
                for iv in intervals
            ]
            self.flight_metadata[flight_id]["occupancy_intervals"] = canonical_intervals
            if flight_id in self.flight_data:
                self.flight_data[flight_id]["occupancy_intervals"] = [dict(iv) for iv in canonical_intervals]
            self._flight_tv_sequence_cache.pop(flight_id, None)

        self._update_delay_metrics(view)

    def _update_delay_metrics(self, view: DeltaOccupancyView) -> None:
        for flight_id, delay_minutes in view.delays.nonzero_items():
            if flight_id not in self.flight_id_to_row:
                continue
            previous = self._delay_by_flight.get(flight_id, 0)
            if delay_minutes == previous:
                continue
            if previous:
                self._decrement_histogram(previous)
            self._delay_by_flight[flight_id] = delay_minutes
            self.delay_histogram[delay_minutes] = self.delay_histogram.get(delay_minutes, 0) + 1
            self.total_delay_assigned_min += delay_minutes - previous

        self.num_delayed_flights = len(self._delay_by_flight)

    def _decrement_histogram(self, delay_minutes: int) -> None:
        count = self.delay_histogram.get(delay_minutes)
        if not count:
            return
        if count <= 1:
            self.delay_histogram.pop(delay_minutes, None)
        else:
            self.delay_histogram[delay_minutes] = count - 1
