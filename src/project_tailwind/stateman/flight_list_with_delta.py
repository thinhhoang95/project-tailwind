"""FlightList extension providing incremental delta application."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from project_tailwind.optimize.eval.flight_list import FlightList

from .delta_view import DeltaOccupancyView


class FlightListWithDelta(FlightList):
    """FlightList variant that can ingest :class:`DeltaOccupancyView` instances."""

    def __init__(self, occupancy_file_path: str, tvtw_indexer_path: str):
        """
        Initialize a FlightListWithDelta and prepare data structures for applying and tracking incremental occupancy deltas.
        
        Parameters:
            occupancy_file_path (str): Path to the occupancy data file used by the base FlightList.
            tvtw_indexer_path (str): Path to the TV/TW indexer used to size internal arrays.
        
        Attributes:
            applied_regulations (List[str]): Regulation identifiers that have been applied, in order.
            delay_histogram (Dict[int, int]): Mapping from delay minutes to count of flights with that delay.
            total_delay_assigned_min (int): Total minutes of delay assigned by applied deltas.
            num_delayed_flights (int): Number of flights currently recorded as delayed.
            num_regulations (int): Number of delta views applied.
            _delta_aggregate (np.ndarray): Dense int64 array aggregating delta values across all tvtws.
            _applied_views (List[DeltaOccupancyView]): List of applied DeltaOccupancyView instances.
            _delay_by_flight (Dict[str, int]): Per-flight current delay in minutes.
        """
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
        """
        Apply one or more DeltaOccupancyView instances to the flight list and optionally finalize occupancy updates.
        
        If no views are provided and `finalize` is True, calls `finalize_occupancy_updates` and returns. Each supplied view is validated to be a DeltaOccupancyView and applied via the class's incremental update logic; after all views are applied, `finalize_occupancy_updates` is called when `finalize` is True.
        
        Parameters:
        	views (DeltaOccupancyView): One or more delta views describing occupancy and delay changes.
        	finalize (bool): If True, call `finalize_occupancy_updates` after applying the views (or immediately if no views).
        
        Raises:
        	TypeError: If any positional argument in `views` is not an instance of DeltaOccupancyView.
        """

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
        """
        Get a copy of the dense aggregate delta vector accumulated from applied views.
        
        Returns:
            np.ndarray: A copy of the internal delta aggregate array (dtype int64), indexed by tvtw.
        """

        return self._delta_aggregate.copy()

    # --- internal helpers --------------------------------------------------------
    def _apply_single_view(self, view: DeltaOccupancyView) -> None:
        """
        Apply a DeltaOccupancyView to the flight list, merging its delta into the aggregate and updating per‑flight occupancy and delay state.
        
        Parameters:
            view (DeltaOccupancyView): Delta view containing a dense delta vector, per-flight occupancy intervals, optional regulation identifier, and delay updates.
        
        Raises:
            ValueError: If the view's dense delta length does not match the flight list occupancy dimensions.
        
        Behavior:
            - Increments the internal aggregate delta with the view's dense delta and records the view as applied.
            - If the view includes a regulation identifier, records it in the applied regulations list and updates the applied count.
            - For each flight listed in the view that exists in the flight list:
                - Replaces that flight's occupancy columns and canonical occupancy_intervals metadata with the view's intervals.
                - Clears any cached per-flight TV sequence data.
            - Updates per-flight delay state and overall delay metrics based on the view's delay entries.
        """
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
        """
        Update the flight-level delay tracking and aggregate delay metrics using delays from the given view.
        
        Processes each (flight_id, delay_minutes) pair from view.delays.nonzero_items():
        - Ignores flights not present in self.flight_id_to_row.
        - If the delay for a flight changed, updates the per-flight delay map, adjusts the delay_histogram (decrementing the count for the previous delay when present and incrementing the count for the new delay), and adds the difference to total_delay_assigned_min.
        After processing, sets num_delayed_flights to the number of flights currently tracked in _delay_by_flight.
        
        Parameters:
            view (DeltaOccupancyView): An occupancy delta view whose `delays.nonzero_items()` yields per-flight delay minutes.
        """
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
        """
        Decrease the count for a specific delay value in the delay histogram.
        
        If there is no entry for the given delay value this does nothing. If the
        entry's count is 1 the key is removed from the histogram; otherwise the
        count is decremented by 1.
        
        Parameters:
            delay_minutes (int): Delay value in minutes whose histogram count should be decremented.
        """
        count = self.delay_histogram.get(delay_minutes)
        if not count:
            return
        if count <= 1:
            self.delay_histogram.pop(delay_minutes, None)
        else:
            self.delay_histogram[delay_minutes] = count - 1
