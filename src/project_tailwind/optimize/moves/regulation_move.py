from typing import Dict, List
from project_tailwind.casa.casa_flightlist import run_readapted_casa
from project_tailwind.impact_eval.operators.delay import delay_operator
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.parser.regulation_parser import Regulation, RegulationParser


class RegulationMove:
    def __init__(
        self,
        regulation_str: str,
        parser: RegulationParser,
        flight_list: FlightList,
        tvtw_indexer: TVTWIndexer,
        so6_occupancy_path: str,
    ):
        self.regulation = Regulation(regulation_str)
        self.parser = parser
        self.flight_list = flight_list
        self.tvtw_indexer = tvtw_indexer
        self.so6_occupancy_path = so6_occupancy_path

    def __call__(self, state: FlightList) -> FlightList:
        """
        Apply the regulation move to the current state (FlightList).

        Args:
            state: The current FlightList.

        Returns:
            A new FlightList with the regulation applied.
        """
        # Find flights matching the regulation
        matched_flights = self.parser.parse(self.regulation)
        if not matched_flights:
            return state  # No change if no flights are matched

        # Calculate delays using C-CASA
        delays = run_readapted_casa(
            flight_list=state,
            identifier_list=matched_flights,
            reference_location=self.regulation.location,
            tvtw_indexer=self.tvtw_indexer,
            hourly_rate=self.regulation.rate,
            active_time_windows=self.regulation.time_windows,
        )

        # Create a deep copy to avoid modifying the original state directly
        new_flight_list = state.copy()

        # Apply delays to the matched flights
        for flight_id, delay_minutes in delays.items():
            if delay_minutes > 0:
                original_occupancy = state.get_occupancy_vector(flight_id)
                if original_occupancy is not None:
                    new_occupancy = delay_operator(
                        original_occupancy,
                        int(delay_minutes),
                        self.tvtw_indexer,
                    )
                    new_flight_list.update_flight(flight_id, {"occupancy_vector": new_occupancy})
        
        return new_flight_list

