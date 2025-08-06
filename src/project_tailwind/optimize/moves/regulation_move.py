from typing import Dict, List
from project_tailwind.casa.casa_flightlist import run_readapted_casa
from project_tailwind.impact_eval.operators.delay import batch_delay_operator
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
    ):
        self.regulation = Regulation(regulation_str)
        self.parser = parser
        self.flight_list = flight_list
        self.tvtw_indexer = tvtw_indexer

    def __call__(self, state: FlightList) -> FlightList:
        """
        Apply the regulation move to the current state (FlightList) in-place.

        Args:
            state: The current FlightList to be modified.

        Returns:
            The modified FlightList.
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

        # Debug: Print delay allocations for debugging
        # if delays:
        #     # Sort flights by delay (most delayed first)
        #     sorted_delays = sorted(delays.items(), key=lambda x: x[1], reverse=True)
            
        #     print(f"\n=== Regulation Move Debug: {self.regulation.location} ===")
        #     print(f"Regulation: {self.regulation}")
        #     print(f"Total matched flights: {len(matched_flights)}")
        #     print(f"Flights with delays: {len([d for d in delays.values() if d > 0])}")
            
        #     # Show top 10 most delayed flights
        #     top_delayed = sorted_delays[:10]
        #     print(f"\nTop {len(top_delayed)} most delayed flights:")
        #     for i, (flight_id, delay_min) in enumerate(top_delayed, 1):
        #         print(f"  {i:2d}. Flight {flight_id}: {delay_min:.1f} minutes")
            
        #     # Show summary statistics
        #     delay_values = [d for d in delays.values() if d > 0]
        #     if delay_values:
        #         total_delay = sum(delay_values)
        #         avg_delay = total_delay / len(delay_values)
        #         max_delay = max(delay_values)
        #         print(f"\nDelay Statistics:")
        #         print(f"  Total delay applied: {total_delay:.1f} minutes")
        #         print(f"  Average delay per affected flight: {avg_delay:.1f} minutes")
        #         print(f"  Maximum delay: {max_delay:.1f} minutes")
        #     print("=" * 50)

        # Apply delays to the matched flights in batch for efficiency
        flight_delays = {
            flight_id: int(delay_minutes) 
            for flight_id, delay_minutes in delays.items() 
            if delay_minutes > 0
        }
        
        if flight_delays:
            batch_delay_operator(
                flight_delays=flight_delays,
                state=state,
                indexer=self.tvtw_indexer,
            )
        
        return state
