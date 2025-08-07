from typing import Dict, List
from collections import defaultdict
from project_tailwind.casa.casa_flightlist import run_readapted_casa
from project_tailwind.impact_eval.operators.delay import batch_delay_operator
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.optimize.network_plan import NetworkPlan


class NetworkPlanMove:
    """
    NetworkPlanMove applies multiple regulations together, collecting all delays
    and applying the highest delay for each flight.
    """
    
    def __init__(
        self,
        network_plan: NetworkPlan,
        parser: RegulationParser,
        flight_list: FlightList,
        tvtw_indexer: TVTWIndexer,
    ):
        """
        Initialize NetworkPlanMove.
        
        Args:
            network_plan: NetworkPlan containing multiple regulations
            parser: RegulationParser for parsing regulations
            flight_list: FlightList for reference
            tvtw_indexer: TVTWIndexer for time-volume indexing
        """
        self.network_plan = network_plan
        self.parser = parser
        self.flight_list = flight_list
        self.tvtw_indexer = tvtw_indexer
    
    def __call__(self, state: FlightList) -> tuple[FlightList, float]:
        """
        Apply all regulations in the network plan to the current state.
        
        For each flight that gets multiple delay values from different regulations,
        the highest delay value is applied.
        
        Args:
            state: The current FlightList to be modified.
            
        Returns:
            Tuple of (modified FlightList, total delay applied)
        """
        if not self.network_plan.regulations:
            return state, 0.0
        
        # Dictionary to collect all delay values for each flight
        flight_delays_by_regulation: Dict[str, List[float]] = defaultdict(list)
        total_regulations_applied = 0
        
        # Process each regulation and collect delays
        for regulation in self.network_plan.regulations:
            # Find flights matching this regulation
            matched_flights = self.parser.parse(regulation)
            if not matched_flights:
                continue  # Skip if no flights match
            
            total_regulations_applied += 1
            
            # Calculate delays using C-CASA for this regulation
            delays = run_readapted_casa(
                flight_list=state,
                identifier_list=matched_flights,
                reference_location=regulation.location,
                tvtw_indexer=self.tvtw_indexer,
                hourly_rate=regulation.rate,
                active_time_windows=regulation.time_windows,
            )
            
            # Collect delays for each flight
            for flight_id, delay_minutes in delays.items():
                if delay_minutes > 0:
                    flight_delays_by_regulation[flight_id].append(delay_minutes)
        
        # Apply the highest delay for each flight
        final_flight_delays = {}
        for flight_id, delay_list in flight_delays_by_regulation.items():
            # Take the maximum delay for this flight
            max_delay = max(delay_list)
            final_flight_delays[flight_id] = int(max_delay)
        
        # Calculate total delay
        total_delay = sum(final_flight_delays.values())
        
        # Apply the final delays in batch
        if final_flight_delays:
            batch_delay_operator(
                flight_delays=final_flight_delays,
                state=state,
                indexer=self.tvtw_indexer,
            )
        
        return state, total_delay
    
    def get_regulation_summary(self) -> Dict:
        """
        Get a summary of the regulations in this network plan.
        
        Returns:
            Dictionary with regulation summary information
        """
        if not self.network_plan.regulations:
            return {"total_regulations": 0, "regulations": []}
        
        regulation_info = []
        for i, reg in enumerate(self.network_plan.regulations):
            regulation_info.append({
                "index": i,
                "location": reg.location,
                "rate": reg.rate,
                "time_windows": reg.time_windows,
                "filter": f"{reg.filter_type}_{reg.filter_value}",
                "raw_string": reg.raw_str
            })
        
        return {
            "total_regulations": len(self.network_plan.regulations),
            "regulations": regulation_info
        }