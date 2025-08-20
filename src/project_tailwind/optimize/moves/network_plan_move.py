from typing import Dict, List, Union, TYPE_CHECKING
from collections import defaultdict
from project_tailwind.optimize.fcfs.scheduler import assign_delays
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.delta_flight_list import DeltaFlightList
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.debug import console, is_debug_enabled

if TYPE_CHECKING:
    from project_tailwind.optimize.alns.pstate import ProblemState


class NetworkPlanMove:
    """
    NetworkPlanMove applies multiple regulations together, collecting all delays
    and applying the highest delay for each flight.
    """
    
    def __init__(
        self,
        network_plan: NetworkPlan,
        parser: RegulationParser,
        tvtw_indexer: TVTWIndexer,
    ):
        """
        Initialize NetworkPlanMove.
        
        Args:
            network_plan: NetworkPlan containing multiple regulations
            parser: RegulationParser for parsing regulations
            tvtw_indexer: TVTWIndexer for time-volume indexing
        """
        self.network_plan = network_plan
        self.parser = parser
        self.tvtw_indexer = tvtw_indexer
    
    def __call__(self, state: Union[FlightList, "ProblemState"]) -> Union[tuple[FlightList, float], tuple["ProblemState", float]]:
        """
        Apply all regulations in the network plan to the current state.
        
        For each flight that gets multiple delay values from different regulations,
        the highest delay value is applied.
        
        Args:
            state: The current FlightList or ProblemState to be modified.
            
        Returns:
            If FlightList input: Tuple of (modified FlightList, total delay applied)
            If ProblemState input: Tuple of (new ProblemState, total delay applied)
        """
        # Handle ProblemState input
        if hasattr(state, 'flight_list'):
            return self._apply_to_problem_state(state)
        
        # Handle FlightList input (original behavior)
        return self._apply_to_flight_list(state)
    
    def _apply_to_problem_state(self, state: "ProblemState") -> tuple["ProblemState", float]:
        """
        Apply regulations to a ProblemState and return a new ProblemState.
        
        Args:
            state: The ProblemState to modify
            
        Returns:
            Tuple containing the new ProblemState with modified FlightList and the total delay applied
        """
        # Build a non-mutating delta view and return a new ProblemState
        delta_view, total_delay = self.build_delta_view(state.flight_list)
        return state.with_flight_list(delta_view), total_delay
        
    def _apply_to_flight_list(self, state: FlightList) -> tuple[FlightList, float]:
        """
        Apply all regulations to a FlightList (original implementation).
        
        Args:
            state: The FlightList to modify
            
        Returns:
            Tuple of (modified FlightList, total delay applied)
        """
        if not self.network_plan.regulations:
            return state, 0.0
        # Return a non-mutating overlay view for incremental evaluation
        return self.build_delta_view(state)

    # --- New: expose a non-mutating evaluation helper ---------------------------
    def compute_final_delays(self, state: FlightList) -> Dict[str, int]:
        """
        Compute, but do not apply, the final per-flight delays implied by the
        current network plan for the given state.
        """
        if not self.network_plan.regulations:
            return {}

        flight_delays_by_regulation: Dict[str, List[float]] = defaultdict(list)

        for regulation in self.network_plan.regulations:
            matched_flights = self.parser.parse(regulation)
            if not matched_flights:
                continue

            delays = assign_delays(
                flight_list=state,
                identifier_list=matched_flights,
                reference_location=regulation.location,
                tvtw_indexer=self.tvtw_indexer,
                hourly_rate=regulation.rate,
                active_time_windows=regulation.time_windows,
            )

            for flight_id, delay_minutes in delays.items():
                if delay_minutes > 0:
                    flight_delays_by_regulation[flight_id].append(delay_minutes)

        # Reduce by max per flight
        final_flight_delays: Dict[str, int] = {}
        for flight_id, delay_list in flight_delays_by_regulation.items():
            final_flight_delays[flight_id] = int(max(delay_list))

        return final_flight_delays

    def build_delta_view(self, state: FlightList) -> tuple[DeltaFlightList, int]:
        """
        Construct a `DeltaFlightList` view for evaluation without mutating `state`.
        Returns the view and the total delay in minutes.
        """
        final_delays = self.compute_final_delays(state)
        total_delay = sum(final_delays.values())
        return DeltaFlightList(state, final_delays), total_delay
    
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