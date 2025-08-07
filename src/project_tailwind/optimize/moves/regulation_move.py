from typing import Dict, List
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.parser.regulation_parser import Regulation, RegulationParser
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.moves.network_plan_move import NetworkPlanMove


class RegulationMove:
    """
    RegulationMove is now a simplified wrapper around NetworkPlanMove
    for single regulation operations.
    """
    
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
        
        # Create a NetworkPlan with single regulation
        self.network_plan = NetworkPlan([self.regulation])
        
        # Create NetworkPlanMove to handle the actual processing
        self.network_plan_move = NetworkPlanMove(
            network_plan=self.network_plan,
            parser=parser,
            flight_list=flight_list,
            tvtw_indexer=tvtw_indexer
        )

    def __call__(self, state: FlightList) -> tuple[FlightList, float]:
        """
        Apply the regulation move to the current state (FlightList) in-place.

        Args:
            state: The current FlightList to be modified.

        Returns:
            Tuple of (modified FlightList, total delay applied)
        """
        return self.network_plan_move(state)
