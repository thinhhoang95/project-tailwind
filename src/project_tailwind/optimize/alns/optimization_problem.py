from __future__ import annotations

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.alns.pstate import ProblemState
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.moves.network_plan_move import NetworkPlanMove
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.parser.regulation_parser import RegulationParser


class OptimizationProblem:
    def __init__(
        self,
        base_flight_list: FlightList,
        regulation_parser: RegulationParser,
        tvtw_indexer: TVTWIndexer,
        objective_weights: dict,
        horizon_time_windows: int,
        base_traffic_volumes=None,
    ):
        self.base_flight_list = base_flight_list
        self.regulation_parser = regulation_parser
        self.tvtw_indexer = tvtw_indexer
        self.objective_weights = objective_weights
        self.horizon_time_windows = horizon_time_windows

        # Store the base traffic volumes
        self.base_traffic_volumes = base_traffic_volumes

    def create_initial_state(self) -> ProblemState:
        """
        Create the initial ProblemState with an empty NetworkPlan and base FlightList.
        """
        return ProblemState(
            network_plan=NetworkPlan(), 
            flight_list=self.base_flight_list,
            optimization_problem=self
        )
