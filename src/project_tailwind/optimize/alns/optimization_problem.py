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
    ):
        self.base_flight_list = base_flight_list
        self.regulation_parser = regulation_parser
        self.tvtw_indexer = tvtw_indexer
        self.objective_weights = objective_weights
        self.horizon_time_windows = horizon_time_windows

        # Memoize the base traffic volumes
        self.base_traffic_volumes = self.tvtw_indexer.get_traffic_volumes_gdf(
            self.base_flight_list, aggregate=True
        )

    def create_initial_state(self) -> ProblemState:
        """
        Create the initial ProblemState with an empty NetworkPlan.
        """
        return ProblemState(network_plan=NetworkPlan(), optimization_problem=self)

    def objective(self, state: ProblemState, debug: bool = False) -> float:
        """
        Computes the objective function value for a given state.
        This method re-evaluates the network plan from scratch.
        """
        # Get the network_plan from the state
        network_plan = state.network_plan

        # Create a deep copy of the base flight_list for simulation
        sim_flight_list = self.base_flight_list.copy()

        # Instantiate and apply a NetworkPlanMove to the copied flight list
        move = NetworkPlanMove(
            network_plan=network_plan,
            parser=self.regulation_parser,
            flight_list=sim_flight_list,
            tvtw_indexer=self.tvtw_indexer,
        )
        modified_flight_list, total_delay = move(sim_flight_list)

        # Use NetworkEvaluator to compute metrics
        network_evaluator = NetworkEvaluator(
            traffic_volumes_gdf=self.base_traffic_volumes,
            flight_list=modified_flight_list,
        )
        metrics = network_evaluator.compute_horizon_metrics(self.horizon_time_windows)

        if debug:
            print("-" * 60)
            print("NETWORK EVALUATOR METRICS")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            print(f"Total Delay (minutes): {total_delay}")
            print("-" * 60)

        # The objective function is a weighted sum of the metrics
        score = (
            self.objective_weights["z_95"] * metrics["z_95"]
            + self.objective_weights["z_sum"] * metrics["z_sum"]
            + self.objective_weights["delay"] * total_delay
        )
        return score
