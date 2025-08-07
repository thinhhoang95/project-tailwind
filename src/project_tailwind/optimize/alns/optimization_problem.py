from typing import List
import geopandas as gpd

from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.alns.pstate import ProblemState


class OptimizationProblem:
    def __init__(
        self,
        traffic_volumes_gdf: gpd.GeoDataFrame,
        flight_list: FlightList,
        horizon_time_windows: int,
        objective_weights: dict
    ):
        self.initial_flight_list = flight_list
        self.horizon_time_windows = horizon_time_windows
        self.traffic_volumes_gdf = traffic_volumes_gdf

        self.weight_z95 = objective_weights['z_95']
        self.weight_zsum = objective_weights['z_sum']
        self.weight_delay_min = objective_weights['delay']

        # Create NetworkEvaluator as an instance variable
        self.network_evaluator = NetworkEvaluator(
            traffic_volumes_gdf=self.traffic_volumes_gdf,
            flight_list=self.initial_flight_list
        )

    def create_initial_state(self) -> ProblemState:
        """
        Create the initial ProblemState with the original flight_list.
        """
        return ProblemState(flight_list=self.initial_flight_list)
    
    def objective(self, state: ProblemState, debug: bool = False) -> float:
        """
        Computes the objective function value for a given state.
        """
        # Update the existing network evaluator with the flight_list from the current state
        self.network_evaluator.update_flight_list(state.flight_list)
        
        metrics = self.network_evaluator.compute_horizon_metrics(
            self.horizon_time_windows
        )

        if debug:
            # For debugging
            print("-" * 60)
            print("NETWORK EVALUATOR METRICS")
            for key, value in metrics.items():
                print(f"{key}: {value}")
            print("-" * 60)

        # The objective function is a weighted sum of the metrics
        return self.weight_z95 * metrics["z_95"] + self.weight_zsum * metrics["z_sum"] + \
            self.weight_delay_min * metrics["total_delay_seconds"] / 60.0
