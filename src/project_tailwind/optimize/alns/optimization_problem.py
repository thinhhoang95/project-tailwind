from typing import List
import geopandas as gpd

from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.flight_list import FlightList

class OptimizationProblem:
    def __init__(self, traffic_volumes_gdf: gpd.GeoDataFrame, flight_list: FlightList, horizon_time_windows: int):
        self.flight_list = flight_list
        self.horizon_time_windows = horizon_time_windows
        self.traffic_volumes_gdf = traffic_volumes_gdf

        self.network_evaluator = NetworkEvaluator(
            traffic_volumes_gdf=self.traffic_volumes_gdf,
            flight_list=self.flight_list
        )

    def objective(self) -> float:
        """
        Computes the objective function value.
        """
        metrics = self.network_evaluator.compute_horizon_metrics(self.horizon_time_windows)
        # Placeholder for a real objective function.
        # For now, let's use a combination of z_max and z_sum.
        return metrics['z_max'] + metrics['z_sum']

