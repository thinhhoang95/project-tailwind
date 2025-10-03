from __future__ import annotations

from typing import TYPE_CHECKING, Any

from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.debug import console, is_debug_enabled

if TYPE_CHECKING:
    from project_tailwind.optimize.alns.optimization_problem import OptimizationProblem
    from project_tailwind.optimize.eval.flight_list import FlightList
    from project_tailwind.optimize.eval.delta_flight_list import DeltaFlightList


class ProblemState:
    """
    Container for the optimization problem state used by ALNS.
    Represents a solution candidate (NetworkPlan + FlightList) and the context for its evaluation.
    """

    def __init__(
        self,
        network_plan: NetworkPlan,
        flight_list: "FlightList",
        optimization_problem: "OptimizationProblem",
        aux: dict[str, Any] | None = None,
    ):
        self.network_plan = network_plan
        self.flight_list = flight_list
        self.optimization_problem = optimization_problem
        self.aux = aux or {}

    def copy(self) -> "ProblemState":
        """
        Return a deep copy of the state, as required by ALNS move semantics.
        """
        # Make copying cheap by keeping a reference to the shared base flight list
        # and copying only the lightweight network plan and aux.
        return ProblemState(
            network_plan=self.network_plan.copy() if hasattr(self.network_plan, 'copy') else self.network_plan,
            flight_list=self.flight_list,
            optimization_problem=self.optimization_problem,
            aux=self.aux.copy(),
        )

    def objective(self) -> float:
        """
        Computes the objective function value for this state.
        This method evaluates the network plan from scratch.
        """
        from project_tailwind.optimize.moves.network_plan_move import NetworkPlanMove
        from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator

        # Build a delta view to avoid deep-copying the base flight list
        move = NetworkPlanMove(
            network_plan=self.network_plan,
            parser=self.optimization_problem.regulation_parser,
            tvtw_indexer=self.optimization_problem.tvtw_indexer,
        )
        # if is_debug_enabled():
        #     console.print(
        #         f"[debug] Objective: building delta view for {len(self.network_plan.regulations)} regulations",
        #         style="debug",
        #     )
        delta_view, total_delay = move.build_delta_view(self.flight_list)

        # Use NetworkEvaluator against the delta view
        network_evaluator = NetworkEvaluator(
            traffic_volumes_gdf=self.optimization_problem.base_traffic_volumes,
            flight_list=delta_view,
        )
        metrics = network_evaluator.compute_horizon_metrics(
            self.optimization_problem.horizon_time_windows
        )

        # The objective function is a weighted sum of the metrics
        score = (
            self.optimization_problem.objective_weights["z_95"] * metrics["z_95"]
            + self.optimization_problem.objective_weights["z_sum"] * metrics["z_sum"]
            + self.optimization_problem.objective_weights["delay"] * total_delay
        )
        if is_debug_enabled():
            console.print(
                (
                    f"[debug] n_reg = {len(self.network_plan.regulations)}, z_95={metrics['z_95']:.3f}, "
                    f"z_sum={metrics['z_sum']:.3f}, delay={total_delay} → score={score:.3f}"
                ),
                style="debug",
            )
        return score

    def with_flight_list(self, new_flight_list: "FlightList") -> "ProblemState":
        """
        Create a new state with updated flight_list.
        
        Args:
            new_flight_list: The new FlightList to use
            
        Returns:
            New ProblemState with updated flight_list
        """
        return ProblemState(
            network_plan=self.network_plan,
            flight_list=new_flight_list,
            optimization_problem=self.optimization_problem,
            aux=self.aux.copy(),
        )

    def __repr__(self) -> str:
        return (
            f"ProblemState(network_plan={self.network_plan}, "
            f"flight_list_flights={self.flight_list.num_flights}, "
            f"aux_keys={list(self.aux.keys())})"
        )
