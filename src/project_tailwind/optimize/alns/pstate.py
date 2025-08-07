from __future__ import annotations

from typing import Any

from project_tailwind.optimize.alns.optimization_problem import OptimizationProblem
from project_tailwind.optimize.network_plan import NetworkPlan


class ProblemState:
    """
    Container for the optimization problem state used by ALNS.
    Represents a solution candidate (NetworkPlan) and the context for its evaluation.
    """

    def __init__(
        self,
        network_plan: NetworkPlan,
        optimization_problem: "OptimizationProblem",
        aux: dict[str, Any] | None = None,
    ):
        self.network_plan = network_plan
        self.optimization_problem = optimization_problem
        self.aux = aux or {}

    def copy(self) -> "ProblemState":
        """
        Return a deep copy of the state, as required by ALNS move semantics.
        """
        return ProblemState(
            network_plan=self.network_plan.copy(),
            optimization_problem=self.optimization_problem,
            aux=self.aux.copy(),
        )

    def objective(self) -> float:
        """
        Delegates objective function computation to the optimization problem.
        """
        return self.optimization_problem.objective(self)

    def __repr__(self) -> str:
        return (
            f"ProblemState(network_plan={self.network_plan}, "
            f"aux_keys={list(self.aux.keys())})"
        )
