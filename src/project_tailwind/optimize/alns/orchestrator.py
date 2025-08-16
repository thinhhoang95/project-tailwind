import random
import sys
from pathlib import Path

import numpy as np
from alns import ALNS
from alns.accept import HillClimbing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.alns.optimization_problem import OptimizationProblem
from project_tailwind.optimize.alns.pstate import ProblemState
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.regulation import Regulation
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
import geopandas as gpd
from project_tailwind.optimize.debug import console, is_debug_enabled, set_debug


DATA_DIR = "output"
TRAFFIC_VOLUMES_PATH = "D:/project-cirrus/cases/traffic_volumes_simplified.geojson"
OBJECTIVE_WEIGHTS = {"z_95": 1.0, "z_sum": 1.0, "delay": 0.01}
HORIZON_TIME_WINDOWS = 100
DEFAULT_REGULATIONS = [
    "TV_MASH5RL IC__ 10 36-45",
    "TV_MASH5RL IC__ 8 41-45",
    "TV_EDMTEG IC__ 60 36-38",
    "TV_LFPPLW1 IC_LFPG_EGLL 30 48-50",
    "TV_EDMTEG IC_LF>_EG> 40 47-53",
    "TV_EBBUELS1 IC_LFP>_> 25 50-52",
]


# -- ALNS Operators --
def remove_regulation(state: ProblemState, rnd_state: np.random.RandomState) -> ProblemState:
    """
    Removes a random regulation from the network plan.
    """
    new_state = state.copy()
    if not new_state.network_plan.regulations:
        return new_state

    idx = rnd_state.randint(len(new_state.network_plan.regulations))
    removed = new_state.network_plan.regulations[idx]
    new_state.network_plan.remove_regulation(idx)

    if is_debug_enabled():
        try:
            parser = state.optimization_problem.regulation_parser
            explanation = parser.explain_regulation(removed)
        except Exception:
            explanation = removed.raw_str
        console.print(
            f"[debug] Destroy operator: remove regulation {idx}: {explanation}",
            style="debug",
        )
    return new_state


def add_regulation(state: ProblemState, rnd_state: np.random.RandomState) -> ProblemState:
    """
    Adds a random regulation to the network plan.
    """
    new_state = state.copy()
    regulation_str = rnd_state.choice(DEFAULT_REGULATIONS)
    regulation = Regulation(regulation_str)
    new_state.network_plan.add_regulation(regulation)

    if is_debug_enabled():
        try:
            parser = state.optimization_problem.regulation_parser
            explanation = parser.explain_regulation(regulation)
        except Exception:
            explanation = regulation.raw_str
        console.print(
            f"[debug] Repair operator: add regulation → {explanation}",
            style="debug",
        )
    return new_state


class AlnsOrchestrator:
    """
    Sets up and runs the ALNS optimization.
    """

    def __init__(self, data_dir: Path, debug: bool = False):
        self.data_dir = data_dir
        self.flight_list = None
        self.tvtw_indexer = None
        self.regulation_parser = None
        self.optimization_problem = None
        self.traffic_volumes = None
        # Configure debug printing
        # set_debug(debug or is_debug_enabled())
        set_debug(True)

    def setup(self):
        """
        Load all necessary data and initialize components.
        """
        print("1. Loading TVTW Indexer...")
        self.tvtw_indexer = TVTWIndexer.load(self.data_dir + "/tvtw_indexer.json")
        print(f"   OK Loaded indexer with {len(self.tvtw_indexer._tv_id_to_idx)} traffic volumes")

        print("2. Loading Flight List...")
        self.flight_list = FlightList(
            occupancy_file_path=self.data_dir + "/so6_occupancy_matrix_with_times.json",
            tvtw_indexer_path=self.data_dir + "/tvtw_indexer.json",
        )
        print(f"   OK Loaded {self.flight_list.num_flights} flights")

        print("3. Initializing Regulation Parser...")
        self.regulation_parser = RegulationParser(
            flights_file=self.data_dir + "/so6_occupancy_matrix_with_times.json",
            tvtw_indexer=self.tvtw_indexer,
        )
        print("   OK Regulation parser initialized")

        print("4. Loading Traffic Volumes...")
        self.traffic_volumes = gpd.read_file(TRAFFIC_VOLUMES_PATH)
        print(f"   OK Loaded {len(self.traffic_volumes)} traffic volumes")

        print("5. Initializing Optimization Problem...")
        self.optimization_problem = OptimizationProblem(
            base_flight_list=self.flight_list,
            regulation_parser=self.regulation_parser,
            tvtw_indexer=self.tvtw_indexer,
            objective_weights=OBJECTIVE_WEIGHTS,
            horizon_time_windows=HORIZON_TIME_WINDOWS,
            base_traffic_volumes=self.traffic_volumes,
        )
        print("   OK Optimization problem initialized")

    def run(self):
        """
        Run the ALNS optimization.
        """
        if not self.optimization_problem:
            print("Error: Orchestrator not set up. Call setup() first.")
            return

        print("\n5. Running ALNS...")
        initial_state = self.optimization_problem.create_initial_state()

        if is_debug_enabled():
            console.rule("ALNS Start")
            console.print(
                f"[info] Initial state with {len(initial_state.network_plan.regulations)} regulations",
                style="info",
            )

        alns = ALNS(np.random.RandomState(42))
        alns.add_destroy_operator(remove_regulation)
        alns.add_repair_operator(add_regulation)

        select = RouletteWheel([5, 2, 1, 0.5], 0.8, 1, 1)
        accept = HillClimbing()
        stop = MaxIterations(10)

        result = alns.iterate(initial_state, select, accept, stop)
        best_solution = result.best_state

        print("\n--- ALNS Run Complete ---")
        print(f"Best objective: {best_solution.objective():.4f}")
        print(f"Best network plan: {best_solution.network_plan}")
        print("-------------------------\n")
        if is_debug_enabled():
            console.rule("ALNS Complete")
            try:
                parser = self.regulation_parser
                for i, reg in enumerate(best_solution.network_plan.regulations):
                    console.print(
                        f"[success] Final regulation {i}: {parser.explain_regulation(reg)}",
                        style="success",
                    )
            except Exception:
                pass
        
        return best_solution


if __name__ == "__main__":
    orchestrator = AlnsOrchestrator(DATA_DIR)
    orchestrator.setup()
    orchestrator.run()

