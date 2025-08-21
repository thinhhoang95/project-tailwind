import sys
from pathlib import Path
import time

# Ensure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    import geopandas as gpd  # type: ignore
    GEOPANDAS_AVAILABLE = True
except Exception:
    GEOPANDAS_AVAILABLE = False
    gpd = None  # type: ignore

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.plan_evaluator import PlanEvaluator
from project_tailwind.optimize.features.flight_features import FlightFeatures
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.regulation import Regulation
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.optimize.tabu.initializer import initialize_plan
from project_tailwind.optimize.tabu.engine import TabuEngine
from project_tailwind.optimize.tabu.config import TabuConfig
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


def _load_tvs_gdf():
    candidates = [
        Path("/Volumes/CrucialX/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"),
        Path("D:/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"),
    ]
    for p in candidates:
        if p.exists():
            return gpd.read_file(str(p)) if GEOPANDAS_AVAILABLE else None
    return None


def _require_inputs():
    occupancy = PROJECT_ROOT / "output/so6_occupancy_matrix_with_times.json"
    indexer = PROJECT_ROOT / "output/tvtw_indexer.json"
    if (not occupancy.exists()) or (not indexer.exists()) or (not GEOPANDAS_AVAILABLE):
        return None
    tvs = _load_tvs_gdf()
    if tvs is None:
        return None
    fl = FlightList(str(occupancy), str(indexer))
    tvtw = TVTWIndexer.load(str(indexer))
    parser = RegulationParser(str(occupancy), tvtw)
    return fl, tvtw, tvs, parser


def test_initializer_and_tabu_smoke():
    deps = _require_inputs()
    if deps is None:
        # Gracefully skip when data or geopandas is unavailable
        print("[SKIP] test_initializer_and_tabu_smoke: prerequisites missing (data or geopandas).")
        return
    fl, tvtw, tvs, parser = deps

    # Baseline metrics
    baseline_eval = NetworkEvaluator(tvs, fl)
    baseline_excess = baseline_eval.compute_excess_traffic_vector()
    baseline_z_sum = float(baseline_excess.sum()) if baseline_excess.size else 0.0

    # Heavy features computed once and reused
    features = FlightFeatures(fl, baseline_eval, overload_threshold=0.0)

    # Initialize a plan
    print("Initializing plan...")
    time_start = time.time()
    init_plan, init_components = initialize_plan(
        traffic_volumes_gdf=tvs,
        flight_list=fl,
        parser=parser,
        tvtw_indexer=tvtw,
        evaluator=baseline_eval,
        features=features,
    )
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

    # Evaluate initial plan objective
    print("Evaluating initial plan objective...")
    time_start = time.time()
    peval = PlanEvaluator(tvs, parser, tvtw)
    init_eval = peval.evaluate_plan(init_plan, fl)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
    assert "objective" in init_eval

    # Run a brief tabu search
    engine = TabuEngine(
        traffic_volumes_gdf=tvs,
        base_flight_list=fl,
        parser=parser,
        tvtw_indexer=tvtw,
        evaluator=baseline_eval,
        features=features,
        config=TabuConfig(max_iterations=5, no_improve_patience=3, beam_width=5, candidate_pool_size=50),
    )
    result = engine.run(initial_plan=init_plan)

    # Basic assertions
    assert isinstance(result.plan, NetworkPlan)
    assert result.objective <= init_eval["objective"] + 1e-6


if __name__ == "__main__":
    test_initializer_and_tabu_smoke()