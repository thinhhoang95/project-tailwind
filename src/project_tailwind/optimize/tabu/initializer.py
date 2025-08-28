from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.plan_evaluator import PlanEvaluator
from project_tailwind.optimize.features.flight_features import FlightFeatures
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.regulation import Regulation
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


def _pick_top_hotspot_hour(evaluator: NetworkEvaluator) -> Optional[Dict]:
    per_hour = evaluator.get_hotspot_flights(threshold=0.0, mode="hour")
    if not per_hour:
        return None
    overloaded = [x for x in per_hour if x.get("is_overloaded")]
    def deficit(item: Dict) -> float:
        return float(item.get("hourly_occupancy", 0.0)) - float(item.get("hourly_capacity", 0.0))
    pool = overloaded if overloaded else per_hour
    pool.sort(key=deficit, reverse=True)
    return pool[0]


def _hour_to_time_windows(hour: int, time_bin_minutes: int) -> List[int]:
    bins_per_hour = 60 // int(time_bin_minutes)
    start = int(hour) * bins_per_hour
    return list(range(start, start + bins_per_hour))


def initialize_plan(
    *,
    traffic_volumes_gdf,
    flight_list: FlightList,
    parser: RegulationParser,
    tvtw_indexer: TVTWIndexer,
    max_candidates: int = 300,
    rate_guesses: Optional[Iterable[int]] = None,
    objective_weights: Optional[Dict[str, float]] = None,
    evaluator: Optional["NetworkEvaluator"] = None,
    features: Optional[FlightFeatures] = None,
) -> Tuple[NetworkPlan, Dict[str, float]]:
    """
    Build an initial NetworkPlan by choosing the top hotspot (tv, hour),
    creating a blanket regulation for that hour, and seeding targeted flights
    using multiplicity/similarity/slack features.

    Returns (plan, objective_components_dict) where objective is evaluated via PlanEvaluator.
    """
    evaluator = evaluator or NetworkEvaluator(traffic_volumes_gdf, flight_list)
    hotspot = _pick_top_hotspot_hour(evaluator)
    if not hotspot:
        # No overloads – return empty plan
        return NetworkPlan([]), {"z_sum": 0.0, "z_max": 0.0, "delay_min": 0.0, "num_regs": 0.0}

    tv_id = str(hotspot["traffic_volume_id"])  # type: ignore[index]
    hour = int(hotspot["hour"])  # type: ignore[index]
    hourly_capacity = int(hotspot.get("hourly_capacity", 0))
    hourly_occupancy = float(hotspot.get("hourly_occupancy", 0.0))

    time_windows = _hour_to_time_windows(hour, flight_list.time_bin_minutes)

    # Choose a rate: default to capacity if available, else a conservative guess from occupancy
    if rate_guesses is None:
        rate_guesses = [max(1, hourly_capacity), max(1, int(hourly_occupancy)), 10, 20, 30]
    rate_guesses = [int(r) for r in rate_guesses if int(r) > 0]

    # Precompute features once; allow dynamic candidate pools by restricting to flights crossing the hotspot hour
    feat = features or FlightFeatures(flight_list, evaluator, overload_threshold=0.0)

    # Candidate pool: flights crossing the hotspot during the active hour
    # Use parser on a temporary blanket regulation to get matching flights
    blanket = Regulation.from_components(location=tv_id, rate=hourly_capacity if hourly_capacity > 0 else 10, time_windows=time_windows)
    candidate_flight_ids = parser.parse(blanket)
    if not candidate_flight_ids:
        # Fallback: empty plan
        return NetworkPlan([]), {"z_sum": 0.0, "z_max": 0.0, "delay_min": 0.0, "num_regs": 0.0}

    # Rank candidates against empty seed (so jaccard uses only multiplicity/slack)
    ranked = feat.rank_candidates(seed_footprint_tv_ids=set(), candidate_flight_ids=candidate_flight_ids, top_k=max_candidates)
    top_flights = [r["flight_id"] for r in ranked]

    # Evaluate a few rate guesses and pick the best
    peval = PlanEvaluator(traffic_volumes_gdf, parser, tvtw_indexer)
    best_tuple: Tuple[float, NetworkPlan, Dict[str, Any]] | None = None
    for rate in rate_guesses:
        reg = Regulation.from_components(location=tv_id, rate=rate, time_windows=time_windows, target_flight_ids=top_flights)
        plan = NetworkPlan([reg])
        res = peval.evaluate_plan(plan, flight_list, weights=objective_weights)
        obj = float(res["objective"])  # type: ignore[index]
        if (best_tuple is None) or (obj < best_tuple[0]):
            best_tuple = (obj, plan, res["objective_components"])  # type: ignore[index]

    assert best_tuple is not None
    return best_tuple[1], best_tuple[2]  # plan, objective components


