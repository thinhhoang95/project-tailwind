from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from project_tailwind.optimize.eval.delta_flight_list import DeltaFlightList
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.fcfs.scheduler import assign_delays
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


class PlanEvaluator:
    """
    Incremental evaluator for a NetworkPlan using FCFS-based per-regulation delays
    and a DeltaFlightList overlay for efficient overload computation.

    Responsibilities:
      - Compute per-regulation FCFS delays and aggregate by max-per-flight
      - Build a DeltaFlightList overlay view over a base FlightList
      - Produce overload and delay metrics using NetworkEvaluator
    """

    def __init__(
        self,
        traffic_volumes_gdf,
        parser: RegulationParser,
        tvtw_indexer: TVTWIndexer,
    ):
        self.traffic_volumes_gdf = traffic_volumes_gdf
        self.parser = parser
        self.tvtw_indexer = tvtw_indexer

        # Simple in-memory cache for FCFS results per regulation fingerprint
        # key: (location, rate, tuple(sorted(windows)), tuple(sorted(flight_ids))) -> Dict[fid, delay]
        self._fcfs_cache: Dict[Tuple[str, int, Tuple[int, ...], Tuple[str, ...]], Dict[str, int]] = {}

    # --- Core computation -----------------------------------------------------
    def compute_aggregated_delays(
        self, network_plan: NetworkPlan, base_flight_list: FlightList
    ) -> Dict[str, int]:
        """
        For a given plan and base flight list, compute the aggregated delays by
        applying FCFS per regulation and combining via max-per-flight.
        """
        if not network_plan.regulations:
            return {}

        delays_by_flight: Dict[str, List[int]] = {}

        for reg in network_plan.regulations:
            # Determine the flights targeted by this regulation
            if getattr(reg, "target_flight_ids", None) is not None:
                matched_flights = reg.target_flight_ids
            else:
                matched_flights = self.parser.parse(reg)
            if not matched_flights:
                continue

            cache_key = (
                reg.location,
                int(reg.rate),
                tuple(sorted(int(w) for w in reg.time_windows)),
                tuple(sorted(matched_flights)),
            )

            if cache_key in self._fcfs_cache:
                per_flight_delays = self._fcfs_cache[cache_key]
            else:
                per_flight_delays = assign_delays(
                    flight_list=base_flight_list,
                    identifier_list=matched_flights,
                    reference_location=reg.location,
                    tvtw_indexer=self.tvtw_indexer,
                    hourly_rate=reg.rate,
                    active_time_windows=reg.time_windows,
                )
                self._fcfs_cache[cache_key] = per_flight_delays

            for fid, dmin in per_flight_delays.items():
                if dmin <= 0:
                    continue
                delays_by_flight.setdefault(fid, []).append(int(dmin))

        # Reduce by max per flight
        aggregated: Dict[str, int] = {}
        for fid, values in delays_by_flight.items():
            aggregated[fid] = int(max(values))
        return aggregated

    def build_delta_flight_list(
        self, base_flight_list: FlightList, aggregated_delays: Dict[str, int]
    ) -> DeltaFlightList:
        """
        Build a lightweight overlay view applying per-flight delays logically.
        """
        return DeltaFlightList(base_flight_list, aggregated_delays)

    # --- High-level API -------------------------------------------------------
    def evaluate_plan(
        self,
        network_plan: NetworkPlan,
        base_flight_list: FlightList,
        *,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the plan and return a dictionary with the overlay and metrics.

        Returns dict with keys:
          - delays_by_flight: Dict[str, int]
          - delta_view: DeltaFlightList
          - excess_vector: np.ndarray
          - delay_stats: Dict[str, float]
          - objective: float
          - objective_components: Dict[str, float]
        """
        delays_by_flight = self.compute_aggregated_delays(network_plan, base_flight_list)
        delta_view = self.build_delta_flight_list(base_flight_list, delays_by_flight)

        evaluator = NetworkEvaluator(self.traffic_volumes_gdf, delta_view)
        excess_vector = evaluator.compute_excess_traffic_vector()
        delay_stats = evaluator.compute_delay_stats()

        # Compute scalar objective and components using default or provided weights
        objective, components = self.compute_objective(
            excess_vector=excess_vector,
            delay_stats=delay_stats,
            num_regs=len(network_plan.regulations),
            weights=weights,
        )

        return {
            "delays_by_flight": delays_by_flight,
            "delta_view": delta_view,
            "excess_vector": excess_vector,
            "delay_stats": delay_stats,
            "objective": objective,
            "objective_components": components,
        }

    # --- Objective -------------------------------------------------------------
    def compute_objective(
        self,
        *,
        excess_vector: Any,
        delay_stats: Dict[str, float],
        num_regs: int,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute scalar objective as a weighted combination of overload and delay.

        Objective: alpha*z_sum + beta*z_max + gamma*delay_min + delta*num_regs

        Args:
            excess_vector: Numpy array of excess per TVTW
            delay_stats: Dict with at least 'total_delay_seconds'
            num_regs: Number of regulations in the plan
            weights: Optional dict overriding coefficients 'alpha','beta','gamma','delta'

        Returns:
            Tuple of (objective_value, components_dict)
        """
        # Defaults from design doc
        default_weights = {"alpha": 1.0, "beta": 2.0, "gamma": 0.1, "delta": 25.0}
        if weights:
            # Shallow override while keeping defaults for unspecified keys
            w = {**default_weights, **weights}
        else:
            w = default_weights

        # Defensive conversion to numpy array without copying if already ndarray
        ev = np.asarray(excess_vector, dtype=float)
        z_sum = float(np.sum(ev)) if ev.size > 0 else 0.0
        z_max = float(np.max(ev)) if ev.size > 0 else 0.0
        delay_min = float(delay_stats.get("total_delay_seconds", 0.0)) / 60.0

        objective = (
            w["alpha"] * z_sum
            + w["beta"] * z_max
            + w["gamma"] * delay_min
            + w["delta"] * float(num_regs)
        )

        components = {
            "z_sum": z_sum,
            "z_max": z_max,
            "delay_min": delay_min,
            "num_regs": float(num_regs),
            "alpha": float(w["alpha"]),
            "beta": float(w["beta"]),
            "gamma": float(w["gamma"]),
            "delta": float(w["delta"]),
        }
        return float(objective), components


__all__ = ["PlanEvaluator"]


