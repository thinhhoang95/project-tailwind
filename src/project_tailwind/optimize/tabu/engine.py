from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.plan_evaluator import PlanEvaluator
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.features.flight_features import FlightFeatures
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.regulation import Regulation
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from .config import TabuConfig


@dataclass
class TabuSolution:
    plan: NetworkPlan
    objective: float
    objective_components: Dict[str, float]


class TabuEngine:
    def __init__(
        self,
        *,
        traffic_volumes_gdf,
        base_flight_list: FlightList,
        parser: RegulationParser,
        tvtw_indexer: TVTWIndexer,
        config: Optional[TabuConfig] = None,
        evaluator: Optional["NetworkEvaluator"] = None,
        features: Optional[FlightFeatures] = None,
    ) -> None:
        self.traffic_volumes_gdf = traffic_volumes_gdf
        self.base_flight_list = base_flight_list
        self.parser = parser
        self.tvtw_indexer = tvtw_indexer
        self.config = config or TabuConfig()

        if self.config.seed is not None:
            random.seed(self.config.seed)

        self.peval = PlanEvaluator(self.traffic_volumes_gdf, self.parser, self.tvtw_indexer)

        # A single FlightFeatures instance can be reused; restrict candidate pools per-move when needed
        # Build from the baseline evaluator state
        self._baseline_evaluator = evaluator or NetworkEvaluator(self.traffic_volumes_gdf, self.base_flight_list)
        self.features = features or FlightFeatures(self.base_flight_list, self._baseline_evaluator, overload_threshold=0.0)

        # Simple tabu list of move keys with remaining tenure
        self._tabu: Dict[Tuple, int] = {}

    # Utility
    def _evaluate(self, plan: NetworkPlan) -> Tuple[float, Dict[str, float]]:
        res = self.peval.evaluate_plan(plan, self.base_flight_list, weights={
            "alpha": self.config.alpha,
            "beta": self.config.beta,
            "gamma": self.config.gamma,
            "delta": self.config.delta,
        })
        return float(res["objective"]), res["objective_components"]  # type: ignore[index]

    def _decrement_tabu(self):
        expired = [k for k, t in self._tabu.items() if t <= 1]
        for k in expired:
            del self._tabu[k]
        for k in list(self._tabu.keys()):
            self._tabu[k] -= 1

    # Neighborhood proposal helpers (beam-limited)
    def _neighbors(self, plan: NetworkPlan) -> List[Tuple[str, NetworkPlan, Tuple]]:
        proposals: List[Tuple[str, NetworkPlan, Tuple]] = []
        bw = max(1, int(self.config.beam_width))

        # AddFlight: choose a regulation and add a top-ranked candidate not present
        for reg_idx, reg in enumerate(plan.regulations):
            existing = set(reg.target_flight_ids or [])
            # Pool: flights crossing this reg's location during its hour windows
            blanket = Regulation.from_components(location=reg.location, rate=reg.rate, time_windows=reg.time_windows)
            pool = self.parser.parse(blanket)
            ranked = self.features.rank_candidates(set(), candidate_flight_ids=pool, top_k=self.config.candidate_pool_size)
            for r in ranked[:bw]:
                fid = str(r["flight_id"])  # type: ignore[index]
                if fid in existing:
                    continue
                new_reg = Regulation.from_components(location=reg.location, rate=reg.rate, time_windows=reg.time_windows, target_flight_ids=list(existing | {fid}))
                new_plan = NetworkPlan([*plan.regulations])
                new_plan.regulations[reg_idx] = new_reg
                key = ("AddFlight", reg_idx, fid)
                proposals.append(("AddFlight", new_plan, key))
                if len(proposals) >= bw:
                    break
            if len(proposals) >= bw:
                break

        # RemoveFlight: remove a random existing flight from a non-empty regulation
        for reg_idx, reg in enumerate(plan.regulations):
            if not reg.target_flight_ids:
                continue
            candidates = list(reg.target_flight_ids)
            random.shuffle(candidates)
            for fid in candidates[:bw]:
                new_targets = [f for f in reg.target_flight_ids if f != fid]
                new_reg = Regulation.from_components(location=reg.location, rate=reg.rate, time_windows=reg.time_windows, target_flight_ids=new_targets)
                new_plan = NetworkPlan([*plan.regulations])
                new_plan.regulations[reg_idx] = new_reg
                key = ("RemoveFlight", reg_idx, fid)
                proposals.append(("RemoveFlight", new_plan, key))
                if len(proposals) >= bw:
                    break
            if len(proposals) >= bw:
                break

        # AdjustRate: +/- step
        for reg_idx, reg in enumerate(plan.regulations):
            for delta in (-self.config.rate_step, self.config.rate_step):
                new_rate = max(1, int(reg.rate + delta))
                if new_rate == reg.rate:
                    continue
                new_reg = Regulation.from_components(location=reg.location, rate=new_rate, time_windows=reg.time_windows, target_flight_ids=reg.target_flight_ids)
                new_plan = NetworkPlan([*plan.regulations])
                new_plan.regulations[reg_idx] = new_reg
                key = ("AdjustRate", reg_idx, delta)
                proposals.append(("AdjustRate", new_plan, key))
                if len(proposals) >= bw:
                    break
            if len(proposals) >= bw:
                break

        return proposals[:bw]

    def run(self, initial_plan: Optional[NetworkPlan] = None) -> TabuSolution:
        # Evaluate or create empty plan
        plan = initial_plan or NetworkPlan([])
        best_obj, best_comp = self._evaluate(plan)
        best = TabuSolution(plan=plan, objective=best_obj, objective_components=best_comp)

        current = best
        since_improve = 0

        for _ in range(self.config.max_iterations):
            self._decrement_tabu()
            candidates = self._neighbors(current.plan)
            if not candidates:
                break

            scored: List[Tuple[float, Tuple[str, NetworkPlan, Tuple], Dict[str, float]]] = []
            for move_name, cand_plan, key in candidates:
                obj, comp = self._evaluate(cand_plan)
                scored.append((obj, (move_name, cand_plan, key), comp))

            # Sort ascending by objective
            scored.sort(key=lambda x: x[0])

            chosen: Optional[Tuple[str, NetworkPlan, Tuple]] = None
            chosen_comp: Optional[Dict[str, float]] = None
            for obj, tup, comp in scored:
                _, _, key = tup
                if (key in self._tabu) and (obj >= best.objective):
                    # Tabu and does not satisfy aspiration (not better than best)
                    continue
                chosen = tup
                chosen_comp = comp
                break

            if chosen is None:
                # All tabu without aspiration; pick the best anyway (aspiration)
                obj, tup, comp = scored[0]
                chosen = tup
                chosen_comp = comp

            move_name, new_plan, key = chosen
            new_obj = float([s for s in scored if s[1] == chosen][0][0])

            # Update tabu with reverse move key
            reverse_key = self._reverse_key(move_name, key)
            self._tabu[reverse_key] = self.config.tenure

            current = TabuSolution(plan=new_plan, objective=new_obj, objective_components=chosen_comp or {})

            if current.objective < best.objective:
                best = current
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= self.config.no_improve_patience:
                    break

        return best

    def _reverse_key(self, move_name: str, key: Tuple) -> Tuple:
        if move_name == "AddFlight":
            _, reg_idx, fid = key
            return ("RemoveFlight", reg_idx, fid)
        if move_name == "RemoveFlight":
            _, reg_idx, fid = key
            return ("AddFlight", reg_idx, fid)
        if move_name == "AdjustRate":
            _, reg_idx, delta = key
            return ("AdjustRate", reg_idx, -delta)
        return ("None",)


