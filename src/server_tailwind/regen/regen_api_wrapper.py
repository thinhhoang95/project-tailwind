from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

import logging

import numpy as np

from parrhesia.flow_agent35.regen.types import RegenConfig
from server_tailwind.core.resources import get_resources

from parrhesia.api.flows import compute_flows
from parrhesia.flow_agent35.regen.config import resolve_weights
from parrhesia.flow_agent35.regen.engine import propose_regulations_for_hotspot
from parrhesia.optim.capacity import normalize_capacities


logger = logging.getLogger(__name__)


class RegenAPIWrapper:
    """Expose regen regulation proposals with shared resources and caches."""

    def __init__(self) -> None:
        self._resources = get_resources()
        self._indexer = self._resources.indexer
        self._flight_list = self._resources.flight_list
        self._tv_to_row = self._flight_list.tv_id_to_idx
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._capacities_by_tv: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    @property
    def time_bin_minutes(self) -> int:
        return int(self._indexer.time_bin_minutes)

    def _format_time_window(self, bin_offset: int) -> str:
        start_total_min = int(bin_offset * self.time_bin_minutes)
        start_hour = start_total_min // 60
        start_min = start_total_min % 60

        end_total_min = start_total_min + self.time_bin_minutes
        if end_total_min == 24 * 60:
            end_str = "24:00"
        else:
            end_hour = (end_total_min // 60) % 24
            end_min = end_total_min % 60
            end_str = f"{end_hour:02d}:{end_min:02d}"

        start_str = f"{start_hour:02d}:{start_min:02d}"
        return f"{start_str}-{end_str}"

    def _format_window_from_bins(self, start_bin: int, end_bin: int) -> str:
        start_label = self._format_time_window(int(start_bin)).split("-")[0]
        end_label = self._format_time_window(int(end_bin)).split("-")[1]
        return f"{start_label}-{end_label}"

    def _parse_time_window_to_bins(self, value: str) -> List[int]:
        s = str(value).strip()
        if "-" not in s:
            raise ValueError("time_window must be in 'HH:MM-HH:MM' form")
        start_token, end_token = [token.strip() for token in s.split("-", 1)]

        def _parse_piece(piece: str) -> Tuple[int, int, int]:
            if ":" not in piece:
                raise ValueError("time_window parts must include ':'")
            parts = piece.split(":")
            if len(parts) not in (2, 3):
                raise ValueError("time_window parts must be HH:MM or HH:MM:SS")
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) == 3 else 0
            if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
                raise ValueError("time_window contains invalid time components")
            return hour, minute, second

        sh, sm, ss = _parse_piece(start_token)
        eh, em, es = _parse_piece(end_token)

        start_dt = datetime(2000, 1, 1, sh, sm, ss)
        end_dt = datetime(2000, 1, 1, eh, em, es)
        if end_dt <= start_dt:
            raise ValueError("time_window end must be after start")

        bins = self._indexer.bin_range_for_interval(start_dt, end_dt)
        if not bins:
            raise ValueError("time_window resolves to no bins for the configured bin size")
        return bins


    # INSTRUCTIONS FOR CODING AGENTS AND DEVELOPERS: This function could be a source of inconsistency
    # since it replicates regen_test_bench_custom_tvtw's _build_capacities_by_tv function
    # If the objective function seem very high, one common culprit is the normalization of missing capacity values
    def _build_capacities_by_tv(self) -> Dict[str, np.ndarray]:
        if self._capacities_by_tv is not None:
            return self._capacities_by_tv
        matrix = self._resources.capacity_per_bin_matrix
        if matrix is None:
            logger.warning("Regen: AppResources capacity matrix is unavailable; capacities_by_tv will be empty")
            self._capacities_by_tv = {}
            return self._capacities_by_tv
        T = int(self._indexer.num_time_bins)
        raw_capacities: Dict[str, np.ndarray] = {}
        for tv_id, row_idx in self._tv_to_row.items():
            try:
                slice_vec = np.asarray(matrix[int(row_idx), :T], dtype=np.float64)
            except Exception:
                slice_vec = np.full((T,), -1.0, dtype=np.float64)
            raw_capacities[str(tv_id)] = slice_vec
        if raw_capacities:
            non_positive = [tv for tv, arr in raw_capacities.items() if float(np.max(arr)) <= 0.0]
            if non_positive:
                logger.warning(
                    "Regen: capacity matrix rows are non-positive for %d/%d TVs; sample=%s",
                    len(non_positive),
                    len(raw_capacities),
                    ",".join(non_positive[:5]),
                )
        # Normalize: treat missing/zero bins as unconstrained
        self._capacities_by_tv = normalize_capacities(raw_capacities)
        if self._capacities_by_tv:
            sample_items = list(self._capacities_by_tv.items())[:5]
            sample_stats = []
            for tv, arr in sample_items:
                arr_np = np.asarray(arr, dtype=np.float64)
                if arr_np.size == 0:
                    sample_stats.append(f"{tv}:empty")
                    continue
                sample_stats.append(
                    f"{tv}:min={float(arr_np.min()):.1f},max={float(arr_np.max()):.1f}"
                )
            print(
                "Regen: normalized capacities ready for %d TVs; samples: %s",
                len(self._capacities_by_tv),
                "; ".join(sample_stats),
            )
        return self._capacities_by_tv

    @staticmethod
    def _flow_id_to_flights(flows_payload: Mapping[str, Any]) -> Dict[int, List[str]]:
        mapping: Dict[int, List[str]] = {}
        for flow_obj in flows_payload.get("flows", []) or []:
            try:
                flow_id = int(flow_obj.get("flow_id"))
            except Exception:
                continue
            flights: List[str] = []
            for spec in flow_obj.get("flights", []) or []:
                fid = spec.get("flight_id")
                if fid is None:
                    continue
                flights.append(str(fid))
            mapping[int(flow_id)] = flights
        return mapping

    # ------------------------------------------------------------------
    async def propose_regulations(
        self,
        *,
        traffic_volume_id: str,
        time_window: str,
        top_k_regulations: Optional[int] = None,
        threshold: Optional[float] = None,
        resolution: Optional[float] = None,
    ) -> Dict[str, Any]:
        tv = str(traffic_volume_id).strip()
        if tv not in self._tv_to_row:
            raise ValueError(f"Unknown traffic_volume_id: {tv}")

        timebins_h = self._parse_time_window_to_bins(time_window)
        dir_opts = {"mode": "coord_cosine", "tv_centroids": self._resources.tv_centroids}
        capacities_by_tv = self._build_capacities_by_tv()
        travel_minutes_map = self._resources.travel_minutes()
        loop = asyncio.get_event_loop()

        def _compute() -> Tuple[Dict[str, Any], Dict[int, List[str]], List[Any]]:
            flows_payload = compute_flows(
                tvs=[tv],
                timebins=list(timebins_h),
                direction_opts=dir_opts,
                threshold=threshold,
                resolution=resolution,
            )
            flow_to_flights = self._flow_id_to_flights(flows_payload)

            num_regulations_limit = int(top_k_regulations) if (top_k_regulations is not None and int(top_k_regulations) > 0) else None
            
            
            my_cfg = RegenConfig(
                g_min=-float("inf"),
                rho_max=float("inf"),
                slack_min=-float("inf"),
                distinct_controls_required=False,
                raise_on_edge_cases=True,
                min_num_flights=4,
                top_k_regulations=num_regulations_limit
            )            


            proposals = propose_regulations_for_hotspot(
                indexer=self._indexer,
                flight_list=self._flight_list,
                capacities_by_tv=capacities_by_tv,
                travel_minutes_map=travel_minutes_map,
                hotspot_tv=tv,
                timebins_h=timebins_h,
                flows_payload=flows_payload,
                flow_to_flights=flow_to_flights,
                weights=None,
                config=my_cfg,
            )


            return flows_payload, flow_to_flights, proposals

        flows_payload, flow_to_flights, proposals = await loop.run_in_executor(self._executor, _compute)

        

        weights = resolve_weights(None)
        weights_dict = {
            "w1": float(weights.w1),
            "w2": float(weights.w2),
            "w3": float(weights.w3),
            "w4": float(weights.w4),
            "w5": float(weights.w5),
            "w6": float(weights.w6),
        }

        proposals_payload: List[Dict[str, Any]] = []
        for proposal in proposals:
            start_bin = int(proposal.window.start_bin)
            end_bin = int(proposal.window.end_bin)
            control_window_label = self._format_window_from_bins(start_bin, end_bin)

            diagnostics = dict(proposal.diagnostics)
            components_before = diagnostics.get("score_components_before", {}) or {}
            components_after = diagnostics.get("score_components_after", {}) or {}
            components_delta: Dict[str, float] = {}
            for key in set(components_before.keys()) | set(components_after.keys()):
                try:
                    delta_val = float(components_after.get(key, 0.0)) - float(components_before.get(key, 0.0))
                except Exception:
                    delta_val = 0.0
                components_delta[str(key)] = delta_val

            per_flow_diag: Mapping[int, Mapping[str, Any]] = diagnostics.get("per_flow", {}) or {}

            flows_section: List[Dict[str, Any]] = []
            for flow_entry in proposal.flows_info:
                flow_id = int(flow_entry.get("flow_id"))
                control_volume = flow_entry.get("control_tv_id")
                diag_features = per_flow_diag.get(flow_id)
                if diag_features is None:
                    diag_features = per_flow_diag.get(str(flow_id), {})

                gH = float(diag_features.get("gH", 0.0))
                v_tilde = float(diag_features.get("v_tilde", 0.0))
                slack15 = float(diag_features.get("slack15", 0.0))
                slack30 = float(diag_features.get("slack30", 0.0))
                rho = float(diag_features.get("rho", 0.0))
                coverage = float(diag_features.get("coverage", 0.0))
                final_score = (
                    weights_dict["w1"] * gH
                    + weights_dict["w2"] * v_tilde
                    + weights_dict["w3"] * slack15
                    + weights_dict["w4"] * slack30
                    - weights_dict["w5"] * rho
                    + weights_dict["w6"] * coverage
                )

                flows_section.append(
                    {
                        "flow_id": flow_id,
                        "flight_ids": list(flow_to_flights.get(flow_id, [])),
                        "control_volume_id": control_volume,
                        "baseline_rate_per_hour": float(flow_entry.get("r0_i", 0.0)),
                        "allowed_rate_per_hour": float(flow_entry.get("R_i", 0.0)),
                        "assigned_cut_per_hour": float(flow_entry.get("lambda_cut_i", 0.0)),
                        "time_window_label": control_window_label,
                        "time_window_bins": [
                            int(diag_features.get("tGl", start_bin)),
                            int(diag_features.get("tGu", end_bin)),
                        ],
                        "features": diag_features,
                        "final_score": float(final_score),
                    }
                )

            proposals_payload.append(
                {
                    "hotspot": {
                        "traffic_volume_id": tv,
                        "input_time_window": time_window,
                        "timebins": list(timebins_h),
                    },
                    "control_window": {
                        "bins": [start_bin, end_bin],
                        "label": control_window_label,
                    },
                    # Regulation-level scope for inspection
                    "target_tvs": list(getattr(proposal, "target_tvs", [])),
                    "target_cells": [
                        [str(tv_id), int(b)] for (tv_id, b) in getattr(proposal, "target_cells", [])
                    ],
                    "ripple_tvs": list(getattr(proposal, "ripple_tvs", [])),
                    "ripple_cells": [
                        [str(tv_id), int(b)] for (tv_id, b) in getattr(proposal, "ripple_cells", [])
                    ],
                    "objective_improvement": {
                        "delta_deficit_per_hour": float(proposal.predicted_improvement.delta_deficit_per_hour),
                        "delta_objective_score": float(proposal.predicted_improvement.delta_objective_score),
                    },
                    "objective_components": {
                        "before": components_before,
                        "after": components_after,
                        "delta": components_delta,
                    },
                    "flows": flows_section,
                    "diagnostics": {
                        "ranking_score": float(diagnostics.get("ranking_score", 0.0)),
                        "E_target": float(diagnostics.get("E_target", 0.0)),
                        "E_target_occupancy": float(diagnostics.get("E_target_occupancy", 0.0)),
                        "diversity_penalty": float(diagnostics.get("diversity_penalty", 0.0)),
                    },
                }
            )

        return {
            "traffic_volume_id": tv,
            "time_window": time_window,
            "time_bin_minutes": self.time_bin_minutes,
            "top_k": limit,
            "weights": weights_dict,
            "num_proposals": len(proposals_payload),
            "proposals": proposals_payload,
        }

    def __del__(self) -> None:
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
