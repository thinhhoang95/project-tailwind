from __future__ import annotations

"""Sandbox environment for Regulation Zero.

This module provides:
- Structural cloning of FlightListWithDelta to isolate per-simulation sandbox state
  without disk reload or Python deepcopy.
- Context managers to safely bind global resources for flows and evaluator calls,
  and to restore them on exit.
- RZSandbox class exposing hotspot extraction, proposal generation, and
  application of selected proposals to mutate the sandboxed flight list.

Design constraints
- Avoid silent fallbacks: validate inputs and states aggressively with explicit
  checks. Let exceptions surface to make debugging predictable.
- Keep the sandbox isolated: every fork gets its own flight list copy while
  sharing read-only resources (indexer, capacities, centroids) by reference.
- Recompute flows and hotspots from the current sandbox state after each move.
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import server_tailwind.core.resources as core_res
from server_tailwind.core.resources import AppResources

from parrhesia.api.flows import compute_flows
from parrhesia.api.resources import get_global_resources, set_global_resources
from parrhesia.flow_agent35.regen.engine import propose_regulations_for_hotspot
from parrhesia.flow_agent35.regen.hotspot_segment_extractor import (
    extract_hotspot_segments_from_resources,
    segment_to_hotspot_payload,
)
from parrhesia.flow_agent35.regen.types import RegenConfig
from parrhesia.optim.capacity import normalize_capacities

from parrhesia.actions.dfplan_evaluator import evaluate_df_regulation_plan
from parrhesia.actions.regulations import DFRegulationPlan
from project_tailwind.stateman.delay_assignment import DelayAssignmentTable
from project_tailwind.stateman.delta_view import DeltaOccupancyView
from project_tailwind.stateman.flight_list_with_delta import FlightListWithDelta

from .types import RZConfig


def _timebins_from_window(window_bins: Iterable[int]) -> List[int]:
    """Converts [start, end_exclusive] to a list of bin indices.

    - If only one value is provided, treat it as a single bin.
    - If end <= start, clamp to a minimum 1-bin span.
    """
    wb = list(int(b) for b in window_bins)
    if not wb:
        return []
    if len(wb) == 1:
        return [wb[0]]
    start = int(wb[0])
    end_exclusive = int(wb[1])
    if end_exclusive <= start:
        end_exclusive = start + 1
    return list(range(start, end_exclusive))


def clone_flight_list_structural(src: FlightListWithDelta) -> FlightListWithDelta:
    """Structural clone of a FlightListWithDelta.

    Copies sparse matrices and small metadata to ensure independence across
    sandboxes. Avoids Python's deepcopy and any disk reload.
    """
    cls = src.__class__
    dst = cls.__new__(cls)

    # Basic identity
    dst.occupancy_file_path = getattr(src, "occupancy_file_path", "")
    dst.tvtw_indexer_path = getattr(src, "tvtw_indexer_path", "")
    # Indexer proxy and mappings
    dst.tvtw_indexer = dict(getattr(src, "tvtw_indexer", {}) or {})
    dst.time_bin_minutes = int(getattr(src, "time_bin_minutes", 60))
    dst.tv_id_to_idx = dict(getattr(src, "tv_id_to_idx", {}) or {})
    dst.idx_to_tv_id = dict(getattr(src, "idx_to_tv_id", {}) or {})
    dst.num_traffic_volumes = int(getattr(src, "num_traffic_volumes", len(dst.tv_id_to_idx)))
    dst.num_time_bins_per_tv = int(getattr(src, "num_time_bins_per_tv", 1))
    dst._indexer = getattr(src, "_indexer", None)

    # Flights/shape
    dst.flight_ids = list(getattr(src, "flight_ids", ()))
    dst.num_flights = int(getattr(src, "num_flights", len(dst.flight_ids)))
    dst.flight_id_to_row = dict(getattr(src, "flight_id_to_row", {}) or {})
    dst.num_tvtws = int(getattr(src, "num_tvtws", 0))

    # Sparse occupancy matrices
    occ_csr = getattr(src, "occupancy_matrix", None)
    dst.occupancy_matrix = occ_csr.copy() if occ_csr is not None else None
    occ_lil = getattr(src, "_occupancy_matrix_lil", None)
    dst._occupancy_matrix_lil = occ_lil.copy() if occ_lil is not None else None
    # Fresh buffer and flags
    dst._lil_matrix_dirty = False
    buf = getattr(src, "_temp_occupancy_buffer", None)
    dst._temp_occupancy_buffer = (
        np.array(buf, copy=True) if buf is not None else np.zeros(dst.num_tvtws, dtype=np.float32)
    )

    # Deep-copy small dicts
    from copy import deepcopy as _dc

    dst.flight_data = _dc(getattr(src, "flight_data", {}) or {})
    dst.flight_metadata = _dc(getattr(src, "flight_metadata", {}) or {})

    # Caches
    dst._flight_tv_sequence_cache = {}

    # Delta bookkeeping snapshot
    dst.applied_regulations = list(getattr(src, "applied_regulations", []) or [])
    dst.delay_histogram = dict(getattr(src, "delay_histogram", {}) or {})
    dst.total_delay_assigned_min = int(getattr(src, "total_delay_assigned_min", 0))
    dst.num_delayed_flights = int(getattr(src, "num_delayed_flights", 0))
    dst.num_regulations = int(getattr(src, "num_regulations", 0))
    agg = getattr(src, "_delta_aggregate", None)
    dst._delta_aggregate = np.array(agg, copy=True) if agg is not None else np.zeros(dst.num_tvtws, dtype=np.int64)
    dst._applied_views = []

    # Verify independence for matrices
    if (src.occupancy_matrix is not None and dst.occupancy_matrix is src.occupancy_matrix) or (
        src._occupancy_matrix_lil is not None and dst._occupancy_matrix_lil is src._occupancy_matrix_lil
    ):
        raise RuntimeError("Clone shares sparse matrix references with source")
    return dst


@contextmanager
def with_fl_resources(indexer: Any, flight_list: Any):
    """Bind flows global resources to the provided indexer/flight_list for the duration.

    Asserts correct binding and full restoration on exit.
    """
    prev = get_global_resources()
    set_global_resources(indexer, flight_list)
    gi, gf = get_global_resources()
    if gi is not indexer or gf is not flight_list:
        raise RuntimeError("Flows globals not bound to sandbox resources")
    try:
        yield
    finally:
        # Restore prior state (possibly Nones)
        set_global_resources(*(prev or (None, None)))
        gi2, gf2 = get_global_resources()
        if prev is None:
            if gi2 is not None or gf2 is not None:
                raise RuntimeError("Flows globals not cleared on exit")
        else:
            if gi2 is not prev[0] or gf2 is not prev[1]:
                raise RuntimeError("Flows globals not restored on exit")


@contextmanager
def with_core_resources(res: AppResources):
    """Temporarily set server core resources for the evaluator to the sandbox."""
    prev = core_res.get_resources()
    core_res._GLOBAL_RESOURCES = res  # direct assignment to swap during eval
    if core_res.get_resources() is not res:
        raise RuntimeError("Core resources not set to sandbox")
    try:
        yield
    finally:
        core_res._GLOBAL_RESOURCES = prev
        if core_res.get_resources() is not prev:
            raise RuntimeError("Core resources not restored after eval")


class RZSandbox:
    """Process-local sandbox wrapper around AppResources.

    Each sandbox has a distinct FlightListWithDelta instance while sharing other
    heavy read-only artifacts by reference.
    """

    def __init__(self, res: AppResources, cfg: RZConfig):
        # Ensure resources are fully loaded to avoid lazy pitfalls while searching
        self._res = res.preload_all()
        self._cfg = cfg
        # Time-axis consistency: indexer width must match capacity width
        T_idx = int(self._res.indexer.num_time_bins)
        T_cap = int(self._res.capacity_per_bin_matrix.shape[1])
        if T_idx != T_cap:
            raise RuntimeError(
                f"Time-axis mismatch: indexer bins={T_idx}, capacity width={T_cap}"
            )

    @property
    def resources(self) -> AppResources:
        """Return the underlying AppResources for this sandbox."""
        return self._res

    def fork(self) -> "RZSandbox":
        """Create a child sandbox with a structural clone of the flight list.

        - Shares indexer, capacities, centroids by reference for efficiency.
        - Copies the FlightListWithDelta matrices and metadata to isolate state.
        """
        parent = self._res
        child = AppResources(parent.paths)
        # Reuse heavy, read-only artifacts by reference
        child._indexer = parent.indexer
        child._traffic_volumes_gdf = parent.traffic_volumes_gdf
        child._hourly_capacity_by_tv = parent.hourly_capacity_by_tv
        child._capacity_per_bin_matrix = parent.capacity_per_bin_matrix
        child._travel_minutes = parent._travel_minutes
        child._tv_centroids = parent.tv_centroids
        # Structural clone of the current flight list
        child._flight_list = clone_flight_list_structural(parent.flight_list)
        return RZSandbox(child, cfg=self._cfg)

    def extract_hotspots(self, k: int) -> List[Dict[str, Any]]:
        """Extract and rank hotspot segments from the sandbox state."""
        segs = extract_hotspot_segments_from_resources(
            threshold=float(self._cfg.hotspot_threshold), resources=self._res
        )
        segs.sort(key=lambda s: float(s.get("max_excess", 0.0)), reverse=True)
        return segs[: int(k)]

    def proposals_for_hotspot(self, hotspot_payload: Mapping[str, Any], k: int):
        """Generate proposals for a hotspot using the current sandbox state.

        Returns a tuple: (proposals, flights_by_flow) suitable for applying.
        """
        res = self._res
        # Validate hotspot payload
        if "control_volume_id" not in hotspot_payload:
            raise ValueError("hotspot_payload missing control_volume_id")
        window_bins = hotspot_payload.get("window_bins")
        if window_bins is None:
            raise ValueError("hotspot_payload missing window_bins")
        timebins_h = _timebins_from_window(window_bins)
        if not timebins_h:
            raise RuntimeError("hotspot window resolves to no bins")
        control_tv = str(hotspot_payload["control_volume_id"])
        if control_tv not in res.flight_list.tv_id_to_idx:
            raise ValueError(f"Unknown control TV: {control_tv}")

        # Compute flows bound to THIS sandbox
        with with_fl_resources(res.indexer, res.flight_list):
            flows_payload = compute_flows(
                tvs=[control_tv],
                timebins=timebins_h,
                threshold=self._cfg.flows_threshold,
                resolution=self._cfg.flows_resolution,
                direction_opts={
                    "mode": self._cfg.direction_mode,
                    "tv_centroids": res.tv_centroids,
                },
            )

        # Validate and transform flows payload
        if flows_payload.get("tvs") != [control_tv]:
            raise RuntimeError("compute_flows returned unexpected TV set")
        tb = list(flows_payload.get("timebins") or [])
        if tb and tb != list(timebins_h):
            raise RuntimeError("compute_flows returned mismatched timebins")
        flows = flows_payload.get("flows") or []
        if self._cfg.fail_fast and len(flows) == 0:
            raise RuntimeError("No flows returned for a non-empty hotspot window")

        flow_to_flights: Dict[str, List[str]] = {
            str(int(flow["flow_id"])): [
                str(spec["flight_id"]) for spec in (flow.get("flights") or []) if spec.get("flight_id") is not None
            ]
            for flow in flows
            if "flow_id" in flow
        }
        if self._cfg.fail_fast and not flow_to_flights:
            raise RuntimeError("Empty flow_to_flights after compute_flows")
        # Validate flight existence in sandbox
        missing = [
            fid
            for fids in flow_to_flights.values()
            for fid in fids
            if fid not in res.flight_list.flight_id_to_row
        ]
        if self._cfg.fail_fast and missing:
            raise RuntimeError(f"Flows reference unknown flights in sandbox: {missing[:5]}...")

        # Capacities: build and normalize against sandbox
        caps = res.capacity_per_bin_matrix
        T = int(res.indexer.num_time_bins)
        cap_map_raw = {str(tv): caps[int(row), :T] for tv, row in res.flight_list.tv_id_to_idx.items()}
        capacities_by_tv = normalize_capacities(cap_map_raw)
        for tv, arr in capacities_by_tv.items():
            if arr.shape[0] != T:
                raise RuntimeError(f"Capacity array length != T for {tv}")
            if np.any(arr <= 0):
                raise RuntimeError("Non-positive capacity detected after normalization")

        proposals = propose_regulations_for_hotspot(
            indexer=res.indexer,
            flight_list=res.flight_list,
            capacities_by_tv=capacities_by_tv,
            travel_minutes_map=res.travel_minutes(),
            hotspot_tv=control_tv,
            timebins_h=timebins_h,
            flows_payload=flows_payload,
            flow_to_flights=flow_to_flights,
            config=RegenConfig(k_proposals=int(k)),
        )
        # Sanity: predicted improvements must be finite
        for p in proposals:
            d = getattr(getattr(p, "predicted_improvement", None), "delta_objective_score", None)
            if d is None or not np.isfinite(float(d)):
                raise RuntimeError("Proposal has non-finite delta_objective_score")
        return proposals, flow_to_flights

    def apply_proposal(self, proposal: Any, flights_by_flow: Mapping[str, List[str]]) -> None:
        """Evaluate a DF plan from proposal and mutate the sandbox flight list.

        This uses evaluator bound to the sandbox's AppResources and then applies
        the resulting delays as a DeltaOccupancyView to the sandbox flight list.
        """
        res = self._res
        tbm = int(res.indexer.time_bin_minutes)
        if tbm <= 0:
            raise RuntimeError("Invalid indexer.time_bin_minutes")

        # Construct DF plan with explicit minutes to avoid fallback defaults
        plan = DFRegulationPlan.from_proposal(
            proposal,
            flights_by_flow=flights_by_flow,
            time_bin_minutes=tbm,
        )

        # Evaluate within sandbox-scoped core resources
        with with_core_resources(res):
            eval_res = evaluate_df_regulation_plan(
                plan,
                indexer_path=str(res.paths.tvtw_indexer_path),
                flights_path=str(res.paths.occupancy_file_path),
            )

        # Convert to delay table, basic sanity
        delays = DelayAssignmentTable.from_dict(eval_res.delays_by_flight)
        if self._cfg.fail_fast and not delays.delays_by_flight:
            raise RuntimeError(
                "Evaluation produced no delays; likely misbound resources or empty targets"
            )

        pre_regs = int(res.flight_list.num_regulations)
        pre_total_delay = int(res.flight_list.total_delay_assigned_min)
        view = DeltaOccupancyView.from_delay_table(res.flight_list, delays, regulation_id="rz")
        res.flight_list.step_by_delay(view, finalize=True)

        # Check that state was actually mutated by the step
        if int(res.flight_list.num_regulations) <= pre_regs:
            raise RuntimeError("step_by_delay did not register a new regulation")
        if int(res.flight_list.total_delay_assigned_min) < pre_total_delay:
            raise RuntimeError("Total delay decreased after applying a regulation")


__all__ = [
    "RZSandbox",
    "clone_flight_list_structural",
    "with_fl_resources",
    "with_core_resources",
    "segment_to_hotspot_payload",
]

