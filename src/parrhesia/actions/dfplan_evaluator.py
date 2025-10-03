from __future__ import annotations

"""
Evaluate a DFRegulationPlan via the flow objective, returning per-flight delays
and objective components.

This module mirrors the evaluation path used by the flow APIs:
  - Build flows from the plan's regulations and explicit flight lists
  - Construct demand and allowance-limited schedules (n_f_t)
  - Build a shared ScoreContext and score baseline vs. regulated

Public API:
    evaluate_df_regulation_plan(plan, *, indexer_path, flights_path,
                                capacities_path=None, weights=None,
                                include_excess_vector=False) -> DFPlanEvalResult
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.optimize.regulation import Regulation
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.eval.plan_evaluator import PlanEvaluator

from parrhesia.actions.regulations import (
    DFRegulationPlan,
    DEFAULT_AUTO_RIPPLE_TIME_BINS,
)
from parrhesia.optim.capacity import build_bin_capacities, normalize_capacities
from parrhesia.optim.objective import (
    ObjectiveWeights,
    build_score_context,
    score_with_context,
)
from parrhesia.optim.ripple import compute_auto_ripple_cells
from parrhesia.optim.sa_optimizer import prepare_flow_scheduling_inputs
from parrhesia.flow_agent35.regen.rates import distribute_hourly_rate_to_bins


@dataclass(frozen=True)
class DFPlanEvalResult:
    """Dataclass to store the results of a DFRegulationPlan evaluation."""

    delays_by_flight: Dict[str, int]
    objective_components: Dict[str, float]
    objective: float
    pre_objective: float
    delta_objective: float


def _time_to_bin(hhmm: str, *, time_bin_minutes: int) -> int:
    """Converts a time string in 'HH:MM' format to a time bin index."""
    parts = str(hhmm).split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time string: {hhmm!r}")
    hh = int(parts[0])
    mm = int(parts[1])
    total_min = hh * 60 + mm
    return int(total_min // int(time_bin_minutes))


def _window_to_bins(start: str, end: str, *, time_bin_minutes: int, T: int) -> List[int]:
    """Converts a time window (start and end 'HH:MM' strings) to a list of time bin indices."""
    # end is exclusive; allow "24:00"
    start_bin = max(0, _time_to_bin(start, time_bin_minutes=time_bin_minutes))
    end_bin_excl = _time_to_bin(end, time_bin_minutes=time_bin_minutes)
    # Treat exact 24:00 as T
    if end.startswith("24"):
        end_bin_excl = T
    end_bin_excl = max(0, min(T, int(end_bin_excl)))
    if end_bin_excl <= start_bin:
        return []
    return list(range(int(start_bin), int(end_bin_excl)))


def _dfplan_to_regulations(plan: DFRegulationPlan, *, indexer: TVTWIndexer, T: int) -> List[Regulation]:
    """Converts a DFRegulationPlan object into a list of Regulation objects."""
    regs: List[Regulation] = []
    tbm = int(indexer.time_bin_minutes)
    for r in plan.regulations:
        # Convert the regulation's time window to a list of bin indices.
        wins = _window_to_bins(r.window_from, r.window_to, time_bin_minutes=tbm, T=T)
        # Create a Regulation object from the plan's components.
        reg = Regulation.from_components(
            location=str(r.tv_id),
            rate=int(r.allowed_rate_per_hour),
            time_windows=[int(b) for b in wins],
            filter_type="IC",  # Assuming 'IC' (internal code) filter type.
            filter_value="__",  # Placeholder filter value.
            target_flight_ids=list(r.flights),
        )
        regs.append(reg)
    return regs


def _cells_from_ranges(
    idx: TVTWIndexer, ranges: Mapping[str, Mapping[str, Any]]
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Convert {tv: {from,to}} mapping to explicit (tv, bin) cells.

    Returns (cells, tvs_considered). Invalid TVs or malformed windows are ignored.
    """
    from datetime import datetime

    def _parse_time_hms(s: str) -> datetime:
        """Helper to parse time strings that may or may not include seconds."""
        ss = str(s).strip()
        fmt = "%H:%M:%S" if ss.count(":") == 2 else "%H:%M"
        # Use a fixed date for time comparison.
        return datetime.strptime(ss, fmt).replace(year=2025, month=1, day=1)

    cells: List[Tuple[str, int]] = []
    tvs: List[str] = []
    for tv, win in (ranges or {}).items():
        tv_id = str(tv)
        # Skip if the traffic volume ID is not in the indexer.
        if tv_id not in idx.tv_id_to_idx:
            continue
        try:
            # Parse start and end times from the window mapping.
            t_from = _parse_time_hms(str(win.get("from")))
            t_to = _parse_time_hms(str(win.get("to")))
        except Exception:
            # Ignore malformed time windows.
            continue
        # Get the list of bin indices for the time interval.
        bins = idx.bin_range_for_interval(t_from, t_to)
        for b in bins:
            cells.append((tv_id, int(b)))
        tvs.append(tv_id)
    return cells, tvs


def _build_capacities_by_tv(
    *,
    indexer: TVTWIndexer,
    flight_list: FlightList,
    capacities_path: Optional[str],
) -> Dict[str, np.ndarray]:
    """Build per-TV per-bin capacities with normalization, mirroring API logic.

    This function sources capacity data from one of three places, in order of precedence:
    1. An explicit file path (`capacities_path`).
    2. Shared server resources, if available.
    3. A default file path located in the project's data directory.
    """
    # 1) Prefer explicit path if provided.
    if capacities_path:
        raw = build_bin_capacities(str(capacities_path), indexer)
        return normalize_capacities(raw)

    # 2) Try to use shared server resources if available.
    # This is useful in a production environment where resources are centrally managed.
    try:
        from server_tailwind.core.resources import get_resources  # type: ignore

        _res = get_resources()
        mat = _res.capacity_per_bin_matrix  # shape [num_tvs, T]
        if mat is not None:
            out: Dict[str, np.ndarray] = {}
            T = int(indexer.num_time_bins)
            for tv_id, row_idx in flight_list.tv_id_to_idx.items():
                arr = np.asarray(mat[int(row_idx), :T], dtype=np.float64)
                out[str(tv_id)] = arr
            return normalize_capacities(out)
    except Exception:
        # If server resources are not available or fail, proceed to the next fallback.
        pass

    # 3) Fallback to a default capacity file path within the project structure.
    # This is useful for local development and testing.
    from pathlib import Path

    here = Path(__file__).resolve()
    root = None
    # Traverse up the directory tree to find the project root.
    for p in here.parents:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            root = p
            break
    if root is None:
        # As a last resort, assume a fixed directory structure.
        root = here.parents[3]
    default_cap_path = root / "data" / "cirrus" / "wxm_sm_ih_maxpool.geojson"
    raw = build_bin_capacities(str(default_cap_path), indexer)
    return normalize_capacities(raw)


def evaluate_df_regulation_plan(
    plan: DFRegulationPlan,
    *,
    indexer_path: str,
    flights_path: str,
    capacities_path: Optional[str] = None,
    weights: Optional[Mapping[str, float]] = None,
    include_excess_vector: bool = False,  # reserved; not returned by result
) -> DFPlanEvalResult:
    """Evaluate a DFRegulationPlan against the flow objective.

    Returns DFPlanEvalResult with per-flight delays and objective breakdown.
    """
    # Step 1: Load essential artifacts for evaluation.
    # The TVTWIndexer provides mappings for time and traffic volumes.
    tvtw_indexer = TVTWIndexer.load(str(indexer_path))
    # The FlightList contains information about all flights.
    flight_list = FlightList.from_json(str(flights_path), tvtw_indexer)
    # The RegulationParser is used to parse and apply regulations.
    parser = RegulationParser(flights_file=str(flights_path), tvtw_indexer=tvtw_indexer)

    # Step 2: Translate the high-level DFRegulationPlan into a list of detailed Regulation objects.
    T = int(tvtw_indexer.num_time_bins)
    regs = _dfplan_to_regulations(plan, indexer=tvtw_indexer, T=T)

    # Step 3 (Optional): Perform a preliminary evaluation of the plan.
    # This step is not required for the final scoring but can provide an
    # intermediate "overlay" view of the plan's impact. It's wrapped in a
    # try-except block to be non-fatal if it fails.
    try:
        network_plan = NetworkPlan(regs)
        evaluator = PlanEvaluator(traffic_volumes_gdf=None, parser=parser, tvtw_indexer=tvtw_indexer)
        _ = evaluator.evaluate_plan(network_plan, base_flight_list=flight_list, weights=None)
    except Exception:
        # If the preliminary evaluation fails, we can still proceed with the
        # more detailed flow-based evaluation.
        pass

    # Step 4: Build the capacity profiles for each traffic volume (TV).
    # Capacities are crucial for determining where and when congestion occurs.
    cap_by_tv = _build_capacities_by_tv(
        indexer=tvtw_indexer, flight_list=flight_list, capacities_path=capacities_path
    )

    # Step 5: Build "flows" by mapping flights to their corresponding regulations.
    # A flow is a group of flights affected by the same regulation.
    # Each flight is assigned to the first regulation that targets it.
    flow_to_reg: Dict[int, Regulation] = {}
    flow_map: Dict[str, int] = {}
    assigned: set[str] = set()
    for idx, reg in enumerate(regs):
        flow_to_reg[idx] = reg
        for fid in reg.target_flight_ids or []:
            s = str(fid)
            if s in assigned:
                continue
            assigned.add(s)
            flow_map[s] = idx

    # Prepare inputs for flow scheduling based on the created flows.
    flights_by_flow_raw, _ = prepare_flow_scheduling_inputs(
        flight_list=flight_list,
        flow_map=flow_map,
        hotspot_ids=[r.location for r in regs],
        flight_ids=list(flow_map.keys()),
    )
    # Filter out any empty flows.
    flights_by_flow: Dict[int, List[Dict[str, Any]]] = {
        int(k): v for k, v in flights_by_flow_raw.items() if v
    }

    # Handle the edge case where there are no flights or flows to evaluate.
    if not flights_by_flow:
        components_zero = {k: 0.0 for k in ("J_cap", "J_delay", "J_reg", "J_tv")}
        return DFPlanEvalResult(
            delays_by_flight={},
            objective_components=components_zero,
            objective=0.0,
            pre_objective=0.0,
            delta_objective=0.0,
        )

    # Step 6: Construct demand and regulated schedule vectors for each flow.
    # `n0_f_t` represents the baseline demand (no regulations).
    # `n_f_t` represents the regulated schedule, capped by allowances.
    bins_per_hour = int(tvtw_indexer.rolling_window_size())
    n_f_t: Dict[int, List[int]] = {}  # Regulated schedule
    n0_f_t: Dict[int, List[int]] = {}  # Baseline demand
    target_cells_set: set[Tuple[str, int]] = set()
    for fid, specs in flights_by_flow.items():
        # Initialize demand vector for the current flow.
        demand = np.zeros(T + 1, dtype=np.int64)
        for spec in specs or []:
            rb = spec.get("requested_bin") if isinstance(spec, dict) else None
            try:
                b = int(rb)
            except Exception:
                continue
            # Clamp bin index to be within valid range [0, T-1].
            if b < 0:
                b = 0
            if b >= T:
                b = T - 1
            demand[b] += 1

        # `baseline` is the initial demand distribution.
        baseline = demand.copy()
        baseline_total = int(np.sum(demand[:T], dtype=np.int64))
        # The last element of the vector stores flights that couldn't be scheduled.
        baseline[T] = max(0, baseline_total - int(np.sum(baseline[:T], dtype=np.int64)))

        # `schedule` starts as a copy of demand and is then modified by regulations.
        schedule = demand.copy()
        reg = flow_to_reg.get(int(fid))
        if reg is not None:
            wins = [int(w) for w in getattr(reg, "time_windows", []) or []]
            wins_in = [w for w in wins if 0 <= int(w) < T]
            if wins_in:
                start_bin = min(wins_in)
                end_bin = max(wins_in)
                # Distribute the hourly rate to get per-bin allowances.
                try:
                    allowance = distribute_hourly_rate_to_bins(
                        int(reg.rate),
                        bins_per_hour=bins_per_hour,
                        start_bin=start_bin,
                        end_bin=end_bin,
                    )
                except Exception:
                    allowance = np.zeros(max(0, end_bin - start_bin + 1), dtype=np.int64)

                # Apply the allowance to cap the schedule in each bin.
                for offset, b in enumerate(range(start_bin, end_bin + 1)):
                    if 0 <= b < T:
                        allow_val = int(allowance[offset]) if offset < allowance.size else 0
                        schedule[b] = int(min(schedule[b], allow_val))
                        # Record the cells (TV, bin) affected by this regulation.
                        target_cells_set.add((str(reg.location), int(b)))

        # Calculate total flights and released flights to find the spillover.
        total_flights = int(np.sum(demand[:T], dtype=np.int64))
        released = int(np.sum(schedule[:T], dtype=np.int64))
        schedule[T] = max(0, total_flights - released)

        # Store the final schedule and baseline vectors.
        n_f_t[int(fid)] = schedule.astype(int).tolist()
        n0_f_t[int(fid)] = baseline.astype(int).tolist()

    # Step 7: Determine ripple effects and the set of Traffic Volumes (TVs) to consider for scoring.
    # Resolve ripples from plan metadata using the same precedence as DFRegulationPlan.
    base_payload = plan.to_base_eval_payload(warn_on_auto_fallback=False)

    # First, try to get target cells and TVs directly from the plan's payload.
    targets_in = base_payload.get("targets") or {}
    target_cells, target_tvs = _cells_from_ranges(tvtw_indexer, targets_in) if targets_in else ([], [])
    if not target_cells:
        # Fallback: if not specified, derive target cells from the regulation windows.
        for reg in regs:
            for w in getattr(reg, "time_windows", []) or []:
                wi = int(w)
                if 0 <= wi < T:
                    target_cells_set.add((str(reg.location), wi))
        target_cells = sorted(target_cells_set, key=lambda x: (x[0], x[1]))
        target_tvs = sorted({tv for tv, _ in target_cells})

    # Determine ripple cells, which are areas affected by the regulations' downstream effects.
    ripple_cells: List[Tuple[str, int]] = []
    ripples_in = base_payload.get("ripples")
    auto_bins = base_payload.get("auto_ripple_time_bins")
    if isinstance(ripples_in, Mapping) and ripples_in:
        # Use explicitly defined ripple cells if provided.
        ripple_cells, ripple_tv_ids = _cells_from_ranges(tvtw_indexer, ripples_in)
    else:
        # Otherwise, compute them automatically based on flight trajectories.
        try:
            w = int(auto_bins) if auto_bins is not None else DEFAULT_AUTO_RIPPLE_TIME_BINS
        except Exception:
            w = DEFAULT_AUTO_RIPPLE_TIME_BINS
        ripple_cells = compute_auto_ripple_cells(
            indexer=tvtw_indexer,
            flight_list=flight_list,
            flight_ids=list(flow_map.keys()),
            window_bins=max(0, int(w)),
        )
        ripple_tv_ids = sorted({str(tv) for (tv, _b) in ripple_cells})

    # The `tv_filter` is the union of target TVs and ripple TVs.
    tv_filter = sorted(set(target_tvs) | set(ripple_tv_ids if "ripple_tv_ids" in locals() else []))
    if not tv_filter:
        # As a conservative fallback, use all TVs that are touched by the affected flights.
        tv_filter = sorted(tvtw_indexer.tv_id_to_idx.keys())

    # Step 8: Score the plan using a shared context for baseline and regulated scenarios.
    # Create an ObjectiveWeights object from the provided weights mapping.
    weight_kwargs: Dict[str, float] = {}
    if isinstance(weights, Mapping):
        for k, v in weights.items():
            if k in ObjectiveWeights.__dataclass_fields__:
                weight_kwargs[str(k)] = float(v)
    weights_obj = ObjectiveWeights(**weight_kwargs) if weight_kwargs else ObjectiveWeights()

    # Build a shared context for scoring. This pre-computes various metrics
    # to make the scoring process more efficient.
    context = build_score_context(
        flights_by_flow,
        indexer=tvtw_indexer,
        capacities_by_tv=cap_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        flight_list=flight_list,
        weights=weights_obj,
        tv_filter=tv_filter,
    )

    # Score the baseline scenario (before regulations).
    # `context.d_by_flow` represents the initial demand.
    J_before, comps_before, arts_before = score_with_context(
        context.d_by_flow,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=cap_by_tv,
        flight_list=flight_list,
        context=context,
        spill_mode="dump_to_next_bin",
    )
    # Score the regulated scenario.
    J_after, comps_after, arts_after = score_with_context(
        n_f_t,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=cap_by_tv,
        flight_list=flight_list,
        context=context,
        spill_mode="dump_to_next_bin",
    )

    # Extract results and package them into the DFPlanEvalResult dataclass.
    delays_min = arts_after.get("delays_min", {}) or {}
    delays_by_flight = {str(fid): int(val) for fid, val in delays_min.items()}
    objective_components = {str(k): float(v) for k, v in (comps_after or {}).items()}
    pre_objective = float(J_before)
    objective = float(J_after)
    delta_objective = float(J_before - J_after)

    return DFPlanEvalResult(
        delays_by_flight=delays_by_flight,
        objective_components=objective_components,
        objective=objective,
        pre_objective=pre_objective,
        delta_objective=delta_objective,
    )


__all__ = ["DFPlanEvalResult", "evaluate_df_regulation_plan"]

