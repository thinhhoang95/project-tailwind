from __future__ import annotations

"""
Evaluate a DFRegulationPlan via the flow objective using shared AppResources.

Overview
--------
- Prefers process-wide AppResources for indexer, flight list, capacities, and
  TV GDF. Raises an exception if resources are not available.
- Uses the resources-backed RegulationParser when regulations do not provide
  explicit target flights.
- Builds flows, constructs demand/regulated schedules, and scores baseline vs
  regulated using a shared ScoreContext.

Notes
-----
- This module intentionally does not call `preload_all()`. If resources weren’t
  preloaded by the caller, lazy access may initialize them from default paths.
- A runtime guard asserts the resources’ indexer `num_time_bins` matches the
  flight list’s `num_time_bins_per_tv` for consistency.

Public API:
    evaluate_df_regulation_plan(plan, *, indexer_path, flights_path,
                                capacities_path=None, weights=None,
                                include_excess_vector=False) -> DFPlanEvalResult

The `indexer_path` and `flights_path` parameters are retained for signature
compatibility but are ignored because this evaluator uses AppResources.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
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

try:
    # Prefer the resources-backed parser
    from project_tailwind.optimize.parser.regulation_parser_with_resources import (  # type: ignore
        RegulationParser,
    )
except Exception as _e:
    RegulationParser = None  # type: ignore[assignment]
    _REG_PARSER_IMPORT_ERR = _e
else:
    _REG_PARSER_IMPORT_ERR = None

try:
    from server_tailwind.core.resources import get_resources  # type: ignore
except Exception as _e:
    get_resources = None  # type: ignore
    _RES_IMPORT_ERR = _e
else:
    _RES_IMPORT_ERR = None


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
    # Split the 'HH:MM' string into hours and minutes.
    parts = str(hhmm).split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time string: {hhmm!r}")
    # Parse integer values for hours and minutes.
    hh = int(parts[0])
    mm = int(parts[1])
    # Calculate the total minutes from midnight.
    total_min = hh * 60 + mm
    # Determine the time bin by integer division.
    return int(total_min // int(time_bin_minutes))


def _window_to_bins(start: str, end: str, *, time_bin_minutes: int, T: int) -> List[int]:
    """Converts a time window (start and end 'HH:MM' strings) to a list of time bin indices."""
    # The end of the window is exclusive. "24:00" is a permissible value for `end`.
    # Calculate the starting bin, ensuring it's not negative.
    start_bin = max(0, _time_to_bin(start, time_bin_minutes=time_bin_minutes))
    # Calculate the ending bin (exclusive).
    end_bin_excl = _time_to_bin(end, time_bin_minutes=time_bin_minutes)
    # Special handling for "24:00" to mean the end of the time horizon.
    if end.startswith("24"):
        end_bin_excl = T
    # Clamp the end bin to be within the valid range [0, T].
    end_bin_excl = max(0, min(T, int(end_bin_excl)))
    # If the window is invalid or has zero length, return an empty list.
    if end_bin_excl <= start_bin:
        return []
    # Generate the list of bin indices for the window.
    return list(range(int(start_bin), int(end_bin_excl)))


def _dfplan_to_regulations(plan: DFRegulationPlan, *, indexer: TVTWIndexer, T: int) -> List[Regulation]:
    """Converts a DFRegulationPlan object into a list of Regulation objects."""
    regs: List[Regulation] = []
    # Get the duration of each time bin in minutes from the indexer.
    tbm = int(indexer.time_bin_minutes)
    # Iterate over each regulation defined in the plan.
    for r in plan.regulations:
        # Convert the regulation's time window from 'HH:MM' strings to a list of bin indices.
        wins = _window_to_bins(r.window_from, r.window_to, time_bin_minutes=tbm, T=T)
        # Create a `Regulation` object, which is a more detailed representation
        # used by the backend optimization and evaluation logic.
        reg = Regulation.from_components(
            location=str(r.tv_id),
            rate=int(r.allowed_rate_per_hour),
            time_windows=[int(b) for b in wins],
            # 'IC' is an internal code, a legacy detail of how regulations are filtered.
            filter_type="IC",  # Assuming 'IC' (internal code) filter type.
            # This filter value is a placeholder, as filters are not fully implemented here.
            filter_value="__",  # Placeholder filter value.
            # Explicitly list the flight IDs targeted by this regulation.
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
        # Choose the format string based on whether seconds are present.
        fmt = "%H:%M:%S" if ss.count(":") == 2 else "%H:%M"
        # Use a fixed date (Jan 1, 2025) to create datetime objects for time comparison.
        # The date itself is arbitrary and only used to provide a consistent frame for time parsing.
        return datetime.strptime(ss, fmt).replace(year=2025, month=1, day=1)

    # `cells` will store tuples of (traffic_volume_id, time_bin_index).
    cells: List[Tuple[str, int]] = []
    # `tvs` will store the IDs of traffic volumes that were successfully processed.
    tvs: List[str] = []
    # Iterate through the provided ranges, where keys are TV IDs and values are time window dicts.
    for tv, win in (ranges or {}).items():
        tv_id = str(tv)
        # Skip if the traffic volume ID is not recognized by the indexer.
        if tv_id not in idx.tv_id_to_idx:
            continue
        try:
            # Parse start and end times from the window mapping.
            t_from = _parse_time_hms(str(win.get("from")))
            t_to = _parse_time_hms(str(win.get("to")))
        except Exception:
            # Ignore malformed time windows.
            continue
        # Use the indexer to convert the datetime objects into a sequence of bin indices.
        bins = idx.bin_range_for_interval(t_from, t_to)
        # For each bin in the sequence, create a (tv_id, bin) tuple.
        for b in bins:
            cells.append((tv_id, int(b)))
        # Add the TV ID to the list of processed TVs.
        tvs.append(tv_id)
    return cells, tvs


def _build_capacities_by_tv(
    *,
    indexer: TVTWIndexer,
    flight_list: FlightList,
    capacities_path: Optional[str],
) -> Dict[str, np.ndarray]:
    """Build per-TV per-bin capacities with normalization using resources.

    Order of precedence:
      1) Use explicit `capacities_path` when provided.
      2) Otherwise, use AppResources.capacity_per_bin_matrix.

    If neither is available, raise an exception rather than falling back to
    project-default files.
    """

    # 1) Prefer an explicit file path if one is provided by the caller.
    # This allows for overriding default capacities for specific evaluations.
    if capacities_path:
        raw = build_bin_capacities(str(capacities_path), indexer)
        # `normalize_capacities` adjusts raw capacity values, e.g., to handle missing data.
        return normalize_capacities(raw)

    # 2) If no path is given, use shared server resources.
    if get_resources is None:
        raise RuntimeError(
            f"AppResources are unavailable (import error: {_RES_IMPORT_ERR}) and capacities_path not provided"
        )
    _res = get_resources()
    mat = getattr(_res, "capacity_per_bin_matrix", None)
    if mat is None:
        raise RuntimeError("AppResources has no capacity_per_bin_matrix; provide capacities_path explicitly")
    # Reconstruct the dictionary mapping TV IDs to their capacity arrays.
    out: Dict[str, np.ndarray] = {}
    T = int(indexer.num_time_bins)
    for tv_id, row_idx in flight_list.tv_id_to_idx.items():
        arr = np.asarray(mat[int(row_idx), :T], dtype=np.float64)
        out[str(tv_id)] = arr
    return normalize_capacities(out)


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

    Uses process-wide AppResources for indexer/flight list. `indexer_path` and
    `flights_path` are ignored and retained only for backward compatibility.

    Returns DFPlanEvalResult with per-flight delays and objective breakdown.
    """
    # Step 1: Load essential artifacts for evaluation from AppResources.
    if get_resources is None:
        raise RuntimeError(
            f"AppResources are unavailable (import error: {_RES_IMPORT_ERR}); cannot evaluate DF plan"
        )
    res = get_resources()
    tvtw_indexer = res.indexer
    flight_list = res.flight_list  # FlightListWithDelta (subclass of FlightList)

    # Guard: Ensure resource time-axis consistency
    T_idx = int(getattr(tvtw_indexer, "num_time_bins"))
    T_fl = int(getattr(flight_list, "num_time_bins_per_tv"))
    if T_idx != T_fl:
        raise RuntimeError(
            f"Resource mismatch: indexer.num_time_bins ({T_idx}) != flight_list.num_time_bins_per_tv ({T_fl})"
        )

    # Initialize resources-backed RegulationParser when available; this is used
    # to expand target flights if a regulation omits them.
    if RegulationParser is None:
        raise RuntimeError(
            f"RegulationParser with resources is unavailable (import error: {_REG_PARSER_IMPORT_ERR})"
        )
    parser = RegulationParser(resources=res)  # type: ignore[call-arg]

    # Step 2: Translate the high-level DFRegulationPlan into a list of detailed
    # `Regulation` objects, which are understood by the core evaluation logic.
    T = int(tvtw_indexer.num_time_bins)
    regs = _dfplan_to_regulations(plan, indexer=tvtw_indexer, T=T)

    # Ensure each regulation has an explicit target flight list; if empty or None,
    # compute it using the resources-backed parser.
    for r in regs:
        if not (getattr(r, "target_flight_ids", None) or []):
            try:
                ids = parser.parse(r)
            except Exception as e:
                raise RuntimeError(f"Failed to expand flights for regulation at {r.location}: {e}") from e
            r.target_flight_ids = list(ids or [])

    # Step 4: Build the capacity profiles for each traffic volume (TV).
    # Capacities represent the maximum number of flights a TV can handle in a given
    # time bin and are crucial for identifying and scoring congestion.
    cap_by_tv = _build_capacities_by_tv(
        indexer=tvtw_indexer, flight_list=flight_list, capacities_path=capacities_path
    )

    # Step 5: Build "flows" by mapping flights to their corresponding regulations.
    # A flow is a group of flights that are all affected by the same regulation.
    # This modeling simplifies the problem by grouping flights with similar constraints.
    # Each flight is assigned to the *first* regulation in the list that targets it.
    flow_to_reg: Dict[int, Regulation] = {}  # Maps flow index to its Regulation object.
    flow_map: Dict[str, int] = {}  # Maps flight ID to its assigned flow index.
    assigned: set[str] = set()  # Tracks flights that have already been assigned to a flow.
    for idx, reg in enumerate(regs):
        flow_to_reg[idx] = reg
        for fid in reg.target_flight_ids or []:
            s = str(fid)
            # Ensure each flight is assigned to only one flow.
            if s in assigned:
                continue
            assigned.add(s)
            flow_map[s] = idx

    # Prepare inputs required for the flow scheduling simulation.
    flights_by_flow_raw, _ = prepare_flow_scheduling_inputs(
        flight_list=flight_list,
        flow_map=flow_map,
        hotspot_ids=[r.location for r in regs],
        flight_ids=list(flow_map.keys()),
    )
    # Filter out any flows that have no flights assigned to them.
    flights_by_flow: Dict[int, List[Dict[str, Any]]] = {
        int(k): v for k, v in flights_by_flow_raw.items() if v
    }

    # Handle the edge case where there are no flights or flows to evaluate.
    # If so, return a result with all metrics set to zero.
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
    # These vectors represent the number of flights in a flow scheduled in each time bin.
    # `n0_f_t` represents the baseline demand (schedule without regulations).
    # `n_f_t` represents the regulated schedule, where counts are capped by allowances.
    bins_per_hour = int(tvtw_indexer.rolling_window_size())
    n_f_t: Dict[int, List[int]] = {}  # Regulated schedule for each flow.
    n0_f_t: Dict[int, List[int]] = {}  # Baseline demand for each flow.
    target_cells_set: set[Tuple[str, int]] = set()  # Collects (TV, bin) pairs affected by regulations.
    for fid, specs in flights_by_flow.items():
        # Initialize a demand vector for the current flow with T+1 bins.
        # The extra bin at index T is used to account for "spillover" flights that
        # cannot be scheduled within the time horizon.
        demand = np.zeros(T + 1, dtype=np.int64)
        for spec in specs or []:
            rb = spec.get("requested_bin") if isinstance(spec, dict) else None
            try:
                b = int(rb)
            except Exception:
                continue
            # Clamp the bin index to be within the valid range [0, T-1].
            if b < 0:
                b = 0
            if b >= T:
                b = T - 1
            demand[b] += 1

        # `baseline` is the initial, unregulated demand distribution.
        baseline = demand.copy()
        baseline_total = int(np.sum(demand[:T], dtype=np.int64))
        # The last element of the vector stores flights that couldn't be scheduled.
        baseline[T] = max(0, baseline_total - int(np.sum(baseline[:T], dtype=np.int64)))

        # `schedule` starts as a copy of demand and is then modified by regulation allowances.
        schedule = demand.copy()
        reg = flow_to_reg.get(int(fid))
        if reg is not None:
            wins = [int(w) for w in getattr(reg, "time_windows", []) or []]
            wins_in = [w for w in wins if 0 <= int(w) < T]
            if wins_in:
                start_bin = min(wins_in)
                end_bin = max(wins_in)
                # Distribute the regulation's hourly rate to get per-bin allowances.
                try:
                    allowance = distribute_hourly_rate_to_bins(
                        int(reg.rate),
                        bins_per_hour=bins_per_hour,
                        start_bin=start_bin,
                        end_bin=end_bin,
                    )
                except Exception:
                    # If rate distribution fails, assume a zero allowance.
                    allowance = np.zeros(max(0, end_bin - start_bin + 1), dtype=np.int64)

                # Apply the allowance to cap the scheduled flight count in each affected bin.
                for offset, b in enumerate(range(start_bin, end_bin + 1)):
                    if 0 <= b < T:
                        allow_val = int(allowance[offset]) if offset < allowance.size else 0
                        schedule[b] = int(min(schedule[b], allow_val))
                        # Record the specific cell (TV, bin) affected by this regulation.
                        target_cells_set.add((str(reg.location), int(b)))

        # Recalculate the spillover for the regulated schedule.
        total_flights = int(np.sum(demand[:T], dtype=np.int64))
        released = int(np.sum(schedule[:T], dtype=np.int64))
        schedule[T] = max(0, total_flights - released)

        # Store the final regulated schedule and baseline demand vectors.
        n_f_t[int(fid)] = schedule.astype(int).tolist()
        n0_f_t[int(fid)] = baseline.astype(int).tolist()

    # Step 7: Determine ripple effects and the set of Traffic Volumes (TVs) to consider for scoring.
    # Ripple effects are the downstream consequences of regulations. Accurately identifying
    # the scope of these effects is key to a comprehensive evaluation.
    # We resolve ripples from plan metadata using the same precedence as in DFRegulationPlan itself.
    base_payload = plan.to_base_eval_payload(warn_on_auto_fallback=False)

    # First, try to get target cells and TVs directly from the plan's payload, if specified.
    targets_in = base_payload.get("targets") or {}
    target_cells, target_tvs = _cells_from_ranges(tvtw_indexer, targets_in) if targets_in else ([], [])
    # If target cells are not explicitly defined in the plan, derive them from the
    # time windows and locations of the regulations themselves.
    if not target_cells:
        for reg in regs:
            for w in getattr(reg, "time_windows", []) or []:
                wi = int(w)
                if 0 <= wi < T:
                    target_cells_set.add((str(reg.location), wi))
        target_cells = sorted(target_cells_set, key=lambda x: (x[0], x[1]))
        target_tvs = sorted({tv for tv, _ in target_cells})

    # Determine ripple cells, which are areas indirectly affected by the regulations'
    # downstream consequences (e.g., delayed flights causing congestion elsewhere).
    ripple_cells: List[Tuple[str, int]] = []
    ripples_in = base_payload.get("ripples")
    auto_bins = base_payload.get("auto_ripple_time_bins")
    if isinstance(ripples_in, Mapping) and ripples_in:
        # Use explicitly defined ripple cells from the plan if they are provided.
        ripple_cells, ripple_tv_ids = _cells_from_ranges(tvtw_indexer, ripples_in)
    else:
        # Otherwise, compute ripple cells automatically based on the trajectories
        # of the flights affected by the regulations.
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

    # The `tv_filter` defines the full set of TVs to consider during scoring. It is
    # the union of TVs directly targeted by regulations and those affected by ripples.
    tv_filter = sorted(set(target_tvs) | set(ripple_tv_ids if "ripple_tv_ids" in locals() else []))
    if not tv_filter:
        # As a conservative fallback, if no target or ripple TVs are identified,
        # use all TVs that are touched by any of the affected flights. This ensures
        # we don't miss any potential impacts.
        tv_filter = sorted(tvtw_indexer.tv_id_to_idx.keys())

    # Step 8: Score the plan using a shared context for both baseline and regulated scenarios.
    # This involves calculating an objective function that balances various costs,
    # such as delay, capacity violations, and regulation adherence.
    # Create an ObjectiveWeights object from the provided weights mapping.
    weight_kwargs: Dict[str, float] = {}
    if isinstance(weights, Mapping):
        for k, v in weights.items():
            if k in ObjectiveWeights.__dataclass_fields__:
                weight_kwargs[str(k)] = float(v)
    weights_obj = ObjectiveWeights(**weight_kwargs) if weight_kwargs else ObjectiveWeights()

    # Build a shared context for scoring. This pre-computes various metrics
    # and matrices (e.g., flight-to-TV mappings) to make the scoring of both
    # baseline and regulated scenarios more efficient.
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

    # Score the baseline scenario (before regulations are applied).
    # `context.d_by_flow` holds the initial demand distribution for each flow.
    J_before, comps_before, arts_before = score_with_context(
        context.d_by_flow,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=cap_by_tv,
        flight_list=flight_list,
        context=context,
        spill_mode="dump_to_next_bin",
    )
    # Score the regulated scenario, using the capped schedule `n_f_t`.
    J_after, comps_after, arts_after = score_with_context(
        n_f_t,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=cap_by_tv,
        flight_list=flight_list,
        context=context,
        spill_mode="dump_to_next_bin",
    )

    # Step 9: Extract final results and package them into the DFPlanEvalResult dataclass.
    # This provides a structured output for the caller.
    delays_min = arts_after.get("delays_min", {}) or {}
    delays_by_flight = {str(fid): int(val) for fid, val in delays_min.items()}
    objective_components = {str(k): float(v) for k, v in (comps_after or {}).items()}
    pre_objective = float(J_before)
    objective = float(J_after)
    # The delta represents the improvement (or degradation) from the regulation plan.
    # A positive delta indicates a reduction in the objective score (cost), which is good.
    delta_objective = float(J_before - J_after)

    return DFPlanEvalResult(
        delays_by_flight=delays_by_flight,
        objective_components=objective_components,
        objective=objective,
        pre_objective=pre_objective,
        delta_objective=delta_objective,
    )


__all__ = ["DFPlanEvalResult", "evaluate_df_regulation_plan"]
