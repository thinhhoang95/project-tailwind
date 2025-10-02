"""Regulation data models and payload builders for downstream APIs."""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from parrhesia.flow_agent35.regen.engine import (  # type: ignore
        AUTO_RIPPLE_DILATION_BINS as _DEFAULT_AUTO_RIPPLE_BINS,
    )
except Exception:  # pragma: no cover - defensive fallback when regen is unavailable
    _DEFAULT_AUTO_RIPPLE_BINS = 2

DEFAULT_AUTO_RIPPLE_TIME_BINS = max(0, int(_DEFAULT_AUTO_RIPPLE_BINS)) or 2


def _normalize_time_str(value: Any) -> str:
    """Ensure time strings follow HH:MM or HH:MM:SS with zero padding.

    Accepts multiple formats and attempts to parse them into a standard,
    lexicographically comparable format.

    Supported input formats:
    - "HHMM"
    - "HHMMSS"
    - "HH:MM"
    - "HH:MM:SS"

    It validates time components (hour, minute, second) and allows "24:00"
    as an exclusive upper bound for time windows.

    Args:
        value: The time string or object to normalize.

    Returns:
        A normalized time string in "HH:MM" or "HH:MM:SS" format.

    Raises:
        ValueError: If the time string is empty or in an invalid format.
    """
    s = str(value).strip()
    if not s:
        raise ValueError("time string cannot be empty")

    # Handle formats without colons, like "HHMM" or "HHMMSS"
    if ":" not in s:
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) == 4:  # HHMM
            s = f"{digits[:2]}:{digits[2:]}"
        elif len(digits) == 6:  # HHMMSS
            s = f"{digits[:2]}:{digits[2:4]}:{digits[4:]}"
        else:
            raise ValueError(f"Invalid time format '{value}'")

    # Parse components from "HH:MM" or "HH:MM:SS"
    parts = s.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid time format '{value}'")

    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid time components in '{value}'") from exc

    # Validate time component ranges
    if hour == 24 and minute == 0 and second == 0:
        hour = 24  # allow 24:00 as exclusive upper bound
    elif not (0 <= hour < 24):
        raise ValueError(f"Hour out of range in '{value}'")
    if not (0 <= minute < 60):
        raise ValueError(f"Minute out of range in '{value}'")
    if not (0 <= second < 60):
        raise ValueError(f"Second out of range in '{value}'")

    # Return in the simplest valid format
    if len(parts) == 2 and second == 0:
        return f"{hour:02d}:{minute:02d}"
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _min_time(a: str, b: str) -> str:
    """Return the earlier of two zero-padded time strings."""
    return a if a <= b else b


def _max_time(a: str, b: str) -> str:
    """Return the later of two zero-padded time strings."""
    return a if a >= b else b


def _normalize_flights(flights: Iterable[Any]) -> Tuple[str, ...]:
    """Convert an iterable of flight identifiers to a tuple of strings."""
    return tuple(str(f) for f in flights)


def _normalize_ripples(
    ripples: Optional[Mapping[Any, Mapping[str, Any]]],
) -> Optional[Dict[str, Dict[str, str]]]:
    """Normalize a ripples mapping to use string keys and validated time windows.

    Args:
        ripples: A mapping from TV-like keys to time window mappings.

    Returns:
        A normalized dictionary with string TV IDs and "from"/"to" time strings,
        or None if the input is empty.

    Raises:
        ValueError: If a ripple window is not a mapping or has invalid time strings.
    """
    if not ripples:
        return None
    normalized: Dict[str, Dict[str, str]] = {}
    for tv, window in ripples.items():
        if not isinstance(window, Mapping):
            raise ValueError("Each ripple window must be a mapping with 'from'/'to'")
        try:
            wf = _normalize_time_str(window.get("from"))
            wt = _normalize_time_str(window.get("to"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid ripple window for TV '{tv}'") from exc
        normalized[str(tv)] = {"from": wf, "to": wt}
    return normalized


def _coerce_auto_bins(value: Any) -> Optional[int]:
    """Coerce a value to a non-negative integer for auto-ripple bins.

    Returns None if coercion fails.
    """
    if value is None:
        return None
    try:
        bins = int(value)
    except Exception:
        return None
    return max(0, int(bins))


def _format_window_from_bins(
    start_bin: int,
    end_bin: int,
    *,
    time_bin_minutes: int,
) -> Tuple[str, str]:
    """Convert time bin indices into a formatted time window ("HH:MM", "HH:MM").

    Args:
        start_bin: The index of the starting time bin.
        end_bin: The index of the ending time bin.
        time_bin_minutes: The duration of each time bin in minutes.

    Returns:
        A tuple containing the start and end time strings.

    Raises:
        ValueError: if time_bin_minutes is not positive.
    """
    if time_bin_minutes <= 0:
        raise ValueError("time_bin_minutes must be positive")

    def _bin_start(bin_idx: int) -> str:
        """Calculate the start time of a bin."""
        minutes = int(bin_idx) * time_bin_minutes
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"

    def _bin_end(bin_idx: int) -> str:
        """Calculate the end time of a bin (exclusive upper bound)."""
        minutes = (int(bin_idx) + 1) * time_bin_minutes
        if minutes >= 24 * 60:
            return "24:00"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"

    start_label = _bin_start(int(start_bin))
    end_label = _bin_end(int(end_bin))
    return start_label, end_label


@dataclass(frozen=True)
class DFRegulation:
    """Represents a single demand-capacity regulation.

    A regulation defines a control on a traffic volume (TV) over a specific
    time window, affecting a list of flights and specifying an allowed hourly rate.

    Attributes:
        id: A unique identifier for the regulation.
        tv_id: The identifier of the traffic volume (TV) being controlled.
        window_from: The start time of the regulation window ("HH:MM" or "HH:MM:SS").
        window_to: The end time of the regulation window ("HH:MM" or "HH:MM:SS").
        flights: An immutable tuple of flight identifiers affected by this regulation.
        allowed_rate_per_hour: The maximum number of flights allowed per hour.
        metadata: An optional dictionary for storing extra information.
    """

    id: str
    tv_id: str
    window_from: str
    window_to: str
    flights: Tuple[str, ...]
    allowed_rate_per_hour: int
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Perform validation and normalization after initialization."""
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "tv_id", str(self.tv_id))
        object.__setattr__(self, "window_from", _normalize_time_str(self.window_from))
        object.__setattr__(self, "window_to", _normalize_time_str(self.window_to))
        object.__setattr__(self, "flights", _normalize_flights(self.flights))
        object.__setattr__(self, "allowed_rate_per_hour", int(self.allowed_rate_per_hour))
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping when provided")

    @staticmethod
    def from_flights(
        *,
        id: Any,
        tv_id: Any,
        window_from: Any,
        window_to: Any,
        flights: Iterable[Any],
        allowed_rate_per_hour: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "DFRegulation":
        """Create a DFRegulation instance with type coercion and normalization.

        This factory method provides a convenient way to construct a regulation
        while ensuring all inputs are converted to their correct types.

        Args:
            id: The regulation identifier.
            tv_id: The controlled TV identifier.
            window_from: The start of the regulation window.
            window_to: The end of the regulation window.
            flights: An iterable of flight identifiers.
            allowed_rate_per_hour: The allowed hourly rate.
            metadata: Optional metadata.

        Returns:
            A new, validated DFRegulation instance.
        """
        return DFRegulation(
            id=str(id),
            tv_id=str(tv_id),
            window_from=_normalize_time_str(window_from),
            window_to=_normalize_time_str(window_to),
            flights=_normalize_flights(flights),
            allowed_rate_per_hour=int(allowed_rate_per_hour),
            metadata=dict(metadata) if metadata is not None else None,
        )

    def targets_map(self) -> Dict[str, Dict[str, str]]:
        """Return the regulation's target window in the format for downstream APIs."""
        return {self.tv_id: {"from": self.window_from, "to": self.window_to}}


@dataclass
class DFRegulationPlan:
    """A collection of DFRegulations with helpers for metrics and payload generation.

    This class acts as a container for a set of regulations that constitute a
    single coherent plan. It provides methods to calculate metrics on the plan
    and to generate payloads for various downstream APIs.

    Attributes:
        regulations: A list of DFRegulation objects in the plan.
        metadata: Optional dictionary for plan-level metadata.
        _auto_ripple_warned: Internal flag to ensure the auto-ripple warning
            is shown only once per instance.
    """

    regulations: List[DFRegulation] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    _auto_ripple_warned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping when provided")

    # --- Metrics -----------------------------------------------------------------
    def number_of_regulations(self) -> int:
        """Return the total number of regulations in the plan."""
        return len(self.regulations)

    def number_of_flows(self) -> int:
        """Return the total number of flows, which is one per regulation."""
        return len(self.regulations)

    def number_of_flights_affected(self) -> int:
        """Return the number of unique flights affected by the plan."""
        unique: set[str] = set()
        for regulation in self.regulations:
            unique.update(regulation.flights)
        return len(unique)

    # --- Mutation helpers --------------------------------------------------------
    def add(self, regulation: DFRegulation) -> None:
        """Add a single regulation to the plan."""
        self.regulations.append(regulation)

    def extend(self, regs: Iterable[DFRegulation]) -> None:
        """Extend the plan with an iterable of regulations."""
        self.regulations.extend(regs)

    # --- Payload builders --------------------------------------------------------
    def _build_flows_and_targets(self) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
        """Construct the 'flows' and 'targets' mappings for API payloads.

        This method iterates through the regulations to create two key structures:
        - flows: A mapping from a stable, zero-padded flow key to a list of
          flight IDs. Each regulation corresponds to one flow.
        - targets: A mapping from a TV ID to its control window. If multiple
          regulations target the same TV, their windows are merged to form the
          union (min of start times, max of end times).

        Returns:
            A tuple containing the flows dictionary and the targets dictionary.
        """
        flows: Dict[str, List[str]] = {}
        targets: Dict[str, Dict[str, str]] = {}
        total = len(self.regulations)
        width = max(1, len(str(total - 1))) if total else 1
        for idx, regulation in enumerate(self.regulations):
            # Assign a stable, zero-padded flow key based on regulation order
            flow_key = f"{idx:0{width}d}"
            flows[flow_key] = list(regulation.flights)

            # Merge regulation windows for the same TV
            if regulation.tv_id in targets:
                current = targets[regulation.tv_id]
                current["from"] = _min_time(current["from"], regulation.window_from)
                current["to"] = _max_time(current["to"], regulation.window_to)
            else:
                targets[regulation.tv_id] = {
                    "from": regulation.window_from,
                    "to": regulation.window_to,
                }
        return flows, targets

    def _resolve_ripples(
        self,
        *,
        ripples: Optional[Mapping[Any, Mapping[str, Any]]],
        auto_ripple_time_bins: Optional[Any],
        warn_on_auto_fallback: bool,
    ) -> Tuple[Optional[Dict[str, Dict[str, str]]], Optional[int]]:
        """Determine the ripple strategy based on inputs and metadata.

        This method implements the precedence logic for ripple effects:
        1. Explicit `ripples` provided as an argument.
        2. `ripples` found in the plan's `metadata`.
        3. Explicit `auto_ripple_time_bins` provided as an argument.
        4. `auto_ripple_time_bins` found in the plan's `metadata`.
        5. If none of the above are provided, it falls back to a default number
           of auto-ripple bins and issues a warning.

        Args:
            ripples: Explicitly provided ripple windows.
            auto_ripple_time_bins: Explicitly provided number of auto-ripple bins.
            warn_on_auto_fallback: Whether to warn on auto-ripple fallback.

        Returns:
            A tuple containing either a normalized ripples mapping or None,
            and either the number of auto-ripple bins or None. Only one of the
            two can be non-None.
        """
        meta = self.metadata or {}
        meta_ripples = meta.get("ripples") if isinstance(meta, Mapping) else None
        meta_auto = meta.get("auto_ripple_time_bins") if isinstance(meta, Mapping) else None

        # Precedence 1 & 2: Explicit ripples from args or metadata
        resolved_ripples = None
        if ripples is not None:
            resolved_ripples = _normalize_ripples(ripples)
        elif meta_ripples is not None:
            resolved_ripples = _normalize_ripples(meta_ripples)

        if resolved_ripples:
            return resolved_ripples, None

        # Precedence 3 & 4: Explicit auto-ripple bins from args or metadata
        resolved_auto: Optional[int] = None
        candidate_auto = auto_ripple_time_bins if auto_ripple_time_bins is not None else meta_auto
        coerced_auto = _coerce_auto_bins(candidate_auto)
        if coerced_auto is not None:
            resolved_auto = coerced_auto
            return None, resolved_auto

        # Precedence 5: Fallback to default auto-ripple bins
        resolved_auto = DEFAULT_AUTO_RIPPLE_TIME_BINS
        if warn_on_auto_fallback and not self._auto_ripple_warned:
            warnings.warn(
                "No explicit ripples provided; defaulting to auto-ripple with "
                f"±{resolved_auto} time bins.",
                UserWarning,
                stacklevel=3,
            )
            # Ensure the warning is only shown once per plan instance
            self._auto_ripple_warned = True
        elif not self._auto_ripple_warned:
            self._auto_ripple_warned = True  # Also mark as warned if warnings are suppressed
        return None, resolved_auto

    def to_base_eval_payload(
        self,
        *,
        ripples: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        auto_ripple_time_bins: Optional[Any] = None,
        warn_on_auto_fallback: bool = True,
        indexer_path: Optional[Any] = None,
        flights_path: Optional[Any] = None,
        capacities_path: Optional[Any] = None,
        weights: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a payload for the base evaluation API.

        Args:
            ripples: Explicit ripple windows to include in the payload.
            auto_ripple_time_bins: Number of bins for automatic ripple calculation.
                If both ripples and auto_ripple_time_bins are unspecified,
                a default auto-ripple setting is used.
            warn_on_auto_fallback: Whether to warn when falling back to default
                auto-ripples.
            indexer_path: Optional path to a TV/time-window indexer artifact.
            flights_path: Optional path to a flight occupancy matrix artifact.
            capacities_path: Optional path to a capacities GeoJSON artifact.
            weights: Optional overrides for objective function weights.
            extra: Optional extra key-value pairs to add to the payload root.

        Returns:
            A dictionary formatted as a payload for the base evaluation API.
        """
        flows, targets = self._build_flows_and_targets()
        payload: Dict[str, Any] = {
            "flows": flows,
            "targets": targets,
        }

        # Determine the ripple strategy and add to payload
        resolved_ripples, resolved_auto = self._resolve_ripples(
            ripples=ripples,
            auto_ripple_time_bins=auto_ripple_time_bins,
            warn_on_auto_fallback=warn_on_auto_fallback,
        )

        if resolved_ripples is not None:
            payload["ripples"] = resolved_ripples
        elif resolved_auto is not None:
            payload["auto_ripple_time_bins"] = int(resolved_auto)

        # Add optional paths for external data artifacts
        if indexer_path is not None:
            payload["indexer_path"] = str(indexer_path)
        if flights_path is not None:
            payload["flights_path"] = str(flights_path)
        if capacities_path is not None:
            payload["capacities_path"] = str(capacities_path)

        # Add optional weight overrides and extra parameters
        if weights:
            payload["weights"] = dict(weights)
        if extra:
            payload.update(dict(extra))
        return payload

    def to_autorate_payload(
        self,
        *,
        ripples: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        auto_ripple_time_bins: Optional[Any] = None,
        warn_on_auto_fallback: bool = True,
        weights: Optional[Mapping[str, Any]] = None,
        sa_params: Optional[Mapping[str, Any]] = None,
        spill_mode: Optional[Any] = None,
        release_rate_for_spills: Optional[Any] = None,
        indexer_path: Optional[Any] = None,
        flights_path: Optional[Any] = None,
        capacities_path: Optional[Any] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a payload for the automatic rate adjustment API.

        This extends the base evaluation payload with additional parameters
        specific to the simulated annealing optimization process.

        Args:
            ripples: Explicit ripple windows.
            auto_ripple_time_bins: Bins for automatic ripple calculation.
            warn_on_auto_fallback: Whether to warn on auto-ripple fallback.
            weights: Optional objective function weight overrides.
            sa_params: Parameters for the simulated annealing algorithm.
            spill_mode: Strategy for handling demand that exceeds capacity.
            release_rate_for_spills: Rate for releasing spilled demand.
            indexer_path: Optional path to an indexer artifact.
            flights_path: Optional path to a flight matrix artifact.
            capacities_path: Optional path to a capacities artifact.
            extra: Optional extra key-value pairs to add to the payload.

        Returns:
            A dictionary formatted as a payload for the auto-rate API.
        """
        payload = self.to_base_eval_payload(
            ripples=ripples,
            auto_ripple_time_bins=auto_ripple_time_bins,
            warn_on_auto_fallback=warn_on_auto_fallback,
            indexer_path=indexer_path,
            flights_path=flights_path,
            capacities_path=capacities_path,
            weights=weights,
            extra=extra,
        )
        # Add parameters specific to the auto-rate adjustment API
        if sa_params:
            payload["sa_params"] = dict(sa_params)
        if spill_mode is not None:
            payload["spill_mode"] = str(spill_mode)
        if release_rate_for_spills is not None:
            payload["release_rate_for_spills"] = int(release_rate_for_spills)
        return payload

    # --- Constructors ------------------------------------------------------------
    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        default_allowed_rate: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "DFRegulationPlan":
        """Construct a DFRegulationPlan from a raw payload dictionary.

        This factory method enables round-tripping from a payload back to a
        structured DFRegulationPlan object. It reconstructs individual regulations
        by associating flows with their corresponding TV targets.

        The association logic is as follows:
        - If `metadata.flow_to_tv` is provided, it is used as the explicit mapping.
        - If there is only one target TV, all flows are assigned to it.
        - If the number of flows equals the number of targets, they are paired by
          their order in the respective dictionaries.
        - Otherwise, the mapping cannot be inferred, and an error is raised.

        Args:
            payload: The input payload, typically from a downstream API.
            default_allowed_rate: The allowed rate to assign to regulations if not
                specified in `metadata.allowed_rates`.
            metadata: Optional metadata to aid in reconstruction, including
                `allowed_rates` and `flow_to_tv` mappings.

        Returns:
            A new DFRegulationPlan instance.

        Raises:
            ValueError: If the payload is missing required keys or if the
                flow-to-TV mapping cannot be determined.
        """
        flows_in = payload.get("flows")
        targets_in = payload.get("targets")
        if not isinstance(flows_in, Mapping) or not flows_in:
            raise ValueError("payload must include non-empty 'flows' mapping")
        if not isinstance(targets_in, Mapping) or not targets_in:
            raise ValueError("payload must include non-empty 'targets' mapping")

        # Prepare for reconstruction by converting input items to lists
        target_items = list((str(tv), window) for tv, window in targets_in.items())
        flow_items = list(flows_in.items())

        # Extract relevant metadata for reconstructing rates and flow-TV mappings
        plan_metadata = dict(metadata) if metadata is not None else {}
        allowed_rates_meta = {}
        flow_to_tv_meta = {}
        if isinstance(metadata, Mapping):
            allowed_rates_meta = {
                str(k): int(v)
                for k, v in (metadata.get("allowed_rates", {}) or {}).items()
            }
            flow_to_tv_meta = {
                str(k): str(v)
                for k, v in (metadata.get("flow_to_tv", {}) or {}).items()
            }

        regulations: List[DFRegulation] = []
        # Pre-normalize all target windows for efficient lookup
        targets_by_tv = {
            str(tv): {
                "from": _normalize_time_str(window.get("from")),
                "to": _normalize_time_str(window.get("to")),
            }
            for tv, window in target_items
            if isinstance(window, Mapping)
        }

        # Re-create each regulation by associating a flow with a target
        for idx, (flow_key, flights_value) in enumerate(flow_items):
            s_flow_key = str(flow_key)
            # Determine the TV for the current flow using metadata or inference
            tv_id = flow_to_tv_meta.get(s_flow_key)
            if tv_id is None:
                if len(target_items) == 1:
                    # If only one TV, all flows map to it
                    tv_id = str(target_items[0][0])
                elif len(target_items) == len(flow_items):
                    # If #flows == #targets, assume a 1-to-1 mapping by order
                    tv_id = str(target_items[idx][0])
                else:
                    raise ValueError("Cannot infer TV for flow; provide metadata.flow_to_tv")
            window = targets_by_tv.get(tv_id)
            if window is None:
                raise ValueError(f"Missing target window for TV '{tv_id}'")

            # Normalize the list of flights for the current flow
            flights_iter: Sequence[Any]
            if isinstance(flights_value, Sequence) and not isinstance(flights_value, (str, bytes)):
                flights_iter = flights_value
            else:
                flights_iter = []

            flights: List[Any] = []
            for item in flights_iter:
                if isinstance(item, Mapping):
                    # Handle cases where flights are provided as dicts (e.g., {"flight_id": "..."})
                    flights.append(item.get("flight_id"))
                else:
                    flights.append(item)

            # Reconstruct the regulation with its allowed rate
            allowed = allowed_rates_meta.get(s_flow_key, default_allowed_rate)
            regulations.append(
                DFRegulation.from_flights(
                    id=s_flow_key,
                    tv_id=tv_id,
                    window_from=window["from"],
                    window_to=window["to"],
                    flights=flights,
                    allowed_rate_per_hour=allowed,
                )
            )

        return cls(regulations=regulations, metadata=plan_metadata or None)

    @classmethod
    def from_proposal(
        cls,
        proposal: "Proposal",
        flights_by_flow: Mapping[Any, Sequence[Any]],
        *,
        time_bin_minutes: Optional[int] = None,
    ) -> "DFRegulationPlan":
        """Construct a DFRegulationPlan from a `regen` proposal object.

        This provides an interoperability path from the `regen` traffic flow
        proposal engine to this regulation plan structure.

        Args:
            proposal: A `Proposal` object from the `regen` engine.
            flights_by_flow: A mapping from flow ID to the sequence of flights
                in that flow.
            time_bin_minutes: The duration of a time bin in minutes. If not
                provided, it is inferred from the proposal's diagnostics.

        Returns:
            A new DFRegulationPlan instance based on the proposal.
        """
        try:
            from parrhesia.flow_agent35.regen.types import Proposal  # noqa: F401
        except Exception:  # pragma: no cover - type check only
            pass

        # Infer time_bin_minutes from diagnostics if not provided
        diag = getattr(proposal, "diagnostics", {}) or {}
        inferred_minutes = time_bin_minutes
        if inferred_minutes is None:
            try:
                inferred_minutes = int(diag.get("time_bin_minutes"))  # type: ignore[arg-type]
            except Exception:
                inferred_minutes = None
        if inferred_minutes is None:
            inferred_minutes = 30  # Fallback if not found in diagnostics

        # Convert the proposal's time bin window to "HH:MM" format
        start_label, end_label = _format_window_from_bins(
            int(proposal.window.start_bin),
            int(proposal.window.end_bin),
            time_bin_minutes=int(inferred_minutes),
        )

        regulations: List[DFRegulation] = []
        # Create a regulation for each flow described in the proposal
        for flow_entry in proposal.flows_info:
            flow_id = flow_entry.get("flow_id")
            tv_id = flow_entry.get("control_tv_id") or proposal.controlled_volume
            allowed_rate = int(flow_entry.get("R_i", 0))

            # Extract additional metadata from the proposal for context
            baseline_rate = flow_entry.get("r0_i")
            lambda_cut = flow_entry.get("lambda_cut_i")

            # Retrieve the list of flights for this flow
            flights_spec = (
                flights_by_flow.get(flow_id)
                or flights_by_flow.get(str(flow_id))
                or []
            )
            flight_ids: List[Any] = []
            for item in flights_spec:
                if isinstance(item, Mapping):
                    flight_ids.append(item.get("flight_id"))
                else:
                    flight_ids.append(item)

            reg_metadata: Dict[str, Any] = {
                "baseline_rate_per_hour": baseline_rate,
                "cut_per_hour": lambda_cut,
            }
            regulations.append(
                DFRegulation.from_flights(
                    id=flow_id,
                    tv_id=tv_id,
                    window_from=start_label,
                    window_to=end_label,
                    flights=flight_ids,
                    allowed_rate_per_hour=allowed_rate,
                    metadata=reg_metadata,
                )
            )

        # Attach proposal-level metadata to the plan
        plan_metadata = {
            "proposal_hotspot_id": proposal.hotspot_id,
            "target_tvs": list(getattr(proposal, "target_tvs", [])),
            "ripple_tvs": list(getattr(proposal, "ripple_tvs", [])),
        }
        return cls(regulations=regulations, metadata=plan_metadata)


__all__ = ["DFRegulation", "DFRegulationPlan", "DEFAULT_AUTO_RIPPLE_TIME_BINS"]
