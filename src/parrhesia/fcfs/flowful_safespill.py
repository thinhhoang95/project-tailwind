"""
Flow‑aware FIFO scheduler under per‑flow integer release rates.

This module implements `assign_delays_flowful`, which performs per‑flow FIFO
releases across time bins according to an integer schedule `n_f(t)` that
includes an overflow bin (index T, where T = indexer.num_time_bins).

For each flow, flights are queued in order of their requested (baseline)
entry times at the controlled volume, denoted r_i. At each time bin t, up to
`n_f(t)` flights that are ready (r_i ≤ end_of_bin(t)) are released. Each
released flight i has a realised start time

    s_i = max(r_i, start_of_bin(t)),

and an assigned pushback delay

    delay_i = ceil((s_i - r_i) / 60 seconds) in minutes.

The overflow bin t = T releases any remaining flights with

    s_i = max(r_i, start_of_bin(T)),

so that flights whose r_i lies beyond the horizon (r_i ≥ t_end) incur zero
delay and are carried as spill.

Spill handling beyond the active window can be customised via ``spill_mode``:

``"one_per_spill_bin"`` (default)
    Place each remaining flight into consecutive bins starting at
    ``last_active_bin + 1``.
``"defined_release_rate_for_spills"``
    Token-bucket release at a configured flights-per-hour rate.
``"same_release_rate_for_spills"``
    Token-bucket release at the mean per-bin capacity observed within the
    active window.
``"dump_to_next_bin"``
    Release every remaining flight in the single bin immediately after the
    active window.

Inputs are intentionally flexible: each flight in `flights_by_flow` may carry
either a concrete datetime for r_i, or a time-bin index. Datetimes are
preferred to compute within-bin delays precisely; when only bin indices are
available, delays are computed as multiples of the bin length (conservative).

Expected shapes
---------------
- `flights_by_flow`: Mapping[flow_id, Sequence[FlightSpec]]
  where each FlightSpec is one of:
    • dict with keys:
        - 'flight_id' (required)
        - One of the following to define r_i:
            * 'requested_dt' (datetime or ISO string)
            * 'requested_bin' (int in [0, T] where T = num_time_bins)
            * 'takeoff_time' (datetime/ISO) AND 'entry_time_s' (float seconds)
              — this pair implies requested_dt = takeoff_time + entry_time_s
    • tuple formats supported for convenience:
        (flight_id, requested_dt)
        (flight_id, requested_bin)
        (flight_id, takeoff_time, entry_time_s)

- `n_f_t`: Mapping[flow_id, Sequence[int] | Mapping[int, int]]
  For each flow, a length-(T+1) integer vector of release counts per bin. A
  mapping is also accepted with integer keys 0..T.

Returns
-------
Tuple[Dict[str, int], Dict[str, object]] where
  - delays_min[flight_id] = integer delay in minutes (ceil of seconds/60)
  - realised_start[flight_id] =
        datetime when requested_dt is provided/derivable, otherwise a dict
        { 'bin': int } indicating the realised start bin index.

Notes
-----
- This routine assumes the provided `n_f(t)` satisfy both non‑anticipativity
  and completeness for each flow. If infeasibilities are detected (e.g.,
  attempting to release more flights than are ready in a bin), the algorithm
  defensively caps releases to the number of ready flights and logs a warning.
  Overflow bin handling will attempt to release all remaining flights.

Examples
--------
Minimal FIFO release across adjacent bins using datetimes for within‑bin
precision:

>>> import datetime as dt
>>> from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
>>> from parrhesia.fcfs.flowful import assign_delays_flowful
>>> idx = TVTWIndexer(time_bin_minutes=30)
>>> base = dt.datetime(2025, 1, 1, 9, 0, 0)  # bin 18 starts at 09:00
>>> flights_by_flow = {
...   "flowA": [
...     {"flight_id": "F1", "requested_dt": base + dt.timedelta(seconds=0)},
...     {"flight_id": "F2", "requested_dt": base + dt.timedelta(seconds=10)},
...     {"flight_id": "F3", "requested_dt": base + dt.timedelta(minutes=1)},
...   ]
... }
>>> sched = [0] * (idx.num_time_bins + 1)
>>> sched[18] = 2  # release two in bin 18
>>> sched[19] = 1  # release one in bin 19
>>> delays, realised = assign_delays_flowful({"flowA": flights_by_flow["flowA"]}, {"flowA": sched}, idx)
>>> delays["F1"], delays["F2"], delays["F3"]
(0, 0, 29)
>>> realised["F3"].hour, realised["F3"].minute
(9, 30)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union
from collections import deque
import logging
from math import ceil
from datetime import datetime, timedelta

from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import _parse_naive_utc


_BinIndex = int
SpillMode = Literal[
    "one_per_spill_bin",
    "defined_release_rate_for_spills",
    "same_release_rate_for_spills",
    "dump_to_next_bin",
]


def _to_len_T_plus_1_array(n_t: Union[Sequence[int], Mapping[int, int]], T: int) -> List[int]:
    """
    Normalize a per‑bin release schedule into a Python list of length T+1.

    T here is the number of within‑day bins (indexer.num_time_bins). The
    returned list has indices 0..T inclusive, where index T is the overflow bin.
    """
    T_plus_1 = T + 1
    out = [0] * T_plus_1
    if isinstance(n_t, Mapping):
        for k, v in n_t.items():
            try:
                kk = int(k)
                if 0 <= kk <= T:
                    out[kk] = int(v)
            except Exception:
                continue
        return out
    # Sequence path
    try:
        L = len(n_t)  # type: ignore[arg-type]
    except Exception:
        raise TypeError("n_f_t must be a Mapping or a sequence of length T+1")
    if L != T_plus_1:
        raise ValueError(f"Expected per-flow schedule length {T_plus_1}, got {L}")
    return [int(x) for x in n_t]  # type: ignore[arg-type]


def _bin_len_minutes(indexer: TVTWIndexer) -> int:
    return int(getattr(indexer, "time_bin_minutes", 0))


def _start_of_bin_for_date(dt: datetime, t: int, bin_minutes: int) -> datetime:
    base = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return base + timedelta(minutes=t * bin_minutes)


def _apply_assignment(
    fid: str,
    r_dt: Optional[datetime],
    r_bin: int,
    assigned_bin: int,
    bin_minutes: int,
    realised_start: Dict[str, object],
    delays_min: Dict[str, int],
) -> None:
    if isinstance(r_dt, datetime):
        start_of_bin_dt = _start_of_bin_for_date(r_dt, assigned_bin, bin_minutes)
        s_dt = r_dt if r_dt >= start_of_bin_dt else start_of_bin_dt
        delay_seconds = max(0.0, (s_dt - r_dt).total_seconds())
        delay_minutes = int(ceil(delay_seconds / 60.0)) if delay_seconds > 0 else 0
        realised_start[fid] = s_dt
        delays_min[fid] = delay_minutes
    else:
        delay_minutes = max(0, (int(assigned_bin) - int(r_bin)) * bin_minutes)
        realised_start[fid] = {"bin": int(assigned_bin)}
        delays_min[fid] = int(delay_minutes)


def _resolve_release_rate(
    release_rate_for_spills: Optional[Union[float, Mapping[Any, float]]],
    flow_id: Any,
) -> Optional[float]:
    if release_rate_for_spills is None:
        return None
    if isinstance(release_rate_for_spills, Mapping):
        for key in (flow_id, str(flow_id)):
            if key in release_rate_for_spills:
                value = release_rate_for_spills[key]
                if value is None:
                    return None
                try:
                    return float(value)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid release_rate_for_spills value for flow {flow_id!r}: {value!r}"
                    ) from exc
        for key in ("default", "DEFAULT"):
            if key in release_rate_for_spills:
                value = release_rate_for_spills[key]
                if value is None:
                    return None
                try:
                    return float(value)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid default release_rate_for_spills value: {value!r}"
                    ) from exc
        return None
    try:
        return float(release_rate_for_spills)
    except Exception as exc:
        raise ValueError(
            f"Invalid release_rate_for_spills scalar value: {release_rate_for_spills!r}"
        ) from exc


def _compute_mean_active_rate(schedule: Sequence[int]) -> Optional[float]:
    if not schedule:
        return None
    T = len(schedule) - 1
    first: Optional[int] = None
    last: Optional[int] = None
    for idx in range(T):
        if float(schedule[idx]) > 0:
            if first is None:
                first = idx
            last = idx
    if first is None or last is None:
        return None
    total = 0.0
    for idx in range(first, last + 1):
        total += float(schedule[idx])
    bins = (last - first + 1)
    if bins <= 0:
        return None
    return total / bins


def _token_bucket_assignments(
    count: int,
    spill_start: int,
    tokens_per_bin: float,
) -> List[int]:
    if tokens_per_bin <= 0:
        raise ValueError("tokens_per_bin must be positive")
    assignments: List[int] = []
    tokens = 0.0
    bin_idx = int(spill_start)
    while len(assignments) < count:
        tokens += tokens_per_bin
        release = int(tokens)
        if release > 0:
            release = min(release, count - len(assignments))
            assignments.extend([bin_idx] * release)
            tokens -= release
        bin_idx += 1
    return assignments


def _assign_spill_flights(
    flow_id: Any,
    remaining: Sequence[Tuple[str, Optional[datetime], int]],
    spill_start: int,
    schedule: Sequence[int],
    bin_minutes: int,
    indexer: TVTWIndexer,
    delays_min: Dict[str, int],
    realised_start: Dict[str, object],
    spill_mode: SpillMode,
    release_rate_for_spills: Optional[Union[float, Mapping[Any, float]]],
) -> None:
    if not remaining:
        return

    del indexer  # unused but kept for signature compatibility

    spill_start = int(spill_start)
    if spill_start < 0:
        spill_start = 0

    mode: SpillMode = spill_mode or "one_per_spill_bin"
    assigned_bins: List[int]

    if mode == "dump_to_next_bin":
        assigned_bins = [spill_start] * len(remaining)
    elif mode == "one_per_spill_bin":
        assigned_bins = [spill_start + i for i in range(len(remaining))]
    elif mode == "defined_release_rate_for_spills":
        rate = _resolve_release_rate(release_rate_for_spills, flow_id)
        if rate is None:
            raise ValueError(
                f"spill_mode 'defined_release_rate_for_spills' requires release_rate_for_spills for flow {flow_id!r}"
            )
        if rate <= 0:
            raise ValueError(
                f"release_rate_for_spills must be positive for flow {flow_id!r}; got {rate}"
            )
        tokens_per_bin = rate * (bin_minutes / 60.0)
        if tokens_per_bin <= 0:
            raise ValueError(
                f"Computed tokens_per_bin <= 0 for flow {flow_id!r}; check bin minutes {bin_minutes}"
            )
        assigned_bins = _token_bucket_assignments(len(remaining), spill_start, tokens_per_bin)
    elif mode == "same_release_rate_for_spills":
        tokens_per_bin_opt = _compute_mean_active_rate(schedule)
        if tokens_per_bin_opt is None or tokens_per_bin_opt <= 0:
            logging.warning(
                "Flow %s: same_release_rate_for_spills requested but no active bins; falling back to one_per_spill_bin",
                str(flow_id),
            )
            assigned_bins = [spill_start + i for i in range(len(remaining))]
        else:
            assigned_bins = _token_bucket_assignments(len(remaining), spill_start, tokens_per_bin_opt)
    else:
        logging.warning(
            "Flow %s: unknown spill_mode %r; defaulting to one_per_spill_bin",
            str(flow_id),
            mode,
        )
        assigned_bins = [spill_start + i for i in range(len(remaining))]

    for (fid, r_dt, r_bin), assigned_bin in zip(remaining, assigned_bins):
        _apply_assignment(fid, r_dt, r_bin, int(assigned_bin), bin_minutes, realised_start, delays_min)


def _normalize_flight_spec(
    spec: Any,
    indexer: TVTWIndexer,
) -> Tuple[str, Optional[datetime], _BinIndex]:
    """
    Convert a flexible flight spec to (flight_id, requested_dt | None, bin_idx).

    If a datetime cannot be recovered, bin_idx is still returned for scheduling;
    realised datetimes will then be unavailable and delays will be computed at
    whole‑bin granularity.

    Examples
    --------
    Dict with requested datetime:

    >>> from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
    >>> idx = TVTWIndexer(time_bin_minutes=30)
    >>> fid, rdt, rbin = _normalize_flight_spec({"flight_id": "X", "requested_dt": "2025-01-01T09:00:00"}, idx)
    >>> fid, isinstance(rdt, datetime), rbin
    ('X', True, 18)

    Tuple with requested bin index:

    >>> _normalize_flight_spec(("Y", 21), idx)[0:3]
    ('Y', None, 21)
    """
    fid: Optional[str] = None
    r_dt: Optional[datetime] = None
    r_bin: Optional[int] = None

    if isinstance(spec, dict):
        fid = spec.get("flight_id") or spec.get("id") or spec.get("fid")
        # Try direct datetime
        r_dt_raw = spec.get("requested_dt") or spec.get("r_dt") or spec.get("scheduled_dt")
        if isinstance(r_dt_raw, str):
            try:
                r_dt = _parse_naive_utc(r_dt_raw)
            except Exception:
                r_dt = None
        elif isinstance(r_dt_raw, datetime):
            r_dt = r_dt_raw
        # Try bin index
        if r_dt is None:
            try:
                r_bin = int(spec.get("requested_bin") if "requested_bin" in spec else spec.get("r_bin"))
            except Exception:
                r_bin = None
        # Try takeoff + offset
        if r_dt is None and ("takeoff_time" in spec and "entry_time_s" in spec):
            tko_raw = spec.get("takeoff_time")
            try:
                tko = _parse_naive_utc(tko_raw) if isinstance(tko_raw, str) else tko_raw
                entry_s = float(spec.get("entry_time_s", 0.0))
                if isinstance(tko, datetime):
                    r_dt = tko + timedelta(seconds=entry_s)
            except Exception:
                r_dt = None
        # Derive bin if we have datetime
        if r_dt is not None and r_bin is None:
            r_bin = int(indexer.bin_of_datetime(r_dt))
    elif isinstance(spec, tuple) and len(spec) >= 2:
        fid = str(spec[0])
        second = spec[1]
        if isinstance(second, datetime):
            r_dt = second
            r_bin = int(indexer.bin_of_datetime(second))
        else:
            try:
                # Could be bin index
                r_bin = int(second)
            except Exception:
                # Or a takeoff time followed by entry_time_s
                try:
                    tko = second if isinstance(second, datetime) else _parse_naive_utc(str(second))
                    entry_s = float(spec[2]) if len(spec) >= 3 else 0.0
                    if isinstance(tko, datetime):
                        r_dt = tko + timedelta(seconds=entry_s)
                        r_bin = int(indexer.bin_of_datetime(r_dt))
                except Exception:
                    pass
    else:
        raise TypeError("Unsupported flight spec; expected dict or tuple")

    if fid is None:
        raise ValueError("Flight spec missing 'flight_id'")
    if r_bin is None:
        # As a last resort when no information is provided, assume bin 0
        logging.warning("Flight %s missing requested time; assuming bin 0", fid)
        r_bin = 0
    return str(fid), r_dt, int(r_bin)


def preprocess_flights_for_scheduler(
    flights_by_flow: Mapping[Any, Sequence[Any]],
    indexer: TVTWIndexer,
) -> Dict[Any, List[Tuple[str, Optional[datetime], int]]]:
    """
    Normalize and sort flights once per flow for scheduling.

    Returns a mapping flow_id -> list of tuples (flight_id, requested_dt | None, requested_bin)
    sorted by (requested_bin, within-bin seconds, flight_id).
    """
    bin_minutes = _bin_len_minutes(indexer)
    out: Dict[Any, List[Tuple[str, Optional[datetime], int]]] = {}

    def _sort_key(item: Tuple[str, Optional[datetime], int]):
        fid, r_dt, r_bin = item
        if isinstance(r_dt, datetime):
            mod_minutes = (r_dt.hour * 60 + r_dt.minute) % bin_minutes
            within_bin_s = mod_minutes * 60 + r_dt.second + r_dt.microsecond / 1e6
        else:
            within_bin_s = 0.0
        return (int(r_bin), float(within_bin_s), str(fid))

    for flow_id, specs in flights_by_flow.items():
        flights_norm: List[Tuple[str, Optional[datetime], int]] = []
        for sp in specs or []:
            fid, r_dt, r_bin = _normalize_flight_spec(sp, indexer)
            flights_norm.append((fid, r_dt, r_bin))
        flights_norm.sort(key=_sort_key)
        out[flow_id] = flights_norm
    return out


def assign_delays_flowful_preparsed(
    flights_sorted_by_flow: Mapping[Any, Sequence[Tuple[str, Optional[datetime], int]]],
    n_f_t: Mapping[Any, Union[Sequence[int], Mapping[int, int]]],
    indexer: TVTWIndexer,
    *,
    spill_mode: SpillMode = "one_per_spill_bin",
    release_rate_for_spills: Optional[Union[float, Mapping[Any, float]]] = None,
) -> Tuple[Dict[str, int], Dict[str, object]]:
    """
    Variant of ``assign_delays_flowful`` for pre-normalized and sorted flights.

    Parameters
    ----------
    flights_sorted_by_flow : Mapping[Any, Sequence[Tuple[str, Optional[datetime], int]]]
        Flights per flow, already normalized via ``preprocess_flights_for_scheduler``.
    n_f_t : Mapping[Any, Sequence[int] | Mapping[int, int]]
        Per-flow integer schedule of length T+1 (overflow at index T).
    indexer : TVTWIndexer
        Provides bin configuration.
    spill_mode : SpillMode, optional
        Policy for allocating spill beyond the last scheduled bin. Defaults to
        ``"one_per_spill_bin"`` for the legacy behaviour.
    release_rate_for_spills : float | Mapping[Any, float], optional
        Required when ``spill_mode="defined_release_rate_for_spills"``. Expressed
        in flights per hour, may be scalar (all flows) or a per-flow mapping.

    Spill Modes
    -----------
    ``"one_per_spill_bin"``
        Assign flights to successive bins starting with ``spill_start``.
    ``"defined_release_rate_for_spills"``
        Token-bucket release outside the active window at the provided rate.
    ``"same_release_rate_for_spills"``
        Token-bucket release using the mean active-window capacity.
    ``"dump_to_next_bin"``
        Queue all remaining flights into ``spill_start`` while preserving FIFO.
    """
    T = int(indexer.num_time_bins)
    bin_minutes = _bin_len_minutes(indexer)
    if bin_minutes <= 0:
        raise ValueError("TVTWIndexer.time_bin_minutes must be positive")

    delays_min: Dict[str, int] = {}
    realised_start: Dict[str, object] = {}

    for flow_id, flights_norm_in in flights_sorted_by_flow.items():
        schedule = _to_len_T_plus_1_array(n_f_t.get(flow_id, []), T)
        flights_norm: List[Tuple[str, Optional[datetime], int]] = list(flights_norm_in)

        # Sanity: completeness (soft check)
        demanded = len(flights_norm)
        scheduled_total = int(sum(schedule))
        if scheduled_total != demanded:
            logging.warning(
                "Flow %s: schedule sum (%d) != number of flights (%d)",
                str(flow_id), scheduled_total, demanded,
            )

        # Two‑phase queueing: push flights into 'ready' as bins advance
        ready = deque()  # type: deque[Tuple[str, Optional[datetime], int]]
        i = 0  # next flight in sorted list to consider
        N = len(flights_norm)

        # Iterate day bins 0..T-1
        for t in range(T):
            # Enqueue flights that are ready by end of bin t
            while i < N and flights_norm[i][2] <= t:
                ready.append(flights_norm[i])
                i += 1

            capacity = int(schedule[t])
            if capacity < 0:
                logging.warning("Flow %s: negative capacity at bin %d; treating as 0", str(flow_id), t)
                capacity = 0
            if capacity > len(ready):
                if len(ready) == 0 and capacity > 0:
                    logging.warning(
                        "Flow %s: n_f(%d)=%d but no ready flights; capping to 0",
                        str(flow_id), t, capacity,
                    )
                capacity = min(capacity, len(ready))

            for _ in range(capacity):
                fid, r_dt, r_bin = ready.popleft()
                _apply_assignment(fid, r_dt, r_bin, t, bin_minutes, realised_start, delays_min)

        # Safe-spill: distribute remaining flights across successive bins
        # immediately after the last active in-window bin, instead of
        # teleporting all to the end-of-day overflow bin T.
        remaining: List[Tuple[str, Optional[datetime], int]] = list(ready)
        if i < N:
            remaining.extend(flights_norm[i:])

        # Determine the last active in-window bin where capacity was explicitly set
        last_active_bin = -1
        for tt in range(T - 1, -1, -1):
            if int(schedule[tt]) > 0:
                last_active_bin = int(tt)
                break
        if last_active_bin < 0:
            # Fallback: use the latest requested bin among remaining flights (if any)
            try:
                last_active_bin = max((int(rb) for (_fid, _rdt, rb) in flights_norm), default=-1)
            except Exception:
                last_active_bin = -1

        # Log diagnostic if schedule[T] disagrees with the actual remainder, but do not truncate
        overflow_cap = int(schedule[T]) if T < len(schedule) else 0
        if overflow_cap != len(remaining):
            logging.warning(
                "Flow %s: overflow count %d != remaining %d; using safe-spill distribution",
                str(flow_id), overflow_cap, len(remaining),
            )

        # Assign each remaining flight to consecutive spill bins starting at
        # (last_active_bin + 1), allowing indices beyond T to represent
        # bins after the planning horizon.
        spill_start = int(last_active_bin + 1)
        _assign_spill_flights(
            flow_id,
            remaining,
            spill_start,
            schedule,
            bin_minutes,
            indexer,
            delays_min,
            realised_start,
            spill_mode,
            release_rate_for_spills,
        )

        # Sanity: ensure we produced outputs for all flights in this flow
        produced = sum(1 for fid, _, _ in flights_norm if fid in delays_min)
        if produced != len(flights_norm):
            logging.warning(
                "Flow %s: produced schedules for %d/%d flights (check feasibility)",
                str(flow_id), produced, len(flights_norm),
            )

    return delays_min, realised_start


def assign_delays_flowful(
    flights_by_flow: Mapping[Any, Sequence[Any]],
    n_f_t: Mapping[Any, Union[Sequence[int], Mapping[int, int]]],
    indexer: TVTWIndexer,
    *,
    spill_mode: SpillMode = "one_per_spill_bin",
    release_rate_for_spills: Optional[Union[float, Mapping[Any, float]]] = None,
) -> Tuple[Dict[str, int], Dict[str, object]]:
    """
    Assign per‑flight delays and realised start times under per‑flow integer
    release plans `n_f(t)` with FIFO within each flow.

    Parameters
    ----------
    flights_by_flow : Mapping[Any, Sequence[Any]]
        For each flow identifier, a sequence of flight specs. Each spec must
        provide at least a flight id and its baseline requested time at the
        controlled volume (as a datetime or a bin index). See module docstring
        for accepted formats.
    n_f_t : Mapping[Any, Sequence[int] | Mapping[int, int]]
        For each flow identifier, per‑bin release counts of length T+1, where
        T = indexer.num_time_bins and index T denotes the overflow bin.
    indexer : TVTWIndexer
        Provider of time bin parameters (bin length and T).
    spill_mode : SpillMode, optional
        Strategy for releasing spill after the last active bin. Defaults to the
        legacy behaviour ``"one_per_spill_bin"``.
    release_rate_for_spills : float | Mapping[Any, float], optional
        Flights-per-hour rate for spill releases. Required when
        ``spill_mode="defined_release_rate_for_spills"``. Scalar applies to all
        flows; a mapping may specify per-flow rates (with optional ``"default"``).

    Spill Modes
    -----------
    ``"one_per_spill_bin"``
        Assign flights to successive bins starting with ``spill_start``.
    ``"defined_release_rate_for_spills"``
        Token-bucket release outside the active window at the provided rate.
    ``"same_release_rate_for_spills"``
        Token-bucket release using the mean active-window capacity.
    ``"dump_to_next_bin"``
        Queue all remaining flights into ``spill_start`` while preserving FIFO.

    Returns
    -------
    (delays_min, realised_start)
        delays_min: Dict[flight_id, int minutes]
        realised_start: Dict[flight_id, datetime | { 'bin': int }]

    Notes
    -----
    - Releases in each bin are capped at the number of ready flights. If
      `n_f(t)` attempts to release more than are ready, a warning is logged.
    - Flights not released by the end of bin T‑1 are released in the overflow
      bin T per `n_f(T)`.
    - This function does not mutate its inputs.

    Examples
    --------
    FIFO with a one‑bin spill for the third flight:

    >>> import datetime as dt
    >>> idx = TVTWIndexer(time_bin_minutes=30)
    >>> base = dt.datetime(2025, 1, 1, 9, 0, 0)
    >>> flights = {
    ...   "flow": [
    ...     ("A1", base + dt.timedelta(seconds=0)),
    ...     ("A2", base + dt.timedelta(seconds=5)),
    ...     ("A3", base + dt.timedelta(minutes=1)),
    ...   ]
    ... }
    >>> sched = [0] * (idx.num_time_bins + 1)
    >>> sched[18] = 2; sched[19] = 1
    >>> dly, rs = assign_delays_flowful(flights, {"flow": sched}, idx)
    >>> dly["A1"], dly["A2"], dly["A3"], (rs["A3"].hour, rs["A3"].minute)
    (0, 0, 29, (9, 30))
    """
    T = int(indexer.num_time_bins)
    bin_minutes = _bin_len_minutes(indexer)
    if bin_minutes <= 0:
        raise ValueError("TVTWIndexer.time_bin_minutes must be positive")

    # Normalize and sort once, then delegate to fast path
    flights_sorted = preprocess_flights_for_scheduler(flights_by_flow, indexer)
    return assign_delays_flowful_preparsed(
        flights_sorted,
        n_f_t,
        indexer,
        spill_mode=spill_mode,
        release_rate_for_spills=release_rate_for_spills,
    )


__all__ = ["assign_delays_flowful", "assign_delays_flowful_preparsed", "preprocess_flights_for_scheduler", "_normalize_flight_spec"]
