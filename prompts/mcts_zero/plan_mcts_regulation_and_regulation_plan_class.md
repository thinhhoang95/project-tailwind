### What this task asks for
```9:13:prompts/mcts_zero/preplan_mcts_scaffold.md
- Based on how the regulation plan (or an individual regulation) is input in `base_evaluation.py` and `automatic_rate_adjustment.py`, design the `DFRegulation` and `DFRegulationPlan` classes. The `DFRegulationPlan` is only a wrapper for many plans in the `DFRegulation`, along with some useful methods such as `number_of_regulations`, `number_of_flows`, `number_of_flights_affected` and a general `metadata` field which is left `None` for the moment.

- Implement these two classes in `src/parrhesia/actions`. 
```

### Downstream payloads these classes must generate
```10:21:src/parrhesia/api/base_evaluation.py
Input payload schema (keys are optional unless noted):
  - flows (required): mapping of flow-id -> list of flight IDs
  - targets (required): mapping of TV id -> {"from": "HH:MM:SS", "to": "HH:MM:SS"}
  - ripples (optional): mapping like targets; used for reduced weights
  - auto_ripple_time_bins (optional): non-negative int; when > 0, override
    'ripples' by auto-deriving ripple cells from the union of all flight
    footprints across TVs, dilated by ±window bins
  - indexer_path (optional): path to tvtw_indexer.json
  - flights_path (optional): path to so6_occupancy_matrix_with_times.json
  - capacities_path (optional): path to capacities GeoJSON
  - weights (optional): partial overrides for ObjectiveWeights
```

```10:19:src/parrhesia/api/automatic_rate_adjustment.py
Request JSON (keys optional unless noted):
  - flows (required): mapping of flow-id -> list of flight IDs
  - targets (required): mapping of TV -> {"from": "HH:MM[:SS]", "to": "HH:MM[:SS]"}
  - ripples (optional): same schema as targets
  - auto_ripple_time_bins (optional): if > 0, overrides ripples by using union of
    footprints of flights across TVs with ±window dilation
  - indexer_path, flights_path, capacities_path (optional): artifact overrides
  - weights (optional): partial overrides for ObjectiveWeights
```

*Note: sa_params is not part of any of these classes. They are proprietary to the SA optimization code*.

### Implementation plan
- Create `src/parrhesia/actions/regulations.py` and `src/parrhesia/actions/__init__.py`.
- Define data models:
  - DFRegulation: one regulation = (control TV id, active window, associated flight list, allowed entry rate).
  - DFRegulationPlan: wrapper for many DFRegulation with helper metrics and payload builders.
- Time handling:
  - Store windows as strings "HH:MM" or "HH:MM:SS". When merging multiple regs on the same TV, union windows per TV by min(from) and max(to).
- Payload builders:
  - `to_base_eval_payload()` and `to_autorate_payload()` producing `flows` and `targets` exactly as those endpoints require.
  - Optional `ripples` support via an optional field; default to none. Allow an `auto_ripple_time_bins` parameter pass-through.
- Metrics:
  - `number_of_regulations()`, `number_of_flows()` (1 flow per regulation), `number_of_flights_affected()` (unique flights across regs).
  - `metadata: Optional[dict] = None`.
- Interop:
  - `from_proposal(proposal: parrhesia.flow_agent35.regen.types.Proposal, flights_by_flow)` helper if you want to bootstrap regs from existing `regen`'s proposal objects.
  - `from_payload(payload: Mapping[str, Any])` for round-tripping.
- Tests:
  - Unit tests to verify metrics and that payloads match the two APIs’ expected shapes, including window union-per-TV and stable flow-id assignment.

### Class skeletons (minimal, typed, ready to drop in)
```python
# src/parrhesia/actions/regulations.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Iterable

def _min_time(a: str, b: str) -> str:
    # "HH:MM[:SS]" compares lexicographically if zero-padded
    return a if a <= b else b

def _max_time(a: str, b: str) -> str:
    return a if a >= b else b

@dataclass(frozen=True)
class DFRegulation:
    id: str
    tv_id: str
    window_from: str  # "HH:MM" or "HH:MM:SS"
    window_to: str    # "HH:MM" or "HH:MM:SS"
    flights: Tuple[str, ...]  # immutable, normalized to strings
    allowed_rate_per_hour: int
    # Optional extras if needed later (ripple TVs, notes, etc.)
    metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_flights(
        *,
        id: str,
        tv_id: str,
        window_from: str,
        window_to: str,
        flights: Iterable[Any],
        allowed_rate_per_hour: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DFRegulation":
        fl_norm = tuple(str(f) for f in flights)
        return DFRegulation(
            id=str(id),
            tv_id=str(tv_id),
            window_from=str(window_from),
            window_to=str(window_to),
            flights=fl_norm,
            allowed_rate_per_hour=int(allowed_rate_per_hour),
            metadata=metadata,
        )

    def targets_map(self) -> Dict[str, Dict[str, str]]:
        return {self.tv_id: {"from": self.window_from, "to": self.window_to}}

@dataclass
class DFRegulationPlan:
    regulations: List[DFRegulation] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    # --- Metrics ---
    def number_of_regulations(self) -> int:
        return len(self.regulations)

    def number_of_flows(self) -> int:
        # 1 flow per regulation by design
        return len(self.regulations)

    def number_of_flights_affected(self) -> int:
        uniq: set[str] = set()
        for r in self.regulations:
            uniq.update(r.flights)
        return len(uniq)

    # --- Mutation helpers ---
    def add(self, reg: DFRegulation) -> None:
        self.regulations.append(reg)

    def extend(self, regs: Iterable[DFRegulation]) -> None:
        self.regulations.extend(regs)

    # --- Payload builders for downstream APIs ---
    def to_base_eval_payload(
        self,
        *,
        ripples: Optional[Mapping[str, Mapping[str, str]]] = None,
        auto_ripple_time_bins: int = 0,
    ) -> Dict[str, Any]:
        flows: Dict[int, List[str]] = {}
        targets: Dict[str, Dict[str, str]] = {}

        # Assign stable flow IDs 0..N-1 in input order
        for i, r in enumerate(self.regulations):
            flows[i] = list(r.flights)
            if r.tv_id in targets:
                cur = targets[r.tv_id]
                cur["from"] = _min_time(cur["from"], r.window_from)
                cur["to"] = _max_time(cur["to"], r.window_to)
            else:
                targets[r.tv_id] = {"from": r.window_from, "to": r.window_to}

        out: Dict[str, Any] = {
            "flows": flows,
            "targets": targets,
        }
        if ripples:
            out["ripples"] = {str(tv): {"from": str(w["from"]), "to": str(w["to"])} for tv, w in ripples.items()}
        if auto_ripple_time_bins and int(auto_ripple_time_bins) > 0:
            out["auto_ripple_time_bins"] = int(auto_ripple_time_bins)
        return out

    def to_autorate_payload(
        self,
        *,
        ripples: Optional[Mapping[str, Mapping[str, str]]] = None,
        auto_ripple_time_bins: int = 0,
        weights: Optional[Mapping[str, Any]] = None,
        sa_params: Optional[Mapping[str, Any]] = None,
        spill_mode: Optional[str] = None,
        release_rate_for_spills: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = self.to_base_eval_payload(
            ripples=ripples, auto_ripple_time_bins=auto_ripple_time_bins
        )
        if weights:
            payload["weights"] = dict(weights)
        if sa_params:
            payload["sa_params"] = dict(sa_params)
        if spill_mode:
            payload["spill_mode"] = str(spill_mode)
        if release_rate_for_spills is not None:
            payload["release_rate_for_spills"] = int(release_rate_for_spills)
        return payload

__all__ = ["DFRegulation", "DFRegulationPlan"]
```

```python
# src/parrhesia/actions/__init__.py
from .regulations import DFRegulation, DFRegulationPlan

__all__ = ["DFRegulation", "DFRegulationPlan"]
```

### Notes on behavior and constraints
- Union windows per TV: both downstream APIs accept only one window per TV; if multiple regs target the same TV with different windows, the plan merges them by min(from), max(to).
- Flow IDs: downstream APIs accept arbitrary ints; we use stable 0..N-1 consistent with input ordering.
- Allowed rates: stored on `DFRegulation` for MCTS/regulation logic, not consumed by the two endpoints; SA will optimize the schedule regardless. Keep it for future integration with a regulation-application path.
- Ripples: optional. Prefer `auto_ripple_time_bins` unless explicit ripple windows are provided.

### Tiny usage example
```python
plan = DFRegulationPlan()
plan.add(DFRegulation.from_flights(
    id="R1", tv_id="TV123", window_from="09:00", window_to="10:15",
    flights=["AF101", "UA202", "DL303"], allowed_rate_per_hour=18
))
base_eval_payload = plan.to_base_eval_payload()
autorate_payload = plan.to_autorate_payload(sa_params={"max_iters": 2000})
```

- base_eval_payload → keys: `flows`, `targets` (and optional ripple/auto-ripple).
- autorate_payload → extends with `weights`, `sa_params`, optional spill controls.

Let me search for where ripple auto-windows are defined and how many bins are used by default, then draft a focused plan for ripple support including the fallback warning and defaults.

[5 tools called]

I’ll grab the exact constant for the default ripple dilation to cite the value of 2, then draft the ripple-only plan with fallback behavior and warning text.

[1 tool called]

I found the default ripple dilation constant (2) in the proposal engine. I’ll now write a focused ripple-support plan with fallback behavior and a one-time warning.

### Clarifying ripples:

- Overview
  - Provide two mutually exclusive ways to specify ripples when building payloads:
    - Explicit ripples: mapping of `tv_id -> {"from": "HH:MM[:SS]", "to": "HH:MM[:SS]"}`
    - Auto-ripple: derive `ripple_cells` from the union of footprints of all flights in all flows, dilated ±w bins; controlled by `auto_ripple_time_bins`
  - Default behavior: when no explicit ripples or explicit auto setting is provided, enable auto-ripple with w=`AUTO_RIPPLE_DILATION_BINS` (see below) and print a one-time warning.
  - Raise Exception when auto-ripple is not enabled and the ripples is None/empty. 

- Default window bins
  - Use `AUTO_RIPPLE_DILATION_BINS` bins for consistency with the proposal engine:
```41:41:src/parrhesia/flow_agent35/regen/engine.py
AUTO_RIPPLE_DILATION_BINS = 2
```

- How the downstream APIs treat auto-ripple (reference semantics)
```215:227:src/parrhesia/api/base_evaluation.py
    try:
        _auto_w = int(payload.get("auto_ripple_time_bins", 0))
    except Exception:
        _auto_w = 0
    if _auto_w > 0:
        ripple_cells = compute_auto_ripple_cells(
            indexer=idx,
            flight_list=fl,
            flight_ids=list(flow_map.keys()),
            window_bins=_auto_w,
        )
```
```310:321:src/parrhesia/api/automatic_rate_adjustment.py
    try:
        auto_w = int(payload.get("auto_ripple_time_bins", 0))
    except Exception:
        auto_w = 0
    if auto_w > 0:
        ripple_cells = compute_auto_ripple_cells(
            indexer=idx,
            flight_list=fl,
            flight_ids=list(flow_map.keys()),
            window_bins=auto_w,
        )
```

- Precedence and behavior
  - If explicit `ripples` (non-empty) is provided:
    - Include `ripples` in the payload.
    - Do not set `auto_ripple_time_bins` (leave unset or 0).
    - No warning.
  - Else if `auto_ripple_time_bins` is explicitly provided by caller (even 0):
    - Set it to `max(0, int(value))`.
    - No warning (explicit instruction).
    - Note: if set to 0, ripples are disabled and no fallback is applied.
  - Else (no explicit ripples, no explicit `auto_ripple_time_bins`):
    - Raise `Exception`.

- Call-site and API design (for the payload builders)
  - Function parameters:
    - `ripples: Optional[Mapping[str, Mapping[str, str]]] = None`
    - `auto_ripple_time_bins: Optional[int] = None`  // None → fallback; int ≥ 0 → use as-is
    - `warn_on_auto_fallback: bool = True`           // one-time warning toggle
  - Metadata support (optional):
    - If `DFRegulationPlan.metadata` contains `{"ripples": {...}}`, treat as explicit ripples (highest precedence).
    - If `DFRegulationPlan.metadata` contains `{"auto_ripple_time_bins": int}`, treat as explicit auto setting.
    - If both are present, prefer explicit `ripples` and ignore the auto setting.

- Validation and coercion
  - Coerce `auto_ripple_time_bins` to `int >= 0` (negative → 0).
  - Treat empty dict `{}` for `ripples` as “not provided”.
  - Time strings must be zero-padded and comparable lexicographically; sanitize to “HH:MM[:SS]”.

- Logging/warning mechanism
  - Use `warnings.warn` with category `UserWarning` (or `logging.warning` if preferred by codebase conventions).
  - One-time guard per plan instance (e.g., `_auto_ripple_warned: bool = False`), so repeat builders don’t spam logs.
  - Do not emit a warning if caller passed an explicit `auto_ripple_time_bins` (even if it equals 2).

- Don’ts
  - Don’t include both `ripples` and `auto_ripple_time_bins > 0` in the same payload; if both are present downstream, raise an Exception. We avoid confusion by enforcing the precedence above.
  - Don’t add non-schema keys to the payload (e.g., no `ripple_strategy_used` field).

- Tests (minimal matrix)
  - Explicit ripples provided:
    - Output has `ripples`; no `auto_ripple_time_bins` key; no warning.
  - No ripples, no auto param:
    - Output has `auto_ripple_time_bins: 2`; one warning emitted once per plan.
  - Explicit `auto_ripple_time_bins = 4`:
    - Output has `auto_ripple_time_bins: 4`; no warning.
  - Explicit `auto_ripple_time_bins = 0`:
    - Output has `auto_ripple_time_bins: 0`; no warning; no `ripples`.
  - Metadata-driven:
    - `metadata.ripples` present → same as explicit ripples.
    - `metadata.auto_ripple_time_bins` present → same as explicit auto.
  - Edge: `ripples={}` (empty) + no auto → fallback to 2 with warning.
  - Edge: invalid auto value (e.g., "abc") → coerce fail → treat as None → fallback to 2 with warning.

- Integration note
  - Keep the default (2) synchronized with the proposal engine’s constant; either import it at runtime or define a local default of 2 with an optional best-effort import. If importing fails, silently fall back to 2.

