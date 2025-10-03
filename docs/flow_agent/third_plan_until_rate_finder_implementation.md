# Flow-Agent Core (State → Rate Finder)

This document captures the concrete implementation that now backs the plan described in
[`third_plan_until_rate_finder.md`](third_plan_until_rate_finder.md). It covers the planning
state model, the symbolic transition system, and the deterministic `RateFinder`, and shows
how they can be exercised with the new unit tests.

---

## Module Inventory

| File | What it owns |
| --- | --- |
| `src/parrhesia/flow_agent/state.py` | Frozen data-carriers (`RegulationSpec`, `HotspotContext`) and the mutable `PlanState` container. |
| `src/parrhesia/flow_agent/actions.py` | Typed action records plus validation guards. |
| `src/parrhesia/flow_agent/transition.py` | `CheapTransition` – applies actions, keeps the cheap residual proxy (`z_hat`) in sync. |
| `src/parrhesia/flow_agent/rate_finder.py` | `RateFinder` and `RateFinderConfig` – coordinate descent over a discrete rate grid with caching and FCFS integration. |
| `tests/flow_agent/test_state_actions.py` | Canonicalisation and guard tests. |
| `tests/flow_agent/test_transition.py` | Transition smoke tests, including `z_hat` bookkeeping and stage changes. |
| `tests/flow_agent/test_rate_finder.py` | Synthetic end-to-end tests for the rate finder, including cache reuse and KPI checks. |

The package is exported in `src/parrhesia/flow_agent/__init__.py` for convenient downstream imports.

---

## Planning State & Data Structures

### `RegulationSpec`

* frozen dataclass; represents a committed regulation.
* stores:
  * `control_volume_id`
  * `window_bins` – half-open `[t0, t1)` window.
  * `flow_ids` – automatically sorted for determinism.
  * `mode` – `"per_flow"` or `"blanket"`.
  * `committed_rates` – int or per-flow mapping once found.
  * optional `diagnostics` payload.
* `to_canonical_dict()` returns a JSON-stable structure that is used for caching.

### `HotspotContext`

* describes the working set while authoring a regulation.
* ensures candidate and selected flow IDs are sorted.
* exposes `add_flow()` / `remove_flow()` convenience methods that return new frozen contexts.
* carries a free-form `metadata` dict. If `metadata["flow_proxies"]` is provided as a mapping `{flow_id -> sequence}`, `CheapTransition` will use those proxy vectors in preference to any constructor-supplied proxies.

### `PlanState`

* mutable snapshot; houses committed plan, the active hotspot context, `z_hat`, and stage metadata.
* important helpers:
  * `copy()` – cheap clone used by the transition system to keep operations pure.
  * `canonical_key()` – stable JSON string (plan, context, stage, `z_hat`). Used heavily by `RateFinder` caches.
  * `reset_hotspot(next_stage=...)` – drops the working context and moves to the given stage while clearing the `awaiting_commit` flag.
  * `set_hotspot()` – activates a context and pre-sizes `z_hat` to the selected window length.
* `canonical_key()` serialises `z_hat` as a list of floats. Any change to `z_hat` (even if not used by the rate finder) will change the cache key.
* `metadata` currently uses the key `"awaiting_commit"` during the confirm stage; it is added by `Continue` and cleared by `Back`/`reset_hotspot`.

Stages are encoded as literals: `idle → select_hotspot → select_flows → confirm → stopped`.

---

## Actions & Guards

`src/parrhesia/flow_agent/actions.py` defines simple dataclasses for every action the plan recognises:
`NewRegulation`, `PickHotspot`, `AddFlow`, `RemoveFlow`, `Continue`, `Back`, `CommitRegulation`, and `Stop`.

Each action has an accompanying guard function that performs semantic checks before state mutation. Examples:

```python
from parrhesia.flow_agent.actions import (
    PickHotspot,
    guard_can_pick_hotspot,
)
from parrhesia.flow_agent.state import PlanState

state = PlanState()
action = PickHotspot(control_volume_id="TV42", window_bins=(12, 16), candidate_flow_ids=("flow_a",))
guard_can_pick_hotspot(state, action)  # raises if stage/window/candidates are invalid
```

These guards are used both in the `CheapTransition` implementation and exposed for higher-level callers if they need pre-flight validation (e.g., UI input checking).

---

## CheapTransition

`CheapTransition` keeps the *symbolic* planning loop fast by avoiding expensive evaluations:

* accepts optional `flow_proxies` – precomputed per-flow vectors representing the “earliest hotspot mass”.
* maintains the cheap proxy `z_hat` in the `PlanState`, slicing / clipping proxies to the currently selected window.
* supports simple decay (`decay` parameter) and clipping (`clip_value`).
* returns a tuple `(next_state, is_commit, is_terminal)` so the caller can trigger expensive work only when necessary (e.g., run the `RateFinder` on commit).

Additional notes and assumptions:

- Proxy precedence: if the active `HotspotContext.metadata` contains `flow_proxies`, those are used first; otherwise proxies passed to the `CheapTransition` constructor are used; if neither is available, a ones vector is used as a fallback for the current window length.
- Window handling: when a proxy time series is longer than the selected window, a deterministic slice `[t0:t1)` is used; if it is shorter, the proxy is padded with zeros to fit the window.
- Decay: before applying an `AddFlow`/`RemoveFlow` proxy, `z_hat` is multiplied by `(1 - decay)` if `decay > 0`.
- Clipping: after applying a proxy (with sign +1 for add, −1 for remove), `z_hat` is clipped elementwise to `[-clip_value, +clip_value]`. Default `clip_value` is 250.0.

### Lifecycle Walkthrough

```python
from parrhesia.flow_agent.transition import CheapTransition
from parrhesia.flow_agent.actions import *  # illustrative only
from parrhesia.flow_agent.state import PlanState

transition = CheapTransition(flow_proxies={"flow_a": [0.3, 1.2, 0.9, 0.1]})
state = PlanState()

state, commit, terminal = transition.step(state, NewRegulation())
# stage -> select_hotspot

state, _, _ = transition.step(
    state,
    PickHotspot(
        control_volume_id="TV42",
        window_bins=(5, 8),
        candidate_flow_ids=("flow_a",),
    ),
)
# z_hat initialised with zeros of length 3

state, _, _ = transition.step(state, AddFlow("flow_a"))
# z_hat becomes proxy slice [0.3, 1.2, 0.9]

state, is_commit, _ = transition.step(state, Continue())
# stage -> confirm, awaiting commit flag set

state, is_commit, _ = transition.step(state, CommitRegulation())
# plan now contains a RegulationSpec, stage -> idle
```

`Stop` transitions always land in the `stopped` stage and signal `is_terminal=True`.

---

## RateFinder

### Overview

The rate finder performs deterministic coordinate descent over a discrete grid of hourly rates.
It reuses the FCFS scheduler, the network evaluator, and a `DeltaFlightList` overlay to avoid
mutating the base flight list.

Key design points:

* **Baseline cache**: keyed by `PlanState.canonical_key()`. The baseline objective is only recomputed once for a given state signature.
* **Candidate cache**: LRU over `(plan_key, tv, window, flow_ids, mode, rate_tuple)`. Avoids recomputing FCFS/evaluator pipelines for identical scenarios.
* **Modes**:
  * `per_flow`: each flow gets its own rate; search loops over flows and grid values.
  * `blanket`: a single rate applies to the union of flights.
* **FCFS integration**: uses `parrhesia.fcfs.scheduler.assign_delays` to generate per-flight delays; aggregates per-flight maximums when multiple flows contribute.
* **Objective**: matches the plan evaluator weights (`alpha`, `beta`, `gamma`, `delta`) with optional overrides via `RateFinderConfig`.

Assumptions, simplifications, and interface expectations:

- Rate grid sentinel: `math.inf` in the `rate_grid` means “no regulation” for that flow/union. Rates `<= 0` and `inf` produce an empty delay set.
- Integerisation: candidate rates are rounded and clamped to integers via `max(1, int(round(rate)))` before FCFS. The returned best rates are floats (including `math.inf`); callers that commit rates into `RegulationSpec` should cast to integers as needed.
- Flow ordering: in `per_flow` mode, flows are visited in descending order of the number of entrants in the active window. Entrant counts are computed via `FlightList.iter_hotspot_crossings(...)` if available; otherwise all flows tie and fall back to lexicographic ordering by flow ID. Ordering only affects search determinism and speed, not correctness.
- Active windows: the `window_bins=(t0, t1)` half-open range is expanded to the consecutive bin list `list(range(t0, t1))` when calling FCFS.
- Regulation penalty: the objective's `num_regs` term uses `len(plan_state.plan)` for the baseline and `len(plan_state.plan) + 1` for candidates (the prospective regulation being evaluated).
- Diagnostics: the result includes `entrants_by_flow`, a `per_flow_history` of `ΔJ` per tried rate, and `aggregate_delays_size` (number of flights delayed in the best candidate), in addition to timing, cache, and objective fields.
- Caching scope: the baseline cache is keyed by the full `PlanState.canonical_key()` (including `z_hat` and any active context); the candidate LRU cache is in-memory and per-`RateFinder` instance.
- Capacity bounds: there is no hard cap tying candidate rates to declared hourly capacities; realism of the grid is left to configuration.
- FlightList/Evaluator interfaces relied upon:
  - Required on the flight list: `flight_ids`, `flight_metadata`, `time_bin_minutes`, `tv_id_to_idx`, `get_total_occupancy_by_tvtw()`, `get_occupancy_vector(fid)`, and `shift_flight_occupancy(fid, delay_min)`.
  - Optional on the flight list: `iter_hotspot_crossings(hotspot_ids, active_windows=...)` to compute entrant counts for ordering.
  - Required on the evaluator: `update_flight_list(fl)`, `compute_excess_traffic_vector()`, and `compute_delay_stats()`.
  - The evaluator currently aggregates occupancy hourly against `hourly_capacity` and redistributes hourly excess to bins proportionally to per-bin occupancy within the hour.

### Example (synthetic data)

```python
from datetime import datetime
import math
import geopandas as gpd

from parrhesia.flow_agent.rate_finder import RateFinder, RateFinderConfig
from parrhesia.flow_agent.state import PlanState
from tests.flow_agent.test_rate_finder import DummyFlightList  # simple fixture helper
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator

# Build the minimal flight context shown in the tests
indexer = TVTWIndexer(time_bin_minutes=30)
indexer._tv_id_to_idx = {"TV1": 0}
indexer._idx_to_tv_id = {0: "TV1"}
indexer._populate_tvtw_mappings()

window_bins = (16, 18)
planes = {
    "F1": {
        "takeoff_time": datetime(2024, 1, 1, 8, 0, 0),
        "occupancy_intervals": [{"tvtw_index": indexer.get_tvtw_index("TV1", 16), "entry_time_s": 0}],
    },
    "F2": {
        "takeoff_time": datetime(2024, 1, 1, 8, 0, 0),
        "occupancy_intervals": [{"tvtw_index": indexer.get_tvtw_index("TV1", 16), "entry_time_s": 120}],
    },
}
flight_list = DummyFlightList(indexer=indexer, flight_metadat
a=planes)

gdf = gpd.GeoDataFrame({
    "traffic_volume_id": ["TV1"],
    "capacity": [{"08:00-09:00": 1}],
    "geometry": [None],
})

evaluator = NetworkEvaluator(gdf, flight_list)
config = RateFinderConfig(rate_grid=(math.inf, 4, 3, 2, 1), passes=2)
rate_finder = RateFinder(evaluator=evaluator, flight_list=flight_list, indexer=indexer, config=config)
plan_state = PlanState()

rates, delta_j, info = rate_finder.find_rates(
    plan_state=plan_state,
    control_volume_id="TV1",
    window_bins=window_bins,
    flows={"cluster": ("F1", "F2")},
    mode="per_flow",
)

print(rates)     # {'cluster': 3.0}, for example
print(delta_j)   # objective change relative to baseline
print(info)
```

The diagnostics dictionary reports evaluator calls, cache hits, per-flow entrant counts, and timing –
useful for KPI enforcement.

### Performance Notes

The unit tests ensure that:

* evaluation calls stay within the configured budget (`max_eval_calls`).
* wall-clock time is well below 0.5 s on the synthetic dataset.
* cached reruns perform no fresh evaluations.

The coordinate-descent passes exit early when the improvement falls below `epsilon × |baseline|` or the evaluation budget is exhausted.

---

## Testing Summary

| Test | Purpose |
| --- | --- |
| `test_state_actions.py::test_regulation_spec_canonicalization_and_rates` | checks deterministic ordering and canonicalisation. |
| `test_state_actions.py::test_plan_state_canonical_key_stable_round_trip` | ensures canonical keys are stable across ordering differences. |
| `test_state_actions.py::test_action_guards_require_valid_state` | validates guard behaviour and error conditions. |
| `test_transition.py::test_transition_pipeline_updates_z_hat_and_plan` | smoke test for the symbolic workflow, including `z_hat` slicing. |
| `test_transition.py::test_back_and_stop_transitions` | verifies stage resets and terminal behaviour. |
| `test_rate_finder.py::test_rate_finder_per_flow_improves_or_matches_baseline` | exercises the full rate finder loop end-to-end with KPI assertions. |
| `test_rate_finder.py::test_rate_finder_reuses_caches` | confirms candidate cache hits eliminate evaluator calls. |

Run all tests (under the `sdrizzle` environment) with:

```bash
conda run -n sdrizzle env PYTHONPATH=src python -m pytest tests/flow_agent -q
```

---

## Next Steps

* Wire these components into higher-level planners (e.g., once MCTS scaffolding is ready).
* Expand the dummy flight fixtures or share them via a reusable builder to support integration tests.
* Consider persisting rate finder diagnostics to the plan state so UI clients can surface them without rerunning evaluations.

---

*Last updated: based on the implementation completing the third-plan scope up to the rate finder.*
