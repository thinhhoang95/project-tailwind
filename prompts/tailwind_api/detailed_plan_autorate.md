### POST `/automatic_rate_adjustment` (Simulated Annealing)

Design an API endpoint that reuses the baseline prep from `/base_evaluation`, then runs Simulated Annealing (SA) to optimize release rates per flow at the controlled volume.

- **Purpose**: Given user flows and target/ripple TVs, compute baseline schedule and objective; run SA to find an improved schedule; return optimized rates per flow plus per-TV post-optimization occupancies.

### Request JSON
- **flows** (required, object): `flow_id -> [flight_id,...]`. Coerce flow ids to deterministic ints.
- **targets** (required, object): `TV_ID -> {"from": "HH:MM[:SS]", "to": "HH:MM[:SS]"}`. Controlled volumes restricted to these TVs.
- **ripples** (optional, object): same schema as `targets`.
- **auto_ripple_time_bins** (optional, int, default 0): if > 0, override `ripples` using union of footprints of all flights in `flows`, dilated by ±`auto_ripple_time_bins`.
- **indexer_path**, **flights_path**, **capacities_path** (optional, string): artifact overrides. Defaults match `/base_evaluation`.
- **weights** (optional, object): partial `ObjectiveWeights` overrides.
- **sa_params** (optional, object): partial `SAParams` overrides:
  - `iterations` (int, default 1000)
  - `warmup_moves` (int, default 50)
  - `alpha_T` (float, default 0.95)
  - `L` (int, temperature update period, default 50)
  - `seed` (int|null, default 0)
  - `attention_bias` (float in [0,1], default 0.8)
  - `max_shift` (int, default 4)
  - `pull_max` (int, default 2)
  - `smooth_window_max` (int, default 3)

Validation (HTTP 400):
- `flows` missing/not object; `targets` missing/empty; malformed time ranges; non-integer `auto_ripple_time_bins`; invalid SA params.

Graceful ignoring:
- Unknown TVs in `targets`/`ripples` are dropped; unknown flight IDs in `flows` are ignored.

### 200 OK Response
Top-level:
- **num_time_bins** (int)
- **tvs** (string[]): TVs from `targets`
- **target_cells** (Array<[string, int]>)
- **ripple_cells** (Array<[string, int]>)
- **flows** (FlowOpt[]): per-flow details
- **objective_baseline**: `{"score": number, "components": {...}}`
- **objective_optimized**: `{"score": number, "components": {...}}`
- **improvement**: `{"absolute": number, "percent": number}`
- **weights_used** (object)
- **sa_params_used** (object)

FlowOpt:
- **flow_id** (int)
- **controlled_volume** (string|null)
- **n0** (int[]): baseline schedule, length `T+1`
- **demand** (int[]): baseline demand, length `T`
- **n_opt** (int[]): optimized schedule, length `T+1`
- **target_demands** (object): baseline earliest-crossing demand per target TV (like `/base_evaluation`)
- **ripple_demands** (object): baseline earliest-crossing demand per ripple TV
- **target_occupancy_opt** (object): realized post-optimization occupancy per target TV (length `T`)
- **ripple_occupancy_opt** (object): realized post-optimization occupancy per ripple TV (length `T`)

Notes:
- “Demands” mirror `/base_evaluation` (earliest crossings), while “occupancy_opt” reflects the optimized schedule’s realized occupancy after delays.
- `n_opt` is the post-optimization release schedule at the controlled volume.

### Implementation plan

- New module: `src/parrhesia/api/automatic_rate_adjustment.py`
- Reuse helpers from `/base_evaluation` to avoid duplication:
  - `_default_paths_from_root`, `_cells_from_ranges`, `_auto_ripple_cells_from_flows`
- Core SA calls from `parrhesia.optim.sa_optimizer`:
  - `prepare_flow_scheduling_inputs(...)`
  - `run_sa(...)`
- Baseline scoring via `parrhesia.optim.objective.score`
- Post-optimization per-flow occupancy via `parrhesia.optim.occupancy.compute_occupancy`

Skeleton (key parts only):
```python
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Mapping, List, Tuple
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from parrhesia.optim.capacity import build_bin_capacities
from parrhesia.optim.objective import ObjectiveWeights, score
from parrhesia.optim.sa_optimizer import SAParams, prepare_flow_scheduling_inputs, run_sa
from parrhesia.optim.occupancy import compute_occupancy
from .base_evaluation import (
    _default_paths_from_root, _cells_from_ranges, _auto_ripple_cells_from_flows
)
from .flows import _load_indexer_and_flights
from .resources import get_global_resources

def compute_automatic_rate_adjustment(payload: Mapping[str, Any]) -> Dict[str, Any]:
    # 1) Load indexer/flight list (prefer explicit paths, else global, else defaults)
    idx_path_default, fl_path_default, cap_path_default = _default_paths_from_root()
    explicit_idx = payload.get("indexer_path")
    explicit_fl = payload.get("flights_path")
    if explicit_idx or explicit_fl:
        idx, fl = _load_indexer_and_flights(indexer_path=explicit_idx or idx_path_default,
                                            flights_path=explicit_fl or fl_path_default)
    else:
        g_idx, g_fl = get_global_resources()
        if g_idx is not None and g_fl is not None:
            idx, fl = g_idx, g_fl  # type: ignore
        else:
            idx, fl = _load_indexer_and_flights(indexer_path=idx_path_default, flights_path=fl_path_default)

    # 1a) Capacities
    cap_path = payload.get("capacities_path")
    if cap_path:
        capacities_by_tv = build_bin_capacities(str(cap_path), idx)
    else:
        capacities_by_tv = None
        try:
            from server_tailwind.core.resources import get_resources as _get_app_resources  # type: ignore
            _res = _get_app_resources()
            mat = _res.capacity_per_bin_matrix
            if mat is not None:
                capacities_by_tv = {}
                for tv_id, row_idx in _res.flight_list.tv_id_to_idx.items():
                    arr = mat[int(row_idx), :]
                    capacities_by_tv[str(tv_id)] = (arr * (arr >= 0.0)).astype(int)
                    capacities_by_tv[str(tv_id)][capacities_by_tv[str(tv_id)] == 0] = 9999
        except Exception:
            capacities_by_tv = capacities_by_tv
        if capacities_by_tv is None:
            capacities_by_tv = build_bin_capacities(str(cap_path_default), idx)

    # 2) Parse cells
    targets_in = payload.get("targets") or {}
    if not isinstance(targets_in, Mapping) or not targets_in:
        raise ValueError("'targets' is required and must be a non-empty mapping")
    target_cells, tvs = _cells_from_ranges(idx, targets_in)

    ripples_in = payload.get("ripples") or {}
    ripple_cells, _ = _cells_from_ranges(idx, ripples_in if isinstance(ripples_in, Mapping) else {})

    # 3) Build flow_map (string flight_id -> int flow_id), ignoring unknown flights
    flows_in = payload.get("flows") or {}
    if not isinstance(flows_in, Mapping):
        raise ValueError("'flows' must be a mapping of flow-id -> [flight_id,...]")
    flow_key_to_int: Dict[str, int] = {}
    next_id = 0
    for k in sorted((str(x) for x in flows_in.keys()), key=str):
        try:
            flow_key_to_int[k] = int(k)
        except Exception:
            flow_key_to_int[k] = next_id; next_id += 1
    flow_map: Dict[str, int] = {}
    for k, flights in flows_in.items():
        fid = flow_key_to_int[str(k)]
        for flid in (flights or []):
            sfl = str(flid)
            if sfl in fl.flight_metadata:
                flow_map[sfl] = fid

    # 4) Scheduling inputs: controlled volume and requested bins
    hotspot_ids = tvs
    flights_by_flow, ctrl_by_flow = prepare_flow_scheduling_inputs(
        flight_list=fl, flow_map=flow_map, hotspot_ids=hotspot_ids
    )

    # 5) Baseline n0/demand
    T = int(idx.num_time_bins)
    n0: Dict[int, List[int]] = {}
    demand: Dict[int, List[int]] = {}
    for f, specs in flights_by_flow.items():
        arr = [0] * (T + 1)
        for sp in specs or []:
            rb = int(sp.get("requested_bin", 0))
            if 0 <= rb <= T:
                arr[rb] += 1
        n0[int(f)] = arr
        demand[int(f)] = arr[:T]

    # 5a) Auto-ripple override
    try:
        auto_w = int(payload.get("auto_ripple_time_bins", 0))
    except Exception:
        auto_w = 0
    if auto_w > 0:
        ripple_cells = _auto_ripple_cells_from_flows(idx, fl, flow_map.keys(), auto_w)

    # 5b) Per-TV baseline demands (same as /base_evaluation)
    target_tv_ids = list(dict.fromkeys(str(tv) for tv in tvs))
    ripple_tv_ids = sorted({str(tv) for (tv, _b) in ripple_cells})
    # (reuse the /base_evaluation loop to build target_demands_by_flow, ripple_demands_by_flow)

    # 6) Baseline objective
    weights = ObjectiveWeights(**(payload.get("weights") or {}))
    J0, comps0, _arts0 = score(
        n0, flights_by_flow=flights_by_flow, indexer=idx,
        capacities_by_tv=capacities_by_tv, target_cells=target_cells, ripple_cells=ripple_cells,
        flight_list=fl, weights=weights
    )

    # 7) SA params + run
    sa_kwargs = dict(payload.get("sa_params") or {})
    params = SAParams(**{k: v for k, v in sa_kwargs.items() if k in SAParams.__dataclass_fields__})
    n_best, J_star, comps_star, arts_star = run_sa(
        flights_by_flow=flights_by_flow, flight_list=fl, indexer=idx,
        capacities_by_tv=capacities_by_tv, target_cells=target_cells, ripple_cells=ripple_cells,
        weights=weights, params=params
    )

    # 8) Per-flow post-optimization occupancy on targets/ripples
    delays = arts_star.get("delays_min", {})  # flight_id -> minutes
    def _occ_for_flow(fid_list: List[str]) -> Dict[str, List[int]]:
        sub_meta = {fid: fl.flight_metadata[fid] for fid in fid_list if fid in fl.flight_metadata}
        class _SubFL: pass
        sub = _SubFL(); sub.flight_metadata = sub_meta
        delays_sub = {fid: int(delays.get(fid, 0)) for fid in fid_list}
        occ = compute_occupancy(sub, delays_sub, idx, tv_filter=set(target_tv_ids) | set(ripple_tv_ids))
        return {tv: occ.get(tv, []).tolist() for tv in (set(target_tv_ids) | set(ripple_tv_ids))}
    target_occ_by_flow: Dict[int, Dict[str, List[int]]] = {}
    ripple_occ_by_flow: Dict[int, Dict[str, List[int]]] = {}
    for f, specs in flights_by_flow.items():
        fids = [str(sp.get("flight_id")) for sp in (specs or [])]
        occ = _occ_for_flow(fids)
        target_occ_by_flow[int(f)] = {tv: occ.get(tv, [0]*T) for tv in target_tv_ids}
        ripple_occ_by_flow[int(f)] = {tv: occ.get(tv, [0]*T) for tv in ripple_tv_ids}

    # 9) Assemble flows + response
    flows_out: List[Dict[str, Any]] = []
    for f in sorted(flights_by_flow.keys(), key=lambda x: int(x)):
        flows_out.append({
            "flow_id": int(f),
            "controlled_volume": (str(ctrl_by_flow.get(f)) if ctrl_by_flow.get(f) is not None else None),
            "n0": n0[int(f)],
            "demand": demand[int(f)],
            "n_opt": list(map(int, n_best.get(f, []).tolist())) if hasattr(n_best.get(f), "tolist") else n_best.get(f, []),
            "target_demands": {},     # fill from baseline per-TV computation
            "ripple_demands": {},     # fill from baseline per-TV computation
            "target_occupancy_opt": target_occ_by_flow.get(int(f), {}),
            "ripple_occupancy_opt": ripple_occ_by_flow.get(int(f), {}),
        })

    improvement_abs = float(J0 - J_star)
    improvement_pct = (improvement_abs / J0 * 100.0) if J0 != 0 else 0.0
    return {
        "num_time_bins": T,
        "tvs": list(hotspot_ids),
        "target_cells": [(str(tv), int(b)) for (tv, b) in target_cells],
        "ripple_cells": [(str(tv), int(b)) for (tv, b) in ripple_cells],
        "flows": flows_out,
        "objective_baseline": {"score": float(J0), "components": comps0},
        "objective_optimized": {"score": float(J_star), "components": comps_star},
        "improvement": {"absolute": improvement_abs, "percent": round(improvement_pct, 2)},
        "weights_used": asdict(weights),
        "sa_params_used": asdict(params),
    }
```

Server route (mirror `/base_evaluation` handling):
```python
# src/server_tailwind/main.py
from parrhesia.api.automatic_rate_adjustment import compute_automatic_rate_adjustment

@app.post("/automatic_rate_adjustment")
def post_automatic_rate_adjustment(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body required")
    if not payload.get("targets"):
        raise HTTPException(status_code=400, detail="'targets' is required")
    if not payload.get("flows"):
        raise HTTPException(status_code=400, detail="'flows' is required")
    try:
        result = compute_automatic_rate_adjustment(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback; print(f"Exception in /automatic_rate_adjustment: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize: {e}")
    return result
```

Export:
```python
# src/parrhesia/api/__init__.py
from .automatic_rate_adjustment import compute_automatic_rate_adjustment
__all__ = ["compute_flows", "compute_base_evaluation", "compute_automatic_rate_adjustment"]
```

### Example
```bash
curl -X POST http://localhost:8000/automatic_rate_adjustment \
  -H 'Content-Type: application/json' \
  -d '{
    "flows": {"0": ["FL1","FL2"], "1": ["FL3"]},
    "targets": {"TV_A": {"from": "08:00", "to": "09:00"}},
    "auto_ripple_time_bins": 2,
    "weights": {"alpha_gt": 10.0, "lambda_delay": 0.1},
    "sa_params": {"iterations": 300, "seed": 0, "attention_bias": 0.8}
  }'
```

### Notes and considerations
- Reuse the exact parsing/paths/capacity-loading behavior from `/base_evaluation` for consistency and performance (app-level cached resources when available).
- Arrays must be JSON-serializable: convert numpy arrays to lists.
- “Demands” vs “occupancy”: keep baseline `target_demands`/`ripple_demands` identical to `/base_evaluation` for comparability; add “target_occupancy_opt”/“ripple_occupancy_opt” to show realized counts under the optimized schedule.
- Determinism: seed SA via `sa_params.seed` for repeatable results.
- Time complexity: per-flow occupancy is computed on reduced flight subsets to avoid N× duplication.

- Add concise docs to `API_README_FLOWS.md` alongside `/base_evaluation` with request/response schema and an example.

- Unit tests: mirror `/base_evaluation` tests plus assertions that `objective_optimized.score <= objective_baseline.score`, shapes of `n_opt`, and presence of per-TV occupancy fields.

- Optional future: expose an `audit` flag to re-run `score` on `n_opt` with `audit_exceedances=True` to return structured exceedance breakdown.


- Implemented plan: New `compute_automatic_rate_adjustment` that loads artifacts, builds flows/controlled volume, computes baseline n0 and objective, runs SA via `run_sa`, and returns `n_opt` plus post-optimization per-TV occupancies and objective deltas. 
- Integration: Added `/automatic_rate_adjustment` route in `src/server_tailwind/main.py` and export in `parrhesia/api/__init__.py`.