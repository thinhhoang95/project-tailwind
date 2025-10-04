### What’s available to build on
- Proposals per hotspot:
```195:208:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/engine.py
def propose_regulations_for_hotspot(
    *,
    indexer,
    flight_list,
    capacities_by_tv: Mapping[str, np.ndarray],
    travel_minutes_map: Mapping[str, Mapping[str, float]],
    hotspot_tv: str,
    timebins_h: Sequence[int],
    flows_payload: Optional[Mapping[str, Any]] = None,
    flow_to_flights: Optional[Mapping[str, Sequence[str]]] = None,
    weights: Optional[FlowScoreWeights] = None,
    config: Optional[RegenConfig] = None,
    verbose_debug: bool = False,
) -> List[Proposal]:
```
```497:511:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/engine.py
        proposals.append(
            Proposal(
                hotspot_id=str(hotspot_tv),
                controlled_volume=str(ctrl_volume),
                window=bundle_variant.window,
                flows_info=flows_info,
                predicted_improvement=improvement,
                diagnostics=proposal_diag,
                target_cells=[(str(tv), int(b)) for (tv, b) in target_cells],
                ripple_cells=list(proposal_diag.get("ripple_cells", [])),
                target_tvs=[str(hotspot_tv)],
                ripple_tvs=list(proposal_diag.get("ripple_tvs", [])),
            )
        )
    return proposals
```
- Hotspot extraction + payload conversion:
```43:57:/mnt/d/project-tailwind/src/parrhesia/flow_agent35/regen/hotspot_segment_extractor.py
def extract_hotspot_segments_from_resources(
    *, threshold: float = 0.0, resources: Optional[AppResources] = None
) -> List[Dict[str, Any]]:
    """Detect hotspot segments using shared AppResources artifacts.
```
- Convert proposal to a regulation plan:
```651:659:/mnt/d/project-tailwind/src/parrhesia/actions/regulations.py
@classmethod
def from_proposal(
    cls,
    proposal: "Proposal",
    flights_by_flow: Mapping[Any, Sequence[Any]],
    *,
    time_bin_minutes: Optional[int] = None,
) -> "DFRegulationPlan":
```
- Evaluate and apply to mutate the state:
```56:69:/mnt/d/project-tailwind/src/project_tailwind/stateman/flight_list_with_delta.py
def step_by_delay(self, *views: DeltaOccupancyView, finalize: bool = True) -> None:
    ...
    for view in views:
        ...
        self._apply_single_view(view)
    if finalize:
        self.finalize_occupancy_updates()
```
```55:66:/mnt/d/project-tailwind/src/project_tailwind/stateman/delta_view.py
@classmethod
def from_delay_table(
    cls,
    flights: FlightList,
    delays: DelayAssignmentTable,
    *,
    regulation_id: Optional[str] = None,
) -> "DeltaOccupancyView":
    num_tvtws = int(flights.num_tvtws)
```
- Cache refresh for shared/global users:
```13:26:/mnt/d/project-tailwind/src/server_tailwind/core/cache_refresh.py
def refresh_after_state_update(
    resources: Any,
    *,
    airspace_wrapper: Optional[Any] = None,
    count_wrapper: Optional[Any] = None,
    query_wrapper: Optional[Any] = None,
) -> None:
    """Refresh global caches so subsequent queries see the latest flight list state."""
    ...
```

### Concrete implementation plan (files, types, flow)
- Bind evaluator resources to the sandbox in `RZSandbox.apply_proposal` using `server_tailwind.core.resources` before calling `evaluate_df_regulation_plan`, then restore.
- Normalize capacities with `normalize_capacities(...)` before passing to `propose_regulations_for_hotspot`.
- Scope flows to the current sandbox when calling `compute_flows(...)` using a tiny context, so they always see the current `flight_list`. Recompute flows and hotspots after every move.
- Do not call `refresh_after_state_update(...)` in the CLI loop; use scoped flows context and `step_by_delay(..., finalize=True)` to keep occupancy/caches correct.
- Root the CLI runner off a locally constructed `AppResources().preload_all()` (not `get_resources()`), ensuring full process isolation from any server instance.
- Keep time-axis consistency with an assertion in the sandbox: `indexer.num_time_bins == capacity_per_bin_matrix.shape[1]` (the evaluator also enforces this).
- Forks use a per-simulation sandbox via structural clone (zero reload; CSR/LIL and small metadata copy); do not use `deepcopy`.
- Ensure `segment_to_hotspot_payload` is imported where used (e.g., in `mcts.py`).
- New package: `src/regulation_zero/`
  - `types.py`:
    - `RZAction`: hotspot_key, proposal_rank, delta_obj (immediate reward), optional `control_tv_id`, `window_bins`.
    - `RZPathKey`: tuple of action keys (stable canonical solution ID).
    - `RZConfig`: max_depth, puct_c, dirichlet_alpha, dirichlet_epsilon, max_hotspots_per_node, k_proposals_per_hotspot, num_simulations.
  - `env.py`:
    - `RZSandbox`: wraps `AppResources` but isolates a `FlightListWithDelta` copy per sandbox.
      - `fork()` creates a child sandbox by cloning the parent’s `flight_list` (deep-copy, no disk reload). Base logic mirrors the API clone:
```111:173:/mnt/d/project-tailwind/src/server_tailwind/airspace/network_evaluator_for_api.py
def _clone_flight_list_for_baseline(self, flight_list: FlightList) -> FlightList:
    """Create a detached snapshot of ``flight_list`` without reloading from disk."""
    ...
    clone.occupancy_matrix = occupancy.copy()
    ...
    clone._occupancy_matrix_lil = lil_matrix.copy()
```
    - `extract_hotspots(sandbox) -> List[Segment]` via `extract_hotspot_segments_from_resources(resources=sandbox.resources)`.
    - `proposals_for_hotspot(sandbox, hotspot_payload, k) -> List[ProposalMeta]`:
      - Call `set_global_resources(sandbox.indexer, sandbox.flight_list)` before `compute_flows(...)` (as in the examples).
      - Call `propose_regulations_for_hotspot(...)` with explicit `indexer`, `flight_list`, `capacities_by_tv`, `travel_minutes_map`.
      - Return list of `(proposal, flights_by_flow, delta_objective_score, rank)`.
    - `apply_action(sandbox, proposal, flights_by_flow) -> None`:
      - `plan = DFRegulationPlan.from_proposal(...)`
      - `eval_res = evaluate_df_regulation_plan(...)`
      - `delays = DelayAssignmentTable.from_dict(eval_res.delays_by_flight)`
      - `view = DeltaOccupancyView.from_delay_table(flights=sandbox.flight_list, delays=delays, regulation_id="rz")`
      - `sandbox.flight_list.step_by_delay(view)`; `refresh_after_state_update(sandbox.resources)`
      - Note: use per-node sandbox; no revert required.
  - `cache.py`:
    - `TranspositionTable`: `Dict[RZPathKey, NodeStats]` (N, W, child edges).
    - `ProposalsCache`: `Dict[Tuple[RZPathKey, HotspotKey], List[ProposalMeta]]`.
    - `HotspotKey`: `f"{tv}:{start_bin}-{end_bin}"` from segment.
  - `mcts.py`:
    - Node struct: `children: Dict[RZAction, ChildStats]` with `P` priors, `N`, `W`, `Q`.
    - Selection: PUCT with root Dirichlet noise: `P_root = (1 - eps) * P + eps * Dir(alpha)`.
    - Expansion: on first visit, enumerate hotspots (top M), for each get up to `k` proposals; compute priors from normalized `delta_objective_score` (e.g., softmax over all child proposals in this node).
    - Backup: leaf value = 0 (no rollout). Backup along path using edge immediate rewards (sum of `delta_obj` seen so far) or equivalently backup 0 and accumulate returns per edge.
    - Termination: when `remaining_depth == 0` or no hotspots or no proposals.
  - `runner.py`:
    - Ties everything together: start from baseline `AppResources().preload_all()`, create root `RZSandbox`, run `num_simulations`, then extract final sequence by repeatedly taking child with max visits at each depth and applying to a fresh sandbox to produce the concrete final solution.

### Design details that satisfy the requests
- State s contains partial canonical solution and remaining depth:
  - Represent the canonical solution as `RZPathKey = tuple[(hotspot_key, proposal_rank)]`.
  - `remaining_depth = cfg.max_depth - len(path)`.
- Action selection is a “combo-action”:
  - We present each child action as `(hotspot segment -> proposal_rank)`.
  - We cache proposals per `(state, hotspot)` to avoid re-running regen.
  - Flow and hotspot parameters (`flows_threshold`, `flows_resolution`, `hotspot_threshold`) are stored in `RZConfig` and applied uniformly at each node expansion.
- Dirichlet noise at root:
  - Injected into the root node’s child priors only, AlphaZero-style.
- Terminal and return:
  - Terminal when `remaining_depth == 0` or no hotspots/proposals.
  - Return is sum of per-edge `delta_objective_score` along the chosen solution path.
- No rollout:
  - Implemented by backing up zero leaf value; the value function is purely the accumulated immediate rewards gathered along traversal.
- Proposal caching and IDs:
  - For each node state, proposals for a hotspot are identified by their rank within that node’s regen call and the hotspot key. We store `proposal_rank`, `delta_obj`, and the serialized `proposal` to reconstruct the plan on apply.

### Guardrails (fail fast, no silent fallbacks)

- Always bind globals to the sandbox, assert they’re active, and restore them.
- Recompute hotspots and flows after each move; never reuse flows across nodes.
- Fail if windows resolve to no bins, flows are empty, or mapping includes unknown flights.
- Normalize and validate capacities once per sandbox; enforce time-axis consistency.
- For DF plan construction, never allow fallback `time_bin_minutes`; pass indexer value and assert.
- After evaluation, step must mutate the sandbox (`num_regulations` increases); assert restoration of global resources post-eval.
- Cache proposals only per `(state_path, hotspot_key)` and verify deltas are finite.
- Structural cloning must deep-copy sparse matrices and clear dirty flags; assert instance independence.

### Edits to add in `src/regulation_zero/types.py`

- Add a fail-fast toggle (default on).

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

HotspotKey = str
ActionKey = Tuple[HotspotKey, int]  # (hotspot_key, proposal_rank)

@dataclass(frozen=True)
class RZAction:
    hotspot_key: HotspotKey
    proposal_rank: int
    delta_obj: float
    control_tv_id: Optional[str] = None
    window_bins: Optional[Tuple[int, int]] = None

RZPathKey = Tuple[ActionKey, ...]

@dataclass
class RZConfig:
    max_depth: int = 3
    puct_c: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_hotspots_per_node: int = 5
    k_proposals_per_hotspot: int = 8
    num_simulations: int = 200

    # Flow clustering and hotspot extraction params
    flows_threshold: Optional[float] = None  # defaults to compute_flows default (0.1) when None
    flows_resolution: Optional[float] = None  # defaults to compute_flows default (1.0) when None
    hotspot_threshold: float = 0.0  # excess threshold in extractor
    direction_mode: str = "coord_cosine"  # direction-aware reweighting mode
    fail_fast: bool = True
```

### Edits to add in `src/regulation_zero/env.py`

- Harden resource contexts, cloning, proposals, and apply stages.

```python
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from contextlib import contextmanager
from server_tailwind.core.resources import AppResources
from parrhesia.api.resources import get_global_resources, set_global_resources
from parrhesia.api.flows import compute_flows
from parrhesia.flow_agent35.regen.hotspot_segment_extractor import (
    extract_hotspot_segments_from_resources,
    segment_to_hotspot_payload,
)
from parrhesia.flow_agent35.regen.engine import propose_regulations_for_hotspot
from parrhesia.flow_agent35.regen.types import RegenConfig
from parrhesia.actions.regulations import DFRegulationPlan
from parrhesia.actions.dfplan_evaluator import evaluate_df_regulation_plan
from project_tailwind.stateman.delay_assignment import DelayAssignmentTable
from project_tailwind.stateman.delta_view import DeltaOccupancyView
from parrhesia.optim.capacity import normalize_capacities
import server_tailwind.core.resources as core_res
from project_tailwind.stateman.flight_list_with_delta import FlightListWithDelta
import numpy as np


def clone_flight_list_structural(src: FlightListWithDelta) -> FlightListWithDelta:
    cls = src.__class__
    dst = cls.__new__(cls)

    # Basic identity and indexer wiring
    dst.occupancy_file_path = getattr(src, "occupancy_file_path", "")
    dst.tvtw_indexer_path = getattr(src, "tvtw_indexer_path", "")
    dst.tvtw_indexer = dict(getattr(src, "tvtw_indexer", {}) or {})
    dst.time_bin_minutes = int(getattr(src, "time_bin_minutes", 60))
    dst.tv_id_to_idx = dict(getattr(src, "tv_id_to_idx", {}) or {})
    dst.idx_to_tv_id = dict(getattr(src, "idx_to_tv_id", {}) or {})
    dst.num_traffic_volumes = int(getattr(src, "num_traffic_volumes", len(dst.tv_id_to_idx)))
    dst.num_time_bins_per_tv = int(getattr(src, "num_time_bins_per_tv", 1))
    dst._indexer = getattr(src, "_indexer", None)

    # Flight ids and shape
    dst.flight_ids = list(getattr(src, "flight_ids", ()))
    dst.num_flights = int(getattr(src, "num_flights", len(dst.flight_ids)))
    dst.flight_id_to_row = dict(getattr(src, "flight_id_to_row", {}) or {})
    dst.num_tvtws = int(getattr(src, "num_tvtws", 0))

    # Sparse matrices (copy arrays, no reload)
    occ_csr = getattr(src, "occupancy_matrix", None)
    dst.occupancy_matrix = occ_csr.copy() if occ_csr is not None else None
    occ_lil = getattr(src, "_occupancy_matrix_lil", None)
    dst._occupancy_matrix_lil = occ_lil.copy() if occ_lil is not None else None
    # Force a clean state to avoid mid-update propagation
    dst._lil_matrix_dirty = False
    buf = getattr(src, "_temp_occupancy_buffer", None)
    dst._temp_occupancy_buffer = np.array(buf, copy=True) if buf is not None else np.zeros(dst.num_tvtws, dtype=np.float32)

    # Small metadata (deep copy for isolation; much smaller than matrices)
    from copy import deepcopy as _dc
    dst.flight_data = _dc(getattr(src, "flight_data", {}) or {})
    dst.flight_metadata = _dc(getattr(src, "flight_metadata", {}) or {})

    # Caches
    dst._flight_tv_sequence_cache = {}

    # Delta bookkeeping (preserve current state)
    dst.applied_regulations = list(getattr(src, "applied_regulations", []) or [])
    dst.delay_histogram = dict(getattr(src, "delay_histogram", {}) or {})
    dst.total_delay_assigned_min = int(getattr(src, "total_delay_assigned_min", 0))
    dst.num_delayed_flights = int(getattr(src, "num_delayed_flights", 0))
    dst.num_regulations = int(getattr(src, "num_regulations", 0))
    agg = getattr(src, "_delta_aggregate", None)
    dst._delta_aggregate = np.array(agg, copy=True) if agg is not None else np.zeros(dst.num_tvtws, dtype=np.int64)
    dst._applied_views = []  # do not retain historical objects

    # Check for matrix sharing
    if (src.occupancy_matrix is not None and dst.occupancy_matrix is src.occupancy_matrix) or \
       (src._occupancy_matrix_lil is not None and dst._occupancy_matrix_lil is src._occupancy_matrix_lil):
        raise RuntimeError("Clone shares sparse matrix references with source")
    return dst

def _timebins_from_window(window_bins):
    """Converts a window [start, end_exclusive] to a list of integer time bins."""
    wb = [int(b) for b in (window_bins or [])]
    if not wb:
        return []
    if len(wb) == 1:
        return wb
    start = int(wb[0])
    end_exclusive = int(wb[1])
    if end_exclusive <= start:
        end_exclusive = start + 1
    return list(range(start, end_exclusive))


@contextmanager
def with_fl_resources(indexer, flight_list):
    prev = get_global_resources()
    set_global_resources(indexer, flight_list)
    gi, gf = get_global_resources()
    if gi is not indexer or gf is not flight_list:
        raise RuntimeError("Flows globals not bound to sandbox resources")
    try:
        yield
    finally:
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
    prev = core_res.get_resources()
    core_res._GLOBAL_RESOURCES = res
    if core_res.get_resources() is not res:
        raise RuntimeError("Core resources not set to sandbox")
    try:
        yield
    finally:
        core_res._GLOBAL_RESOURCES = prev
        if core_res.get_resources() is not prev:
            raise RuntimeError("Core resources not restored after eval")


class RZSandbox:
    def __init__(self, res: AppResources, cfg: Optional[RZConfig] = None):
        self._res = res.preload_all()
        self._cfg = cfg or RZConfig()
        T_idx = int(self._res.indexer.num_time_bins)
        T_cap = int(self._res.capacity_per_bin_matrix.shape[1])
        if T_idx != T_cap:
            raise RuntimeError(f"Time-axis mismatch: indexer bins={T_idx}, capacity width={T_cap}")

    @property
    def resources(self) -> AppResources:
        return self._res

    def fork(self) -> "RZSandbox":
        parent = self._res
        child = AppResources(parent.paths)
        # reuse heavy, read-only artifacts by reference
        child._indexer = parent.indexer
        child._traffic_volumes_gdf = parent.traffic_volumes_gdf
        child._hourly_capacity_by_tv = parent.hourly_capacity_by_tv
        child._capacity_per_bin_matrix = parent.capacity_per_bin_matrix
        child._travel_minutes = parent._travel_minutes
        child._tv_centroids = parent.tv_centroids
        # structural clone of the current flight_list (no deepcopy, no disk reload)
        child._flight_list = clone_flight_list_structural(parent.flight_list)
        return RZSandbox(child, cfg=self._cfg)

    def extract_hotspots(self, k: int) -> List[Dict[str, Any]]:
        segs = extract_hotspot_segments_from_resources(
            threshold=float(self._cfg.hotspot_threshold), resources=self._res
        )
        segs.sort(key=lambda s: float(s.get("max_excess", 0.0)), reverse=True)
        return segs[:k]

    def proposals_for_hotspot(self, hotspot_payload: Dict[str, Any], k: int):
        res = self._res
        # Validate hotspot payload
        if "control_volume_id" not in hotspot_payload:
            raise ValueError("hotspot_payload missing control_volume_id")
        window_bins = hotspot_payload.get("window_bins", [])
        timebins_h = _timebins_from_window(window_bins)
        if not timebins_h:
            raise RuntimeError("hotspot window resolves to no bins")
        control_tv = str(hotspot_payload["control_volume_id"])
        if control_tv not in res.flight_list.tv_id_to_idx:
            raise ValueError(f"Unknown control TV: {control_tv}")

        # Flows must be bound to THIS sandbox
        with with_fl_resources(res.indexer, res.flight_list):
            flows_payload = compute_flows(
                tvs=[control_tv],
                timebins=timebins_h,
                threshold=self._cfg.flows_threshold,
                resolution=self._cfg.flows_resolution,
                direction_opts={"mode": self._cfg.direction_mode, "tv_centroids": res.tv_centroids},
            )
        # Validate flows payload
        if flows_payload.get("tvs") != [control_tv]:
            raise RuntimeError("compute_flows returned unexpected TV set")
        tb = list(flows_payload.get("timebins") or [])
        if tb and tb != list(timebins_h):
            raise RuntimeError("compute_flows returned mismatched timebins")
        flows = flows_payload.get("flows") or []
        if self._cfg.fail_fast and len(flows) == 0:
            raise RuntimeError("No flows returned for a non-empty hotspot window")

        flow_to_flights = {
            str(int(flow["flow_id"])): [str(spec["flight_id"]) for spec in (flow.get("flights") or []) if spec.get("flight_id") is not None]
            for flow in flows
            if "flow_id" in flow
        }
        if self._cfg.fail_fast and not flow_to_flights:
            raise RuntimeError("Empty flow_to_flights after compute_flows")
        # Validate that all flights exist in this sandbox
        missing = [
            fid for fids in flow_to_flights.values() for fid in fids
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
                raise RuntimeError(f"Non-positive capacity detected after normalization for {tv}")

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
        # Predicted improvements must be finite
        for p in proposals:
            d = getattr(getattr(p, "predicted_improvement", None), "delta_objective_score", None)
            if d is None or not np.isfinite(float(d)):
                raise RuntimeError("Proposal has non-finite delta_objective_score")
        return proposals, flow_to_flights

    def apply_proposal(self, proposal, flights_by_flow: Dict[str, List[str]]) -> None:
        res = self._res
        tbm = int(res.indexer.time_bin_minutes)
        if tbm <= 0:
            raise RuntimeError("Invalid indexer.time_bin_minutes")
        # Ensure we never use DFRegulationPlan fallback minutes (30)
        plan = DFRegulationPlan.from_proposal(
            proposal,
            flights_by_flow=flights_by_flow,
            time_bin_minutes=tbm,
        )
        # Evaluate in sandbox-scoped core resources
        with with_core_resources(res):
            eval_res = evaluate_df_regulation_plan(
                plan,
                indexer_path=str(res.paths.tvtw_indexer_path),
                flights_path=str(res.paths.occupancy_file_path),
            )
        # Basic sanity on eval result
        delays = DelayAssignmentTable.from_dict(eval_res.delays_by_flight)
        if self._cfg.fail_fast and not delays.delays_by_flight:
            raise RuntimeError("Evaluation produced no delays; likely misbound resources or empty targets")

        pre_regs = int(res.flight_list.num_regulations)
        pre_total_delay = int(res.flight_list.total_delay_assigned_min)
        view = DeltaOccupancyView.from_delay_table(res.flight_list, delays, regulation_id="rz")
        res.flight_list.step_by_delay(view, finalize=True)
        if int(res.flight_list.num_regulations) <= pre_regs:
            raise RuntimeError("step_by_delay did not register a new regulation")
        if int(res.flight_list.total_delay_assigned_min) < pre_total_delay:
            raise RuntimeError("Total delay decreased after applying a regulation")
```
- `src/regulation_zero/cache.py`
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from .types import RZPathKey, ActionKey

@dataclass
class ChildStats:
    P: float
    N: int = 0
    W: float = 0.0  # total value
    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

@dataclass
class NodeStats:
    children: Dict[ActionKey, ChildStats] = field(default_factory=dict)
    expanded: bool = False

TranspositionTable = Dict[RZPathKey, NodeStats]
ProposalsCache = Dict[Tuple[RZPathKey, str], List[Any]]  # (state, hotspot_key) -> proposals
```
- `src/regulation_zero/mcts.py` (outline)
```python
from __future__ import annotations
import math, random
from typing import List, Tuple
import numpy as np
from .types import RZConfig, RZPathKey, RZAction, ActionKey
from .cache import NodeStats, ChildStats, TranspositionTable, ProposalsCache
from parrhesia.flow_agent35.regen.hotspot_segment_extractor import segment_to_hotspot_payload

def softmax(xs: List[float], tau: float = 1.0) -> List[float]:
    arr = np.asarray(xs, dtype=np.float64) / max(tau, 1e-6)
    arr -= arr.max() if arr.size else 0.0
    ex = np.exp(arr)
    Z = ex.sum()
    return [(x / Z) if Z > 0 else 1.0 / len(xs) for x in ex]

class MCTS:
    def __init__(self, env_factory, cfg: RZConfig):
        self.env_factory = env_factory  # Should provide a fresh RZSandbox fork
        self.cfg = cfg
        self.tt: TranspositionTable = {}
        self.cache: ProposalsCache = {}

    def run(self) -> List[RZAction]:
        for _ in range(self.cfg.num_simulations):
            # Fork once per simulation, then mutate the sandbox down the path.
            env = self.env_factory()
            self._simulate((), env)

        # Derive final plan by greedy descent on visits
        actions: List[RZAction] = []
        state: RZPathKey = ()
        depth = 0
        while depth < self.cfg.max_depth and state in self.tt and self.tt[state].children:
            child = max(self.tt[state].children.items(), key=lambda kv: kv[1].N)
            ak = child[0]
            # store stub RZAction; delta filled later when applied
            actions.append(RZAction(hotspot_key=ak[0], proposal_rank=ak[1], delta_obj=0.0))
            state = tuple(list(state) + [ak])  # descend
            depth += 1
        return actions

    def _simulate(self, state: RZPathKey, env: RZSandbox) -> float:
        depth = len(state)
        if depth >= self.cfg.max_depth:
            return 0.0

        node = self.tt.setdefault(state, NodeStats())
        if not node.expanded:
            # Expand: use the sandbox, which is already at the correct state for this node.
            segments = env.extract_hotspots(self.cfg.max_hotspots_per_node)
            actions: List[Tuple[ActionKey, float]] = []
            for seg in segments:
                hk = f'{seg["traffic_volume_id"]}:{int(seg["start_bin"])}-{int(seg["end_bin"])}'
                hot_payload = segment_to_hotspot_payload(seg)  # import where defined
                key = (state, hk)
                if key not in self.cache:
                    proposals, f2f = env.proposals_for_hotspot(hot_payload, self.cfg.k_proposals_per_hotspot)
                    # keep tuple of (proposal, f2f, delta_obj)
                    ranked = [(p, f2f, float(p.predicted_improvement.delta_objective_score)) for p in proposals]
                    self.cache[key] = ranked
                ranked = self.cache[key]
                for rank, (_, _, delta) in enumerate(ranked[: self.cfg.k_proposals_per_hotspot]):
                    if not np.isfinite(float(delta)):
                        raise RuntimeError("Non-finite delta_objective_score in proposals cache")
                    actions.append(((hk, rank), float(delta)))
            if not actions:
                node.expanded = True
                return 0.0
            priors = softmax([d for _, d in actions])
            for (ak, _), p in zip(actions, priors):
                node.children.setdefault(ak, ChildStats(P=float(p)))
            node.expanded = True

            # Apply Dirichlet noise at root
            if depth == 0 and node.children:
                eps = self.cfg.dirichlet_epsilon
                alpha = self.cfg.dirichlet_alpha
                noise = np.random.default_rng().dirichlet([alpha] * len(node.children))
                for (cs, n) in zip(node.children.values(), noise):
                    cs.P = float((1 - eps) * cs.P + eps * n)

        # Selection
        total_N = sum(cs.N for cs in self.tt[state].children.values()) + 1
        best_ak, best_cs = max(
            self.tt[state].children.items(),
            key=lambda kv: kv[1].Q + self.cfg.puct_c * kv[1].P * math.sqrt(total_N) / (1 + kv[1].N),
        )
        # Recurse and get reward
        r = self._roll(state, best_ak, env)
        # Backup
        best_cs.N += 1
        best_cs.W += r
        return r

    def _roll(self, state: RZPathKey, ak: ActionKey, env: RZSandbox) -> float:
        # Immediate reward for this edge from cached proposal
        hk, rank = ak
        key = (state, hk)
        if key not in self.cache:
            raise RuntimeError("Cache miss for selected action; expansion logic broken")
        ranked = self.cache[key]
        if rank < 0 or rank >= len(ranked):
            raise IndexError("Selected proposal rank out of range")
        prop, f2f, delta = ranked[rank]
        if not np.isfinite(float(delta)):
            raise RuntimeError("Selected action has non-finite immediate reward")
        # Apply action to the sandbox, mutating it for the child state.
        env.apply_proposal(prop, f2f)
        # Next state
        child_state: RZPathKey = tuple(list(state) + [ak])
        v = self._simulate(child_state, env)
        return float(delta) + v
```
- `src/regulation_zero/runner.py` (glue)
```python
from __future__ import annotations
from server_tailwind.core.resources import AppResources
from regulation_zero.mcts import MCTS
from regulation_zero.types import RZConfig
from regulation_zero.env import RZSandbox


def make_env_factory(root_res: AppResources, cfg: RZConfig):
    root = RZSandbox(root_res.preload_all(), cfg=cfg)

    def factory():
        return root.fork()

    return factory


def run_search():
    cfg = RZConfig()
    # Local, process-isolated resources for the CLI
    res = AppResources().preload_all()
    mcts = MCTS(env_factory=make_env_factory(res, cfg), cfg=cfg)
    actions = mcts.run()
    return actions


### Notes and assumptions
- We use a "fork once per simulation" strategy. A sandbox is forked from the root at the start of each simulation and then mutated along the search path. This contains memory usage to one `flight_list` copy per simulation, avoiding the high cost of forking at each node.
- Hotspots are re-extracted and flows recomputed at each node expansion from the current sandbox state; flows use `with_fl_resources(...)` so they cannot accidentally bind to a stale `flight_list`.
- `compute_flows` must see the node’s sandbox flight list: we call `set_global_resources(indexer, flight_list)` before it.
- We use predicted improvement `delta_objective_score` from regen as immediate rewards and as priors (via softmax); leaf value is 0 per “no rollout”.
- Transpositions are keyed by canonical solution path, not structural state equivalence. That’s sufficient to cache `regen` results and node stats for our search.
- Optimizations (optional): instead of forking from the baseline for each node, replay actions on a fork to avoid recomputing all previous steps; or add an undo stack using stored delay tables.
- Structural clone mirrors the evaluator’s baseline cloning semantics for `FlightList` (copy sparse CSR/LIL and small dicts; no disk reload). This keeps forks isolated without `deepcopy`, while preserving current post-move state.

- Confirmed behaviors used:
  - Hotspot extraction and ranking via resources.
  - Proposal generation on explicit `indexer/flight_list` without relying on global singletons.
  - State stepping via `DeltaOccupancyView` and `step_by_delay`, then `refresh_after_state_update`.

- Integration smoke test: replicate the flow in `examples/regen/regen_second_order.py` within one MCTS playout to validate stepping.

- Final solution materialization: after MCTS decides a sequence, rebuild the actual `DFRegulationPlan` objects by querying the cached `(proposal, flights_by_flow)` per selected `(state, hotspot_key, rank)` and apply on a fresh sandbox or emit payloads.

- Safety: handle “no proposals” gracefully; cap hotspots per node; thread-safety not required for single-process runs.

- Tuning defaults: `max_depth=3`, `k_proposals_per_hotspot=8`, `max_hotspots_per_node=5`, `num_simulations=200` as a starting point.

Summary
- Designed `src/regulation_zero/` with `types`, `env`, `cache`, `mcts`, and `runner`.
- Integrated existing APIs: hotspot extraction, regen proposals, plan evaluation, and state stepping.
- Defined canonical solution keys, proposal caching by `(state, hotspot)`, and root Dirichlet noise.
- Provided code outlines to implement per-node sandboxes and the PUCT search without rollouts.

### Guardrails justified by existing fallback behavior

- Flows fall back to ambient globals or disk load unless bound:

```46:63:/Volumes/CrucialX/project-tailwind/src/parrhesia/api/flows.py
def _load_indexer_and_flights(
    *,
    indexer_path: Optional[Path] = None,
    flights_path: Optional[Path] = None,
) -> Tuple[TVTWIndexer, FlightList]:
    # 1) Try shared resources first
    g_idx, g_fl = get_global_resources()
    if g_idx is not None and g_fl is not None:
        return g_idx, g_fl  # type: ignore[return-value]
```

- DFRegulationPlan silently falls back to 30-minute bins; prevent this:

```679:689:/Volumes/CrucialX/project-tailwind/src/parrhesia/actions/regulations.py
inferred_minutes = time_bin_minutes
if inferred_minutes is None:
    try:
        inferred_minutes = int(diag.get("time_bin_minutes"))  # type: ignore[arg-type]
    except Exception:
        inferred_minutes = None
if inferred_minutes is None:
    inferred_minutes = 30  # Fallback if not found in diagnostics
```

### Notes

- Use `RuntimeError`/`ValueError` with clear context; avoid prints/warnings.
- Keep `fail_fast=True` by default; make it configurable via `RZConfig` if you need to relax in experiments.
- If proposals legitimately return empty, treat it as terminal (don’t raise) unless `fail_fast` is strictly enforced for debugging.

- Summary
  - Added a “Guardrails (fail fast)” section with concrete edits: stricter resource contexts, hotspot/flows/capacity validation, no-minute-fallback plan construction, evaluation and apply invariants, cache and reward checks, and structural clone independence.

### Minimal scaffolding (new code)
- `src/regulation_zero/types.py`
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

HotspotKey = str
ActionKey = Tuple[HotspotKey, int]  # (hotspot_key, proposal_rank)

@dataclass(frozen=True)
class RZAction:
    hotspot_key: HotspotKey
    proposal_rank: int
    delta_obj: float
    control_tv_id: Optional[str] = None
    window_bins: Optional[Tuple[int, int]] = None

RZPathKey = Tuple[ActionKey, ...]

@dataclass
class RZConfig:
    max_depth: int = 3
    puct_c: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_hotspots_per_node: int = 5
    k_proposals_per_hotspot: int = 8
    num_simulations: int = 200

    # Flow clustering and hotspot extraction params
    flows_threshold: Optional[float] = None  # defaults to compute_flows default (0.1) when None
    flows_resolution: Optional[float] = None  # defaults to compute_flows default (1.0) when None
    hotspot_threshold: float = 0.0  # excess threshold in extractor
    direction_mode: str = "coord_cosine"  # direction-aware reweighting mode
    fail_fast: bool = True
```
- `src/regulation_zero/env.py`
```python
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from contextlib import contextmanager
from server_tailwind.core.resources import AppResources
from parrhesia.api.resources import get_global_resources, set_global_resources
from parrhesia.api.flows import compute_flows
from parrhesia.flow_agent35.regen.hotspot_segment_extractor import (
    extract_hotspot_segments_from_resources,
    segment_to_hotspot_payload,
)
from parrhesia.flow_agent35.regen.engine import propose_regulations_for_hotspot
from parrhesia.flow_agent35.regen.types import RegenConfig
from parrhesia.actions.regulations import DFRegulationPlan
from parrhesia.actions.dfplan_evaluator import evaluate_df_regulation_plan
from project_tailwind.stateman.delay_assignment import DelayAssignmentTable
from project_tailwind.stateman.delta_view import DeltaOccupancyView
from parrhesia.optim.capacity import normalize_capacities
import server_tailwind.core.resources as core_res
from project_tailwind.stateman.flight_list_with_delta import FlightListWithDelta
import numpy as np


def clone_flight_list_structural(src: FlightListWithDelta) -> FlightListWithDelta:
    cls = src.__class__
    dst = cls.__new__(cls)

    # Basic identity and indexer wiring
    dst.occupancy_file_path = getattr(src, "occupancy_file_path", "")
    dst.tvtw_indexer_path = getattr(src, "tvtw_indexer_path", "")
    dst.tvtw_indexer = dict(getattr(src, "tvtw_indexer", {}) or {})
    dst.time_bin_minutes = int(getattr(src, "time_bin_minutes", 60))
    dst.tv_id_to_idx = dict(getattr(src, "tv_id_to_idx", {}) or {})
    dst.idx_to_tv_id = dict(getattr(src, "idx_to_tv_id", {}) or {})
    dst.num_traffic_volumes = int(getattr(src, "num_traffic_volumes", len(dst.tv_id_to_idx)))
    dst.num_time_bins_per_tv = int(getattr(src, "num_time_bins_per_tv", 1))
    dst._indexer = getattr(src, "_indexer", None)

    # Flight ids and shape
    dst.flight_ids = list(getattr(src, "flight_ids", ()))
    dst.num_flights = int(getattr(src, "num_flights", len(dst.flight_ids)))
    dst.flight_id_to_row = dict(getattr(src, "flight_id_to_row", {}) or {})
    dst.num_tvtws = int(getattr(src, "num_tvtws", 0))

    # Sparse matrices (copy arrays, no reload)
    occ_csr = getattr(src, "occupancy_matrix", None)
    dst.occupancy_matrix = occ_csr.copy() if occ_csr is not None else None
    occ_lil = getattr(src, "_occupancy_matrix_lil", None)
    dst._occupancy_matrix_lil = occ_lil.copy() if occ_lil is not None else None
    # Force a clean state to avoid mid-update propagation
    dst._lil_matrix_dirty = False
    buf = getattr(src, "_temp_occupancy_buffer", None)
    dst._temp_occupancy_buffer = np.array(buf, copy=True) if buf is not None else np.zeros(dst.num_tvtws, dtype=np.float32)

    # Small metadata (deep copy for isolation; much smaller than matrices)
    from copy import deepcopy as _dc
    dst.flight_data = _dc(getattr(src, "flight_data", {}) or {})
    dst.flight_metadata = _dc(getattr(src, "flight_metadata", {}) or {})

    # Caches
    dst._flight_tv_sequence_cache = {}

    # Delta bookkeeping (preserve current state)
    dst.applied_regulations = list(getattr(src, "applied_regulations", []) or [])
    dst.delay_histogram = dict(getattr(src, "delay_histogram", {}) or {})
    dst.total_delay_assigned_min = int(getattr(src, "total_delay_assigned_min", 0))
    dst.num_delayed_flights = int(getattr(src, "num_delayed_flights", 0))
    dst.num_regulations = int(getattr(src, "num_regulations", 0))
    agg = getattr(src, "_delta_aggregate", None)
    dst._delta_aggregate = np.array(agg, copy=True) if agg is not None else np.zeros(dst.num_tvtws, dtype=np.int64)
    dst._applied_views = []  # do not retain historical objects

    # Check for matrix sharing
    if (src.occupancy_matrix is not None and dst.occupancy_matrix is src.occupancy_matrix) or \
       (src._occupancy_matrix_lil is not None and dst._occupancy_matrix_lil is src._occupancy_matrix_lil):
        raise RuntimeError("Clone shares sparse matrix references with source")
    return dst

def _timebins_from_window(window_bins):
    """Converts a window [start, end_exclusive] to a list of integer time bins."""
    wb = [int(b) for b in (window_bins or [])]
    if not wb:
        return []
    if len(wb) == 1:
        return wb
    start = int(wb[0])
    end_exclusive = int(wb[1])
    if end_exclusive <= start:
        end_exclusive = start + 1
    return list(range(start, end_exclusive))


@contextmanager
def with_fl_resources(indexer, flight_list):
    prev = get_global_resources()
    set_global_resources(indexer, flight_list)
    gi, gf = get_global_resources()
    if gi is not indexer or gf is not flight_list:
        raise RuntimeError("Flows globals not bound to sandbox resources")
    try:
        yield
    finally:
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
    prev = core_res.get_resources()
    core_res._GLOBAL_RESOURCES = res
    if core_res.get_resources() is not res:
        raise RuntimeError("Core resources not set to sandbox")
    try:
        yield
    finally:
        core_res._GLOBAL_RESOURCES = prev
        if core_res.get_resources() is not prev:
            raise RuntimeError("Core resources not restored after eval")


class RZSandbox:
    def __init__(self, res: AppResources, cfg: Optional[RZConfig] = None):
        self._res = res.preload_all()
        self._cfg = cfg or RZConfig()
        T_idx = int(self._res.indexer.num_time_bins)
        T_cap = int(self._res.capacity_per_bin_matrix.shape[1])
        if T_idx != T_cap:
            raise RuntimeError(f"Time-axis mismatch: indexer bins={T_idx}, capacity width={T_cap}")

    @property
    def resources(self) -> AppResources:
        return self._res

    def fork(self) -> "RZSandbox":
        parent = self._res
        child = AppResources(parent.paths)
        # reuse heavy, read-only artifacts by reference
        child._indexer = parent.indexer
        child._traffic_volumes_gdf = parent.traffic_volumes_gdf
        child._hourly_capacity_by_tv = parent.hourly_capacity_by_tv
        child._capacity_per_bin_matrix = parent.capacity_per_bin_matrix
        child._travel_minutes = parent._travel_minutes
        child._tv_centroids = parent.tv_centroids
        # structural clone of the current flight_list (no deepcopy, no disk reload)
        child._flight_list = clone_flight_list_structural(parent.flight_list)
        return RZSandbox(child, cfg=self._cfg)

    def extract_hotspots(self, k: int) -> List[Dict[str, Any]]:
        segs = extract_hotspot_segments_from_resources(
            threshold=float(self._cfg.hotspot_threshold), resources=self._res
        )
        segs.sort(key=lambda s: float(s.get("max_excess", 0.0)), reverse=True)
        return segs[:k]

    def proposals_for_hotspot(self, hotspot_payload: Dict[str, Any], k: int):
        res = self._res
        # Validate hotspot payload
        if "control_volume_id" not in hotspot_payload:
            raise ValueError("hotspot_payload missing control_volume_id")
        window_bins = hotspot_payload.get("window_bins", [])
        timebins_h = _timebins_from_window(window_bins)
        if not timebins_h:
            raise RuntimeError("hotspot window resolves to no bins")
        control_tv = str(hotspot_payload["control_volume_id"])
        if control_tv not in res.flight_list.tv_id_to_idx:
            raise ValueError(f"Unknown control TV: {control_tv}")

        # Flows must be bound to THIS sandbox
        with with_fl_resources(res.indexer, res.flight_list):
            flows_payload = compute_flows(
                tvs=[control_tv],
                timebins=timebins_h,
                threshold=self._cfg.flows_threshold,
                resolution=self._cfg.flows_resolution,
                direction_opts={"mode": self._cfg.direction_mode, "tv_centroids": res.tv_centroids},
            )
        # Validate flows payload
        if flows_payload.get("tvs") != [control_tv]:
            raise RuntimeError("compute_flows returned unexpected TV set")
        tb = list(flows_payload.get("timebins") or [])
        if tb and tb != list(timebins_h):
            raise RuntimeError("compute_flows returned mismatched timebins")
        flows = flows_payload.get("flows") or []
        if self._cfg.fail_fast and len(flows) == 0:
            raise RuntimeError("No flows returned for a non-empty hotspot window")

        flow_to_flights = {
            str(int(flow["flow_id"])): [str(spec["flight_id"]) for spec in (flow.get("flights") or []) if spec.get("flight_id") is not None]
            for flow in flows
            if "flow_id" in flow
        }
        if self._cfg.fail_fast and not flow_to_flights:
            raise RuntimeError("Empty flow_to_flights after compute_flows")
        # Validate that all flights exist in this sandbox
        missing = [
            fid for fids in flow_to_flights.values() for fid in fids
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
                raise RuntimeError(f"Non-positive capacity detected after normalization for {tv}")

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
        # Predicted improvements must be finite
        for p in proposals:
            d = getattr(getattr(p, "predicted_improvement", None), "delta_objective_score", None)
            if d is None or not np.isfinite(float(d)):
                raise RuntimeError("Proposal has non-finite delta_objective_score")
        return proposals, flow_to_flights

    def apply_proposal(self, proposal, flights_by_flow: Dict[str, List[str]]) -> None:
        res = self._res
        tbm = int(res.indexer.time_bin_minutes)
        if tbm <= 0:
            raise RuntimeError("Invalid indexer.time_bin_minutes")
        # Ensure we never use DFRegulationPlan fallback minutes (30)
        plan = DFRegulationPlan.from_proposal(
            proposal,
            flights_by_flow=flights_by_flow,
            time_bin_minutes=tbm,
        )
        # Evaluate in sandbox-scoped core resources
        with with_core_resources(res):
            eval_res = evaluate_df_regulation_plan(
                plan,
                indexer_path=str(res.paths.tvtw_indexer_path),
                flights_path=str(res.paths.occupancy_file_path),
            )
        # Basic sanity on eval result
        delays = DelayAssignmentTable.from_dict(eval_res.delays_by_flight)
        if self._cfg.fail_fast and not delays.delays_by_flight:
            raise RuntimeError("Evaluation produced no delays; likely misbound resources or empty targets")

        pre_regs = int(res.flight_list.num_regulations)
        pre_total_delay = int(res.flight_list.total_delay_assigned_min)
        view = DeltaOccupancyView.from_delay_table(res.flight_list, delays, regulation_id="rz")
        res.flight_list.step_by_delay(view, finalize=True)
        if int(res.flight_list.num_regulations) <= pre_regs:
            raise RuntimeError("step_by_delay did not register a new regulation")
        if int(res.flight_list.total_delay_assigned_min) < pre_total_delay:
            raise RuntimeError("Total delay decreased after applying a regulation")
```
- `src/regulation_zero/cache.py`
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from .types import RZPathKey, ActionKey

@dataclass
class ChildStats:
    P: float
    N: int = 0
    W: float = 0.0  # total value
    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

@dataclass
class NodeStats:
    children: Dict[ActionKey, ChildStats] = field(default_factory=dict)
    expanded: bool = False

TranspositionTable = Dict[RZPathKey, NodeStats]
ProposalsCache = Dict[Tuple[RZPathKey, str], List[Any]]  # (state, hotspot_key) -> proposals
```
- `src/regulation_zero/mcts.py` (outline)
```python
from __future__ import annotations
import math, random
from typing import List, Tuple
import numpy as np
from .types import RZConfig, RZPathKey, RZAction, ActionKey
from .cache import NodeStats, ChildStats, TranspositionTable, ProposalsCache
from parrhesia.flow_agent35.regen.hotspot_segment_extractor import segment_to_hotspot_payload

def softmax(xs: List[float], tau: float = 1.0) -> List[float]:
    arr = np.asarray(xs, dtype=np.float64) / max(tau, 1e-6)
    arr -= arr.max() if arr.size else 0.0
    ex = np.exp(arr)
    Z = ex.sum()
    return [(x / Z) if Z > 0 else 1.0 / len(xs) for x in ex]

class MCTS:
    def __init__(self, env_factory, cfg: RZConfig):
        self.env_factory = env_factory  # Should provide a fresh RZSandbox fork
        self.cfg = cfg
        self.tt: TranspositionTable = {}
        self.cache: ProposalsCache = {}

    def run(self) -> List[RZAction]:
        for _ in range(self.cfg.num_simulations):
            # Fork once per simulation, then mutate the sandbox down the path.
            env = self.env_factory()
            self._simulate((), env)

        # Derive final plan by greedy descent on visits
        actions: List[RZAction] = []
        state: RZPathKey = ()
        depth = 0
        while depth < self.cfg.max_depth and state in self.tt and self.tt[state].children:
            child = max(self.tt[state].children.items(), key=lambda kv: kv[1].N)
            ak = child[0]
            # store stub RZAction; delta filled later when applied
            actions.append(RZAction(hotspot_key=ak[0], proposal_rank=ak[1], delta_obj=0.0))
            state = tuple(list(state) + [ak])  # descend
            depth += 1
        return actions

    def _simulate(self, state: RZPathKey, env: RZSandbox) -> float:
        depth = len(state)
        if depth >= self.cfg.max_depth:
            return 0.0

        node = self.tt.setdefault(state, NodeStats())
        if not node.expanded:
            # Expand: use the sandbox, which is already at the correct state for this node.
            segments = env.extract_hotspots(self.cfg.max_hotspots_per_node)
            actions: List[Tuple[ActionKey, float]] = []
            for seg in segments:
                hk = f'{seg["traffic_volume_id"]}:{int(seg["start_bin"])}-{int(seg["end_bin"])}'
                hot_payload = segment_to_hotspot_payload(seg)  # import where defined
                key = (state, hk)
                if key not in self.cache:
                    proposals, f2f = env.proposals_for_hotspot(hot_payload, self.cfg.k_proposals_per_hotspot)
                    # keep tuple of (proposal, f2f, delta_obj)
                    ranked = [(p, f2f, float(p.predicted_improvement.delta_objective_score)) for p in proposals]
                    self.cache[key] = ranked
                ranked = self.cache[key]
                for rank, (_, _, delta) in enumerate(ranked[: self.cfg.k_proposals_per_hotspot]):
                    if not np.isfinite(float(delta)):
                        raise RuntimeError("Non-finite delta_objective_score in proposals cache")
                    actions.append(((hk, rank), float(delta)))
            if not actions:
                node.expanded = True
                return 0.0
            priors = softmax([d for _, d in actions])
            for (ak, _), p in zip(actions, priors):
                node.children.setdefault(ak, ChildStats(P=float(p)))
            node.expanded = True

            # Apply Dirichlet noise at root
            if depth == 0 and node.children:
                eps = self.cfg.dirichlet_epsilon
                alpha = self.cfg.dirichlet_alpha
                noise = np.random.default_rng().dirichlet([alpha] * len(node.children))
                for (cs, n) in zip(node.children.values(), noise):
                    cs.P = float((1 - eps) * cs.P + eps * n)

        # Selection
        total_N = sum(cs.N for cs in self.tt[state].children.values()) + 1
        best_ak, best_cs = max(
            self.tt[state].children.items(),
            key=lambda kv: kv[1].Q + self.cfg.puct_c * kv[1].P * math.sqrt(total_N) / (1 + kv[1].N),
        )
        # Recurse and get reward
        r = self._roll(state, best_ak, env)
        # Backup
        best_cs.N += 1
        best_cs.W += r
        return r

    def _roll(self, state: RZPathKey, ak: ActionKey, env: RZSandbox) -> float:
        # Immediate reward for this edge from cached proposal
        hk, rank = ak
        key = (state, hk)
        if key not in self.cache:
            raise RuntimeError("Cache miss for selected action; expansion logic broken")
        ranked = self.cache[key]
        if rank < 0 or rank >= len(ranked):
            raise IndexError("Selected proposal rank out of range")
        prop, f2f, delta = ranked[rank]
        if not np.isfinite(float(delta)):
            raise RuntimeError("Selected action has non-finite immediate reward")
        # Apply action to the sandbox, mutating it for the child state.
        env.apply_proposal(prop, f2f)
        # Next state
        child_state: RZPathKey = tuple(list(state) + [ak])
        v = self._simulate(child_state, env)
        return float(delta) + v
```
- `src/regulation_zero/runner.py` (glue)
```python
from __future__ import annotations
from server_tailwind.core.resources import AppResources
from regulation_zero.mcts import MCTS
from regulation_zero.types import RZConfig
from regulation_zero.env import RZSandbox


def make_env_factory(root_res: AppResources, cfg: RZConfig):
    root = RZSandbox(root_res.preload_all(), cfg=cfg)

    def factory():
        return root.fork()

    return factory


def run_search():
    cfg = RZConfig()
    # Local, process-isolated resources for the CLI
    res = AppResources().preload_all()
    mcts = MCTS(env_factory=make_env_factory(res, cfg), cfg=cfg)
    actions = mcts.run()
    return actions


### Notes and assumptions
- We use a "fork once per simulation" strategy. A sandbox is forked from the root at the start of each simulation and then mutated along the search path. This contains memory usage to one `flight_list` copy per simulation, avoiding the high cost of forking at each node.
- Hotspots are re-extracted and flows recomputed at each node expansion from the current sandbox state; flows use `with_fl_resources(...)` so they cannot accidentally bind to a stale `flight_list`.
- `compute_flows` must see the node’s sandbox flight list: we call `set_global_resources(indexer, flight_list)` before it.
- We use predicted improvement `delta_objective_score` from regen as immediate rewards and as priors (via softmax); leaf value is 0 per “no rollout”.
- Transpositions are keyed by canonical solution path, not structural state equivalence. That’s sufficient to cache `regen` results and node stats for our search.
- Optimizations (optional): instead of forking from the baseline for each node, replay actions on a fork to avoid recomputing all previous steps; or add an undo stack using stored delay tables.
- Structural clone mirrors the evaluator’s baseline cloning semantics for `FlightList` (copy sparse CSR/LIL and small dicts; no disk reload). This keeps forks isolated without `deepcopy`, while preserving current post-move state.

- Confirmed behaviors used:
  - Hotspot extraction and ranking via resources.
  - Proposal generation on explicit `indexer/flight_list` without relying on global singletons.
  - State stepping via `DeltaOccupancyView` and `step_by_delay`, then `refresh_after_state_update`.

- Integration smoke test: replicate the flow in `examples/regen/regen_second_order.py` within one MCTS playout to validate stepping.

- Final solution materialization: after MCTS decides a sequence, rebuild the actual `DFRegulationPlan` objects by querying the cached `(proposal, flights_by_flow)` per selected `(state, hotspot_key, rank)` and apply on a fresh sandbox or emit payloads.

- Safety: handle “no proposals” gracefully; cap hotspots per node; thread-safety not required for single-process runs.

- Tuning defaults: `max_depth=3`, `k_proposals_per_hotspot=8`, `max_hotspots_per_node=5`, `num_simulations=200` as a starting point.

Summary
- Designed `src/regulation_zero/` with `types`, `env`, `cache`, `mcts`, and `runner`.
- Integrated existing APIs: hotspot extraction, regen proposals, plan evaluation, and state stepping.
- Defined canonical solution keys, proposal caching by `(state, hotspot)`, and root Dirichlet noise.
- Provided code outlines to implement per-node sandboxes and the PUCT search without rollouts.