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
- Dirichlet noise at root:
  - Injected into the root node’s child priors only, AlphaZero-style.
- Terminal and return:
  - Terminal when `remaining_depth == 0` or no hotspots/proposals.
  - Return is sum of per-edge `delta_objective_score` along the chosen solution path.
- No rollout:
  - Implemented by backing up zero leaf value; the value function is purely the accumulated immediate rewards gathered along traversal.
- Proposal caching and IDs:
  - For each node state, proposals for a hotspot are identified by their rank within that node’s regen call and the hotspot key. We store `proposal_rank`, `delta_obj`, and the serialized `proposal` to reconstruct the plan on apply.

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
```
- `src/regulation_zero/env.py`
```python
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from server_tailwind.core.resources import AppResources, get_resources
from server_tailwind.core.cache_refresh import refresh_after_state_update
from parrhesia.api.resources import set_global_resources
from parrhesia.api.flows import compute_flows
from parrhesia.flow_agent35.regen.hotspot_segment_extractor import (
    extract_hotspot_segments_from_resources,
    segment_to_hotspot_payload,
)
from parrhesia.flow_agent35.regen.engine import propose_regulations_for_hotspot
from parrhesia.flow_agent35.regen.types import RegenConfig
from parrhesia.flow_agent35.regen.exceedance import compute_hotspot_exceedance
from parrhesia.actions.regulations import DFRegulationPlan
from parrhesia.actions.dfplan_evaluator import evaluate_df_regulation_plan
from project_tailwind.stateman.delay_assignment import DelayAssignmentTable
from project_tailwind.stateman.delta_view import DeltaOccupancyView

class RZSandbox:
    def __init__(self, res: AppResources):
        self._res = res.preload_all()

    @property
    def resources(self) -> AppResources:
        return self._res

    def fork(self) -> "RZSandbox":
        # clone FlightListWithDelta (deep-copy) and reuse other cached artifacts
        parent = self._res
        child = AppResources(parent.paths)
        # Attach parent's already-built caches
        child._indexer = parent.indexer
        child._traffic_volumes_gdf = parent.traffic_volumes_gdf
        child._hourly_capacity_by_tv = parent.hourly_capacity_by_tv
        child._capacity_per_bin_matrix = parent.capacity_per_bin_matrix
        child._travel_minutes = parent._travel_minutes
        child._tv_centroids = parent.tv_centroids
        # Deep-copy the flight list only
        from copy import deepcopy
        fl = parent.flight_list
        fl_copy = deepcopy(fl)
        child._flight_list = fl_copy
        return RZSandbox(child)

    def extract_hotspots(self, k: int) -> List[Dict[str, Any]]:
        segs = extract_hotspot_segments_from_resources(resources=self._res)
        segs.sort(key=lambda s: float(s.get("max_excess", 0.0)), reverse=True)
        return segs[:k]

    def proposals_for_hotspot(self, hotspot_payload: Dict[str, Any], k: int):
        res = self._res
        set_global_resources(res.indexer, res.flight_list)
        timebins_h = list(hotspot_payload.get("window_bins", []))
        control_tv = str(hotspot_payload["control_volume_id"])
        flows_payload = compute_flows(
            tvs=[control_tv],
            timebins=timebins_h,
            direction_opts={"mode": "coord_cosine", "tv_centroids": res.tv_centroids},
        )
        flow_to_flights = {
            str(int(flow["flow_id"])): [str(spec["flight_id"]) for spec in (flow.get("flights") or []) if spec.get("flight_id") is not None]
            for flow in (flows_payload.get("flows") or [])
            if "flow_id" in flow
        }
        caps = res.capacity_per_bin_matrix
        # normalize and align capacities as in examples if needed
        proposals = propose_regulations_for_hotspot(
            indexer=res.indexer,
            flight_list=res.flight_list,
            capacities_by_tv={str(tv): caps[int(row), :] for tv, row in res.flight_list.tv_id_to_idx.items()},
            travel_minutes_map=res.travel_minutes(),
            hotspot_tv=control_tv,
            timebins_h=timebins_h,
            flows_payload=flows_payload,
            flow_to_flights=flow_to_flights,
            config=RegenConfig(k_proposals=int(k)),
        )
        return proposals, flow_to_flights

    def apply_proposal(self, proposal, flights_by_flow: Dict[str, List[str]]) -> None:
        res = self._res
        plan = DFRegulationPlan.from_proposal(
            proposal,
            flights_by_flow=flights_by_flow,
            time_bin_minutes=int(res.indexer.time_bin_minutes),
        )
        eval_res = evaluate_df_regulation_plan(
            plan,
            indexer_path=str(res.paths.tvtw_indexer_path),
            flights_path=str(res.paths.occupancy_file_path),
        )
        delays = DelayAssignmentTable.from_dict(eval_res.delays_by_flight)
        view = DeltaOccupancyView.from_delay_table(res.flight_list, delays, regulation_id="rz")
        res.flight_list.step_by_delay(view)
        refresh_after_state_update(res)
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
                    actions.append(((hk, rank), delta))
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
        ranked = self.cache[key]
        prop, f2f, delta = ranked[rank]
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

def make_env_factory(root_res: AppResources):
    # Keep a stabilized baseline to fork from.
    root = RZSandbox(root_res.preload_all())
    def factory():
        # The factory provides a fresh fork of the root state.
        return root.fork()
    return factory

def run_search():
    cfg = RZConfig()
    res = AppResources().preload_all()
    mcts = MCTS(env_factory=make_env_factory(res), cfg=cfg)
    actions = mcts.run()
    return actions
```

### Notes and assumptions
- We use a "fork once per simulation" strategy. A sandbox is forked from the root at the start of each simulation and then mutated along the search path. This contains memory usage to one `flight_list` copy per simulation, avoiding the high cost of forking at each node.
- `compute_flows` must see the node’s sandbox flight list: we call `set_global_resources(indexer, flight_list)` before it.
- We use predicted improvement `delta_objective_score` from regen as immediate rewards and as priors (via softmax); leaf value is 0 per “no rollout”.
- Transpositions are keyed by canonical solution path, not structural state equivalence. That’s sufficient to cache `regen` results and node stats for our search.
- Optimizations (optional): instead of forking from the baseline for each node, replay actions on a fork to avoid recomputing all previous steps; or add an undo stack using stored delay tables.

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