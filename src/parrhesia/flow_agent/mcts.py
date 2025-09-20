from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .actions import (
    Action,
    AddFlow,
    Back,
    CommitRegulation,
    Continue,
    NewRegulation,
    PickHotspot,
    RemoveFlow,
    Stop,
)
from .state import PlanState
from .transition import CheapTransition
from .rate_finder import RateFinder


# ------------------------------- Config types -------------------------------


@dataclass
class MCTSConfig:
    # PUCT + widening
    c_puct: float = 2.0
    alpha: float = 0.7
    k0: int = 4
    k1: float = 1.0
    widen_batch_size: int = 2
    # Budgets
    commit_depth: int = 1
    max_sims: int = 24
    max_time_s: float = 20.0
    commit_eval_limit: int = 3
    # Priors
    priors_temperature: float = 1.0
    # Shaping
    phi_scale: float = 1.0
    # RNG
    seed: int = 0
    # Debug prints
    debug_prints: bool = False


@dataclass
class EdgeStats:
    N: int = 0
    W: float = 0.0
    Q: float = 0.0


@dataclass
class TreeNode:
    key: str
    P: Dict[Tuple, float] = field(default_factory=dict)  # action_signature -> prior
    edges: Dict[Tuple, EdgeStats] = field(default_factory=dict)
    children: Dict[Tuple, str] = field(default_factory=dict)
    widened: int = 0
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    phi: float = 0.0  # cached potential


# ------------------------------- Core search --------------------------------


class MCTS:
    """
    Lightweight MCTS with progressive widening and potential shaping.

    Assumptions
    - State mutations happen only through `CheapTransition.step`, which returns a copy.
    - Commit evaluation integrates RateFinder once per commit action and caches results.
    - Flow proxies for shaping are provided via `HotspotContext.metadata['flow_proxies']`.
    - Flight grouping per flow (for commits) is provided via `HotspotContext.metadata['flow_to_flights']`.
    """

    def __init__(
        self,
        *,
        transition: CheapTransition,
        rate_finder: RateFinder,
        config: Optional[MCTSConfig] = None,
        rng: Optional[np.random.Generator] = None,
        timer: Optional[Callable[[str], ContextManager[Any]]] = None,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.transition = transition
        self.rate_finder = rate_finder
        self.cfg = config or MCTSConfig()
        self.rng = rng or np.random.default_rng(int(self.cfg.seed))
        self.nodes: Dict[str, TreeNode] = {}
        self._commit_eval_cache: Dict[Tuple, Tuple[Dict[str, float] | int, float, Dict[str, Any]]] = {}
        self._commit_calls = 0
        self._best_commit: Optional[Tuple[CommitRegulation, float]] = None  # (action, deltaJ)
        self._timer_factory = timer
        self._action_counts: Dict[str, int] = {}
        self._progress_cb = progress_cb
        self._last_delta_j: Optional[float] = None

    def _timed(self, name: str) -> ContextManager[Any]:
        if self._timer_factory is None:
            return nullcontext()
        return self._timer_factory(name)

    # ------------------------------- Debug helpers -------------------------
    def _dbg(self, message: str) -> None:
        if getattr(self.cfg, "debug_prints", False):
            try:
                print(message)
            except Exception:
                pass

    @staticmethod
    def _sig_to_label(sig: Tuple) -> str:
        try:
            if not isinstance(sig, (list, tuple)) or not sig:
                return str(sig)
            kind = sig[0]
            if kind == "add":
                return f"add({sig[1]})"
            if kind == "rem":
                return f"rem({sig[1]})"
            if kind == "cont":
                return "cont"
            if kind == "back":
                return "back"
            if kind == "commit":
                return f"commit({max(0, len(sig)-1)})"
            if kind == "new_reg":
                return "new_reg"
            if kind == "hotspot":
                return f"hotspot({sig[1]},{sig[2]}-{sig[3]})"
            if kind == "stop":
                return "stop"
            return str(sig)
        except Exception:
            return str(sig)

    @staticmethod
    def _state_brief(state: PlanState) -> str:
        try:
            n_regs = len(getattr(state, "plan", []) or [])
            stage = getattr(state, "stage", "?")
            ctx = getattr(state, "hotspot_context", None)
            tv = getattr(ctx, "control_volume_id", None) if ctx is not None else None
            wb = getattr(ctx, "window_bins", None) if ctx is not None else None
            n_flows = len(getattr(ctx, "selected_flow_ids", []) or []) if ctx is not None else 0
            tv_s = str(tv) if tv is not None else "—"
            wb_s = f"{int(wb[0])}-{int(wb[1])}" if isinstance(wb, (tuple, list)) and len(wb) == 2 else "—"
            return f"plan_regs={n_regs} stage={stage} tv={tv_s} win={wb_s} sel_flows={n_flows}"
        except Exception:
            return "plan_regs=? stage=?"

    @property
    def action_counts(self) -> Dict[str, int]:
        return dict(self._action_counts)

    def _record_action(self, action: Action) -> None:
        key = type(action).__name__
        self._action_counts[key] = self._action_counts.get(key, 0) + 1

    # ------------------------------- Public API ---------------------------
    def run(self, root: PlanState, *, max_sims: Optional[int] = None, commit_depth: Optional[int] = None) -> CommitRegulation:
        sims = int(max_sims if max_sims is not None else self.cfg.max_sims)
        depth_limit = int(commit_depth if commit_depth is not None else self.cfg.commit_depth)
        self._commit_calls = 0
        self._best_commit = None
        self._action_counts.clear()
        self._last_delta_j = None

        t_start = time.perf_counter()
        t_end = t_start + float(self.cfg.max_time_s)

        self._dbg(f"[MCTS] run start: sims={sims} depth={depth_limit} {self._state_brief(root)}")

        for i in range(sims):
            now = time.perf_counter()
            if now > t_end:
                break
            self._dbg(f"[MCTS] simulate[{i+1}/{sims}] start nodes={len(self.nodes)} {self._state_brief(root)}")
            last_ret = self._simulate(root, depth_limit)
            self._dbg(f"[MCTS] simulate[{i+1}/{sims}] end   return={float(last_ret):.3f} best_dJ={(self._best_commit[1] if self._best_commit is not None else None)} nodes={len(self.nodes)}")
            # Progress callback (best-effort, non-fatal)
            if self._progress_cb is not None:
                try:
                    root_key = root.canonical_key()
                    root_node = self.nodes.get(root_key)
                    root_children = 0
                    root_top: List[Tuple] = []
                    if root_node is not None:
                        root_children = len(root_node.children)
                        tmp: List[Tuple] = []
                        for sig, est in root_node.edges.items():
                            p = float(root_node.P.get(sig, 0.0))
                            tmp.append((sig, int(est.N), float(est.Q), p))
                        tmp.sort(key=lambda x: (-x[1], -x[2]))
                        root_top = tmp[:5]
                    payload = {
                        "sims_done": i + 1,
                        "sims_total": sims,
                        "elapsed_s": now - t_start,
                        "eta_s": max(0.0, t_end - now),
                        "nodes": len(self.nodes),
                        "root_visits": (root_node.N if root_node is not None else 0),
                        "root_children": root_children,
                        "root_top": root_top,
                        "commit_evals": self._commit_calls,
                        "best_delta_j": (self._best_commit[1] if self._best_commit is not None else None),
                        "last_delta_j": self._last_delta_j,
                        "last_return": float(last_ret),
                        "action_counts": dict(self._action_counts),
                    }
                    self._progress_cb(payload)
                except Exception:
                    pass

        if self._best_commit is None:
            raise RuntimeError("MCTS did not evaluate any commit; increase sims or adjust state")
        return self._best_commit[0]

    # ----------------------------- Simulation loop ------------------------
    def _simulate(self, root: PlanState, commit_depth: int) -> float:
        state = root
        path: List[Tuple[str, Tuple]] = []  # (node_key, action_signature)
        commits_used = 0
        total_return = 0.0

        while True:
            key = state.canonical_key()
            node = self.nodes.get(key)
            if node is None:
                node = self._create_node(state)
                self._dbg(f"[MCTS/expand] create_leaf node={key[:24]}… phi={float(node.phi):.3f} {self._state_brief(state)}")
                # Leaf bootstrap: return -phi
                v = -node.phi
                self._dbg(f"[MCTS/simulate] leaf_bootstrap return={float(v):.3f} {self._state_brief(state)} path_len={len(path)}")
                self._backup(path, v)
                return v

            # Expand/widen
            candidates = self._enumerate_actions(state)
            priors = self._compute_priors(state, candidates)
            # Store priors deterministically
            for sig, p in priors.items():
                if sig not in node.P:
                    node.P[sig] = float(p)
            m_allow = int(self.cfg.k0 + self.cfg.k1 * (node.N ** self.cfg.alpha))
            m_allow = max(1, m_allow)
            self._dbg(f"[MCTS/expand] node={key[:24]}… allow={m_allow} priors={len(priors)} children_before={len(node.children)} {self._state_brief(state)}")
            while len(node.children) < m_allow and len(node.children) < len(priors):
                # Expand top-ranked by prior not already expanded
                for sig, _ in sorted(priors.items(), key=lambda kv: (-kv[1], kv[0])):
                    if sig in node.children:
                        continue
                    node.children[sig] = "?"  # placeholder child key; will be set after step
                    node.edges[sig] = node.edges.get(sig, EdgeStats())
                    self._dbg(f"[MCTS/expand]   + child {self._sig_to_label(sig)} P={float(priors.get(sig, 0.0)):.3f} children_now={len(node.children)}")
                    break
                else:
                    break

            # Selection: PUCT over expanded children only
            if not node.children:
                # No viable moves -> terminal; bootstrap
                v = -node.phi
                self._dbg(f"[MCTS/select] no_children -> terminal bootstrap={float(v):.3f} {self._state_brief(state)}")
                self._backup(path, v)
                return v

            # Filter by commit quota
            viable: List[Tuple[Tuple, EdgeStats]] = []
            for sig, est in node.edges.items():
                kind = sig[0]
                if kind == "commit" and commits_used >= commit_depth:
                    continue
                viable.append((sig, est))
            if not viable:
                v = -node.phi
                self._dbg(f"[MCTS/select] no_viable_moves -> bootstrap={float(v):.3f} commits_used={commits_used}/{commit_depth} {self._state_brief(state)}")
                self._backup(path, v)
                return v

            sqrtN = math.sqrt(max(1, node.N))
            best_sig = None
            best_score = -1e30
            for sig, est in viable:
                P = node.P.get(sig, 0.0)
                U = est.Q + float(self.cfg.c_puct) * P * (sqrtN / (1.0 + est.N))
                # Deterministic tie-breaker using signature ordering
                if U > best_score or (U == best_score and (best_sig is None or sig < best_sig)):
                    best_score = U
                    best_sig = sig
            assert best_sig is not None
            est_sel = node.edges.get(best_sig, EdgeStats())
            self._dbg(
                f"[MCTS/select] pick {self._sig_to_label(best_sig)} U={float(best_score):.3f} Q={float(est_sel.Q):.3f} P={float(node.P.get(best_sig, 0.0)):.3f} N_edge={int(est_sel.N)} node_N={int(node.N)} children={len(node.children)} {self._state_brief(state)}"
            )

            # Materialize action from signature and step
            action = self._action_from_signature(state, best_sig)

            # Commit requires evaluation first, to embed committed_rates
            r_base = 0.0
            if isinstance(action, CommitRegulation):
                if commits_used >= commit_depth:
                    # Treat as terminal; bootstrap
                    v = -node.phi
                    self._dbg(f"[MCTS/simulate] commit_blocked budget commits_used={commits_used}/{commit_depth} -> bootstrap={float(v):.3f}")
                    self._backup(path, v)
                    return v
                self._dbg(
                    f"[MCTS/simulate] evaluate_commit flows={len(getattr(getattr(state, 'hotspot_context', None), 'selected_flow_ids', []) or [])} {self._state_brief(state)}"
                )
                commit_action, delta_j = self._evaluate_commit(state)
                self._dbg(f"[MCTS/simulate] evaluate_commit_done ΔJ={float(delta_j):.3f} calls={int(self._commit_calls)}")
                action = commit_action
                r_base = -float(delta_j)
                # Track best commit observed across sims (at root or deeper)
                if self._best_commit is None or delta_j < self._best_commit[1]:
                    self._best_commit = (commit_action, float(delta_j))

            self._record_action(action)

            next_state, is_commit, is_terminal = self.transition.step(state, action)
            # Set child key now that next_state exists
            child_key = next_state.canonical_key()
            node.children[best_sig] = child_key

            # Shaped reward: Δphi + base
            phi_s = node.phi
            phi_sp = self._phi(next_state)
            r_shaped = r_base + (phi_sp - phi_s)
            total_return += r_shaped
            self._dbg(
                f"[MCTS/step] {type(action).__name__} commit={bool(is_commit)} term={bool(is_terminal)} r_base={float(r_base):.3f} Δphi={(phi_sp - phi_s):.3f} r={float(r_shaped):.3f} G={float(total_return):.3f} {self._state_brief(next_state)}"
            )

            # Push to path and continue
            path.append((key, best_sig))
            state = next_state
            if is_commit:
                commits_used += 1
            if is_terminal or commits_used >= commit_depth:
                # Terminal or used up commit budget -> stop and bootstrap at leaf
                leaf_v = -self._phi(state)
                total_return += leaf_v
                self._dbg(
                    f"[MCTS/simulate] terminal_or_budget commits_used={commits_used}/{commit_depth} leaf_bootstrap={float(leaf_v):.3f} return={float(total_return):.3f} path_len={len(path)}"
                )
                self._backup(path, total_return)
                return total_return

    # ----------------------------- Node helpers ----------------------------
    def _create_node(self, state: PlanState) -> TreeNode:
        key = state.canonical_key()
        node = TreeNode(key=key)
        node.phi = self._phi(state)
        self.nodes[key] = node
        return node

    def _phi(self, state: PlanState) -> float:
        z = state.z_hat
        if z is None:
            return 0.0
        arr = np.asarray(z, dtype=float)
        # Use only positive overload and scale
        val = -float(np.dot(np.maximum(arr, 0.0), np.maximum(arr, 0.0)))
        return self.cfg.phi_scale * val

    # ------------------------------ Priors ---------------------------------
    def _enumerate_actions(self, state: PlanState) -> List[Action]:
        actions: List[Action] = []
        ctx = state.hotspot_context

        if state.stage == "idle":
            # Start a new regulation or stop
            actions.append(NewRegulation())
            actions.append(Stop())
            return actions

        if state.stage == "select_hotspot":
            # Enumerate candidate hotspots from inventory metadata
            cands = state.metadata.get("hotspot_candidates") if isinstance(state.metadata, dict) else None
            if isinstance(cands, list):
                for item in cands:
                    try:
                        tv = str(item.get("control_volume_id"))
                        wb = item.get("window_bins") or []
                        t0, t1 = int(wb[0]), int(wb[1])
                    except Exception:
                        continue
                    actions.append(PickHotspot(
                        control_volume_id=tv,
                        window_bins=(t0, t1),
                        candidate_flow_ids=tuple(str(x) for x in (item.get("candidate_flow_ids") or [])),
                        mode=str(item.get("mode", "per_flow")),
                        metadata=dict(item.get("metadata", {})),
                    ))
            actions.append(Stop())
            return actions

        if state.stage == "select_flows" and ctx is not None:
            selected = set(ctx.selected_flow_ids)
            for fid in ctx.candidate_flow_ids:
                fid_s = str(fid)
                if fid_s not in selected:
                    actions.append(AddFlow(fid_s))
            for fid in ctx.selected_flow_ids:
                actions.append(RemoveFlow(str(fid)))
            if ctx.selected_flow_ids:
                actions.append(Continue())
            actions.append(Stop())
            return actions

        if state.stage == "confirm":
            actions.append(CommitRegulation())
            actions.append(Back())
            actions.append(Stop())
            return actions

        # Fallback terminal when not in an expected stage
        actions.append(Stop())
        return actions

    def _compute_priors(self, state: PlanState, candidates: Sequence[Action]) -> Dict[Tuple, float]:
        # Feature: sum of proxy histogram for AddFlow
        ctx = state.hotspot_context
        proxies: Mapping[str, Sequence[float]] = {}
        if ctx is not None and isinstance(ctx.metadata, Mapping):
            proxies = ctx.metadata.get("flow_proxies", {}) or {}

        logits: Dict[Tuple, float] = {}
        for a in candidates:
            sig = self._signature_for_action(state, a)
            if isinstance(a, AddFlow):
                proxy = np.asarray(proxies.get(a.flow_id, []), dtype=float)
                logits[sig] = float(proxy.sum()) if proxy.size else 1.0
            elif isinstance(a, PickHotspot):
                # Read prior from candidate metadata when available
                prior = None
                if isinstance(state.metadata, Mapping):
                    for item in (state.metadata.get("hotspot_candidates") or []):
                        try:
                            tv = str(item.get("control_volume_id"))
                            wb = item.get("window_bins") or []
                            t0, t1 = int(wb[0]), int(wb[1])
                        except Exception:
                            continue
                        if tv == a.control_volume_id and (t0, t1) == tuple(int(b) for b in a.window_bins):
                            try:
                                prior = float(item.get("hotspot_prior", 1.0))
                            except Exception:
                                prior = 1.0
                            break
                logits[sig] = float(prior if prior is not None else 1.0)
            elif isinstance(a, RemoveFlow):
                logits[sig] = 0.5  # mildly discouraged initially
            elif isinstance(a, Continue):
                logits[sig] = 1.0
            elif isinstance(a, CommitRegulation):
                logits[sig] = 0.5
            elif isinstance(a, Back):
                logits[sig] = 0.25
            elif isinstance(a, Stop):
                logits[sig] = 0.1
            else:
                logits[sig] = 1.0

        # Softmax with temperature
        T = max(1e-6, float(self.cfg.priors_temperature))
        max_logit = max(logits.values()) if logits else 0.0
        exps: Dict[Tuple, float] = {}
        for k, v in logits.items():
            exps[k] = math.exp((v - max_logit) / T)
        Z = sum(exps.values()) or 1.0
        priors = {k: (v / Z) for k, v in exps.items()}
        return priors

    # ------------------------------ Commits --------------------------------
    def _evaluate_commit(self, state: PlanState) -> Tuple[CommitRegulation, float]:
        ctx = state.hotspot_context
        if ctx is None:
            raise RuntimeError("Commit attempted without hotspot context")
        if not ctx.selected_flow_ids:
            raise RuntimeError("Commit attempted without any selected flows")

        # Build flows map from metadata
        meta = ctx.metadata or {}
        flow_to_flights: Mapping[str, Sequence[str]] = meta.get("flow_to_flights", {}) or {}
        flows: Dict[str, Sequence[str]] = {
            str(fid): tuple(str(x) for x in flow_to_flights.get(str(fid), ())) for fid in ctx.selected_flow_ids
        }

        # Key for caching evaluator results
        base_key = (
            state.canonical_key(),
            str(ctx.control_volume_id),
            tuple(int(b) for b in ctx.window_bins),
            tuple(sorted(flows.keys())),
        )
        cached = self._commit_eval_cache.get(base_key)
        if cached is not None:
            rates, delta_j, info = cached
        else:
            if self._commit_calls >= int(self.cfg.commit_eval_limit):
                # Treat as no-op commit with zero improvement to avoid extra cost
                rates, delta_j, info = ({}, 0.0, {"reason": "eval_budget_exhausted"})
            else:
                with self._timed("mcts.rate_finder.find_rates"):
                    rates, delta_j, info = self.rate_finder.find_rates(
                        plan_state=state,
                        control_volume_id=str(ctx.control_volume_id),
                        window_bins=tuple(int(b) for b in ctx.window_bins),
                        flows=flows,
                        mode="per_flow" if ctx.mode == "per_flow" else "blanket",
                    )
                self._commit_calls += 1
                self._last_delta_j = float(delta_j)
                self._commit_eval_cache[base_key] = (rates, delta_j, info)

        # Sanitize committed rates for serialization and canonicalization
        sanitized: Dict[str, int] | int | None
        if isinstance(rates, dict):
            cleaned: Dict[str, int] = {}
            for k, v in rates.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isinf(fv) or math.isnan(fv):
                    # Skip infinite/NaN rates (equivalent to no regulation on that flow)
                    continue
                cleaned[str(k)] = int(round(fv))
            sanitized = cleaned
        elif isinstance(rates, (int, float)):
            fv = float(rates)
            sanitized = int(round(fv)) if (not math.isinf(fv) and not math.isnan(fv)) else 0
        else:
            sanitized = {}

        commit_action = CommitRegulation(committed_rates=sanitized, diagnostics={"rate_finder": info})
        return commit_action, float(delta_j)

    # --------------------------- Signatures/decoding -----------------------
    def _signature_for_action(self, state: PlanState, action: Action) -> Tuple:
        if isinstance(action, AddFlow):
            return ("add", str(action.flow_id))
        if isinstance(action, RemoveFlow):
            return ("rem", str(action.flow_id))
        if isinstance(action, Continue):
            return ("cont",)
        if isinstance(action, Back):
            return ("back",)
        if isinstance(action, CommitRegulation):
            ctx = state.hotspot_context
            chosen = tuple(sorted(ctx.selected_flow_ids)) if ctx is not None else tuple()
            return ("commit",) + chosen
        if isinstance(action, NewRegulation):
            return ("new_reg",)
        if isinstance(action, PickHotspot):
            t0, t1 = int(action.window_bins[0]), int(action.window_bins[1])
            return ("hotspot", str(action.control_volume_id), t0, t1)
        if isinstance(action, Stop):
            return ("stop",)
        return (type(action).__name__,)

    def _action_from_signature(self, state: PlanState, sig: Tuple) -> Action:
        kind = sig[0]
        if kind == "add":
            return AddFlow(str(sig[1]))
        if kind == "rem":
            return RemoveFlow(str(sig[1]))
        if kind == "cont":
            return Continue()
        if kind == "back":
            return Back()
        if kind == "commit":
            return CommitRegulation()
        if kind == "new_reg":
            return NewRegulation()
        if kind == "hotspot":
            # Reconstruct payload from metadata candidates
            tv = str(sig[1])
            t0, t1 = int(sig[2]), int(sig[3])
            if isinstance(state.metadata, Mapping):
                for item in (state.metadata.get("hotspot_candidates") or []):
                    try:
                        tv_i = str(item.get("control_volume_id"))
                        wb = item.get("window_bins") or []
                        tt0, tt1 = int(wb[0]), int(wb[1])
                    except Exception:
                        continue
                    if tv_i == tv and tt0 == t0 and tt1 == t1:
                        return PickHotspot(
                            control_volume_id=tv,
                            window_bins=(t0, t1),
                            candidate_flow_ids=tuple(str(x) for x in (item.get("candidate_flow_ids") or [])),
                            mode=str(item.get("mode", "per_flow")),
                            metadata=dict(item.get("metadata", {})),
                        )
            # Fallback minimal action if metadata missing
            return PickHotspot(control_volume_id=tv, window_bins=(t0, t1), candidate_flow_ids=tuple(), metadata={})
        if kind == "stop":
            return Stop()
        # Fallback should not occur in tests
        return Stop()

    # ------------------------------- Backup --------------------------------
    def _backup(self, path: Sequence[Tuple[str, Tuple]], value: float) -> None:
        v = float(value)
        for node_key, sig in path:
            node = self.nodes.get(node_key)
            if node is None:
                continue
            node.N += 1
            node.W += v
            node.Q = node.W / max(1, node.N)
            est = node.edges.get(sig)
            if est is None:
                est = EdgeStats()
                node.edges[sig] = est
            est.N += 1
            est.W += v
            est.Q = est.W / max(1, est.N)
        if getattr(self.cfg, "debug_prints", False):
            try:
                root_key = path[0][0] if path else None
                root_node = self.nodes.get(root_key) if root_key is not None else None
                if root_node is not None:
                    print(
                        f"[MCTS/backprop] value={v:.3f} path_len={len(path)} root_visits={int(root_node.N)} root_Q={float(root_node.Q):.3f}"
                    )
                else:
                    print(f"[MCTS/backprop] value={v:.3f} path_len={len(path)} (no_root)")
            except Exception:
                pass


__all__ = ["MCTS", "MCTSConfig", "TreeNode", "EdgeStats"]
