from __future__ import annotations

import hashlib
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

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
from .logging import SearchLogger
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
        debug_logger: Optional[SearchLogger] = None,
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
        self._debug_logger = debug_logger
        self._current_sim: Optional[int] = None

    def _timed(self, name: str) -> ContextManager[Any]:
        if self._timer_factory is None:
            return nullcontext()
        return self._timer_factory(name)

    # ------------------------------- Debug helpers -------------------------
    def _log_debug_event(
        self,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        sim_index: Optional[int] = None,
        step_index: Optional[int] = None,
    ) -> None:
        logger = self._debug_logger
        if logger is None:
            return
        row = dict(payload or {})
        if sim_index is None:
            sim_index = self._current_sim
        if sim_index is not None:
            row.setdefault("sim", int(sim_index))
        if step_index is not None:
            row.setdefault("step", int(step_index))
        try:
            logger.event(kind, row)
        except Exception:
            pass

    @staticmethod
    def _short_hash(value: Optional[bytes | str], length: int = 12) -> Optional[str]:
        if value is None:
            return None
        try:
            if isinstance(value, str):
                value = value.encode("utf-8", "ignore")
            digest = hashlib.sha1(value).hexdigest()
            return digest[:length]
        except Exception:
            return None

    @staticmethod
    def _sig_to_json(sig: Tuple) -> List[Any]:
        try:
            return [sig[i] for i in range(len(sig))]
        except Exception:
            return [str(sig)]

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

        root_hash = self._short_hash(root.canonical_key())
        self._log_debug_event(
            "search_run_start",
            {
                "root_hash": root_hash,
                "max_sims": sims,
                "commit_depth": depth_limit,
                "time_budget_s": float(self.cfg.max_time_s),
            },
        )

        self._dbg(f"[MCTS] run start: sims={sims} depth={depth_limit} {self._state_brief(root)}")

        simulations_run = 0
        for i in range(sims):
            now = time.perf_counter()
            if now > t_end:
                self._log_debug_event(
                    "sim_time_budget_exhausted",
                    {
                        "index": i + 1,
                        "elapsed_s": now - t_start,
                        "time_budget_s": float(self.cfg.max_time_s),
                    },
                )
                break
            self._current_sim = i + 1
            self._log_debug_event(
                "sim_start",
                {
                    "index": i + 1,
                    "max_sims": sims,
                    "nodes": len(self.nodes),
                    "root_hash": root_hash,
                },
                sim_index=i + 1,
            )
            self._dbg(f"[MCTS] simulate[{i+1}/{sims}] start nodes={len(self.nodes)} {self._state_brief(root)}")
            last_ret = self._simulate(root, depth_limit, sim_index=i + 1)
            elapsed = time.perf_counter() - t_start
            self._dbg(
                f"[MCTS] simulate[{i+1}/{sims}] end   return={float(last_ret):.3f} best_dJ={(self._best_commit[1] if self._best_commit is not None else None)} nodes={len(self.nodes)}"
            )
            self._log_debug_event(
                "sim_end",
                {
                    "index": i + 1,
                    "return": float(last_ret),
                    "best_delta_j": (self._best_commit[1] if self._best_commit is not None else None),
                    "nodes": len(self.nodes),
                    "elapsed_s": elapsed,
                },
                sim_index=i + 1,
            )
            simulations_run += 1
            self._current_sim = None
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

        self._log_debug_event(
            "search_run_end",
            {
                "simulations": simulations_run,
                "best_delta_j": (self._best_commit[1] if self._best_commit is not None else None),
                "nodes": len(self.nodes),
            },
        )

        if self._best_commit is None:
            raise RuntimeError("MCTS did not evaluate any commit; increase sims or adjust state")
        return self._best_commit[0]

    # ----------------------------- Simulation loop ------------------------
    def _simulate(self, root: PlanState, commit_depth: int, *, sim_index: int) -> float:
        state = root
        path: List[Tuple[str, Tuple]] = []  # (node_key, action_signature)
        commits_used = 0
        total_return = 0.0
        step_index = 0
        created_nodes: Set[str] = set()

        while True:
            step_index += 1
            key = state.canonical_key()
            node = self.nodes.get(key)
            node_hash = self._short_hash(key)
            ctx = state.hotspot_context
            selected_flows = tuple(ctx.selected_flow_ids) if ctx is not None else tuple()
            candidate_flows = tuple(ctx.candidate_flow_ids) if ctx is not None else tuple()

            z_hash = None
            if state.z_hat is not None:
                arr = np.asarray(state.z_hat, dtype=float)
                if not np.all(np.isfinite(arr)):
                    self._log_debug_event(
                        "invalid_z_hat",
                        {
                            "node_hash": node_hash,
                            "nan_count": int(np.isnan(arr).sum()),
                            "inf_count": int(np.isinf(arr).sum()),
                        },
                        sim_index=sim_index,
                        step_index=step_index,
                    )
                try:
                    z_hash = self._short_hash(arr.tobytes())
                except Exception:
                    z_hash = None

            visit_payload: Dict[str, Any] = {
                "stage": state.stage,
                "node_hash": node_hash,
                "existing": bool(node is not None),
                "node_visits": int(node.N) if node is not None else 0,
                "commits_used": commits_used,
                "total_return": float(total_return),
                "path_len": len(path),
                "z_hash": z_hash,
            }
            if selected_flows:
                visit_payload["selected_flows"] = list(selected_flows)
            if candidate_flows and state.stage in {"select_flows", "confirm"}:
                visit_payload["candidate_flows"] = list(candidate_flows)
            self._log_debug_event("node_visit", visit_payload, sim_index=sim_index, step_index=step_index)

            if node is None:
                node = self._create_node(state)
                created_nodes.add(key)
                self._dbg(f"[MCTS/expand] create_leaf node={key[:24]}… phi={float(node.phi):.3f} {self._state_brief(state)}")
                self._log_debug_event(
                    "node_created",
                    {
                        "node_hash": node_hash,
                        "stage": state.stage,
                        "phi": float(node.phi),
                        "selected_flows": list(selected_flows),
                        "z_hash": z_hash,
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
                v = -node.phi
                self._dbg(f"[MCTS/simulate] leaf_bootstrap return={float(v):.3f} {self._state_brief(state)} path_len={len(path)}")
                self._log_debug_event(
                    "leaf_bootstrap",
                    {
                        "node_hash": node_hash,
                        "value": float(v),
                        "phi": float(node.phi),
                        "path_len": len(path),
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
                self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="leaf_bootstrap")
                self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="leaf_bootstrap")
                self._log_debug_event(
                    "simulate_return",
                    {
                        "reason": "leaf_bootstrap",
                        "value": float(v),
                        "path_len": len(path),
                        "node_hash": node_hash,
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
                return v

            candidates = self._enumerate_actions(state)
            priors = self._compute_priors(state, candidates)
            invalid_priors = {
                self._sig_to_label(sig): float(p)
                for sig, p in priors.items()
                if not math.isfinite(float(p))
            }
            if invalid_priors:
                self._log_debug_event(
                    "invalid_priors",
                    {"node_hash": node_hash, "priors": invalid_priors},
                    sim_index=sim_index,
                    step_index=step_index,
                )
            for sig, p in priors.items():
                if sig not in node.P:
                    node.P[sig] = float(p)
            m_allow = int(self.cfg.k0 + self.cfg.k1 * (node.N ** self.cfg.alpha))
            m_allow = max(1, m_allow)
            self._dbg(f"[MCTS/expand] node={key[:24]}… allow={m_allow} priors={len(priors)} children_before={len(node.children)} {self._state_brief(state)}")
            self._log_debug_event(
                "expand_overview",
                {
                    "node_hash": node_hash,
                    "stage": state.stage,
                    "priors_count": len(priors),
                    "children_before": len(node.children),
                    "m_allow": m_allow,
                    "node_visits": int(node.N),
                },
                sim_index=sim_index,
                step_index=step_index,
            )
            while len(node.children) < m_allow and len(node.children) < len(priors):
                for sig, _ in sorted(priors.items(), key=lambda kv: (-kv[1], kv[0])):
                    if sig in node.children:
                        continue
                    node.children[sig] = "?"
                    node.edges[sig] = node.edges.get(sig, EdgeStats())
                    self._dbg(
                        f"[MCTS/expand]   + child {self._sig_to_label(sig)} P={float(priors.get(sig, 0.0)):.3f} children_now={len(node.children)}"
                    )
                    break
                else:
                    break

            children_snapshot: List[Dict[str, Any]] = []
            for sig, child_key in node.children.items():
                est = node.edges.get(sig)
                child_hash = self._short_hash(child_key) if isinstance(child_key, str) and child_key not in {"?"} else None
                children_snapshot.append(
                    {
                        "sig": self._sig_to_json(sig),
                        "label": self._sig_to_label(sig),
                        "child_hash": child_hash,
                        "P": float(node.P.get(sig, 0.0)),
                        "N": int(est.N if est else 0),
                        "Q": float(est.Q if est else 0.0),
                    }
                )

            if not node.children:
                v = -node.phi
                self._dbg(f"[MCTS/select] no_children -> terminal bootstrap={float(v):.3f} {self._state_brief(state)}")
                self._log_debug_event(
                    "no_children_terminal",
                    {
                        "node_hash": node_hash,
                        "stage": state.stage,
                        "value": float(v),
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
                self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="no_children")
                self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="no_children")
                self._log_debug_event(
                    "simulate_return",
                    {
                        "reason": "no_children",
                        "value": float(v),
                        "path_len": len(path),
                        "node_hash": node_hash,
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
                return v

            viable: List[Tuple[Tuple, EdgeStats]] = []
            for sig, est in node.edges.items():
                if sig[0] == "commit" and commits_used >= commit_depth:
                    continue
                viable.append((sig, est))
            if not viable:
                v = -node.phi
                self._dbg(
                    f"[MCTS/select] no_viable_moves -> bootstrap={float(v):.3f} commits_used={commits_used}/{commit_depth} {self._state_brief(state)}"
                )
                self._log_debug_event(
                    "no_viable_moves",
                    {
                        "node_hash": node_hash,
                        "stage": state.stage,
                        "value": float(v),
                        "commits_used": commits_used,
                        "commit_depth": commit_depth,
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
                self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="no_viable_moves")
                self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="no_viable_moves")
                self._log_debug_event(
                    "simulate_return",
                    {
                        "reason": "no_viable_moves",
                        "value": float(v),
                        "path_len": len(path),
                        "node_hash": node_hash,
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
                return v

            sqrtN = math.sqrt(max(1, node.N))
            best_sig: Optional[Tuple] = None
            best_score = -1e30
            selection_records: List[Dict[str, Any]] = []
            for sig, est in viable:
                P = float(node.P.get(sig, 0.0))
                explore = float(self.cfg.c_puct) * P * (sqrtN / (1.0 + est.N))
                score = float(est.Q) + explore
                selection_records.append(
                    {
                        "sig": self._sig_to_json(sig),
                        "label": self._sig_to_label(sig),
                        "Q": float(est.Q),
                        "P": P,
                        "N_edge": int(est.N),
                        "explore": float(explore),
                        "score": float(score),
                    }
                )
                if score > best_score or (score == best_score and (best_sig is None or sig < best_sig)):
                    best_score = score
                    best_sig = sig
            assert best_sig is not None
            est_sel = node.edges.get(best_sig, EdgeStats())
            created_flag = key in created_nodes

            if state.stage == "confirm":
                commit_sig = self._signature_for_action(state, CommitRegulation())
                back_sig = self._signature_for_action(state, Back())
                stop_sig = self._signature_for_action(state, Stop())
                confirm_payload = {
                    "node_hash": node_hash,
                    "node_N": int(node.N),
                    "child_count": len(node.children),
                    "created_this_sim": created_flag,
                    "z_hash": z_hash,
                    "edges": {},
                    "priors": {},
                }
                for label, sig in (("commit", commit_sig), ("back", back_sig), ("stop", stop_sig)):
                    est_edge = node.edges.get(sig)
                    confirm_payload["edges"][label] = {
                        "P": float(node.P.get(sig, 0.0)),
                        "N": int(est_edge.N if est_edge else 0),
                        "Q": float(est_edge.Q if est_edge else 0.0),
                    }
                    confirm_payload["priors"][label] = float(priors.get(sig, node.P.get(sig, 0.0)))
                self._log_debug_event("confirm_snapshot", confirm_payload, sim_index=sim_index, step_index=step_index)

            selection_payload = {
                "stage": state.stage,
                "node_hash": node_hash,
                "node_N": int(node.N),
                "created_this_sim": created_flag,
                "sqrtN": float(sqrtN),
                "records": selection_records,
                "children": children_snapshot,
                "chosen": {
                    "sig": self._sig_to_json(best_sig),
                    "label": self._sig_to_label(best_sig),
                    "score": float(best_score),
                },
                "viable_count": len(viable),
                "path_len": len(path),
            }
            self._log_debug_event("selection_snapshot", selection_payload, sim_index=sim_index, step_index=step_index)

            self._dbg(
                f"[MCTS/select] pick {self._sig_to_label(best_sig)} U={float(best_score):.3f} Q={float(est_sel.Q):.3f} P={float(node.P.get(best_sig, 0.0)):.3f} N_edge={int(est_sel.N)} node_N={int(node.N)} children={len(node.children)} {self._state_brief(state)}"
            )

            action = self._action_from_signature(state, best_sig)

            r_base = 0.0
            if isinstance(action, CommitRegulation):
                if commits_used >= commit_depth:
                    v = -node.phi
                    self._dbg(
                        f"[MCTS/simulate] commit_blocked budget commits_used={commits_used}/{commit_depth} -> bootstrap={float(v):.3f}"
                    )
                    self._log_debug_event(
                        "commit_blocked_quota",
                        {
                            "node_hash": node_hash,
                            "value": float(v),
                            "commits_used": commits_used,
                            "commit_depth": commit_depth,
                        },
                        sim_index=sim_index,
                        step_index=step_index,
                    )
                    self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="commit_blocked")
                    self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="commit_blocked")
                    self._log_debug_event(
                        "simulate_return",
                        {
                            "reason": "commit_blocked",
                            "value": float(v),
                            "path_len": len(path),
                            "node_hash": node_hash,
                        },
                        sim_index=sim_index,
                        step_index=step_index,
                    )
                    return v
                self._log_debug_event(
                    "commit_eval_start",
                    {
                        "node_hash": node_hash,
                        "selected_flows": len(selected_flows),
                        "commit_calls": int(self._commit_calls),
                        "commit_eval_limit": int(self.cfg.commit_eval_limit),
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
                self._dbg(
                    f"[MCTS/simulate] evaluate_commit flows={len(getattr(getattr(state, 'hotspot_context', None), 'selected_flow_ids', []) or [])} {self._state_brief(state)}"
                )
                commit_action, delta_j = self._evaluate_commit(state, sim_index=sim_index, step_index=step_index)
                self._dbg(f"[MCTS/simulate] evaluate_commit_done ΔJ={float(delta_j):.3f} calls={int(self._commit_calls)}")
                self._log_debug_event(
                    "commit_eval_done",
                    {
                        "node_hash": node_hash,
                        "delta_j": float(delta_j),
                        "commit_calls": int(self._commit_calls),
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
                action = commit_action
                r_base = -float(delta_j)
                if self._best_commit is None or delta_j < self._best_commit[1]:
                    self._best_commit = (commit_action, float(delta_j))

            self._record_action(action)

            next_state, is_commit, is_terminal = self.transition.step(state, action)
            child_key = next_state.canonical_key()
            node.children[best_sig] = child_key

            phi_s = node.phi
            phi_sp = self._phi(next_state)
            delta_phi = phi_sp - phi_s
            r_shaped = r_base + delta_phi
            total_return += r_shaped
            self._dbg(
                f"[MCTS/step] {type(action).__name__} commit={bool(is_commit)} term={bool(is_terminal)} r_base={float(r_base):.3f} Δphi={float(delta_phi):.3f} r={float(r_shaped):.3f} G={float(total_return):.3f} {self._state_brief(next_state)}"
            )

            next_ctx = next_state.hotspot_context
            next_selected = tuple(next_ctx.selected_flow_ids) if next_ctx is not None else tuple()
            next_z_hash = None
            if next_state.z_hat is not None:
                try:
                    next_z_hash = self._short_hash(np.asarray(next_state.z_hat, dtype=float).tobytes())
                except Exception:
                    next_z_hash = None

            future_commits = commits_used + (1 if is_commit else 0)
            self._log_debug_event(
                "step_result",
                {
                    "node_hash": node_hash,
                    "stage": state.stage,
                    "action_sig": self._sig_to_json(best_sig),
                    "action_label": self._sig_to_label(best_sig),
                    "is_commit": bool(is_commit),
                    "is_terminal": bool(is_terminal),
                    "r_base": float(r_base),
                    "phi_s": float(phi_s),
                    "phi_sp": float(phi_sp),
                    "delta_phi": float(delta_phi),
                    "reward": float(r_shaped),
                    "total_return": float(total_return),
                    "child_hash": self._short_hash(child_key),
                    "commits_used_after": future_commits,
                    "selected_flows_before": list(selected_flows),
                    "selected_flows_after": list(next_selected),
                    "next_z_hash": next_z_hash,
                },
                sim_index=sim_index,
                step_index=step_index,
            )

            path.append((key, best_sig))
            state = next_state
            if is_commit:
                commits_used += 1

            if is_terminal or commits_used >= commit_depth:
                leaf_key = state.canonical_key()
                leaf_hash = self._short_hash(leaf_key)
                leaf_v = -self._phi(state)
                total_return += leaf_v
                self._dbg(
                    f"[MCTS/simulate] terminal_or_budget commits_used={commits_used}/{commit_depth} leaf_bootstrap={float(leaf_v):.3f} return={float(total_return):.3f} path_len={len(path)}"
                )
                reason = "terminal" if is_terminal else "commit_budget"
                self._log_debug_event(
                    "terminal_leaf",
                    {
                        "node_hash": leaf_hash,
                        "reason": reason,
                        "leaf_value": float(leaf_v),
                        "total_return": float(total_return),
                        "commits_used": commits_used,
                        "commit_depth": commit_depth,
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
                self._log_backup_path(
                    path,
                    total_return,
                    sim_index=sim_index,
                    step_index=step_index,
                    reason="terminal_or_budget",
                )
                self._backup(
                    path,
                    total_return,
                    sim_index=sim_index,
                    step_index=step_index,
                    reason="terminal_or_budget",
                )
                self._log_debug_event(
                    "simulate_return",
                    {
                        "reason": "terminal_or_budget",
                        "value": float(total_return),
                        "path_len": len(path),
                        "node_hash": leaf_hash,
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
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
    def _evaluate_commit(
        self,
        state: PlanState,
        *,
        sim_index: Optional[int] = None,
        step_index: Optional[int] = None,
    ) -> Tuple[CommitRegulation, float]:
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
            self._log_debug_event(
                "commit_eval_cached",
                {
                    "flows": len(ctx.selected_flow_ids),
                    "delta_j": float(delta_j),
                },
                sim_index=sim_index,
                step_index=step_index,
            )
        else:
            if self._commit_calls >= int(self.cfg.commit_eval_limit):
                # Treat as no-op commit with zero improvement to avoid extra cost
                rates, delta_j, info = ({}, 0.0, {"reason": "eval_budget_exhausted"})
                self._log_debug_event(
                    "commit_eval_budget_exhausted",
                    {
                        "commit_calls": int(self._commit_calls),
                        "commit_eval_limit": int(self.cfg.commit_eval_limit),
                        "flows": len(ctx.selected_flow_ids),
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )
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
                self._log_debug_event(
                    "commit_eval_result",
                    {
                        "commit_calls": int(self._commit_calls),
                        "delta_j": float(delta_j),
                        "flows": len(ctx.selected_flow_ids),
                    },
                    sim_index=sim_index,
                    step_index=step_index,
                )

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

    def _log_backup_path(
        self,
        path: Sequence[Tuple[str, Tuple]],
        value: float,
        *,
        sim_index: Optional[int] = None,
        step_index: Optional[int] = None,
        reason: str = "",
    ) -> None:
        if self._debug_logger is None:
            return
        entries: List[Dict[str, Any]] = []
        for depth, (node_key, sig) in enumerate(path):
            entries.append(
                {
                    "depth": depth,
                    "node_hash": self._short_hash(node_key),
                    "action": self._sig_to_label(sig),
                    "sig": self._sig_to_json(sig),
                }
            )
        self._log_debug_event(
            "backup_path",
            {
                "value": float(value),
                "reason": reason,
                "path_len": len(path),
                "path": entries,
            },
            sim_index=sim_index,
            step_index=step_index,
        )

    # ------------------------------- Backup --------------------------------
    def _backup(
        self,
        path: Sequence[Tuple[str, Tuple]],
        value: float,
        *,
        sim_index: Optional[int] = None,
        step_index: Optional[int] = None,
        reason: str = "",
    ) -> None:
        v = float(value)
        updates: List[Dict[str, Any]] = []
        for depth, (node_key, sig) in enumerate(path):
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
            updates.append(
                {
                    "depth": depth,
                    "node_hash": self._short_hash(node_key),
                    "action": self._sig_to_label(sig),
                    "sig": self._sig_to_json(sig),
                    "node_N": int(node.N),
                    "node_Q": float(node.Q),
                    "edge_N": int(est.N),
                    "edge_Q": float(est.Q),
                }
            )
        if updates:
            self._log_debug_event(
                "backup_update",
                {
                    "value": v,
                    "reason": reason,
                    "updates": updates,
                },
                sim_index=sim_index,
                step_index=step_index,
            )
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
