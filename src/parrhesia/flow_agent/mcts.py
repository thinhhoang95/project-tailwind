from __future__ import annotations

import hashlib
import math
import threading
import time
from concurrent.futures import Executor, Future
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
from .state import PlanState
from .transition import CheapTransition
from .rate_finder import RateFinder


# ------------------------------- Config types -------------------------------


@dataclass
class MCTSConfig:
    # PUCT + widening
    c_puct: float = 6400.0 # prev. 4.0
    alpha: float = 0.7
    k0: int = 4
    k1: float = 1.0
    widen_batch_size: int = 2
    # Budgets
    commit_depth: int = 1 # controlled externally
    # Priors
    priors_temperature: float = 8.0 # higher = more uniform over the actions such as RemoveFlow, AddFlow, etc.
    root_dirichlet_epsilon: float = 0.25
    root_dirichlet_alpha: float = 0.3
    hotspot_dirichlet_epsilon: float = 0.3
    hotspot_dirichlet_alpha: float = 0.4
    flow_dirichlet_epsilon: float = 0.3
    flow_dirichlet_alpha: float = 0.4
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
        commit_executor: Optional[Executor] = None,
    ) -> None:
        self.transition = transition
        self.rate_finder = rate_finder
        self.cfg = config or MCTSConfig()
        self.rng = rng or np.random.default_rng(int(self.cfg.seed))
        self.nodes: Dict[str, TreeNode] = {}
        self._commit_eval_cache: Dict[Tuple, Tuple[Dict[str, float] | int, float, Dict[str, Any]]] = {}
        self._best_commit: Optional[Tuple[CommitRegulation, float]] = None  # (action, deltaJ)
        self._timer_factory = timer
        self._action_counts: Dict[str, int] = {}
        self._action_counts_by_stage: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._progress_cb = progress_cb
        self._current_sim: Optional[int] = None
        self._max_commits_path: int = 0
        self._last_run_stats: Dict[str, Any] = {}
        self._run_index: Optional[int] = None
        self._root_noise_applied_run: bool = False
        self._flow_noise_nodes_run: Set[str] = set()
        self._hotspot_noise_nodes_run: Set[str] = set()
        self._commit_executor = commit_executor
        self._commit_lock = threading.Lock()
        self._pending_commit_jobs: Dict[Tuple, Future] = {}

    def _pending_commit_count(self) -> int:
        with self._commit_lock:
            return sum(1 for fut in self._pending_commit_jobs.values() if not fut.done())

    def _commit_eval_job(
        self,
        plan_state: PlanState,
        control_volume_id: str,
        window_bins: Tuple[int, int],
        flows: Mapping[str, Sequence[str]],
        mode: str,
    ) -> Tuple[Dict[str, float] | int, float, Dict[str, Any]]:
        with self._timed("mcts.rate_finder.find_rates"):
            result = self.rate_finder.find_rates(
                plan_state=plan_state,
                control_volume_id=control_volume_id,
                window_bins=window_bins,
                flows=flows,
                mode=mode,
            )
        return result

    def _finalize_commit_future(
        self,
        key: Tuple,
        future: Future,
        *,
        reason: str,
        sim_index: Optional[int] = None,
        step_index: Optional[int] = None,
    ) -> None:
        try:
            rates, delta_j, info = future.result()
        except Exception as exc:
            rates = {}
            delta_j = 0.0
            info = {"reason": "async_error", "error": repr(exc)}
        with self._commit_lock:
            existing = self._pending_commit_jobs.get(key)
            if existing is not future:
                return
            self._pending_commit_jobs.pop(key, None)
            self._commit_eval_cache[key] = (rates, float(delta_j), info)

    def _commit_future_callback(
        self,
        key: Tuple,
        reason: str,
        sim_index: Optional[int],
        step_index: Optional[int],
    ) -> Callable[[Future], None]:
        def _cb(fut: Future) -> None:
            self._finalize_commit_future(key, fut, reason=reason, sim_index=sim_index, step_index=step_index)

        return _cb

    def wait_for_pending_commit_evals(self, timeout: Optional[float] = None) -> None:
        futures: List[Future]
        with self._commit_lock:
            futures = list(self._pending_commit_jobs.values())
        for fut in futures:
            fut.result(timeout=timeout)

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
        # Logging disabled in simplified mode
        return

    def _log_cold_event(
        self,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        sim_index: Optional[int] = None,
        step_index: Optional[int] = None,
    ) -> None:
        return

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

    def _record_action(
        self,
        state: PlanState,
        action: Action,
        *,
        reward: float = 0.0,
        q_value: float = 0.0,
        prior: float = 0.0,
        stage_key: Optional[str] = None,
    ) -> None:
        sig = self._signature_for_action(state, action)
        label = self._sig_to_label(sig)
        key = label
        self._action_counts[key] = self._action_counts.get(key, 0) + 1
        stage = stage_key if stage_key is not None else getattr(state, "stage", "?")
        stage_map = self._action_counts_by_stage.setdefault(stage, {})
        entry = stage_map.setdefault(
            key,
            {
                "label": key,
                "signature": sig,
                "action_type": type(action).__name__,
                "stage": stage,
                "plan_state": self._state_brief(state),
                "N": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
                "total_q": 0.0,
                "avg_q": 0.0,
                "total_prior": 0.0,
                "avg_prior": 0.0,
                "min_q": float("inf"),
                "max_q": float("-inf"),
                "min_prior": float("inf"),
                "max_prior": float("-inf"),
                "min_reward": float("inf"),
                "max_reward": float("-inf"),
            },
        )
        entry["N"] += 1
        entry["total_reward"] += float(reward)
        entry["avg_reward"] = entry["total_reward"] / max(entry["N"], 1)
        entry["total_q"] += float(q_value)
        entry["avg_q"] = entry["total_q"] / max(entry["N"], 1)
        entry["total_prior"] += float(prior)
        entry["avg_prior"] = entry["total_prior"] / max(entry["N"], 1)
        entry["min_q"] = min(entry["min_q"], float(q_value)) if math.isfinite(entry["min_q"]) else float(q_value)
        entry["max_q"] = max(entry["max_q"], float(q_value)) if math.isfinite(entry["max_q"]) else float(q_value)
        entry["min_prior"] = min(entry["min_prior"], float(prior)) if math.isfinite(entry["min_prior"]) else float(prior)
        entry["max_prior"] = max(entry["max_prior"], float(prior)) if math.isfinite(entry["max_prior"]) else float(prior)
        entry["min_reward"] = min(entry["min_reward"], float(reward)) if math.isfinite(entry["min_reward"]) else float(reward)
        entry["max_reward"] = max(entry["max_reward"], float(reward)) if math.isfinite(entry["max_reward"]) else float(reward)

    # ------------------------------- Public API ---------------------------
    def run(self, root: PlanState, *, commit_depth: Optional[int] = None, run_index: Optional[int] = None) -> CommitRegulation:
        depth_limit = int(commit_depth if commit_depth is not None else self.cfg.commit_depth)
        self._best_commit = None
        self._action_counts.clear()
        self._action_counts_by_stage.clear()
        self._max_commits_path = 0
        self._last_run_stats = {}
        self._run_index = run_index
        self._root_noise_applied_run = False
        self._flow_noise_nodes_run = set()
        self._hotspot_noise_nodes_run = set()

        root_hash = self._short_hash(root.canonical_key())
        self._dbg(f"[MCTS] run start: depth={depth_limit} {self._state_brief(root)}")

        self._simulate(root, depth_limit, sim_index=1)

        if self._best_commit is None:
            raise RuntimeError("MCTS did not evaluate any commit")
        return self._best_commit[0]

    @property
    def last_run_stats(self) -> Dict[str, Any]:
        return dict(self._last_run_stats)

    # ----------------------------- Simulation loop ------------------------
    def _simulate(self, root: PlanState, commit_depth: int, *, sim_index: int) -> float:
        state = root
        path: List[Tuple[str, Tuple]] = []  # (node_key, action_signature)
        commits_used = 0
        total_return = 0.0
        step_index = 0
        created_nodes: Set[str] = set()
        visited_keys: Dict[str, int] = {root.canonical_key(): 0}
        flow_noise_applied = False
        hotspot_noise_applied = False

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
                try:
                    z_hash = self._short_hash(arr.tobytes())
                except Exception:
                    z_hash = None

            if node is None:
                node = self._create_node(state)
                created_nodes.add(key)
                self._dbg(f"[MCTS/expand] create_leaf node={key[:24]}… phi={float(node.phi):.3f} {self._state_brief(state)}")
                v = -node.phi
                self._dbg(f"[MCTS/simulate] leaf_bootstrap return={float(v):.3f} {self._state_brief(state)} path_len={len(path)}")
        self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="leaf_bootstrap")
        self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="leaf_bootstrap")
                return v

            candidates = self._enumerate_actions(state)
            has_addflow = any(isinstance(a, AddFlow) for a in candidates)
            has_hotspot_pick = any(isinstance(a, PickHotspot) for a in candidates)
            priors = self._compute_priors(state, candidates)
            override_priors = False
            noise_info: Optional[Tuple[str, float, float]] = None
            if (
                not path
                and not self._root_noise_applied_run
                and float(self.cfg.root_dirichlet_epsilon) > 0.0
                and len(priors) > 0
            ):
                eps = float(self.cfg.root_dirichlet_epsilon)
                alpha = float(self.cfg.root_dirichlet_alpha)
                priors = self._apply_dirichlet_noise(priors, epsilon=eps, alpha=alpha)
                self._root_noise_applied_run = True
                override_priors = True
                noise_info = ("root", eps, alpha)
            elif (
                state.stage == "select_hotspot"
                and has_hotspot_pick
                and (not hotspot_noise_applied or key not in self._hotspot_noise_nodes_run)
                and float(self.cfg.hotspot_dirichlet_epsilon) > 0.0
                and len(priors) > 0
            ):
                eps = float(self.cfg.hotspot_dirichlet_epsilon)
                alpha = float(self.cfg.hotspot_dirichlet_alpha)
                priors = self._apply_dirichlet_noise(priors, epsilon=eps, alpha=alpha)
                self._hotspot_noise_nodes_run.add(key)
                override_priors = True
                noise_info = ("hotspot", eps, alpha)
                hotspot_noise_applied = True
            elif (
                state.stage == "select_flows"
                and has_addflow
                and (not flow_noise_applied or key not in self._flow_noise_nodes_run)
                and float(self.cfg.flow_dirichlet_epsilon) > 0.0
                and len(priors) > 0
            ):
                eps = float(self.cfg.flow_dirichlet_epsilon)
                alpha = float(self.cfg.flow_dirichlet_alpha)
                priors = self._apply_dirichlet_noise(priors, epsilon=eps, alpha=alpha)
                self._flow_noise_nodes_run.add(key)
                override_priors = True
                noise_info = ("flow", eps, alpha)
                flow_noise_applied = True
            if noise_info is not None:
                kind, eps, alpha = noise_info
            invalid_priors = {
                self._sig_to_label(sig): float(p)
                for sig, p in priors.items()
                if not math.isfinite(float(p))
            }
            if override_priors:
                for stale in list(node.P.keys()):
                    if stale not in priors:
                        node.P.pop(stale, None)
            for sig, p in priors.items():
                if override_priors or sig not in node.P:
                    node.P[sig] = float(p)
            m_allow = int(self.cfg.k0 + self.cfg.k1 * (node.N ** self.cfg.alpha))
            m_allow = max(1, m_allow)
            self._dbg(f"[MCTS/expand] node={key[:24]}… allow={m_allow} priors={len(priors)} children_before={len(node.children)} {self._state_brief(state)}")
            while len(node.children) < m_allow and len(node.children) < len(priors):
                for sig, _ in sorted(priors.items(), key=lambda kv: (-kv[1], kv[0])):
                    if sig in node.children:
                        continue
                    node.children[sig] = "?"
                    node.edges[sig] = node.edges.get(sig, EdgeStats())
                    # Count expansion of a new child as one action
                    self._dbg(
                        f"[MCTS/expand]   + child {self._sig_to_label(sig)} P={float(priors.get(sig, 0.0)):.3f} children_now={len(node.children)}"
                    )
                    break
                else:
                    break

            if not node.children:
                v = -node.phi
                self._dbg(f"[MCTS/select] no_children -> terminal bootstrap={float(v):.3f} {self._state_brief(state)}")
                self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="no_children")
                self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="no_children")
                return v

            viable: List[Tuple[Tuple, EdgeStats]] = []
            for sig, est in node.edges.items():
                if sig[0] == "commit" and commits_used >= commit_depth:
                    continue
                viable.append((sig, est))
            viable_map = {sig: est for sig, est in viable}
            if not viable:
                v = -node.phi
                self._dbg(
                    f"[MCTS/select] no_viable_moves -> bootstrap={float(v):.3f} commits_used={commits_used}/{commit_depth} {self._state_brief(state)}"
                )
                self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="no_viable_moves")
                self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="no_viable_moves")
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
                        "sig_tuple": sig,
                        "label": self._sig_to_label(sig),
                        "Q": float(est.Q),
                        "P": P,
                        "N_edge": int(est.N),
                        "explore": float(explore),
                        "U": float(explore),
                        "score": float(score),
                    }
                )
                if score > best_score or (score == best_score and (best_sig is None or sig < best_sig)):
                    best_score = score
                    best_sig = sig
            assert best_sig is not None
            forced_back = False
            if (
                best_sig is not None
                and best_sig[0] == "commit"
            ):
                ctx_commit = state.hotspot_context
                if ctx_commit is not None:
                    signature = tuple(sorted(str(fid) for fid in ctx_commit.selected_flow_ids))
                    if signature in self._commit_eval_cache:
                        back_sig = self._signature_for_action(state, Back())
                        if back_sig in viable_map:
                            best_sig = back_sig
                            best_score = float(viable_map[back_sig].Q if hasattr(viable_map[back_sig], "Q") else 0.0)
                            forced_back = True
            est_sel = node.edges.get(best_sig, EdgeStats())
            created_flag = key in created_nodes

            # Chosen edge score components and tie-break info
            P_best = float(node.P.get(best_sig, 0.0))
            U_best = float(self.cfg.c_puct) * P_best * (sqrtN / (1.0 + est_sel.N))
            tie_candidates = [rec for rec in selection_records if abs(float(rec.get("score", 0.0)) - float(best_score)) <= 1e-12]
            tie_break_applied = len(tie_candidates) > 1

            self._dbg(
                f"[MCTS/select] pick {self._sig_to_label(best_sig)} U={float(best_score):.3f} Q={float(est_sel.Q):.3f} P={float(node.P.get(best_sig, 0.0)):.3f} N_edge={int(est_sel.N)} node_N={int(node.N)} children={len(node.children)} {self._state_brief(state)}"
            )

            # Count selection of the best action
            action = self._action_from_signature(state, best_sig)

            r_base = 0.0
            if isinstance(action, CommitRegulation):
                if commits_used >= commit_depth:
                    v = -node.phi
                    self._dbg(
                        f"[MCTS/simulate] commit_blocked budget commits_used={commits_used}/{commit_depth} -> bootstrap={float(v):.3f}"
                    )
            self._log_backup_path(path, v, sim_index=sim_index, step_index=step_index, reason="commit_blocked")
            self._backup(path, v, sim_index=sim_index, step_index=step_index, reason="commit_blocked")
                    return v
                self._dbg(
                    f"[MCTS/simulate] evaluate_commit flows={len(getattr(getattr(state, 'hotspot_context', None), 'selected_flow_ids', []) or [])} {self._state_brief(state)}"
                )
                commit_action, delta_j = self._evaluate_commit(state, sim_index=sim_index, step_index=step_index)
                self._dbg(f"[MCTS/simulate] evaluate_commit_done ΔJ={float(delta_j):.3f}")
                action = commit_action
                r_base = -float(delta_j)
                diag_payload = commit_action.diagnostics or {}
                banned_commit = False
                if isinstance(diag_payload, Mapping):
                    banned_commit = bool(diag_payload.get("banned_regulation"))
                    if not banned_commit:
                        rf_diag = diag_payload.get("rate_finder")
                        if isinstance(rf_diag, Mapping):
                            banned_commit = bool(rf_diag.get("banned_regulation"))
                if (not banned_commit) and (self._best_commit is None or delta_j < self._best_commit[1]):
                    self._best_commit = (commit_action, float(delta_j))

            state_before = state

            # Count environment/state transition as one action
            next_state, is_commit, is_terminal = self.transition.step(state, action)
            child_key = next_state.canonical_key()
            node.children[best_sig] = child_key

            phi_s = node.phi
            phi_sp = self._phi(next_state)
            delta_phi = phi_sp - phi_s
            r_shaped = r_base + delta_phi
            total_return += r_shaped

            self._record_action(
                state_before,
                action,
                reward=float(r_shaped),
                q_value=float(est_sel.Q),
                prior=float(node.P.get(best_sig, 0.0)),
            )
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
            path.append((key, best_sig))
            state = next_state
            if is_commit:
                commits_used += 1
            # Track maximum number of commits reached along any path for this run
            future_commits = commits_used
            if hasattr(self, "_max_commits_path"):
                if int(future_commits) > int(getattr(self, "_max_commits_path", 0)):
                    self._max_commits_path = int(future_commits)

            prev_visit_depth = visited_keys.get(child_key)
            if prev_visit_depth is not None:
                self._dbg(
                    "[MCTS/simulate] cycle_detected node=%s depth=%s path_len=%s return=%.3f"
                    % (child_key[:24], int(prev_visit_depth), len(path), float(total_return))
                )
                self._log_backup_path(
                    path,
                    total_return,
                    sim_index=sim_index,
                    step_index=step_index,
                    reason="cycle",
                )
                self._backup(
                    path,
                    total_return,
                    sim_index=sim_index,
                    step_index=step_index,
                    reason="cycle",
                )
                return total_return

            visited_keys[child_key] = len(path)

            if is_terminal or commits_used >= commit_depth:
                leaf_key = state.canonical_key()
                leaf_v = -self._phi(state)
                total_return += leaf_v
                self._dbg(
                    f"[MCTS/simulate] terminal_or_budget commits_used={commits_used}/{commit_depth} leaf_bootstrap={float(leaf_v):.3f} return={float(total_return):.3f} path_len={len(path)}"
                )
                reason = "terminal"
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
            ctx_confirm = state.hotspot_context
            if not self._commit_is_banned(state):
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
                if proxy.size:
                    proxy = np.nan_to_num(proxy, nan=0.0, posinf=0.0, neginf=0.0)
                    total = max(float(proxy.sum()), 0.0)
                    logits[sig] = math.log1p(total)
                else:
                    logits[sig] = math.log1p(1.0)
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
                logits[sig] = 0.5
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

    def _apply_dirichlet_noise(
        self,
        priors: Mapping[Tuple, float],
        *,
        epsilon: float,
        alpha: float,
    ) -> Dict[Tuple, float]:
        if not priors:
            return {}
        eps = float(epsilon)
        if eps <= 0.0:
            return dict(priors)
        keys = list(priors.keys())
        base = np.array([max(float(priors[k]), 0.0) for k in keys], dtype=float)
        base_sum = base.sum()
        if not np.isfinite(base_sum) or base_sum <= 0.0:
            base = np.full(len(keys), 1.0 / len(keys), dtype=float)
        else:
            base /= base_sum
        alpha_val = max(float(alpha), 1e-6)
        dirichlet = self.rng.dirichlet(np.full(len(keys), alpha_val, dtype=float))
        mixed = (1.0 - eps) * base + eps * dirichlet
        mixed = np.maximum(mixed, 0.0)
        total = mixed.sum()
        if not np.isfinite(total) or total <= 0.0:
            mixed = np.full(len(keys), 1.0 / len(keys), dtype=float)
        else:
            mixed /= total
        return {keys[i]: float(mixed[i]) for i in range(len(keys))}

    # ------------------------------ Commits --------------------------------
    def _commit_is_banned(self, state: PlanState) -> bool:
        ctx = state.hotspot_context
        if ctx is None:
            return False
        metadata = state.metadata if isinstance(state.metadata, Mapping) else {}
        banned_entries = metadata.get("banned_regulations") or []
        try:
            selected_flows = tuple(sorted(str(fid) for fid in ctx.selected_flow_ids))
        except Exception:
            selected_flows = tuple()
        window_bins = tuple(int(b) for b in ctx.window_bins)
        mode = "per_flow" if ctx.mode == "per_flow" else "blanket"
        tv = str(ctx.control_volume_id)
        for entry in banned_entries:
            try:
                entry_tv = str(entry.get("control_volume_id"))
                raw_window = entry.get("window_bins") or []
                if len(raw_window) >= 2:
                    entry_window = (int(raw_window[0]), int(raw_window[1]))
                else:
                    entry_window = window_bins
                entry_mode = "per_flow" if str(entry.get("mode", "per_flow")) == "per_flow" else "blanket"
                entry_flows = tuple(sorted(str(fid) for fid in (entry.get("flow_ids") or [])))
            except Exception:
                continue
            if (
                entry_tv == tv
                and entry_window == window_bins
                and entry_mode == mode
                and entry_flows == selected_flows
            ):
                return True
        return False

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

        selected_flows_norm = tuple(sorted(str(fid) for fid in ctx.selected_flow_ids))
        mode_norm = "per_flow" if ctx.mode == "per_flow" else "blanket"
        window_norm = (int(ctx.window_bins[0]), int(ctx.window_bins[1]))
        existing_reg = None
        for reg in getattr(state, "plan", []) or []:
            try:
                reg_tv = str(reg.control_volume_id)
                reg_window = (int(reg.window_bins[0]), int(reg.window_bins[1]))
                reg_mode = "per_flow" if getattr(reg, "mode", "per_flow") == "per_flow" else "blanket"
                reg_flows = tuple(sorted(str(fid) for fid in reg.flow_ids))
            except Exception:
                continue
            if (
                reg_tv == str(ctx.control_volume_id)
                and reg_window == window_norm
                and reg_mode == mode_norm
                and reg_flows == selected_flows_norm
            ):
                existing_reg = reg
                break

        if existing_reg is not None:
            info = {
                "control_volume_id": str(ctx.control_volume_id),
                "window_bins": [window_norm[0], window_norm[1]],
                "mode": mode_norm,
                "banned_regulation": True,
                "reason": "duplicate_in_plan",
            }
            commit_action = CommitRegulation(
                committed_rates={},
                diagnostics={"rate_finder": info, "banned_regulation": True},
            )
            return commit_action, 0.0

        if self._commit_is_banned(state):
            info = {
                "control_volume_id": str(ctx.control_volume_id),
                "window_bins": [window_norm[0], window_norm[1]],
                "mode": mode_norm,
                "banned_regulation": True,
                "reason": "banned_regulation",
            }
            commit_action = CommitRegulation(
                committed_rates={},
                diagnostics={"rate_finder": info, "banned_regulation": True},
            )
            return commit_action, 0.0

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
        future = self._pending_commit_jobs.get(base_key)

        if cached is not None:
            rates, delta_j, info = cached
        elif future is not None:
            pending_diag = {
                "control_volume_id": str(ctx.control_volume_id),
                "window_bins": [window_norm[0], window_norm[1]],
                "mode": mode_norm,
                "reason": "async_pending",
            }
            commit_action = CommitRegulation(
                committed_rates={},
                diagnostics={"rate_finder": pending_diag},
            )
            return commit_action, 0.0
        else:
            if self._commit_executor is not None:
                job_state = state.copy()
                window_tuple = tuple(int(b) for b in ctx.window_bins)
                mode_value = "per_flow" if ctx.mode == "per_flow" else "blanket"
                future_job = self._commit_executor.submit(
                    self._commit_eval_job,
                    job_state,
                    str(ctx.control_volume_id),
                    window_tuple,
                    flows,
                    mode_value,
                )
                with self._commit_lock:
                    self._pending_commit_jobs[base_key] = future_job
                future_job.add_done_callback(
                    self._commit_future_callback(
                        base_key,
                        reason="scheduled",
                        sim_index=sim_index,
                        step_index=step_index,
                    )
                )
                pending_diag = {
                    "control_volume_id": str(ctx.control_volume_id),
                    "window_bins": [window_norm[0], window_norm[1]],
                    "mode": mode_norm,
                    "reason": "async_pending",
                }
                commit_action = CommitRegulation(
                    committed_rates={},
                    diagnostics={"rate_finder": pending_diag},
                )
                return commit_action, 0.0
            else:
                with self._timed("mcts.rate_finder.find_rates"):
                    rates, delta_j, info = self.rate_finder.find_rates(
                        plan_state=state,
                        control_volume_id=str(ctx.control_volume_id),
                        window_bins=tuple(int(b) for b in ctx.window_bins),
                        flows=flows,
                        mode="per_flow" if ctx.mode == "per_flow" else "blanket",
                    )
                with self._commit_lock:
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

    def _seed_extra_commit_eval(
        self,
        state: PlanState,
        ctx: Any,
        base_flows: Mapping[str, Sequence[str]],
        *,
        sim_index: Optional[int] = None,
        step_index: Optional[int] = None,
    ) -> None:
        return

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
                    "action_sig": self._sig_to_json(sig),
                    "sig": self._sig_to_json(sig),
                }
            )
        return

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
        # Count a single backpropagation pass as one action regardless of path length
        v = float(value)
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

    def _build_action_stats(self) -> Dict[str, Dict[str, Any]]:
        aggregated: Dict[str, Dict[str, Any]] = {}
        for stage, actions in self._action_counts_by_stage.items():
            for label, stats in actions.items():
                key = f"{stage}:{label}"
                record = dict(stats)
                record["stage"] = stage
                aggregated[key] = record
        return aggregated

    def get_action_stats(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return {stage: {label: dict(stats) for label, stats in action_map.items()} for stage, action_map in self._action_counts_by_stage.items()}


__all__ = ["MCTS", "MCTSConfig", "TreeNode", "EdgeStats"]
