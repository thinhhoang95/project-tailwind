from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .state import HotspotContext, PlanState, RegulationSpec
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
    guard_can_add_flow,
    guard_can_back,
    guard_can_commit,
    guard_can_continue,
    guard_can_pick_hotspot,
    guard_can_remove_flow,
    guard_can_start_new_regulation,
)


class CheapTransition:
    """Symbolic transition model that keeps a cheap residual proxy in sync."""

    def __init__(
        self,
        flow_proxies: Optional[Mapping[str, Sequence[float]]] = None,
        *,
        clip_value: float = 250.0,
        decay: float = 0.0,
    ) -> None:
        self._flow_proxies: Dict[str, np.ndarray] = {
            str(k): np.asarray(v, dtype=float) for k, v in (flow_proxies or {}).items()
        }
        self.clip_value = float(clip_value)
        self.decay = max(0.0, float(decay))

    # --- Public API ---------------------------------------------------------
    def step(self, state: PlanState, action: Action) -> Tuple[PlanState, bool, bool]:
        next_state = state.copy()
        is_commit = False
        is_terminal = False

        if isinstance(action, NewRegulation):
            guard_can_start_new_regulation(next_state)
            next_state.reset_hotspot(next_stage="select_hotspot")
            next_state.metadata.update(action.context_hint or {})
            return next_state, False, False

        if isinstance(action, PickHotspot):
            guard_can_pick_hotspot(next_state, action)
            context = HotspotContext(
                control_volume_id=action.control_volume_id,
                window_bins=tuple(int(b) for b in action.window_bins),
                candidate_flow_ids=tuple(str(fid) for fid in action.candidate_flow_ids),
                mode=action.mode,
                metadata=dict(action.metadata),
            )
            win_len = context.window_bins[1] - context.window_bins[0]
            next_state.set_hotspot(context, z_hat_shape=win_len)
            next_state.stage = "select_flows"
            return next_state, False, False

        if isinstance(action, AddFlow):
            guard_can_add_flow(next_state, action.flow_id)
            context = next_state.hotspot_context
            assert context is not None  # guarded above
            self._decay(next_state)
            proxy = self._lookup_proxy(action.flow_id, context)
            self._apply_proxy(next_state, proxy, sign=1.0)
            next_state.hotspot_context = context.add_flow(action.flow_id)
            return next_state, False, False

        if isinstance(action, RemoveFlow):
            guard_can_remove_flow(next_state, action.flow_id)
            context = next_state.hotspot_context
            assert context is not None
            self._decay(next_state)
            proxy = self._lookup_proxy(action.flow_id, context)
            self._apply_proxy(next_state, proxy, sign=-1.0)
            next_state.hotspot_context = context.remove_flow(action.flow_id)
            return next_state, False, False

        if isinstance(action, Continue):
            guard_can_continue(next_state)
            next_state.stage = "confirm"
            next_state.metadata["awaiting_commit"] = True
            return next_state, False, False

        if isinstance(action, Back):
            guard_can_back(next_state)
            if next_state.stage == "confirm":
                next_state.stage = "select_flows"
                next_state.metadata.pop("awaiting_commit", None)
            elif next_state.stage == "select_flows":
                # Drop current hotspot and return to selection stage
                next_state.reset_hotspot(next_stage="select_hotspot")
            return next_state, False, False

        if isinstance(action, CommitRegulation):
            guard_can_commit(next_state)
            context = next_state.hotspot_context
            assert context is not None

            rates = action.committed_rates
            rates_to_store: Optional[object] = None
            valid = False

            if isinstance(rates, dict):
                cleaned: Dict[str, int] = {}
                for k, v in (rates or {}).items():
                    try:
                        iv = int(v)
                    except Exception:
                        iv = 0
                    if iv > 0:
                        cleaned[str(k)] = iv
                if cleaned and context.selected_flow_ids:
                    rates_to_store = cleaned
                    valid = True
            elif rates is not None:
                try:
                    iv = int(round(float(rates)))
                except Exception:
                    iv = 0
                if iv > 0 and context.selected_flow_ids:
                    rates_to_store = iv
                    valid = True

            if valid:
                regulation = RegulationSpec(
                    control_volume_id=context.control_volume_id,
                    window_bins=context.window_bins,
                    flow_ids=context.selected_flow_ids,
                    mode=context.mode,
                    committed_rates=rates_to_store,
                    diagnostics=dict(action.diagnostics),
                )
                next_state.plan.append(regulation)

            next_state.reset_hotspot(next_stage="idle")
            next_state.metadata.pop("awaiting_commit", None)
            is_commit = True
            return next_state, is_commit, False

        if isinstance(action, Stop):
            next_state.reset_hotspot(next_stage="stopped")
            is_terminal = True
            return next_state, False, is_terminal

        raise TypeError(f"Unsupported action type {type(action)!r}")

    # --- Internal helpers ---------------------------------------------------
    def _decay(self, state: PlanState) -> None:
        if state.z_hat is None or self.decay <= 0.0:
            return
        state.z_hat *= max(0.0, 1.0 - self.decay)

    def _lookup_proxy(self, flow_id: str, context: HotspotContext) -> np.ndarray:
        fid = str(flow_id)
        # Metadata override takes precedence if available
        meta_map = context.metadata.get("flow_proxies") if context.metadata else None
        raw = None
        if isinstance(meta_map, Mapping) and fid in meta_map:
            raw = np.asarray(meta_map[fid], dtype=float)
        elif fid in self._flow_proxies:
            raw = self._flow_proxies[fid]

        win_len = context.window_bins[1] - context.window_bins[0]
        if raw is None:
            return np.ones(win_len, dtype=float)

        if raw.size == win_len:
            return np.asarray(raw, dtype=float)

        start, end = context.window_bins
        if raw.size >= end:
            return np.asarray(raw[start:end], dtype=float)

        # Fallback: pad or trim to window length deterministically
        padded = np.zeros(win_len, dtype=float)
        upto = min(win_len, raw.size)
        padded[:upto] = np.asarray(raw[:upto], dtype=float)
        return padded

    def _apply_proxy(self, state: PlanState, proxy: np.ndarray, *, sign: float) -> None:
        if state.z_hat is None:
            state.z_hat = np.zeros(proxy.shape, dtype=float)
        if state.z_hat.shape != proxy.shape:
            state.z_hat = np.zeros(proxy.shape, dtype=float)
        state.z_hat += sign * proxy
        np.clip(state.z_hat, -self.clip_value, self.clip_value, out=state.z_hat)


__all__ = ["CheapTransition"]
