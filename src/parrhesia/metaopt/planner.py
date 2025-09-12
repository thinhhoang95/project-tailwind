from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import Hotspot, FlowSpec, RegulationProposal


def choose_active_window(
    x_G: np.ndarray,
    t_G: int,
    overloaded_window_at_hotspot_aligned: Optional[Sequence[int]] = None,
    *,
    min_frac_of_peak: float = 0.5,
    max_span: int = 12,
) -> List[int]:
    """
    Select active bins centered around t_G where x_G(t) exceeds a fraction of its peak.
    If an overloaded window aligned to the hotspot is provided, union it with the
    fraction-of-peak window.
    """
    T = x_G.size
    tG = max(0, min(T - 1, int(t_G)))
    peak = float(np.max(x_G)) if T > 0 else 0.0
    thresh = peak * float(min_frac_of_peak)
    # Find contiguous region around tG above threshold
    left = tG
    while left - 1 >= 0 and x_G[left - 1] >= thresh and (tG - (left - 1) <= max_span):
        left -= 1
    right = tG
    while right + 1 < T and x_G[right + 1] >= thresh and ((right + 1) - tG <= max_span):
        right += 1
    base = set(range(left, right + 1))
    if overloaded_window_at_hotspot_aligned:
        base.update(int(b) for b in overloaded_window_at_hotspot_aligned if 0 <= int(b) < T)
    return sorted(base)


def make_proposals(
    hotspot: Hotspot,
    flows: Sequence[FlowSpec],
    group_labels: Mapping[Hashable, int],
    xG_map: Mapping[Hashable, np.ndarray],
    tG_map: Mapping[Hashable, int],
    ctrl_by_flow: Mapping[Hashable, Optional[str]],
    *,
    default_rate_guess: int = 1,
) -> List[RegulationProposal]:
    """
    Build a simple set of RegulationProposals: one per group. Control volume is
    selected as the modal control among group members; active window is the union
    of chosen windows across members. Rate guess is a simple constant.
    """
    # Group members by label
    members: Dict[int, List[FlowSpec]] = {}
    for fs in flows:
        gid = int(group_labels.get(fs.flow_id, -1))
        members.setdefault(gid, []).append(fs)

    proposals: List[RegulationProposal] = []
    for gid, group in members.items():
        if not group:
            continue
        # Modal control volume id among members
        ctrls = [ctrl_by_flow.get(fs.flow_id) for fs in group]
        ctrls = [c for c in ctrls if c is not None]
        if not ctrls:
            continue
        ctrl_modal = sorted(ctrls)[0]
        # Active window: union of member windows around their phases
        active: set[int] = set()
        for fs in group:
            xG = xG_map.get(fs.flow_id)
            tG = tG_map.get(fs.flow_id, 0)
            if xG is None:
                continue
            wins = choose_active_window(xG, int(tG))
            active.update(wins)
        proposals.append(
            RegulationProposal(
                flow_ids=[fs.flow_id for fs in group],
                control_tv_id=str(ctrl_modal),
                active_bins=sorted(active),
                rate_guess=int(default_rate_guess),
                meta={"hotspot": {"tv_id": hotspot.tv_id, "bin": hotspot.bin}, "group_id": gid},
            )
        )
    return proposals


def to_rate_optimizer_inputs(
    proposals: Sequence[RegulationProposal],
    flights_by_flow: Mapping[Any, Sequence[Mapping[str, Any]]],
    ctrl_by_flow: Mapping[Any, Optional[str]],
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Optional[str]]]:
    """
    Filter `flights_by_flow` and `ctrl_by_flow` to only include flows present in
    proposals. Bins are not enforced here; downstream code should filter by
    `active_bins` as needed.
    """
    keep: set = set()
    for p in proposals:
        keep.update(p.flow_ids)
    flights_out: Dict[int, List[Dict[str, Any]]] = {}
    ctrl_out: Dict[int, Optional[str]] = {}
    for f, specs in flights_by_flow.items():
        if f in keep:
            flights_out[int(f)] = [dict(s) for s in specs]
            ctrl_out[int(f)] = ctrl_by_flow.get(f)
    return flights_out, ctrl_out

