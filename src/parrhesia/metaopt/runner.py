from __future__ import annotations

from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import Hotspot, FlowSpec, HyperParams, RegulationProposal
from .base_caches import build_base_caches, attention_mask_from_cells
from .travel_offsets import minutes_to_bin_offsets, flow_offsets_from_ctrl
from .flow_signals import build_xG_series
from .per_flow_features import phase_time, score as score_single
from .pairwise_features import (
    temporal_overlap,
    offset_orthogonality,
    slack_profile,
    slack_corr,
    price_gap,
)
from .grouping import cluster_flows
from ..optim.capacity import build_bin_capacities


def rank_flows_and_plan(
    flight_list: Any,
    indexer: Any,
    travel_minutes_map: Mapping[str, Mapping[str, float]],
    flights_by_flow: Mapping[int, Sequence[Mapping[str, Any]]],
    ctrl_by_flow: Mapping[int, Optional[str]],
    hotspot: Hotspot,
    params: Optional[HyperParams] = None,
    capacities_by_tv: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[List[RegulationProposal], Dict[str, Any]]:
    """
    High-level convenience runner to compute features and assemble a rough set of proposals.
    This is a minimal implementation for experimentation and is not wired to SA.
    """
    P = params or HyperParams()

    # Capacities
    if capacities_by_tv is None:
        raise ValueError("capacities_by_tv must be provided (tv_id -> per-bin capacity array)")

    caches = build_base_caches(flight_list, capacities_by_tv=capacities_by_tv, indexer=indexer)

    # Bin offsets τ from control to all TVs per flow
    bin_offsets = minutes_to_bin_offsets(travel_minutes_map, time_bin_minutes=int(indexer.time_bin_minutes))

    # Build x_G for each flow and compute t_G
    T = int(indexer.num_time_bins)
    xG_map: Dict[int, np.ndarray] = {}
    tG_map: Dict[int, int] = {}
    tau_map: Dict[int, Dict[int, int]] = {}
    row_map = {str(tv): int(r) for tv, r in flight_list.tv_id_to_idx.items()}
    h_row = int(row_map.get(hotspot.tv_id, -1))
    if h_row < 0:
        raise ValueError(f"Unknown hotspot tv_id: {hotspot.tv_id}")
    for f in flights_by_flow.keys():
        xG = build_xG_series(flights_by_flow, ctrl_by_flow, int(f), T)
        xG_map[int(f)] = xG
        ctrl_tv = ctrl_by_flow.get(f)
        tmap = flow_offsets_from_ctrl(ctrl_tv, row_map, bin_offsets) or {}
        tau_map[int(f)] = tmap
        ctrl_row = None if ctrl_tv is None else int(row_map.get(str(ctrl_tv), -1))
        ctrl_row = None if (ctrl_row is not None and ctrl_row < 0) else ctrl_row
        tG = phase_time(ctrl_row, hotspot, tmap, T)
        tG_map[int(f)] = tG

    # Compute per-flow score for rough ranking
    theta = attention_mask_from_cells((hotspot.tv_id, hotspot.bin), tv_id_to_idx=flight_list.tv_id_to_idx, T=T)
    scores: Dict[int, float] = {}
    H_bool = caches["hourly_excess_bool"]
    S_mat = caches["slack_per_bin_matrix"]
    for f in flights_by_flow.keys():
        s = score_single(
            t_G=tG_map[int(f)],
            hotspot_row=h_row,
            hotspot_bin=int(hotspot.bin),
            tau_row_to_bins=tau_map[int(f)],
            hourly_excess_bool=H_bool,
            slack_per_bin_matrix=S_mat,
            params=P,
            xG=xG_map[int(f)],
            theta_mask=theta,
            use_soft_eligibility=True,
        )
        scores[int(f)] = float(s)

    # Pairwise features for simple grouping
    keys = list(flights_by_flow.keys())
    pair_feats: Dict[Tuple[int, int], Dict[str, float]] = {}
    window_bins = list(range(-P.window_left, P.window_right + 1))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            fi, fj = int(keys[i]), int(keys[j])
            # Temporal overlap around aligned phases
            # Shift xGj by Δ = tGi - tGj into Gi's frame
            d = int(tG_map[fi] - tG_map[fj])
            xi = xG_map[fi]
            xj = xG_map[fj]
            # Build window around tGi
            W = [int(tG_map[fi] + w) for w in window_bins]
            # Shift xj by d
            xj_shift = np.zeros_like(xj)
            if d >= 0:
                xj_shift[d:] = xj[: xj.size - d]
            else:
                xj_shift[: xj.size + d] = xj[-d :]
            ov = temporal_overlap(xi, xj_shift, window_bins=W)
            # Orthogonality
            orth = offset_orthogonality(h_row, int(hotspot.bin), tau_map[fi], tau_map[fj], H_bool)
            # Slack correlation
            Si = slack_profile(tG_map[fi], tau_map[fi], S_mat, window_bins)
            Sj = slack_profile(tG_map[fj], tau_map[fj], S_mat, window_bins)
            sc = slack_corr(Si, Sj)
            # Price gap based on v_G at t_G
            # Approximate v_G via primary term only for speed
            vi = scores[fi]  # already combines a and rho; still usable comparatively
            vj = scores[fj]
            pg = price_gap(vi, vj, eps=P.eps)

            pair_feats[(fi, fj)] = {
                "overlap": float(ov),
                "orth": float(orth),
                "slack_corr": float(sc),
                "price_gap": float(pg),
            }

    # Group flows
    labels = cluster_flows(keys, pair_feats, thresholds={
        "tau_ov": 0.0,  # default conservative
        "tau_sl": 0.0,
        "tau_pr": 0.5,
        "tau_orth": 0.8,
    })

    # Build proposals
    flows = [FlowSpec(flow_id=int(f), control_tv_id=ctrl_by_flow.get(f), flight_ids=[sp.get("flight_id", "") for sp in flights_by_flow.get(f, [])]) for f in keys]
    from .planner import make_proposals
    proposals = make_proposals(hotspot, flows, labels, xG_map, tG_map, ctrl_by_flow)

    diagnostics = {
        "scores_by_flow": scores,
        "pairwise_features": pair_feats,
        "labels": labels,
    }
    return proposals, diagnostics

