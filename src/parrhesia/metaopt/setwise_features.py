from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np

from .types import HyperParams, Hotspot
from .travel_offsets import flow_offsets_from_ctrl
from .per_flow_features import phase_time, price_contrib_v_tilde
from .pairwise_features import (
    temporal_overlap,
    offset_orthogonality,
    slack_profile,
    slack_corr,
    price_gap,
)
from .flow_signals import build_x_series_at_tv


def compare_flow_against_flight_set(
    *,
    anchor_tG: int,
    anchor_tau: Mapping[int, int],
    anchor_xG: np.ndarray,
    v_tilde_anchor: float,
    set_flight_ids: Sequence[str],
    control_tv_for_set: str,
    hotspot_tv_id: str,
    hotspot_row: int,
    hotspot_bin: int,
    bin_offsets: Mapping[str, Mapping[str, int]],
    hourly_excess_bool: np.ndarray,
    slack_per_bin_matrix: np.ndarray,
    params: HyperParams,
    flight_list: Any,
    row_map: Mapping[str, int],
    tv_centroids: Optional[Mapping[str, Tuple[float, float]]] = None,
    rolling_occ_by_bin: Optional[np.ndarray] = None,
    hourly_capacity_matrix: Optional[np.ndarray] = None,
    bins_per_hour: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute overlap/orth/slack_corr/price_gap between an anchor flow and an arbitrary set of flights.
    Accumulators (summing over bins) should be handled by the caller.
    """
    T = int(slack_per_bin_matrix.shape[1])

    # τ for the set (signs inferred from flight order; geometric fallback if needed)
    tau_set_raw = flow_offsets_from_ctrl(
        control_tv_id=str(control_tv_for_set),
        tv_id_to_idx=row_map,
        bin_offsets=bin_offsets,
        flow_flight_ids=[str(f) for f in (set_flight_ids or [])],
        flight_list=flight_list,
        hotspots=[str(hotspot_tv_id)],
        trim_policy="earliest_hotspot",
        direction_sign_mode="order_vs_ctrl",
        tv_centroids=tv_centroids,
    ) or {}

    # Restrict τ to rows actually touched by the set; keep hotspot row to retain τ_ref
    visited_rows: set[int] = set()
    if set_flight_ids:
        for fid in set_flight_ids:
            try:
                seq = flight_list.get_flight_tv_sequence_indices(str(fid))
                if not isinstance(seq, np.ndarray):
                    seq = np.asarray(seq, dtype=np.int64)
            except Exception:
                continue
            if seq.size == 0:
                continue
            # Optional trim to earliest hotspot
            if hotspot_row is not None:
                cut = None
                for i, v in enumerate(seq.tolist()):
                    if int(v) == int(hotspot_row):
                        cut = i
                        break
                if cut is not None:
                    seq = seq[: cut + 1]
            visited_rows.update(int(v) for v in seq.tolist())

    if hotspot_row is not None:
        visited_rows.add(int(hotspot_row))
    tau_set = {int(r): int(off) for r, off in tau_set_raw.items() if int(r) in visited_rows}

    # x_set(t) at the chosen control TV
    x_set = build_x_series_at_tv(
        flight_list=flight_list,
        flight_ids=list(set_flight_ids or []),
        tv_id=str(control_tv_for_set),
        num_time_bins_per_tv=T,
    )

    # Phase times and alignment window
    tG_set = int(phase_time(int(hotspot_row), Hotspot(tv_id=str(hotspot_tv_id), bin=int(hotspot_bin)), tau_set, T))
    window = list(range(-int(params.window_left), int(params.window_right) + 1))
    W = [int(anchor_tG) + int(k) for k in window]

    # Shift x_set into anchor's phase frame
    d = int(anchor_tG - tG_set)
    x_set_shift = np.zeros_like(x_set)
    if d >= 0:
        x_set_shift[d:] = x_set[: T - d]
    else:
        x_set_shift[: T + d] = x_set[-d:]

    ov = temporal_overlap(anchor_xG, x_set_shift, window_bins=W)
    orth = offset_orthogonality(int(hotspot_row), int(hotspot_bin), anchor_tau, tau_set, hourly_excess_bool)

    Si = slack_profile(int(anchor_tG), anchor_tau, slack_per_bin_matrix, window)
    Sj = slack_profile(int(tG_set), tau_set, slack_per_bin_matrix, window)
    sc = slack_corr(Si, Sj)

    v_tilde_set = price_contrib_v_tilde(
        int(tG_set),
        int(hotspot_row),
        int(hotspot_bin),
        tau_set,
        hourly_excess_bool,
        slack_per_bin_matrix,
        x_set,
        theta_mask=None,
        w_sum=float(params.w_sum),
        w_max=float(params.w_max),
        kappa=float(params.kappa),
        eps=float(params.eps),
        verbose_debug=False,
        idx_to_tv_id=None,
        rolling_occ_by_bin=rolling_occ_by_bin,
        hourly_capacity_matrix=hourly_capacity_matrix,
        bins_per_hour=bins_per_hour,
    )
    pg = price_gap(float(v_tilde_anchor), float(v_tilde_set), eps=float(params.eps))

    return {
        "overlap": float(ov),
        "orth": float(orth),
        "slack_corr": float(sc),
        "price_gap": float(pg),
    }

