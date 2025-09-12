from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


def temporal_overlap(xGi: np.ndarray, xGj: np.ndarray, window_bins: Optional[Sequence[int]] = None) -> float:
    """
    Overlap_{ij} = ∑_{t∈W} min{x_i(t), x_j(t)}. If window_bins is None, use all bins.
    """
    xi = np.asarray(xGi, dtype=np.float64)
    xj = np.asarray(xGj, dtype=np.float64)
    L = min(xi.size, xj.size)
    if window_bins is None:
        return float(np.minimum(xi[:L], xj[:L]).sum())
    w = np.zeros(L, dtype=bool)
    for t in window_bins:
        tt = int(t)
        if 0 <= tt < L:
            w[tt] = True
    if not np.any(w):
        return 0.0
    return float(np.minimum(xi[:L], xj[:L])[w].sum())


def offset_orthogonality(
    hotspot_row: int,
    hotspot_bin: int,
    tau_i: Mapping[int, int],
    tau_j: Mapping[int, int],
    hourly_excess_bool: np.ndarray,
    tv_universe_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Orth_{ij} = 1 − |T_i ∩ T_j| / |T_all_overloaded|, with
    T_i = {s: o_s(t* + τ_{G_i,s} − τ_{G_i,s*})>0} and T_all_overloaded = T_i ∪ T_j by default.
    """
    V, T = hourly_excess_bool.shape
    rows = np.arange(V, dtype=np.int32)

    # Helper to compute set of overloaded rows for a given tau
    def _set_for_tau(tau_map: Mapping[int, int]) -> np.ndarray:
        tau = np.zeros(V, dtype=np.int32)
        for r, off in tau_map.items():
            if 0 <= int(r) < V:
                tau[int(r)] = int(off)
        tau_ref = int(tau[int(hotspot_row)]) if 0 <= int(hotspot_row) < V else 0
        t_idx = np.clip(int(hotspot_bin) + (tau - tau_ref), 0, T - 1)
        mask = hourly_excess_bool[rows, t_idx]
        mask[int(hotspot_row)] = False  # exclude the hotspot row
        if tv_universe_mask is not None and tv_universe_mask.size == V:
            mask = np.logical_and(mask, tv_universe_mask.astype(bool))
        return mask

    Ti = _set_for_tau(tau_i)
    Tj = _set_for_tau(tau_j)

    denom = np.logical_or(Ti, Tj)
    denom_count = int(np.sum(denom))
    if denom_count == 0:
        return 1.0  # perfectly orthogonal when no overlap or no overloaded TVs
    inter = np.logical_and(Ti, Tj)
    return float(1.0 - (float(np.sum(inter)) / float(denom_count)))


def slack_profile(
    t_G: int,
    tau_row_to_bins: Mapping[int, int],
    slack_per_bin_matrix: np.ndarray,
    window_bins: Sequence[int],
) -> np.ndarray:
    """
    S_G(Δ) = Slack_G(t_G + Δ) over a symmetric/selected window of Δ values.
    """
    V, T = slack_per_bin_matrix.shape
    tau = np.zeros(V, dtype=np.int32)
    for r, off in tau_row_to_bins.items():
        if 0 <= int(r) < V:
            tau[int(r)] = int(off)
    out = np.zeros(len(window_bins), dtype=np.float64)
    rows = np.arange(V, dtype=np.int32)
    for k, d in enumerate(window_bins):
        tt = np.clip(int(t_G) + int(d) + tau, 0, T - 1)
        vals = slack_per_bin_matrix[rows, tt]
        out[k] = float(np.min(vals)) if vals.size > 0 else 0.0
    return out


def slack_corr(profile_i: np.ndarray, profile_j: np.ndarray) -> float:
    """Pearson correlation between two same-length profiles; returns 0.0 if degenerate."""
    a = np.asarray(profile_i, dtype=np.float64)
    b = np.asarray(profile_j, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    da = a - a.mean()
    db = b - b.mean()
    sa = float(np.sqrt(np.sum(da * da)))
    sb = float(np.sqrt(np.sum(db * db)))
    if sa <= 0 or sb <= 0:
        return 0.0
    return float(np.sum(da * db) / (sa * sb))


def price_gap(vGi: float, vGj: float, eps: float = 1e-6) -> float:
    """
    PriceGap_{ij} = |vGi − vGj| / (vGi + vGj + ε). Returns 0 if both zero.
    """
    num = abs(float(vGi) - float(vGj))
    den = float(vGi) + float(vGj) + float(eps)
    if den <= 0:
        return 0.0
    return float(num / den)

