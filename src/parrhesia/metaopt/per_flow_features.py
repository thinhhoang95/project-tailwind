from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import Hotspot, HyperParams


def phase_time(
    hotspot_row: Optional[int],
    hotspot: Hotspot,
    tau_row_to_bins: Optional[Mapping[int, int]],
    T: int,
) -> int:
    """
    Compute t_G = t* − τ_{G,s*} for a flow controlled at `control_tv_row`.

    Parameters
    - hotspot_row: row index of the hotspot TV s* (None => 0 shift)
    - hotspot: (s*, t*) in ids/bins; we use only t*
    - tau_row_to_bins: mapping row -> offset bins from control row to that row
    - T: number of bins per TV
    """
    t_star = int(hotspot.bin)
    if hotspot_row is None or tau_row_to_bins is None:
        return max(0, min(int(T - 1), t_star))
    # Offset from control to hotspot row: τ_{G,s*}
    tau_ctrl_to_hstar = int(tau_row_to_bins.get(int(hotspot_row), 0))
    tG = int(t_star) - int(tau_ctrl_to_hstar)
    return max(0, min(int(T - 1), tG))


def _gather_bool_at_offsets(
    bool_matrix: np.ndarray,
    tau_row_to_bins: Mapping[int, int],
    t_ref: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each row s, compute index t_s = clamp(t_ref + τ_{G,s}, [0..T-1]) and gather bool_matrix[s, t_s].
    Returns (mask_overloaded[rows], rows_indices).
    """
    V, T = bool_matrix.shape
    rows = np.arange(V, dtype=np.int32)
    tau = np.zeros(V, dtype=np.int32)
    for r, off in tau_row_to_bins.items():
        if 0 <= int(r) < V:
            tau[int(r)] = int(off)
    t_idx = np.clip(int(t_ref) + tau, 0, T - 1)
    mask = bool_matrix[rows, t_idx]
    return mask, rows


def price_kernel_vG(
    t_G: int,
    tau_row_to_bins: Mapping[int, int],
    hourly_excess_bool: np.ndarray,
    theta_mask: Optional[Mapping[Tuple[int, int], float]] = None,
    *,
    w_sum: float = 1.0,
    w_max: float = 1.0,
    verbose_debug: bool = False,
    idx_to_tv_id: Optional[Mapping[int, str]] = None,
) -> float:
    """
    v_G(t_G) = sum_s [w_sum + w_max θ_{s, t_G + τ_{G,s}}] · 1{overload(s, t_G+τ)}.
    """
    if verbose_debug:
        print(f"\n--- Computing price_kernel_vG for t_G={t_G} ---")
    overloaded, rows = _gather_bool_at_offsets(hourly_excess_bool, tau_row_to_bins, int(t_G))
    if not np.any(overloaded):
        if verbose_debug:
            print("No overloaded TVs at phase-shifted times. v_G = 0.0")
        return 0.0
    V, T = hourly_excess_bool.shape

    overloaded_rows_indices = rows[overloaded]

    # Build gather indices per row to get t_s = t_G + tau_s
    tau = np.zeros(V, dtype=np.int32)
    for r, off in tau_row_to_bins.items():
        if 0 <= int(r) < V:
            tau[int(r)] = int(off)
    t_idx = np.clip(int(t_G) + tau, 0, T - 1)

    if verbose_debug:
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        console.print(f"Found {len(overloaded_rows_indices)} overloaded TVs at their phase-shifted times:")
        
        if len(overloaded_rows_indices) > 0:
            table = Table(title="Overloaded TVs")
            table.add_column("TV ID", style="cyan")
            table.add_column("Bin", style="magenta")
            
            if idx_to_tv_id:
                for r_idx in overloaded_rows_indices:
                    tv_id = idx_to_tv_id.get(int(r_idx), f'Row {r_idx}')
                    bin_val = str(t_idx[int(r_idx)])
                    table.add_row(tv_id, bin_val)
            else:
                # Fallback to original if no map is provided
                for r_idx in overloaded_rows_indices:
                    tv_id = f'Row {r_idx}'
                    bin_val = str(t_idx[int(r_idx)])
                    table.add_row(tv_id, bin_val)
            
            console.print(table)

    # Sum base weights over overloaded rows
    base = float(w_sum) * float(np.sum(overloaded))
    if verbose_debug:
        print(f"Base price component: w_sum * num_overloaded = {w_sum} * {np.sum(overloaded)} = {base}")

    theta_sum = 0.0
    if theta_mask:
        if verbose_debug:
            print("Theta mask component calculation:")
        for r in overloaded_rows_indices:
            r_int = int(r)
            t_s = int(t_idx[r_int])
            w = theta_mask.get((r_int, t_s))
            if w is not None:
                theta_sum += float(w)
                if verbose_debug:
                    print(
                        f"  - TV row {r_int}: t_s = t_G + τ_{{G,s={r_int}}} = {t_G} + {tau[r_int]} = {t_s}, "
                        f"θ({r_int}, {t_s}) = {w:.4f}. Cumulative theta_sum = {theta_sum:.4f}"
                    )
            elif verbose_debug:
                print(f"  - TV row {r_int}: t_s = {t_s}, θ({r_int}, {t_s}) not in mask.")

    total_price = float(base + float(w_max) * float(theta_sum))
    if verbose_debug:
        print(f"Theta sum component: w_max * theta_sum = {w_max} * {theta_sum} = {float(w_max) * float(theta_sum)}")
        print(f"Total v_G = base + theta_component = {base} + {float(w_max) * float(theta_sum)} = {total_price}")
        print("--- End price_kernel_vG ---")

    return total_price


def price_to_hotspot_vGH(
    hotspot_row: int,
    hotspot_bin: int,
    tau_row_to_bins: Mapping[int, int],
    hourly_excess_bool: np.ndarray,
    theta_mask: Optional[Mapping[Tuple[int, int], float]] = None,
    *,
    w_sum: float = 1.0,
    w_max: float = 1.0,
    kappa: float = 0.25,
) -> float:
    """
    v_{G→H} = [w_sum + w_max θ_{s*,t*}] 1{o_{s*}(t*)>0} + κ ∑_{s≠s*} [w_sum + w_max θ_{s, t* + τ_{G,s} − τ_{G,s*}}] 1{o_s(·)>0}
    """
    V, T = hourly_excess_bool.shape
    # Primary term at hotspot cell (s*, t*)
    primary = 0.0
    if 0 <= int(hotspot_row) < V and 0 <= int(hotspot_bin) < T and bool(hourly_excess_bool[int(hotspot_row), int(hotspot_bin)]):
        theta = 0.0
        if theta_mask:
            theta = float(theta_mask.get((int(hotspot_row), int(hotspot_bin)), 0.0))
        primary = float(w_sum) + float(w_max) * float(theta)

    # Secondary ripple terms
    # Build τ relative to hotspot row: τ_{G,s} − τ_{G,s*}
    tau = np.zeros(V, dtype=np.int32)
    for r, off in tau_row_to_bins.items():
        if 0 <= int(r) < V:
            tau[int(r)] = int(off)
    tau_ref = int(tau[int(hotspot_row)]) if 0 <= int(hotspot_row) < V else 0
    rel_tau = tau - int(tau_ref)
    t_idx = np.clip(int(hotspot_bin) + rel_tau, 0, T - 1)
    rows = np.arange(V, dtype=np.int32)
    mask = hourly_excess_bool[rows, t_idx]
    # Exclude the hotspot row itself
    mask[int(hotspot_row)] = False

    base_sum = float(w_sum) * float(np.sum(mask))
    theta_sum = 0.0
    if theta_mask:
        for r in rows[mask]:
            w = theta_mask.get((int(r), int(t_idx[int(r)])))
            if w is not None:
                theta_sum += float(w)
    return float(primary + float(kappa) * (base_sum + float(w_max) * float(theta_sum)))


def slack_G_at(
    t: int,
    tau_row_to_bins: Mapping[int, int],
    slack_per_bin_matrix: np.ndarray,
) -> float:
    """
    Slack_G(t) = min_s s_s(t + τ_{G,s}), where s_s(·) drawn from per-bin slack matrix [V, T].
    """
    V, T = slack_per_bin_matrix.shape
    tau = np.zeros(V, dtype=np.int32)
    for r, off in tau_row_to_bins.items():
        if 0 <= int(r) < V:
            tau[int(r)] = int(off)
    t_idx = np.clip(int(t) + tau, 0, T - 1)
    vals = slack_per_bin_matrix[np.arange(V, dtype=np.int32), t_idx]
    # If no rows, return 0; else min slack across affected rows
    if vals.size == 0:
        return 0.0
    return float(np.min(vals))


def eligibility_a(
    xG: np.ndarray,
    t_G: int,
    q0: float,
    gamma: float,
    soft: bool = False,
) -> float:
    """
    Hard: 1{x_G(t_G) ≥ q0}, Soft: σ(γ(x_G(t_G) − q0)) with σ(z) = 1/(1+e^{−z}).
    """
    val = float(xG[int(max(0, min(len(xG) - 1, int(t_G))))])
    if not soft:
        return 1.0 if val >= float(q0) else 0.0
    z = float(gamma) * (val - float(q0))
    # Stable logistic
    if z >= 0:
        ez = np.exp(-z)
        return float(1.0 / (1.0 + ez))
    else:
        ez = np.exp(z)
        return float(ez / (1.0 + ez))


def slack_penalty(
    t_G: int,
    tau_row_to_bins: Mapping[int, int],
    slack_per_bin_matrix: np.ndarray,
    S0: float,
) -> float:
    """
    ρ_{G→H} = [1 − Slack_G(t_G)/S0]_+
    """
    s = slack_G_at(int(t_G), tau_row_to_bins, slack_per_bin_matrix)
    if S0 <= 0:
        return 0.0
    return float(max(0.0, 1.0 - float(s) / float(S0)))


def score(
    t_G: int,
    hotspot_row: int,
    hotspot_bin: int,
    tau_row_to_bins: Mapping[int, int],
    hourly_excess_bool: np.ndarray,
    slack_per_bin_matrix: np.ndarray,
    params: HyperParams,
    xG: Optional[np.ndarray] = None,
    theta_mask: Optional[Mapping[Tuple[int, int], float]] = None,
    use_soft_eligibility: bool = False,
) -> float:
    """
    Matched-filter net score:
      Score(G | H) = α · a_{G→H} · v_{G→H} − β · ρ_{G→H} − λ_delay (delay term omitted here)
    """
    v = price_to_hotspot_vGH(
        hotspot_row=hotspot_row,
        hotspot_bin=hotspot_bin,
        tau_row_to_bins=tau_row_to_bins,
        hourly_excess_bool=hourly_excess_bool,
        theta_mask=theta_mask,
        w_sum=params.w_sum,
        w_max=params.w_max,
        kappa=params.kappa,
    )
    a = 1.0
    if xG is not None:
        a = eligibility_a(xG, int(t_G), q0=params.q0, gamma=params.gamma, soft=use_soft_eligibility)
    rho = slack_penalty(int(t_G), tau_row_to_bins, slack_per_bin_matrix, S0=params.S0)
    return float(params.alpha) * float(a) * float(v) - float(params.beta) * float(rho)
