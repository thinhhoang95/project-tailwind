from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import Hotspot, HyperParams

# --- Rev1 helpers and pricing (feat_eng_rev1.md) ---
def _aligned_indices(
    hotspot_row: int,
    hotspot_bin: int,
    t_G: int,
    tau_row_to_bins: Mapping[int, int],
    V: int,
    T: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build alignment helpers restricted to rows touched by G (present in tau_row_to_bins):
      - tau: int32 [V] with provided offsets on touched rows, 0 elsewhere
      - touched_rows: bool [V] mask where row is in tau_row_to_bins
      - t_idx_hotspot: int32 [V] with indices hotspot_bin + (tau - tau_ref) clamped
      - t_idx_control: int32 [V] with indices t_G + tau clamped

    This aligns each row's relevant time for hotspot-based effects (ripple) and
    control-based presence checks.
    """
    tau = np.zeros(int(V), dtype=np.int32)
    touched_rows = np.zeros(int(V), dtype=np.bool_)
    for r, off in tau_row_to_bins.items():
        r_int = int(r)
        if 0 <= r_int < int(V):
            tau[r_int] = int(off)
            touched_rows[r_int] = True

    tau_ref = int(tau[int(hotspot_row)]) if 0 <= int(hotspot_row) < int(V) else 0
    rel_tau = tau - int(tau_ref)
    t_idx_hotspot = np.clip(int(hotspot_bin) + rel_tau, 0, int(T) - 1)
    t_idx_control = np.clip(int(t_G) + tau, 0, int(T) - 1)
    return tau, touched_rows, t_idx_hotspot, t_idx_control


def _exceedance_from_slack(slack_vals: np.ndarray) -> np.ndarray:
    """
    Derive exceedance magnitudes D = max(0, -slack). If provided slack is already
    non-negative (typical with current caches), this yields zeros, which
    gracefully reduces ω to 0/eps or 1 depending on presence.
    """
    return np.maximum(0.0, -np.asarray(slack_vals, dtype=np.float64))


def mass_weight_gH(
    xG: np.ndarray,
    t_G: int,
    hotspot_row: int,
    hotspot_bin: int,
    slack_per_bin_matrix: np.ndarray,
    *,
    eps: float = 1e-6,
    rolling_occ_by_bin: Optional[np.ndarray] = None,
    hourly_capacity_matrix: Optional[np.ndarray] = None,
    bins_per_hour: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute mass weight at the hotspot per feat_eng_rev1.md:
      - x̂_GH = xG(t_G)
      - D_H = max(0, -slack[hotspot_row, hotspot_bin])
      - g_H = x̂_GH / (x̂_GH + D_H) with guard for small denominator

    Returns (x_hat_GH, D_H, g_H).
    """
    V, T = slack_per_bin_matrix.shape
    tG = int(max(0, min(int(T) - 1, int(t_G))))
    x_hat = float(xG[int(tG)]) if xG.size > 0 else 0.0
    D_H = 0.0
    if 0 <= int(hotspot_row) < int(V) and 0 <= int(hotspot_bin) < int(T):
        if (
            rolling_occ_by_bin is not None
            and hourly_capacity_matrix is not None
            and bins_per_hour is not None
        ):
            hidx = int(int(hotspot_bin) // int(bins_per_hour))
            if hidx < 0:
                hidx = 0
            elif hidx > 23:
                hidx = 23
            try:
                roll_val = float(rolling_occ_by_bin[int(hotspot_row), int(hotspot_bin)])
                cap_val = float(hourly_capacity_matrix[int(hotspot_row), int(hidx)])
                D_H = float(max(0.0, roll_val - cap_val))
            except Exception:
                sl = float(slack_per_bin_matrix[int(hotspot_row), int(hotspot_bin)])
                D_H = float(max(0.0, -sl))
        else:
            sl = float(slack_per_bin_matrix[int(hotspot_row), int(hotspot_bin)])
            D_H = float(max(0.0, -sl))
    denom = float(x_hat + D_H)
    if denom <= float(eps):
        g = 0.0
    else:
        g = float(x_hat / denom)
    return x_hat, D_H, g


def _unit_price_matrix(
    hourly_excess_bool: np.ndarray,
    theta_mask: Optional[Mapping[Tuple[int, int], float]],
    *,
    w_sum: float,
    w_max: float,
) -> np.ndarray:
    """
    P_s(t) = (w_sum + w_max * θ_{s,t}) · 1{hourly_excess_bool[s,t]}
    Returns a dense [V, T] matrix of unit prices.
    """
    V, T = hourly_excess_bool.shape
    P = np.zeros((int(V), int(T)), dtype=np.float64)
    # Start with base for overloaded cells
    mask = hourly_excess_bool.astype(np.bool_, copy=False)
    P[mask] = float(w_sum)
    if theta_mask:
        for (r, t), w in theta_mask.items():
            r_int, t_int = int(r), int(t)
            if 0 <= r_int < int(V) and 0 <= t_int < int(T) and mask[r_int, t_int]:
                P[r_int, t_int] += float(w_max) * float(w)
    return P


def price_contrib_v_tilde(
    t_G: int,
    hotspot_row: int,
    hotspot_bin: int,
    tau_row_to_bins: Mapping[int, int],
    hourly_excess_bool: np.ndarray,
    slack_per_bin_matrix: np.ndarray,
    xG: np.ndarray,
    theta_mask: Optional[Mapping[Tuple[int, int], float]] = None,
    *,
    w_sum: float = 1.0,
    w_max: float = 1.0,
    kappa: float = 0.25,
    eps: float = 1e-6,
    verbose_debug: bool = False,
    idx_to_tv_id: Optional[Mapping[int, str]] = None,
    rolling_occ_by_bin: Optional[np.ndarray] = None,
    hourly_capacity_matrix: Optional[np.ndarray] = None,
    bins_per_hour: Optional[int] = None,
) -> float:
    """
    Main Formula for v_tilde

    The value v_tilde_{G → H} is composed of a primary term (for the hotspot sector itself) and a secondary ripple term (for other sectors affected by the flow).

    Let s* be the hotspot sector and t* be the hotspot time bin. The formula is:

    v_tilde_{G → H} = [P_s(t) · ω_s,G|H] + [κ · Σ_{s in touched by G, s ≠ s} P_s(t_hot(s)) · ω_s,G|H]
    Primary Term Secondary Ripple Term

    Here's what each component means:

    P_s(t): The unit price at a sector s and time t. This is non-zero when there is congestion (an "overload").

    Formula for Price P_s(t)

    The formula for the price at a specific sector s and time bin t is:

    P_s(t) = (w_sum + w_max · θ_s,t) · 1{hourly_excess_bool[s,t]}

    ω_s,G|H: The "addressable share," which measures how much the flow G contributes to the congestion at sector s. We'll look at this in more detail below.

    κ: A parameter that weights the secondary ripple effect.

    t_hot(s): The time at sector s that is aligned with the hotspot event at t. It's calculated as t_hot(s) = t + τ_G,s - τ_G,s*, where τ_G,s is the travel time for flow G to sector s.

    Addressable Share (ω)

    The addressable share ω is key to understanding v_tilde. It's defined as:

    ω_s,G|H = min(1, x_s,G / (D_s + ε))

    x_s,G: The demand from flow G at sector s at the hotspot-aligned time, t_hot(s). It represents the amount of traffic the flow is sending through that sector at that time. This is xG[t_hot(s)] in the code.

    D_s: The demand exceedance (i.e., congestion) at sector s. It's the amount by which demand exceeds capacity. It's calculated as D_s = max(0, occupancy - capacity), which is equivalent to max(0, -slack). This is D_s_vec in the code.

    ε: A small constant (eps) to prevent division by zero.

    The code for calculating ω is at 195:201:src/parrhesia/metaopt/per_flow_features.py.

    Intuition

    In simple terms, v_tilde is high if a flow sends a significant amount of traffic to sectors that are already heavily congested.

    The addressable share ω will be close to 1 if the flow's demand (x_s,G) is a large fraction of the congestion (D_s).

    The price P is high for congested sectors.

    The total v_tilde is a sum of price * share. The primary term captures the direct impact on the hotspot, and the secondary term captures the ripple effects on other sectors the flow passes through.

    In src/parrhesia/metaopt/feats/flow_features.py, the per-bin v_tilde values are calculated and then summed up over the entire hotspot period to get the final v_tilde feature for a flow.
    """
    
    V, T = hourly_excess_bool.shape
    tau, touched, t_idx_hot, t_idx_ctl = _aligned_indices(
        hotspot_row, hotspot_bin, t_G, tau_row_to_bins, V, T
    )

    # Unit prices at all cells
    P = _unit_price_matrix(hourly_excess_bool, theta_mask, w_sum=w_sum, w_max=w_max)

    rows = np.arange(int(V), dtype=np.int32)
    touched_rows = rows[touched]

    # x_{s,G} at hotspot-aligned times (use same indices as P/D to avoid clamp mismatches)
    x_sG = np.zeros(int(V), dtype=np.float64)
    if xG.size > 0:
        x_sG = xG[np.asarray(t_idx_hot, dtype=np.int32)]

    # Exceedance magnitudes at hotspot-aligned times for all rows
    t_hot_idx = np.asarray(t_idx_hot, dtype=np.int32)
    if (
        rolling_occ_by_bin is not None
        and hourly_capacity_matrix is not None
        and bins_per_hour is not None
    ):
        hour_idx = np.clip(t_hot_idx // int(bins_per_hour), 0, 23)
        try:
            roll_vals = rolling_occ_by_bin[rows, t_hot_idx]
            cap_vals = hourly_capacity_matrix[rows, hour_idx]
            D_s_vec = np.maximum(0.0, roll_vals - cap_vals)
        except Exception:
            slack_slice = slack_per_bin_matrix[rows, t_hot_idx]
            D_s_vec = _exceedance_from_slack(slack_slice)
    else:
        slack_slice = slack_per_bin_matrix[rows, t_hot_idx]
        D_s_vec = _exceedance_from_slack(slack_slice)

    # Addressable shares ω
    denom = D_s_vec + float(eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.minimum(1.0, np.where(denom > 0.0, x_sG / denom, 0.0))
    omega = omega.astype(np.float64, copy=False)
    # Restrict to sectors touched by G
    omega[~touched] = 0.0

    # Primary term at hotspot cell
    primary = 0.0
    if 0 <= int(hotspot_row) < int(V):
        P_h = float(P[int(hotspot_row), int(hotspot_bin)])
        omega_h = float(omega[int(hotspot_row)])
        primary = P_h * omega_h

    # Secondary ripple on touched rows excluding s*
    mask_sec = touched.copy()
    if 0 <= int(hotspot_row) < int(V):
        mask_sec[int(hotspot_row)] = False
    if np.any(mask_sec):
        P_sec = P[rows[mask_sec], np.asarray(t_idx_hot[mask_sec], dtype=np.int32)]
        omega_sec = omega[mask_sec]
        secondary = float(kappa) * float(np.sum(P_sec * omega_sec))
    else:
        secondary = 0.0

    total = float(primary + secondary)

    if verbose_debug:
        try:
            # Debug info about time alignment
            print(f"\nTime alignment for flow G:")
            print(f"  t_G (phase time): {t_G}")
            print(f"  t* (hotspot bin): {hotspot_bin}")
            print(f"  τ_{{G,s*}} (travel time to hotspot): {tau_row_to_bins.get(hotspot_row, 'N/A')}")
            print(f"  Relationship: t_G = t* - τ_{{G,s*}} = {hotspot_bin} - {tau_row_to_bins.get(hotspot_row, 'N/A')} = {t_G}")
            print(f"\nTime calculations per sector:")
            print(f"  - t_hot(s) = clamp(t* + τ_{{G,s}} - τ_{{G,s*}}, 0, T-1): hotspot-aligned time used to read price/slack at row s. In other words: if there is a hotspot at time t*, then what is the relevant time bin we should read for s when traffic goes from s to the hotspot or vice versa?")
            print(f"  - t_ctl(s) = clamp(t_G + τ_{{G,s}}, 0, T-1): control-aligned time where flow's presence x_{{s,G}} is sampled")
            print(f"  - Algebraically (ignoring clamping), t_hot(s) = t* + τ_{{G,s}} - τ_{{G,s*}} = t_G + τ_{{G,s}} = t_ctl(s)")
            print(f"  - t_G itself is constant per flow and defined as t_G = t* - τ_{{G,s*}}")
            if not np.array_equal(t_idx_hot, t_idx_ctl):
                print("  - Warning: t_hot(s) != t_ctl(s) for some rows due to boundary clamping; indexing x_{s,G} with t_hot(s) to match P and D indices.")
            print()
            from rich.console import Console
            from rich.table import Table
            console = Console()
            table = Table(title="ṽ components (nonzero P or x)")
            table.add_column("TV")
            table.add_column("t_hot")
            table.add_column("t_ctl")
            table.add_column("roll_occ")
            table.add_column("cap_hour")
            table.add_column("P")
            table.add_column("x_{s,G}")
            table.add_column("D_s")
            table.add_column("ω")
            for r in rows[touched]:
                th = int(t_idx_hot[int(r)])
                tc = int(t_idx_ctl[int(r)])
                Pval = float(P[int(r), th])
                xval = float(x_sG[int(r)])
                Dval = float(D_s_vec[int(r)])
                oval = float(omega[int(r)])
                # Rolling-hour occupancy at (row r, bin th) if provided
                occ_val: Optional[float] = None
                if rolling_occ_by_bin is not None:
                    try:
                        occ_val = float(rolling_occ_by_bin[int(r), th])
                    except Exception:
                        occ_val = None
                # Hourly capacity for bin th if provided
                cap_val: Optional[float] = None
                if hourly_capacity_matrix is not None and bins_per_hour is not None:
                    try:
                        hour_idx = int(th // int(bins_per_hour))
                        hour_idx = 0 if hour_idx < 0 else (23 if hour_idx > 23 else hour_idx)
                        cap_val = float(hourly_capacity_matrix[int(r), hour_idx])
                    except Exception:
                        cap_val = None
                # Highlight rows that satisfy the original inclusion condition in yellow
                # Original condition: (P>0 and ω>0) or (row==hotspot_row and (P>0 or x>0))
                highlight = (Pval > 0.0 and oval > 0.0) or (
                    int(r) == int(hotspot_row) and (Pval > 0.0 or xval > 0.0)
                )
                name = idx_to_tv_id.get(int(r), str(r)) if idx_to_tv_id else str(int(r))
                table.add_row(
                    name,
                    str(th),
                    str(tc),
                    (f"{occ_val:.4f}" if occ_val is not None else "N/A"),
                    (f"{cap_val:.4f}" if cap_val is not None else "N/A"),
                    f"{Pval:.4f}",
                    f"{xval:.4f}",
                    f"{Dval:.4f}",
                    f"{oval:.4f}",
                    style=("yellow" if highlight else None),
                )
            console.print(table)
            console.print(f"Primary: {primary:.4f}; Secondary: {secondary:.4f}; Total ṽ: {total:.4f}")
        except Exception:
            pass

    return total


def score_rev1(
    t_G: int,
    hotspot_row: int,
    hotspot_bin: int,
    tau_row_to_bins: Mapping[int, int],
    hourly_excess_bool: np.ndarray,
    slack_per_bin_matrix: np.ndarray,
    params: HyperParams,
    xG: np.ndarray,
    theta_mask: Optional[Mapping[Tuple[int, int], float]] = None,
    *,
    verbose_debug: bool = False,
    idx_to_tv_id: Optional[Mapping[int, str]] = None,
    rolling_occ_by_bin: Optional[np.ndarray] = None,
    hourly_capacity_matrix: Optional[np.ndarray] = None,
    bins_per_hour: Optional[int] = None,
) -> float:
    """
    Rev1 matched-filter score:
      Score_rev1 = α · g_H(x̂_GH) · ṽ_{G→H} − β · ρ_{G→H}

    with:
      - g_H derived from xG and hotspot deficit D_H
      - ṽ from contribution-weighted price using ω shares
      - ρ unchanged from legacy
    """
    x_hat, D_H, gH = mass_weight_gH(
        xG,
        int(t_G),
        int(hotspot_row),
        int(hotspot_bin),
        slack_per_bin_matrix,
        eps=params.eps,
        rolling_occ_by_bin=rolling_occ_by_bin,
        hourly_capacity_matrix=hourly_capacity_matrix,
        bins_per_hour=bins_per_hour,
    )
    v_tilde = price_contrib_v_tilde(
        int(t_G),
        int(hotspot_row),
        int(hotspot_bin),
        tau_row_to_bins,
        hourly_excess_bool,
        slack_per_bin_matrix,
        xG,
        theta_mask=theta_mask,
        w_sum=params.w_sum,
        w_max=params.w_max,
        kappa=params.kappa,
        eps=params.eps,
        verbose_debug=verbose_debug,
        idx_to_tv_id=idx_to_tv_id,
        rolling_occ_by_bin=rolling_occ_by_bin,
        hourly_capacity_matrix=hourly_capacity_matrix,
        bins_per_hour=bins_per_hour,
    )
    rho = slack_penalty(
        int(t_G),
        tau_row_to_bins,
        slack_per_bin_matrix,
        S0=params.S0,
        xG=xG,
        S0_mode=getattr(params, "S0_mode", "x_at_argmin"),
        verbose_debug=verbose_debug,
        idx_to_tv_id=idx_to_tv_id,
        rolling_occ_by_bin=rolling_occ_by_bin,
        hourly_capacity_matrix=hourly_capacity_matrix,
        bins_per_hour=bins_per_hour,
    )
    return float(params.alpha) * float(gH) * float(v_tilde) - float(params.beta) * float(rho)


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
    *,
    rolling_occ_by_bin: Optional[np.ndarray] = None,
    hourly_capacity_matrix: Optional[np.ndarray] = None,
    bins_per_hour: Optional[int] = None,
) -> float:
    """
    Slack_G(t) = min_{s ∈ touched} s_s(t + τ_{G,s}).

    When rolling-hour occupancy and hourly capacity matrices are provided, the per-TV
    slack slice is derived from ``capacity − occupancy`` so negative slacks are
    preserved. Otherwise the cached ``slack_per_bin_matrix`` is used (which is
    non-negative by construction).
    """
    V, T = slack_per_bin_matrix.shape
    tau = np.zeros(int(V), dtype=np.int32)
    touched = np.zeros(int(V), dtype=np.bool_)
    for r, off in tau_row_to_bins.items():
        r_int = int(r)
        if 0 <= r_int < int(V):
            tau[r_int] = int(off)
            touched[r_int] = True

    if not np.any(touched):
        return 0.0

    t_idx_all = np.clip(int(t) + tau, 0, int(T) - 1)
    rows_all = np.arange(int(V), dtype=np.int32)
    rows = rows_all[touched]
    t_idx = t_idx_all[touched]

    try:
        if (
            rolling_occ_by_bin is not None
            and hourly_capacity_matrix is not None
            and bins_per_hour is not None
        ):
            hour_idx = np.clip(t_idx // int(bins_per_hour), 0, 23)
            roll_vals = rolling_occ_by_bin[rows, t_idx]
            cap_vals = hourly_capacity_matrix[rows, hour_idx]
            vals = cap_vals - roll_vals
        else:
            vals = slack_per_bin_matrix[rows, t_idx]
    except Exception:
        vals = slack_per_bin_matrix[rows, t_idx]

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
    S0: Optional[float] = None,
    *,
    xG: Optional[np.ndarray] = None,
    S0_mode: str = "x_at_argmin",
    verbose_debug: bool = True,
    idx_to_tv_id: Optional[Mapping[int, str]] = None,
    rolling_occ_by_bin: Optional[np.ndarray] = None,
    hourly_capacity_matrix: Optional[np.ndarray] = None,
    bins_per_hour: Optional[int] = None,
) -> float:
    """
    ρ_{G→H} = [1 − Slack_G(t_G)/S0]_+

    Dynamic S0 modes (default = "x_at_argmin"):
      - "x_at_argmin": S0 = x_G(t̂) where t̂ corresponds to the row/time attaining min slack in Slack_G(t_G).
      - "x_at_control": S0 = x_G(t_G).
      - "constant": use provided S0 as-is.

    If the chosen mode requires xG but xG is None or empty, falls back to the constant S0.

    Notes
    - Rows considered in Slack_G(t_G) are restricted to those "touched" by the flow
      (rows present in τ) to align with the domain used by ṽ components.
    - When rolling-hour occupancy and hourly capacity matrices are provided, per-row
      slack is derived as (capacity − occupancy) at the aligned time; otherwise the
      cached slack_per_bin_matrix is used.

    Caveats:
    1. You could have S_eff = 0 for x_at_argmin mode
    - Slack selection is over “touched” rows only, i.e., rows present in τ for the flow. But it does not require the flow to have volume at that aligned time.
    - So a flow can “touch” a TV, yet at the argmin-slack aligned time t̂ its flow-level demand xG[t̂] is 0. With S0_mode="x_at_argmin", that makes S0=0 → ρ=0 by design. This is why you see a TVTW selected for slack while the contribution is zero.

    """
    # Compute aligned indices and per-row slack at t_G + τ, then Slack_G(t_G)
    # Restrict to rows touched by the flow (present in τ), to align with ṽ components

    if S0_mode != "constant" and (S0 != None or S0 != 0.0):
        import warnings 
        warnings.warn(f"[per_flow_features/slack_penalty] S0_mode is not 'constant' but S0 is not None or 0.0. S0 will be ignored. S0_mode: {S0_mode}, S0: {S0}")

    V, T = slack_per_bin_matrix.shape
    tau = np.zeros(int(V), dtype=np.int32)
    touched = np.zeros(int(V), dtype=np.bool_)
    for r, off in tau_row_to_bins.items():
        r_int = int(r)
        if 0 <= r_int < int(V):
            tau[r_int] = int(off)
            touched[r_int] = True

    if not np.any(touched):
        return 0.0

    t_idx_vec_all = np.clip(int(t_G) + tau, 0, int(T) - 1)
    rows_all = np.arange(int(V), dtype=np.int32)
    rows = rows_all[touched]
    t_idx_vec = t_idx_vec_all[touched]

    # Prefer deriving slack from rolling occupancy and hourly capacity if provided
    # so that Slack_G(t) basis matches the exceedance basis used in ṽ.
    try:
        if (
            rolling_occ_by_bin is not None
            and hourly_capacity_matrix is not None
            and bins_per_hour is not None
        ):
            hour_idx = np.clip(t_idx_vec // int(bins_per_hour), 0, 23)
            roll_vals = rolling_occ_by_bin[rows, t_idx_vec]
            cap_vals = hourly_capacity_matrix[rows, hour_idx]
            slack_slice = cap_vals - roll_vals
        else:
            slack_slice = slack_per_bin_matrix[rows, t_idx_vec]
    except Exception:
        # Fallback to cached slack if any indexing fails
        slack_slice = slack_per_bin_matrix[rows, t_idx_vec]

    if slack_slice.size == 0:
        return 0.0
    local_idx = int(np.argmin(slack_slice))
    r_hat = int(rows[int(local_idx)])
    t_hat = int(t_idx_vec[int(local_idx)])
    s = float(slack_slice[int(local_idx)])

    # Decide normalization S0 according to mode
    S0_eff = float(S0) if S0 is not None else 0.0
    mode = str(S0_mode or "").strip().lower()
    if mode == "x_at_control":
        if xG is not None and xG.size > 0:
            t_idx = int(max(0, min(len(xG) - 1, int(t_G))))
            S0_eff = float(xG[int(t_idx)])
    elif mode == "x_at_argmin":
        if xG is not None and xG.size > 0:
            # Guard index against xG length
            # print(f"[per_flow_features/slack_penalty] xG.size: {xG.size}, t_hat: {t_hat}")
            if 0 <= int(t_hat) < int(xG.size):
                S0_eff = float(xG[int(t_hat)]) 
                # print(f"[per_flow_features/slack_penalty] S0_eff: {S0_eff}")
    else:
        # mode == "constant" or unrecognized → keep provided S0
        pass

    if S0_eff <= 0.0:
        if verbose_debug:
            try:
                tv_name = (
                    idx_to_tv_id.get(int(r_hat), str(int(r_hat)))
                    if idx_to_tv_id is not None
                    else str(int(r_hat))
                )
                from rich.console import Console
                console = Console()
                from rich.table import Table
                table = Table(title="Slack Penalty (S0_eff <= 0)")
                table.add_column("Parameter", style="cyan")
                table.add_column("Value", style="yellow")
                table.add_row("t_G", str(int(t_G)))
                table.add_row("argmin TV", f"{tv_name} (row {int(r_hat)})")
                table.add_row("t̂", str(int(t_hat)))
                table.add_row("Slack_G", f"{float(s):.4f}")
                table.add_row("S0_eff", f"{float(S0_eff):.4f}")
                console.print(table)
            except Exception:
                print(f"[per_flow_features/slack_penalty] Error in slack_penalty: {e}")
                pass
        return 0.0

    rho = float(max(0.0, 1.0 - float(s) / float(S0_eff)))

    if verbose_debug:
        try:
            tv_name = (
                idx_to_tv_id.get(int(r_hat), str(int(r_hat)))
                if idx_to_tv_id is not None
                else str(int(r_hat))
            )
            occ_val = None
            if rolling_occ_by_bin is not None:
                try:
                    occ_val = float(rolling_occ_by_bin[int(r_hat), int(t_hat)])
                except Exception:
                    occ_val = None
            cap_val = None
            if hourly_capacity_matrix is not None and bins_per_hour is not None and int(bins_per_hour) > 0:
                try:
                    hidx = int(int(t_hat) // int(bins_per_hour))
                    hidx = 0 if hidx < 0 else (23 if hidx > 23 else hidx)
                    cap_val = float(hourly_capacity_matrix[int(r_hat), int(hidx)])
                except Exception:
                    cap_val = None
            from rich.console import Console
            from rich.table import Table
            console = Console()
            table = Table(title="Slack Penalty Debug")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="yellow")
            table.add_row("t_G", str(int(t_G)))
            table.add_row("argmin TV", f"{tv_name} (row {int(r_hat)})")
            table.add_row("t̂", str(int(t_hat)))
            table.add_row("Slack_G", f"{float(s):.4f}")
            table.add_row("S0_mode", str(mode))
            table.add_row("S0_eff", f"{float(S0_eff):.4f}")
            table.add_row("roll_occ", ('N/A' if occ_val is None else f'{occ_val:.4f}'))
            table.add_row("hour_cap", ('N/A' if cap_val is None else f'{cap_val:.4f}'))
            table.add_row("rho", f"{rho:.4f}")
            console.print(table)
        except Exception:
            pass

    return rho


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
    # Rev1 feature flag dispatch
    if getattr(params, "use_rev1", False) and xG is not None:
        return score_rev1(
            t_G=t_G,
            hotspot_row=hotspot_row,
            hotspot_bin=hotspot_bin,
            tau_row_to_bins=tau_row_to_bins,
            hourly_excess_bool=hourly_excess_bool,
            slack_per_bin_matrix=slack_per_bin_matrix,
            params=params,
            xG=xG,
            theta_mask=theta_mask,
        )
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
    rho = slack_penalty(
        int(t_G),
        tau_row_to_bins,
        slack_per_bin_matrix,
        S0=params.S0,
        xG=xG,
        S0_mode=getattr(params, "S0_mode", "x_at_argmin"),
        verbose_debug=False,
        idx_to_tv_id=None,
        rolling_occ_by_bin=None,
        hourly_capacity_matrix=None,
        bins_per_hour=None,
    )
    return float(params.alpha) * float(a) * float(v) - float(params.beta) * float(rho)

# Back-compatibility aliases for legacy implementations
price_kernel_vG_legacy = price_kernel_vG
price_to_hotspot_vGH_legacy = price_to_hotspot_vGH
score_legacy = score
