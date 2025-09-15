"""
End-to-end example: compute flows via API and MetaOpt features.

This mirrors docs/flows_metaopt_end_to_end.md as a runnable script. It:
- Loads shared resources (indexer, flight list, travel minutes)
- Calls the compute_flows API wrapper for a chosen hotspot cell
- Computes per-flow and pairwise features against the hotspot using parrhesia.metaopt

Slack_G+k features (k ∈ {0, 15, 30, 45} minutes) are summed over the selected
time bins for each flow, matching Slack_G aggregation semantics.

Adjust the hotspot TV and time as needed, and ensure your TV GeoJSON path
resolves (see the geojson_path block below).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, List

import numpy as np
from rich.console import Console
from rich.table import Table
import argparse


def main() -> None:
    # 1) Load shared resources (indexer, flight list, travel minutes)
    from server_tailwind.core.resources import get_resources

    res = get_resources().preload_all()
    indexer = res.indexer
    flight_list = res.flight_list
    travel_minutes_map = res.travel_minutes(speed_kts=475.0)

    # 2) Ensure compute_flows uses the same resources
    from parrhesia.api.resources import set_global_resources

    set_global_resources(indexer, flight_list)

    # 3) Parse CLI: hotspot and time range [start, end) at 15-min granularity
    parser = argparse.ArgumentParser(description="Flows + MetaOpt end-to-end over a time range")
    parser.add_argument("--hotspot", default="LSGL13W", help="Hotspot TV ID (e.g., LSGL13W)")
    parser.add_argument("--start", default="11:15", help="Range start time HH:MM (inclusive)")
    parser.add_argument("--end", default="11:45", help="Range end time HH:MM (exclusive); defaults to start+1 bin")
    parser.add_argument("--per-bin-debug", action="store_true", help="Print per-bin debug for pricing/slack")
    args = parser.parse_args()

    def _parse_hhmm(s: str) -> Tuple[int, int]:
        try:
            h, m = s.strip().split(":", 1)
            return int(h), int(m)
        except Exception:
            raise ValueError(f"Invalid time format '{s}'. Expected HH:MM")

    hotspot_tv = str(args.hotspot)
    bin_size = int(indexer.time_bin_minutes)
    start_h, start_m = _parse_hhmm(args.start)
    if start_m % bin_size != 0:
        raise ValueError(f"Start minutes must be a multiple of bin size ({bin_size})")
    start_total = start_h * 60 + start_m
    start_bin = start_total // bin_size

    if args.end is None:
        end_total = start_total + bin_size
    else:
        end_h, end_m = _parse_hhmm(args.end)
        if end_m % bin_size != 0:
            raise ValueError(f"End minutes must be a multiple of bin size ({bin_size})")
        end_total = end_h * 60 + end_m
    if not (0 <= start_total < end_total <= 24 * 60):
        raise ValueError("Time range must be within a single day and end > start.")
    end_bin = end_total // bin_size
    if end_bin > int(indexer.num_time_bins):
        raise ValueError("End time exceeds indexer's configured time bins.")
    timebins: List[int] = list(range(int(start_bin), int(end_bin)))
    if len(timebins) == 0:
        raise ValueError("Empty time bin range; check start/end values.")

    # 4) Compute flows via the API wrapper
    from parrhesia.api.flows import compute_flows

    flows_payload: Dict[str, Any] = compute_flows(
        tvs=[hotspot_tv],
        timebins=timebins,
        direction_opts={
            "mode": "coord_cosine",
            "tv_centroids": res.tv_centroids,
        },
    )
    print(
        f"Found {len(flows_payload['flows'])} flows for {hotspot_tv} @ bins {timebins[0]}–{timebins[-1]}"
    )

    console = Console()
    for flow in flows_payload["flows"]:
        num_flights = len(flow["flights"])
        table = Table(
            title=f"Flow {flow['flow_id']} (Controlled Volume: {flow['controlled_volume']}) - {num_flights} flights"
        )
        table.add_column("Flight ID")
        table.add_column("Requested Bin")
        table.add_column("Earliest Crossing Time")

        # Limit to showing 10 flights to avoid huge tables
        for flight in flow["flights"][:10]:
            table.add_row(
                flight["flight_id"],
                str(flight["requested_bin"]),
                flight["earliest_crossing_time"] or "N/A",
            )
        console.print(table)

    # 5) Build capacities and base caches for features
    from parrhesia.optim.capacity import build_bin_capacities
    from parrhesia.metaopt import (
        Hotspot,
        HyperParams,
        build_base_caches,
        minutes_to_bin_offsets,
        flow_offsets_from_ctrl,
        phase_time,
        price_kernel_vG,
        price_to_hotspot_vGH,
        # Rev1
        mass_weight_gH,
        price_contrib_v_tilde,
        score_rev1,
        slack_G_at,
        eligibility_a,
        slack_penalty,
        score as score_flow,
        temporal_overlap,
        offset_orthogonality,
        slack_profile,
        slack_corr,
        price_gap,
    )

    # Capacities from the TV GeoJSON used by your app
    # Prefer configured path from AppResources; fallback to the fallback path.
    geojson_path = None
    try:
        p = res.paths.traffic_volumes_path
        if p.exists():
            geojson_path = str(p)
        else:
            p2 = res.paths.fallback_traffic_volumes_path
            if p2.exists():
                geojson_path = str(p2)
            else:
                p3 = res.paths.fallback_traffic_volumes_path_2
                if p3.exists():
                    geojson_path = str(p3)
    except Exception:
        geojson_path = None
    if not geojson_path:
        raise RuntimeError(
            "Could not resolve a TV GeoJSON path. Update res.paths.traffic_volumes_path."
        )

    capacities_by_tv = build_bin_capacities(geojson_path, indexer)
    # Print some capacities
    table = Table(title=f"Capacities at bin {timebins[0]}")
    table.add_column("Traffic Volume")
    table.add_column("Capacity")

    # Show a few sample TVs and their capacity at the hotspot time
    # Then show the hotspot TV itself
    tvs_to_show = list(capacities_by_tv.keys())[:5]
    if hotspot_tv in tvs_to_show:
        tvs_to_show.remove(hotspot_tv)
    else:
        tvs_to_show = tvs_to_show[:4]

    for tv in tvs_to_show:
        capacity = capacities_by_tv[tv][timebins[0]]
        table.add_row(tv, f"{capacity:.2f}")

    hotspot_capacity = capacities_by_tv[hotspot_tv][timebins[0]]
    table.add_row(hotspot_tv, f"{hotspot_capacity:.2f}", style="bold magenta")
    console.print(table)
    

    caches = build_base_caches(flight_list, capacities_by_tv, indexer)
    H_bool = caches["hourly_excess_bool"]
    S_mat = caches["slack_per_bin_matrix"]
    T = indexer.num_time_bins
    row_map = flight_list.tv_id_to_idx
    idx_to_tv_id = {idx: tv for tv, idx in row_map.items()}

    # Prepare rolling-hour occupancy and hourly capacity for debug verification
    num_tvs = int(caches.get("num_tvs", len(row_map)))
    bins_per_hour = int(caches.get("bins_per_hour", max(1, 60 // int(indexer.time_bin_minutes))))
    hourly_capacity_matrix = caches.get("hourly_capacity_matrix")
    # Prefer cached rolling occupancy if present
    rolling_occ_by_bin = caches.get("rolling_occ_by_bin")

    # 6) Convert travel minutes to bin offsets
    bin_offsets = minutes_to_bin_offsets(
        travel_minutes_map, time_bin_minutes=indexer.time_bin_minutes
    )

    # 7) Rebuild x_G and τ maps from compute_flows payload
    flows = flows_payload["flows"]

    xG_map: Dict[int, np.ndarray] = {}
    tau_map: Dict[int, Dict[int, int]] = {}
    ctrl_by_flow: Dict[int, str] = {}

    print("\n--- Preparing τ maps and demand series (x_G) ---")
    hotspot_tv_row = row_map.get(hotspot_tv)

    # Feature flag to enable signed τ by relative direction vs control
    enable_signed_tau = True

    for fobj in flows:
        fid = int(fobj["flow_id"])  # flow id
        ctrl = fobj.get("controlled_volume")
        if ctrl:
            ctrl_by_flow[fid] = str(ctrl)
        # Rebuild x_G from flights' requested_bin
        flights_specs = fobj.get("flights", [])
        flow_flight_ids = [str(sp.get("flight_id")) for sp in flights_specs if sp.get("flight_id")]
        x = np.zeros(T, dtype=float)
        for sp in flights_specs:
            rb = int(sp.get("requested_bin", 0))
            if 0 <= rb < T:
                x[rb] += 1.0
        xG_map[fid] = x

        # τ map: control TV to all TVs (row -> Δbins), optionally signed per-flow
        if enable_signed_tau:
            tau = flow_offsets_from_ctrl(
                ctrl,
                row_map,
                bin_offsets,
                flow_flight_ids=flow_flight_ids,
                flight_list=flight_list,
                hotspots=[hotspot_tv],
                trim_policy="earliest_hotspot",
                direction_sign_mode="order_vs_ctrl",
                tv_centroids=res.tv_centroids,
            ) or {}
        else:
            tau = flow_offsets_from_ctrl(ctrl, row_map, bin_offsets) or {}

        # Restrict τ to TVs actually visited by the flow (plus hotspot row for alignment safety)
        visited_rows = set()
        for fid_str in flow_flight_ids:
            try:
                seq = flight_list.get_flight_tv_sequence_indices(str(fid_str))
                if not isinstance(seq, np.ndarray):
                    seq = np.asarray(seq, dtype=np.int64)
            except Exception:
                continue
            if seq.size == 0:
                continue
            # Optional trim to earliest hotspot, mirroring offsets' trim
            if hotspot_tv_row is not None:
                cut = None
                for i, v in enumerate(seq.tolist()):
                    if int(v) == int(hotspot_tv_row):
                        cut = i
                        break
                if cut is not None:
                    seq = seq[: cut + 1]
            visited_rows.update(int(v) for v in seq.tolist())

        # Keep hotspot row to preserve primary term and consistent indexing
        if hotspot_tv_row is not None:
            visited_rows.add(int(hotspot_tv_row))

        tau_filtered = {int(r): int(off) for r, off in (tau or {}).items() if int(r) in visited_rows}
        tau_map[fid] = tau_filtered

    # 8) Aggregate per-flow features over bins in [start_bin, end_bin)
    params = HyperParams(S0=5.0, S0_mode="x_at_argmin")  # default mode can be omitted
    h_row = int(row_map[hotspot_tv])

    minutes_per_bin = int(indexer.time_bin_minutes)
    slack_shift_bins = [1, 2, 3]
    slack_shift_labels = {
        shift: f"slackG+{shift * minutes_per_bin}" for shift in slack_shift_bins
    }

    per_flow_sums: Dict[int, Dict[str, float]] = {}
    # Pre-initialize sums for all flows
    for fid in xG_map.keys():
        per_flow_sums[int(fid)] = {
            "xhat_GH": 0.0,
            "D_H": 0.0,
            "g_H": 0.0,
            "v_tilde": 0.0,
            "slackG": 0.0,
            "eligibility": 0.0,
            "rho": 0.0,
            "score_rev1": 0.0,
        }
        for label in slack_shift_labels.values():
            per_flow_sums[int(fid)][label] = 0.0

    # Bin loop: compute per-flow contributions and accumulate
    for b in timebins:
        hot = Hotspot(tv_id=hotspot_tv, bin=int(b))
        # Build phase times for this bin
        tG_b: Dict[int, int] = {}
        vtilde_b: Dict[int, float] = {}
        for fid in xG_map.keys():
            tau = tau_map[int(fid)]
            xG = xG_map[int(fid)]
            tG = int(phase_time(hotspot_tv_row, hot, tau, T))
            tG_b[int(fid)] = tG

            # Rev1 mass weight and contribution-weighted price
            xhat, DH, gH = mass_weight_gH(
                xG,
                tG,
                h_row,
                hot.bin,
                S_mat,
                eps=params.eps,
                rolling_occ_by_bin=rolling_occ_by_bin,
                hourly_capacity_matrix=hourly_capacity_matrix,
                bins_per_hour=bins_per_hour,
            )
            v_tilde = price_contrib_v_tilde(
                tG,
                h_row,
                hot.bin,
                tau,
                H_bool,
                S_mat,
                xG,
                theta_mask=None,
                w_sum=params.w_sum,
                w_max=params.w_max,
                kappa=params.kappa,
                eps=params.eps,
                verbose_debug=bool(args.per_bin_debug),
                idx_to_tv_id=idx_to_tv_id,
                rolling_occ_by_bin=rolling_occ_by_bin,
                hourly_capacity_matrix=hourly_capacity_matrix,
                bins_per_hour=bins_per_hour,
            )
            vtilde_b[int(fid)] = float(v_tilde)

            # Slack + eligibility
            sG = slack_G_at(
                tG,
                tau,
                S_mat,
                rolling_occ_by_bin=rolling_occ_by_bin,
                hourly_capacity_matrix=hourly_capacity_matrix,
                bins_per_hour=bins_per_hour,
            )
            slack_shifts: Dict[int, float] = {}
            for shift in slack_shift_bins:
                tG_shift = tG + shift
                if not (0 <= tG_shift < T):
                    continue
                slack_shifts[shift] = float(
                    slack_G_at(
                        tG_shift,
                        tau,
                        S_mat,
                        rolling_occ_by_bin=rolling_occ_by_bin,
                        hourly_capacity_matrix=hourly_capacity_matrix,
                        bins_per_hour=bins_per_hour,
                    )
                )

            # Optional: per-bin debug tables for Slack_G+15/30/45, similar to Slack Penalty Debug
            if bool(args.per_bin_debug) and len(slack_shifts) > 0:
                try:
                    # Reproduce argmin details at each shifted time like slack_penalty() debug
                    V, _T = S_mat.shape
                    tau_vec = np.zeros(int(V), dtype=np.int32)
                    touched = np.zeros(int(V), dtype=np.bool_)
                    for r_row, off in tau.items():
                        r_int = int(r_row)
                        if 0 <= r_int < int(V):
                            tau_vec[r_int] = int(off)
                            touched[r_int] = True
                    if np.any(touched):
                        rows_all = np.arange(int(V), dtype=np.int32)
                        rows = rows_all[touched]
                        for shift, _val in slack_shifts.items():
                            tG_shift = int(tG + shift)
                            if not (0 <= tG_shift < T):
                                continue
                            t_idx_vec_all = np.clip(int(tG_shift) + tau_vec, 0, int(T) - 1)
                            t_idx_vec = t_idx_vec_all[touched]
                            try:
                                if (
                                    rolling_occ_by_bin is not None
                                    and hourly_capacity_matrix is not None
                                    and bins_per_hour is not None
                                ):
                                    hour_idx = np.clip(t_idx_vec // int(bins_per_hour), 0, 23)
                                    roll_vals = rolling_occ_by_bin[rows, t_idx_vec]
                                    cap_vals = hourly_capacity_matrix[rows, hour_idx]
                                    vals = cap_vals - roll_vals
                                else:
                                    vals = S_mat[rows, t_idx_vec]
                            except Exception:
                                vals = S_mat[rows, t_idx_vec]
                            if vals.size == 0:
                                continue
                            local_idx = int(np.argmin(vals))
                            r_hat = int(rows[int(local_idx)])
                            t_hat = int(t_idx_vec[int(local_idx)])
                            s_shift = float(vals[int(local_idx)])

                            # Pull friendly TV name and occupancy/capacity at t_hat for the row
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
                            if (
                                hourly_capacity_matrix is not None
                                and bins_per_hour is not None
                                and int(bins_per_hour) > 0
                            ):
                                try:
                                    hidx = int(int(t_hat) // int(bins_per_hour))
                                    hidx = 0 if hidx < 0 else (23 if hidx > 23 else hidx)
                                    cap_val = float(hourly_capacity_matrix[int(r_hat), int(hidx)])
                                except Exception:
                                    cap_val = None

                            # Print a Slack_G+Δ debug table mirroring the style of Slack Penalty Debug
                            shift_min = int(shift * minutes_per_bin)
                            from rich.table import Table as _Table
                            dbg = _Table(title=f"Slack_G+{shift_min} Debug")
                            dbg.add_column("Parameter", style="cyan")
                            dbg.add_column("Value", style="yellow")
                            dbg.add_row("t_G", str(int(tG)))
                            dbg.add_row("Δ", f"+{shift_min}")
                            dbg.add_row("t_G+Δ", str(int(tG_shift)))
                            dbg.add_row("argmin TV", f"{tv_name} (row {int(r_hat)})")
                            dbg.add_row("t̂", str(int(t_hat)))
                            dbg.add_row("Slack_G+Δ", f"{float(s_shift):.4f}")
                            dbg.add_row("roll_occ", ("N/A" if occ_val is None else f"{occ_val:.4f}"))
                            dbg.add_row("hour_cap", ("N/A" if cap_val is None else f"{cap_val:.4f}"))
                            console.print(dbg)
                except Exception:
                    pass
            aG = eligibility_a(xG, tG, q0=params.q0, gamma=params.gamma, soft=True)
            rho = slack_penalty(
                tG,
                tau,
                S_mat,
                S0=params.S0,
                xG=xG,
                S0_mode=params.S0_mode,
                verbose_debug=bool(args.per_bin_debug),
                idx_to_tv_id=idx_to_tv_id,
                rolling_occ_by_bin=rolling_occ_by_bin,
                hourly_capacity_matrix=hourly_capacity_matrix,
                bins_per_hour=bins_per_hour,
            )
            # Rev1 score (α g_H ṽ − β ρ)
            scr_rev1 = score_rev1(
                t_G=tG,
                hotspot_row=h_row,
                hotspot_bin=hot.bin,
                tau_row_to_bins=tau,
                hourly_excess_bool=H_bool,
                slack_per_bin_matrix=S_mat,
                params=params,
                xG=xG,
                theta_mask=None,
                verbose_debug=True,
                idx_to_tv_id=idx_to_tv_id,
                rolling_occ_by_bin=rolling_occ_by_bin,
                hourly_capacity_matrix=hourly_capacity_matrix,
                bins_per_hour=bins_per_hour,
            )

            acc = per_flow_sums[int(fid)]
            acc["xhat_GH"] += float(xhat)
            acc["D_H"] += float(DH)
            acc["g_H"] += float(gH)
            acc["v_tilde"] += float(v_tilde)
            acc["slackG"] += float(sG)
            for shift, val in slack_shifts.items():
                acc[slack_shift_labels[shift]] += float(val)
            acc["eligibility"] += float(aG)
            acc["rho"] += float(rho)
            acc["score_rev1"] += float(scr_rev1)

        # 9) Pairwise features for this bin; accumulate across bins
        if "pairwise" not in locals():
            pairwise: Dict[Tuple[int, int], Dict[str, float]] = {}
        flow_ids = list(xG_map.keys())
        window = list(range(-params.window_left, params.window_right + 1))
        for i in range(len(flow_ids)):
            for j in range(i + 1, len(flow_ids)):
                fi, fj = int(flow_ids[i]), int(flow_ids[j])
                # Build window around tGi in absolute bins
                W = [int(tG_b[fi]) + int(k) for k in window]
                # Align xGj to fi’s phase by shifting
                d = int(tG_b[fi] - tG_b[fj])
                xj_shift = np.zeros_like(xG_map[fj])
                if d >= 0:
                    xj_shift[d:] = xG_map[fj][: T - d]
                else:
                    xj_shift[: T + d] = xG_map[fj][-d:]
                ov = temporal_overlap(xG_map[fi], xj_shift, window_bins=W)
                orth = offset_orthogonality(h_row, hot.bin, tau_map[fi], tau_map[fj], H_bool)
                Si = slack_profile(tG_b[fi], tau_map[fi], S_mat, window)
                Sj = slack_profile(tG_b[fj], tau_map[fj], S_mat, window)
                sc = slack_corr(Si, Sj)
                pg = price_gap(vtilde_b[fi], vtilde_b[fj], eps=params.eps)
                key = (fi, fj)
                if key not in pairwise:
                    pairwise[key] = {"overlap": 0.0, "orth": 0.0, "slack_corr": 0.0, "price_gap": 0.0}
                pairwise[key]["overlap"] += float(ov)
                pairwise[key]["orth"] += float(orth)
                pairwise[key]["slack_corr"] += float(sc)
                pairwise[key]["price_gap"] += float(pg)

    if per_flow_sums:
        feature_columns = list(next(iter(per_flow_sums.values())).keys())
        per_flow_table = Table(
            title=f"Per-flow features (summed across bins {timebins[0]}–{timebins[-1]})"
        )
        per_flow_table.add_column("Flow ID", justify="left", style="bold")
        for col in feature_columns:
            per_flow_table.add_column(col, justify="right")

        for fid, metrics in per_flow_sums.items():
            per_flow_table.add_row(
                str(fid),
                *[f"{metrics[col]:.3f}" for col in feature_columns],
            )
        console.print(per_flow_table)
    else:
        console.print("No per-flow features computed.")

    if "pairwise" in locals() and pairwise:
        pairwise_table = Table(title="Pairwise features (summed across bins)")
        pairwise_table.add_column("Flow Pair", style="bold")
        pairwise_columns = list(next(iter(pairwise.values())).keys())
        for col in pairwise_columns:
            pairwise_table.add_column(col, justify="right")

        for (fi, fj), metrics in pairwise.items():
            pairwise_table.add_row(
                f"{fi} ↔ {fj}",
                *[f"{metrics[col]:.3f}" for col in pairwise_columns],
            )
        console.print(pairwise_table)
    else:
        console.print("No pairwise features computed.")


if __name__ == "__main__":
    main()
