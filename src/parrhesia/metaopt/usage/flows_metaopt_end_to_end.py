"""
End-to-end example: compute flows via API and MetaOpt features.

This mirrors docs/flows_metaopt_end_to_end.md as a runnable script. It:
- Loads shared resources (indexer, flight list, travel minutes)
- Calls the compute_flows API wrapper for a chosen hotspot cell
- Computes per-flow and pairwise features against the hotspot using parrhesia.metaopt

Adjust the hotspot TV and time as needed, and ensure your TV GeoJSON path
resolves (see the geojson_path block below).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table


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

    # 3) Choose hotspot and convert time to bin index
    hotspot_tv = "LSGL13W"
    # For 15-minute bins, 09:00–09:15 starts at bin = 9*60/15 = 36
    start_h, start_m = 9, 0
    bin_size = indexer.time_bin_minutes
    hotspot_bin = (start_h * 60 + start_m) // bin_size

    # 4) Compute flows via the API wrapper
    from parrhesia.api.flows import compute_flows

    flows_payload: Dict[str, Any] = compute_flows(
        tvs=[hotspot_tv],
        timebins=[hotspot_bin],
        direction_opts={
            "mode": "coord_cosine",
            "tv_centroids": res.tv_centroids,
        },
    )
    print(
        f"Found {len(flows_payload['flows'])} flows for {hotspot_tv} @ bin {hotspot_bin}"
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
    table = Table(title=f"Capacities at bin {hotspot_bin}")
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
        capacity = capacities_by_tv[tv][hotspot_bin]
        table.add_row(tv, f"{capacity:.2f}")

    hotspot_capacity = capacities_by_tv[hotspot_tv][hotspot_bin]
    table.add_row(hotspot_tv, f"{hotspot_capacity:.2f}", style="bold magenta")
    console.print(table)
    

    caches = build_base_caches(flight_list, capacities_by_tv, indexer)
    H_bool = caches["hourly_excess_bool"]
    S_mat = caches["slack_per_bin_matrix"]
    T = indexer.num_time_bins
    row_map = flight_list.tv_id_to_idx
    idx_to_tv_id = {idx: tv for tv, idx in row_map.items()}

    # 6) Convert travel minutes to bin offsets
    bin_offsets = minutes_to_bin_offsets(
        travel_minutes_map, time_bin_minutes=indexer.time_bin_minutes
    )

    # 7) Rebuild x_G and τ maps from compute_flows payload
    hot = Hotspot(tv_id=hotspot_tv, bin=hotspot_bin)
    flows = flows_payload["flows"]

    xG_map: Dict[int, np.ndarray] = {}
    tau_map: Dict[int, Dict[int, int]] = {}
    tG_map: Dict[int, int] = {}
    ctrl_by_flow: Dict[int, str] = {}

    print("\n--- Detailed t_G Calculation ---")
    hotspot_tv_row = row_map.get(hot.tv_id)

    for fobj in flows:
        fid = int(fobj["flow_id"])  # flow id
        ctrl = fobj.get("controlled_volume")
        if ctrl:
            ctrl_by_flow[fid] = str(ctrl)
        # Rebuild x_G from flights' requested_bin
        flights_specs = fobj.get("flights", [])
        x = np.zeros(T, dtype=float)
        for sp in flights_specs:
            rb = int(sp.get("requested_bin", 0))
            if 0 <= rb < T:
                x[rb] += 1.0
        xG_map[fid] = x

        # τ map: control TV to all TVs (row -> Δbins)
        tau = flow_offsets_from_ctrl(ctrl, row_map, bin_offsets) or {}
        tau_map[fid] = tau

        # Phase alignment t_G
        ctrl_row = None
        if ctrl is not None and ctrl in row_map:
            ctrl_row = int(row_map[ctrl])
        tG_map[fid] = phase_time(ctrl_row, hot, tau, T)

        # Sanity check: print detailed breakdown of t_G
        tG = tG_map[fid]
        tau_G_s_star = tau.get(hotspot_tv_row) if hotspot_tv_row is not None else None

        print("-" * 60)
        print(f"Calculating phase alignment time t_G for flow {fid}:")
        print("Formula: t_G = t* - τ_{G,s*}")
        print("\n  Components:")
        print("  - Hotspot H = (s*, t*):")
        print(
            f"    - s* (hotspot_tv): '{hot.tv_id}'"
            + (f" (row: {hotspot_tv_row})" if hotspot_tv_row is not None else " (TV not found in row_map)")
        )
        print(f"    - t* (hotspot_bin): {hot.bin}")

        print(f"\n  - Flow G = {fid}:")
        print(
            f"    - Control volume s_ctrl: '{ctrl}'"
            + (f" (row: {ctrl_row})" if ctrl_row is not None else " (Control TV not found or not specified)")
        )

        print("\n  - Travel time τ_{G,s*}:")
        if tau_G_s_star is not None:
            print(f"    - Offset from s_ctrl to s* in time bins: {tau_G_s_star}")
            print("\n  Calculation:")
            print(f"  t_G = {hot.bin} - {tau_G_s_star} = {tG}")
        else:
            print("    - Could not determine travel time offset.")
            print(f"\n  Final t_G value: {tG}")
        print("-" * 60)

    # 8) Per-flow features wrt hotspot
    params = HyperParams(S0=5.0)  # tune as needed
    h_row = int(row_map[hotspot_tv])

    per_flow_feats: Dict[int, Dict[str, float]] = {}
    for fid in xG_map.keys():
        tG = int(tG_map[fid])
        tau = tau_map[fid]
        xG = xG_map[fid]
        # Price kernels
        vG = price_kernel_vG(
            tG,
            tau,
            H_bool,
            theta_mask=None,
            w_sum=params.w_sum,
            w_max=params.w_max,
            verbose_debug=True,
            idx_to_tv_id=idx_to_tv_id,
        )
        vGH = price_to_hotspot_vGH(
            h_row,
            hot.bin,
            tau,
            H_bool,
            theta_mask=None,
            w_sum=params.w_sum,
            w_max=params.w_max,
            kappa=params.kappa,
        )
        # Slack + eligibility
        sG = slack_G_at(tG, tau, S_mat)
        aG = eligibility_a(xG, tG, q0=params.q0, gamma=params.gamma, soft=True)
        rho = slack_penalty(tG, tau, S_mat, S0=params.S0)
        # Net score (α a v_{G→H} − β ρ)
        scr = score_flow(
            t_G=tG,
            hotspot_row=h_row,
            hotspot_bin=hot.bin,
            tau_row_to_bins=tau,
            hourly_excess_bool=H_bool,
            slack_per_bin_matrix=S_mat,
            params=params,
            xG=xG,
            theta_mask=None,
            use_soft_eligibility=True,
        )
        per_flow_feats[fid] = {
            "tG": float(tG),
            "vG": float(vG),
            "vGH": float(vGH),
            "slackG": float(sG),
            "eligibility": float(aG),
            "rho": float(rho),
            "score": float(scr),
        }

    print("Per-flow features:")
    for fid, d in per_flow_feats.items():
        print(fid, d)

    # 9) Pairwise features across flows
    pairwise: Dict[Tuple[int, int], Dict[str, float]] = {}
    flow_ids = list(xG_map.keys())
    window = list(range(-params.window_left, params.window_right + 1))
    for i in range(len(flow_ids)):
        for j in range(i + 1, len(flow_ids)):
            fi, fj = flow_ids[i], flow_ids[j]
            # Build window around tGi in absolute bins
            W = [int(tG_map[fi]) + int(k) for k in window]
            # Align xGj to fi’s phase by shifting
            d = int(tG_map[fi] - tG_map[fj])
            xj_shift = np.zeros_like(xG_map[fj])
            if d >= 0:
                xj_shift[d:] = xG_map[fj][: T - d]
            else:
                xj_shift[: T + d] = xG_map[fj][-d:]
            ov = temporal_overlap(xG_map[fi], xj_shift, window_bins=W)
            orth = offset_orthogonality(h_row, hot.bin, tau_map[fi], tau_map[fj], H_bool)
            Si = slack_profile(tG_map[fi], tau_map[fi], S_mat, window)
            Sj = slack_profile(tG_map[fj], tau_map[fj], S_mat, window)
            sc = slack_corr(Si, Sj)
            pg = price_gap(per_flow_feats[fi]["vG"], per_flow_feats[fj]["vG"], eps=params.eps)
            pairwise[(fi, fj)] = {
                "overlap": float(ov),
                "orth": float(orth),
                "slack_corr": float(sc),
                "price_gap": float(pg),
            }

    print("Pairwise features:")
    for k, d in pairwise.items():
        print(k, d)


if __name__ == "__main__":
    main()

