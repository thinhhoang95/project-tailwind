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

from collections import Counter
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
    parser.add_argument("--threshold", type=float, default=0.25, help="Threshold for flow extraction")
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

    console = Console()

    # Log key inputs before invoking the API wrapper so users can trace context quickly.
    inputs_table = Table(title="Input Summary")
    inputs_table.add_column("Parameter", style="cyan")
    inputs_table.add_column("Value", style="yellow")
    end_h_display = end_total // 60
    end_m_display = end_total % 60
    inputs_table.add_row("Hotspot TV", hotspot_tv)
    inputs_table.add_row("Start (HH:MM)", f"{start_h:02d}:{start_m:02d}")
    inputs_table.add_row("End (HH:MM)", f"{end_h_display:02d}:{end_m_display:02d}")
    inputs_table.add_row("Bin Size (minutes)", str(bin_size))
    inputs_table.add_row("Bin Range", f"[{timebins[0]}, {timebins[-1] + 1})")
    inputs_table.add_row("Direction Mode", "coord_cosine")
    inputs_table.add_row("TV Centroids", "provided" if res.tv_centroids is not None else "missing")
    console.print(inputs_table)

    # 4) Compute flows via the API wrapper
    from parrhesia.api.flows import compute_flows

    flows_payload: Dict[str, Any] = compute_flows(
        tvs=[hotspot_tv],
        timebins=timebins,
        direction_opts={
            "mode": "coord_cosine",
            "tv_centroids": res.tv_centroids,
        },
        threshold=args.threshold,
    )
    print(
        f"Found {len(flows_payload['flows'])} flows for {hotspot_tv} @ bins {timebins[0]}–{timebins[-1]}"
    )

    flows = flows_payload["flows"]
    if len(flows) < 2:
        raise RuntimeError("Expected at least two flows to compare; received fewer.")

    flows_sorted = sorted(flows, key=lambda f: len(f.get("flights", [])), reverse=True)
    top_flows = flows_sorted[:2]

    # Display all flows with their flight counts in a rich table
    all_flows_table = Table(title="All Flows Overview")
    all_flows_table.add_column("Flow ID", style="bold cyan")
    all_flows_table.add_column("Control TV", style="green")
    all_flows_table.add_column("# Flights", justify="right", style="yellow")
    
    for flow in flows_sorted:
        all_flows_table.add_row(
            str(flow.get("flow_id", "?")),
            str(flow.get("controlled_volume", "?")),
            str(len(flow.get("flights", [])))
        )
    
    console.print(all_flows_table)

    # Ensure both flows have enough traffic; focus on meaningful comparisons only.
    for f in top_flows:
        if len(f.get("flights", [])) < 3:
            raise RuntimeError(
                f"Flow {f.get('flow_id')} does not have the minimum required 3 flights."
            )

    largest_flow = top_flows[0]
    anchor_flow = top_flows[1]

    def _sample_flights(flow: Dict[str, Any]) -> str:
        flights = flow.get("flights", [])
        sample = [flight.get("flight_id", "?") for flight in flights[:10]]
        return ", ".join(sample) if sample else "<none>"

    def _requested_bin_hist(flow: Dict[str, Any]) -> str:
        bins = [int(f.get("requested_bin", -1)) for f in flow.get("flights", [])]
        filtered = [b for b in bins if b >= 0]
        if not filtered:
            return "<empty>"
        counts = Counter(filtered)
        top_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{b}:{c}" for b, c in top_items[:6])

    flows_table = Table(title="Selected Top-2 Flows")
    flows_table.add_column("Role", style="bold magenta")
    flows_table.add_column("Flow ID")
    flows_table.add_column("Control TV")
    flows_table.add_column("# Flights", justify="right")
    flows_table.add_column("Sample Flight IDs")
    flows_table.add_column("Requested Bin Histogram")
    flows_table.add_row(
        "Largest",
        str(largest_flow.get("flow_id")),
        str(largest_flow.get("controlled_volume")),
        str(len(largest_flow.get("flights", []))),
        _sample_flights(largest_flow),
        _requested_bin_hist(largest_flow),
    )
    flows_table.add_row(
        "Anchor",
        str(anchor_flow.get("flow_id")),
        str(anchor_flow.get("controlled_volume")),
        str(len(anchor_flow.get("flights", []))),
        _sample_flights(anchor_flow),
        _requested_bin_hist(anchor_flow),
    )
    console.print(flows_table)

    # 5) Build capacities and base caches for features
    from parrhesia.optim.capacity import build_bin_capacities
    from parrhesia.metaopt import (
        Hotspot,
        HyperParams,
        build_base_caches,
        minutes_to_bin_offsets,
        flow_offsets_from_ctrl,
        phase_time,
        price_contrib_v_tilde,
    )
    from parrhesia.metaopt.setwise_features import compare_flow_against_flight_set

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
    bins_per_hour = int(caches.get("bins_per_hour", max(1, 60 // int(indexer.time_bin_minutes))))
    hourly_capacity_matrix = caches.get("hourly_capacity_matrix")
    # Prefer cached rolling occupancy if present
    rolling_occ_by_bin = caches.get("rolling_occ_by_bin")

    # 6) Convert travel minutes to bin offsets
    bin_offsets = minutes_to_bin_offsets(
        travel_minutes_map, time_bin_minutes=indexer.time_bin_minutes
    )

    # 7) Rebuild x_G and τ maps from compute_flows payload
    xG_map: Dict[int, np.ndarray] = {}
    tau_map: Dict[int, Dict[int, int]] = {}
    ctrl_by_flow: Dict[int, str] = {}
    flights_by_flow: Dict[int, List[str]] = {}

    print("\n--- Preparing τ maps and demand series (x_G) ---")
    hotspot_tv_row = row_map.get(hotspot_tv)

    # Feature flag to enable signed τ by relative direction vs control
    enable_signed_tau = True

    for fobj in top_flows:
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
        flights_by_flow[int(fid)] = list(flow_flight_ids)

        # τ map: control TV to all TVs (row -> Δbins), optionally signed per-flow
        if enable_signed_tau:
            tau = flow_offsets_from_ctrl(
                ctrl,
                row_map,
                bin_offsets,
                flow_flight_ids=flow_flight_ids,
                flight_list=flight_list,
                hotspots=None,
                trim_policy=None,
                direction_sign_mode="order_vs_ctrl",
                tv_centroids=res.tv_centroids,
            ) or {}
        else:
            tau = flow_offsets_from_ctrl(ctrl, row_map, bin_offsets) or {}

        # Restrict τ to TVs actually visited by the flow anywhere it can touch (plus hotspot row for alignment safety)
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
            # Do not trim; include full sequence of visited TVs
            visited_rows.update(int(v) for v in seq.tolist())

        # Keep hotspot row to preserve primary term and consistent indexing
        if hotspot_tv_row is not None:
            visited_rows.add(int(hotspot_tv_row))

        tau_filtered = {int(r): int(off) for r, off in (tau or {}).items() if int(r) in visited_rows}
        tau_map[fid] = tau_filtered

    params = HyperParams(S0=5.0, S0_mode="x_at_argmin")
    h_row = int(row_map[hotspot_tv])

    anchor_id = int(anchor_flow["flow_id"])
    largest_id = int(largest_flow["flow_id"])

    tau_summary_table = Table(title="τ Map Summary")
    tau_summary_table.add_column("Role", style="bold magenta")
    tau_summary_table.add_column("Flow ID")
    tau_summary_table.add_column("Control TV")
    tau_summary_table.add_column("Visited Rows", justify="right")
    tau_summary_table.add_column("Sample τ (row→Δ)")

    for role, flow in [("Largest", largest_flow), ("Anchor", anchor_flow)]:
        fid = int(flow["flow_id"])
        tau_entries = tau_map.get(fid, {})
        sample_items = list(tau_entries.items())[:5]
        sample = ", ".join(f"{idx_to_tv_id.get(r, r)}→{off}" for r, off in sample_items) if sample_items else "<none>"
        tau_summary_table.add_row(
            role,
            str(fid),
            ctrl_by_flow.get(fid, hotspot_tv),
            str(len(tau_entries)),
            sample,
        )
    console.print(tau_summary_table)

    set_control_tv = ctrl_by_flow.get(largest_id, hotspot_tv)

    setwise_acc = {"overlap": 0.0, "orth": 0.0, "slack_corr": 0.0, "price_gap": 0.0}

    minutes_per_bin = int(indexer.time_bin_minutes)

    for b in timebins:
        hot = Hotspot(tv_id=hotspot_tv, bin=int(b))
        anchor_tau = tau_map[anchor_id]
        anchor_xG = xG_map[anchor_id]
        tG_anchor = int(phase_time(int(h_row), hot, anchor_tau, T))
        v_tilde_anchor = float(
            price_contrib_v_tilde(
                tG_anchor,
                h_row,
                hot.bin,
                anchor_tau,
                H_bool,
                S_mat,
                anchor_xG,
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
        )

        feats = compare_flow_against_flight_set(
            anchor_tG=tG_anchor,
            anchor_tau=anchor_tau,
            anchor_xG=anchor_xG,
            v_tilde_anchor=v_tilde_anchor,
            set_flight_ids=flights_by_flow[largest_id],
            control_tv_for_set=str(set_control_tv),
            hotspot_tv_id=hotspot_tv,
            hotspot_row=int(h_row),
            hotspot_bin=int(hot.bin),
            bin_offsets=bin_offsets,
            hourly_excess_bool=H_bool,
            slack_per_bin_matrix=S_mat,
            params=params,
            flight_list=flight_list,
            row_map=row_map,
            tv_centroids=res.tv_centroids,
            rolling_occ_by_bin=rolling_occ_by_bin,
            hourly_capacity_matrix=hourly_capacity_matrix,
            bins_per_hour=bins_per_hour,
        )

        for key in setwise_acc:
            setwise_acc[key] += float(feats[key])

        if args.per_bin_debug:
            tau_entries = list(anchor_tau.items())
            tau_entries_sorted = sorted(tau_entries, key=lambda kv: kv[0])
            tau_sample = ", ".join(
                f"{idx_to_tv_id.get(r, r)}→{off}" for r, off in tau_entries_sorted[:5]
            ) or "<none>"
            debug_table = Table(title=f"Setwise Bin Debug (bin {b})")
            debug_table.add_column("Field", style="cyan")
            debug_table.add_column("Value", style="yellow")
            debug_table.add_row("Hotspot TV", hotspot_tv)
            debug_table.add_row("Minutes per Bin", str(minutes_per_bin))
            debug_table.add_row("Anchor t_G", str(tG_anchor))
            debug_table.add_row("|Anchor Flights|", str(len(flights_by_flow[anchor_id])))
            debug_table.add_row("|Set Flights|", str(len(flights_by_flow[largest_id])))
            debug_table.add_row("τ Sample", tau_sample)
            debug_table.add_row("v_tilde(anchor)", f"{v_tilde_anchor:.4f}")
            debug_table.add_row("overlap", f"{feats['overlap']:.4f}")
            debug_table.add_row("orth", f"{feats['orth']:.4f}")
            debug_table.add_row("slack_corr", f"{feats['slack_corr']:.4f}")
            debug_table.add_row("price_gap", f"{feats['price_gap']:.4f}")
            console.print(debug_table)

    summary_table = Table(title="Setwise Feature Sums (Anchor vs Largest Set)")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")
    for key, val in setwise_acc.items():
        summary_table.add_row(key, f"{val:.4f}")
    console.print(summary_table)

    recap_table = Table(title="Computation Recap")
    recap_table.add_column("Field", style="cyan")
    recap_table.add_column("Value", style="yellow")
    recap_table.add_row("Hotspot TV", hotspot_tv)
    recap_table.add_row("Bin Window", f"[{timebins[0]}, {timebins[-1] + 1})")
    recap_table.add_row("Minutes per Bin", str(minutes_per_bin))
    recap_table.add_row("Bins per Hour", str(bins_per_hour))
    recap_table.add_row("Anchor Flow", f"{anchor_id} ({len(flights_by_flow[anchor_id])} flights)")
    recap_table.add_row("Set Flow", f"{largest_id} ({len(flights_by_flow[largest_id])} flights)")
    recap_table.add_row("Set Control TV", str(set_control_tv))
    console.print(recap_table)


if __name__ == "__main__":
    main()
