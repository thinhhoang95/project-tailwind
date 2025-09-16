"""
End-to-end example using FlowFeaturesExtractor and optional manual cross-check.

This script mirrors the structure of flows_features_example_revised_pairwise.py
but delegates per-flow feature aggregation to FlowFeaturesExtractor, and compares
against a manual per-bin computation for verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import argparse
import numpy as np
from rich.console import Console
from rich.table import Table


def main() -> None:
    # 1) Load shared resources
    from server_tailwind.core.resources import get_resources
    res = get_resources().preload_all()
    indexer = res.indexer
    flight_list = res.flight_list
    travel_minutes_map = res.travel_minutes(speed_kts=475.0)

    # Ensure compute_flows uses the same resources
    from parrhesia.api.resources import set_global_resources
    set_global_resources(indexer, flight_list)

    # 2) Parse CLI args
    parser = argparse.ArgumentParser(description="FlowFeaturesExtractor example over a time range")
    parser.add_argument("--hotspot", default="LSGL13W", help="Hotspot TV ID (e.g., LSGL13W)")
    parser.add_argument("--start", default="11:15", help="Range start time HH:MM (inclusive)")
    parser.add_argument("--end", default="11:45", help="Range end time HH:MM (exclusive)")
    parser.add_argument("--compare-manual", action="store_true", help="Compute manual sums to cross-check extractor")
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

    # 3) Compute flows for the hotspot and time range
    from parrhesia.api.flows import compute_flows
    flows_payload: Dict[str, Any] = compute_flows(
        tvs=[hotspot_tv],
        timebins=timebins,
        direction_opts={
            "mode": "coord_cosine",
            "tv_centroids": res.tv_centroids,
        },
    )

    # 4) Capacities and caches
    from parrhesia.optim.capacity import build_bin_capacities
    from parrhesia.metaopt import HyperParams
    capacities_by_tv = build_bin_capacities(str(res.paths.traffic_volumes_path if res.paths.traffic_volumes_path.exists() else (res.paths.fallback_traffic_volumes_path if res.paths.fallback_traffic_volumes_path.exists() else res.paths.fallback_traffic_volumes_path_2)), indexer)

    # 5) Use FlowFeaturesExtractor
    from parrhesia.metaopt.feats import FlowFeaturesExtractor
    params = HyperParams(S0_mode="x_at_argmin")
    extractor = FlowFeaturesExtractor(indexer, flight_list, capacities_by_tv, travel_minutes_map, params=params)
    feats_by_flow = extractor.compute_for_hotspot(hotspot_tv, timebins, flows_payload=flows_payload, direction_opts={
        "mode": "coord_cosine",
        "tv_centroids": res.tv_centroids,
    })

    console = Console()
    table = Table(title=f"FlowFeaturesExtractor results for {hotspot_tv} @ bins {timebins[0]}–{timebins[-1]}")
    cols = [
        "flow_id", "ctrl", "tGl", "tGu", "xGH", "DH", "gH(derived)", "gH_sum", "gH_avg", "v_tilde", "gH*v_tilde",
        "Slack_G0", "G0_row", "Slack_G15", "G15_row", "Slack_G30", "G30_row", "Slack_G45", "G45_row", "rho"
    ]
    for c in cols:
        table.add_column(c)
    for fid, feat in feats_by_flow.items():
        table.add_row(
            str(fid),
            str(feat.control_tv_id or ""),
            str(feat.tGl),
            str(feat.tGu),
            f"{feat.xGH:.3f}",
            f"{feat.DH:.3f}",
            f"{feat.gH:.3f}",
            f"{feat.gH_sum:.3f}",
            f"{feat.gH_avg:.3f}",
            f"{feat.v_tilde:.3f}",
            f"{feat.gH_v_tilde:.3f}",
            f"{feat.Slack_G0:.3f}",
            ("None" if feat.Slack_G0_row is None else str(int(feat.Slack_G0_row))),
            f"{feat.Slack_G15:.3f}",
            ("None" if feat.Slack_G15_row is None else str(int(feat.Slack_G15_row))),
            f"{feat.Slack_G30:.3f}",
            ("None" if feat.Slack_G30_row is None else str(int(feat.Slack_G30_row))),
            f"{feat.Slack_G45:.3f}",
            ("None" if feat.Slack_G45_row is None else str(int(feat.Slack_G45_row))),
            f"{feat.rho:.3f}",
        )
    console.print(table)

    if args.compare_manual:
        # Manual reproduction of per-bin sums to cross-check extractor
        from parrhesia.metaopt import (
            Hotspot,
            build_base_caches,
            minutes_to_bin_offsets,
            flow_offsets_from_ctrl,
            phase_time,
            mass_weight_gH,
            price_contrib_v_tilde,
            slack_G_at,
            slack_penalty,
        )

        caches = build_base_caches(flight_list, capacities_by_tv, indexer)
        H_bool = caches["hourly_excess_bool"]
        S_mat = caches["slack_per_bin_matrix"]
        T = int(indexer.num_time_bins)
        row_map = flight_list.tv_id_to_idx
        idx_to_tv_id = {idx: tv for tv, idx in row_map.items()}
        h_row = int(row_map[hotspot_tv])
        rolling_occ_by_bin = caches.get("rolling_occ_by_bin")
        hourly_capacity_matrix = caches.get("hourly_capacity_matrix")
        bins_per_hour = int(caches.get("bins_per_hour", max(1, 60 // int(indexer.time_bin_minutes))))
        bin_offsets = minutes_to_bin_offsets(travel_minutes_map, time_bin_minutes=indexer.time_bin_minutes)

        # Rebuild xG and τ maps
        flows = flows_payload["flows"]
        xG_map: Dict[int, np.ndarray] = {}
        tau_map: Dict[int, Dict[int, int]] = {}
        for fobj in flows:
            fid = int(fobj["flow_id"])  # flow id
            ctrl = fobj.get("controlled_volume")
            flights_specs = fobj.get("flights", [])
            flow_flight_ids = [str(sp.get("flight_id")) for sp in flights_specs if sp.get("flight_id")]
            x = np.zeros(T, dtype=float)
            for sp in flights_specs:
                rb = int(sp.get("requested_bin", 0))
                if 0 <= rb < T:
                    x[rb] += 1.0
            xG_map[fid] = x
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
            # Keep hotspot row and restrict to visited
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
                # optional trim as above
                cut = None
                for i, v in enumerate(seq.tolist()):
                    if int(v) == int(h_row):
                        cut = i
                        break
                if cut is not None:
                    seq = seq[: cut + 1]
                visited_rows.update(int(v) for v in seq.tolist())
            if h_row not in visited_rows:
                visited_rows.add(int(h_row))
            tau_map[fid] = {int(r): int(off) for r, off in (tau or {}).items() if int(r) in visited_rows}

        # Aggregation structures
        minutes_per_bin = int(indexer.time_bin_minutes)
        def _shift_bins(mins: int) -> int:
            if mins == 0:
                return 0
            return max(1, int(round(mins / float(minutes_per_bin))))
        delta_min_list = [0, 15, 30, 45]
        delta_to_shift = {m: _shift_bins(m) for m in delta_min_list}

        sums: Dict[int, Dict[str, float]] = {}
        tGl: Dict[int, int] = {}
        tGu: Dict[int, int] = {}
        slack_sum: Dict[int, Dict[int, float]] = {int(fid): {m: 0.0 for m in delta_min_list} for fid in xG_map.keys()}
        for fid in xG_map.keys():
            sums[int(fid)] = {"xGH": 0.0, "DH": 0.0, "gH_sum": 0.0, "v_tilde": 0.0, "rho": 0.0}
            tGl[int(fid)] = T - 1
            tGu[int(fid)] = 0

        for b in timebins:
            hot = Hotspot(tv_id=hotspot_tv, bin=int(b))
            for fid in xG_map.keys():
                tau = tau_map[int(fid)]
                xG = xG_map[int(fid)]
                tG = int(phase_time(h_row, hot, tau, T))
                tGl[int(fid)] = min(tGl[int(fid)], tG)
                tGu[int(fid)] = max(tGu[int(fid)], tG)

                xhat, DH, gH = mass_weight_gH(
                    xG,
                    tG,
                    h_row,
                    hot.bin,
                    S_mat,
                    eps=1e-6,
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
                    w_sum=1.0,
                    w_max=1.0,
                    kappa=0.25,
                    eps=1e-6,
                    verbose_debug=False,
                    idx_to_tv_id=idx_to_tv_id,
                    rolling_occ_by_bin=rolling_occ_by_bin,
                    hourly_capacity_matrix=hourly_capacity_matrix,
                    bins_per_hour=bins_per_hour,
                )
                rho = slack_penalty(
                    tG,
                    tau,
                    S_mat,
                    S0=1.0,
                    xG=xG,
                    S0_mode="x_at_argmin",
                    verbose_debug=False,
                    idx_to_tv_id=idx_to_tv_id,
                    rolling_occ_by_bin=rolling_occ_by_bin,
                    hourly_capacity_matrix=hourly_capacity_matrix,
                    bins_per_hour=bins_per_hour,
                )
                sums[int(fid)]["xGH"] += float(xhat)
                sums[int(fid)]["DH"] += float(DH)
                sums[int(fid)]["gH_sum"] += float(gH)
                sums[int(fid)]["v_tilde"] += float(v_tilde)
                sums[int(fid)]["rho"] += float(rho)
                for mins in delta_min_list:
                    t_eval = tG + delta_to_shift[mins]
                    if 0 <= t_eval < T:
                        s_val = float(slack_G_at(
                            t_eval,
                            tau,
                            S_mat,
                            rolling_occ_by_bin=rolling_occ_by_bin,
                            hourly_capacity_matrix=hourly_capacity_matrix,
                            bins_per_hour=bins_per_hour,
                        ))
                        slack_sum[int(fid)][int(mins)] += s_val

        # Compare side-by-side with extractor
        cmp = Table(title="Manual vs Extractor (selected fields)")
        for c in ["flow", "xGH(man)", "xGH(ext)", "DH(man)", "DH(ext)", "gH_sum(man)", "gH_sum(ext)", "v(man)", "v(ext)", "rho(man)", "rho(ext)", "G0(man)", "G0(ext)"]:
            cmp.add_column(c)
        for fid, feat in feats_by_flow.items():
            cmp.add_row(
                str(fid),
                f"{sums[fid]['xGH']:.3f}", f"{feat.xGH:.3f}",
                f"{sums[fid]['DH']:.3f}", f"{feat.DH:.3f}",
                f"{sums[fid]['gH_sum']:.3f}", f"{feat.gH_sum:.3f}",
                f"{sums[fid]['v_tilde']:.3f}", f"{feat.v_tilde:.3f}",
                f"{sums[fid]['rho']:.3f}", f"{feat.rho:.3f}",
                f"{slack_sum[fid][0]:.3f}", f"{feat.Slack_G0:.3f}",
            )
        console.print(cmp)


if __name__ == "__main__":
    main()

