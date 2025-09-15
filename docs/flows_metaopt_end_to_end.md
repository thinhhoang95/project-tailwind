# End-to-End Example: Compute Flows via API and MetaOpt Features

This example shows how to:
- Take a single hotspot (TV + time bin like “EGLSW13 09:00–09:15”)
- Retrieve the flight list and run flow extraction using the `compute_flows` API wrapper
- Compute per‑flow and pairwise features against the hotspot using `parrhesia.metaopt`

It uses project resources (indexer, flight list, TV GeoJSON) via the server resource loader. Adjust paths if needed.

## Code

```python
from typing import Dict, Any
import numpy as np

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
hotspot_tv = "EGLSW13"
# For 15-minute bins, 09:00–09:15 starts at bin = 9*60/15 = 36
# If your time_bin_minutes differs, compute via:
start_h, start_m = 9, 0
bin_size = indexer.time_bin_minutes
hotspot_bin = (start_h * 60 + start_m) // bin_size

# 4) Compute flows via the API wrapper
from parrhesia.api.flows import compute_flows
# Pass direction-aware options with TV centroids, like server wrappers do
flows_payload: Dict[str, Any] = compute_flows(
    tvs=[hotspot_tv],
    timebins=[hotspot_bin],
    direction_opts={
        "mode": "coord_cosine",
        "tv_centroids": res.tv_centroids,
    },
)
print(f"Found {len(flows_payload['flows'])} flows for {hotspot_tv} @ bin {hotspot_bin}")

# 5) Build capacities and base caches for features
from parrhesia.optim.capacity import build_bin_capacities
from parrhesia.metaopt import (
    Hotspot, HyperParams,
    build_base_caches, attention_mask_from_cells,
    minutes_to_bin_offsets, flow_offsets_from_ctrl,
    build_xG_series, phase_time,
    price_kernel_vG, price_to_hotspot_vGH,
    slack_G_at, eligibility_a, slack_penalty, score as score_flow,
    temporal_overlap, offset_orthogonality, slack_profile, slack_corr, price_gap,
)

# Capacities from the TV GeoJSON used by your app
# If you already have a path, replace this with your file path
geojson_path = str(get_resources().traffic_volumes_gdf)
# If the above is not a path string, set it explicitly:
# geojson_path = "/path/to/traffic_volumes.geojson"
capacities_by_tv = build_bin_capacities(geojson_path, indexer)

caches = build_base_caches(flight_list, capacities_by_tv, indexer)
H_bool = caches["hourly_excess_bool"]
S_mat = caches["slack_per_bin_matrix"]
T = indexer.num_time_bins
row_map = flight_list.tv_id_to_idx

# 6) Convert travel minutes to bin offsets
bin_offsets = minutes_to_bin_offsets(travel_minutes_map, time_bin_minutes=indexer.time_bin_minutes)

# 7) Rebuild x_G and τ maps from compute_flows payload
hot = Hotspot(tv_id=hotspot_tv, bin=hotspot_bin)
flows = flows_payload["flows"]

xG_map: Dict[int, np.ndarray] = {}
tau_map: Dict[int, Dict[int, int]] = {}
tG_map: Dict[int, int] = {}
ctrl_by_flow: Dict[int, str] = {}

enable_signed_tau = True  # feature flag to compare behaviors
for fobj in flows:
    fid = int(fobj["flow_id"])  # flow id
    ctrl = fobj.get("controlled_volume")
    if ctrl:
        ctrl_by_flow[fid] = str(ctrl)
    # Rebuild x_G from flights' requested_bin
    # compute_flows returns flights: [{ flight_id, requested_bin, ... }]
    flights_specs = fobj.get("flights", [])
    flow_flight_ids = [sp["flight_id"] for sp in flights_specs if sp.get("flight_id")]
    x = np.zeros(T, dtype=float)
    for sp in flights_specs:
        rb = int(sp.get("requested_bin", 0))
        if 0 <= rb < T:
            x[rb] += 1.0
    xG_map[fid] = x

    # τ map: control TV to all TVs (row -> Δbins), optionally signed
    if enable_signed_tau:
        tau = flow_offsets_from_ctrl(
            ctrl, row_map, bin_offsets,
            flow_flight_ids=flow_flight_ids,
            flight_list=flight_list,
            hotspots=[hotspot_tv],
            trim_policy="earliest_hotspot",
            direction_sign_mode="order_vs_ctrl",
            tv_centroids=res.tv_centroids,
        ) or {}
    else:
        tau = flow_offsets_from_ctrl(ctrl, row_map, bin_offsets) or {}
    tau_map[fid] = tau

    # Phase alignment t_G
    h_row = int(row_map[hotspot_tv])
    tG_map[fid] = phase_time(h_row, hot, tau, T)

# 8) Per-flow features wrt hotspot
params = HyperParams(S0=5.0, S0_mode="x_at_argmin")  # default mode can be omitted
h_row = int(row_map[hotspot_tv])

per_flow_feats: Dict[int, Dict[str, float]] = {}
for fid in xG_map.keys():
    tG = int(tG_map[fid])
    tau = tau_map[fid]
    xG = xG_map[fid]
    # Price kernels
    vG = price_kernel_vG(tG, tau, H_bool, theta_mask=None, w_sum=params.w_sum, w_max=params.w_max)
    vGH = price_to_hotspot_vGH(h_row, hot.bin, tau, H_bool, theta_mask=None, w_sum=params.w_sum, w_max=params.w_max, kappa=params.kappa)
    # Slack + eligibility
    sG = slack_G_at(tG, tau, S_mat)
    aG = eligibility_a(xG, tG, q0=params.q0, gamma=params.gamma, soft=True)
    rho = slack_penalty(tG, tau, S_mat, S0=params.S0, xG=xG, S0_mode=params.S0_mode)
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
        "tG": tG,
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
pairwise: Dict[tuple, Dict[str, float]] = {}
flow_ids = list(xG_map.keys())
window = list(range(-params.window_left, params.window_right + 1))
for i in range(len(flow_ids)):
    for j in range(i + 1, len(flow_ids)):
        fi, fj = flow_ids[i], flow_ids[j]
        # Build window around tGi in absolute bins
        W = [tG_map[fi] + k for k in window]
        # Align xGj to fi’s phase by shifting
        d = int(tG_map[fi] - tG_map[fj])
        xj_shift = np.zeros_like(xG_map[fj])
        if d >= 0:
            xj_shift[d:] = xG_map[fj][: T - d]
        else:
            xj_shift[: T + d] = xG_map[fj][-d :]
        ov = temporal_overlap(xG_map[fi], xj_shift, window_bins=W)
        orth = offset_orthogonality(h_row, hot.bin, tau_map[fi], tau_map[fj], H_bool)
        Si = slack_profile(tG_map[fi], tau_map[fi], S_mat, window)
        Sj = slack_profile(tG_map[fj], tau_map[fj], S_mat, window)
        sc = slack_corr(Si, Sj)
        pg = price_gap(per_flow_feats[fi]["vG"], per_flow_feats[fj]["vG"], eps=params.eps)
        pairwise[(fi, fj)] = {"overlap": ov, "orth": orth, "slack_corr": sc, "price_gap": pg}

print("Pairwise features:")
for k, d in pairwise.items():
    print(k, d)
```

## Notes
- `compute_flows` clusters flights into flows and selects a control TV per flow using the earliest‑median policy. The example reconstructs `x_G(t)` from the per‑flow flights’ `requested_bin`.
- Travel minutes use centroid great‑circle distances from the server resources; adjust if you maintain your own travel‑time matrix.
- Slack uses hourly capacities distributed uniformly within the hour; change `S0` to scale slack penalties for your environment.
- For additional planning/grouping and proposals, see `docs/metaopt_overview.md` and `docs/metaopt_examples.md`.
