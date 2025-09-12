# MetaOpt Examples

This guide shows end‑to‑end usage of `parrhesia.metaopt` components with small, concrete code snippets. Adjust paths and objects to your environment.

## Setup

```python
from parrhesia.metaopt import (
    Hotspot, HyperParams,
    build_base_caches, attention_mask_from_cells,
    minutes_to_bin_offsets, flow_offsets_from_ctrl,
    build_xG_series, phase_time, score,
)
from parrhesia.optim.capacity import build_bin_capacities

# Assume you already have these objects
flight_list = ...   # project_tailwind FlightList
indexer = ...       # TVTWIndexer
travel_minutes_map = ...  # nested dict src->dst->minutes
flights_by_flow = ...  # flow_id -> [ {flight_id, requested_bin, ...}, ... ]
ctrl_by_flow = ...     # flow_id -> control_tv_id

geojson_path = "/path/to/traffic_volumes.geojson"
capacities_by_tv = build_bin_capacities(geojson_path, indexer)
caches = build_base_caches(flight_list, capacities_by_tv, indexer)
```

## Scoring a Single Flow for a Hotspot

```python
hot = Hotspot(tv_id="TV001", bin=36)
T = indexer.num_time_bins

# Build τ for the flow from its control TV
row_map = flight_list.tv_id_to_idx
bin_offsets = minutes_to_bin_offsets(travel_minutes_map, indexer.time_bin_minutes)
tau = flow_offsets_from_ctrl(ctrl_by_flow[0], row_map, bin_offsets)

# Build activity x_G at control
xG = build_xG_series(flights_by_flow, ctrl_by_flow, flow_id=0, num_time_bins_per_tv=T)

# Phase alignment
ctrl_row = row_map.get(ctrl_by_flow[0], None)
tG = phase_time(ctrl_row, hot, tau, T)

# Attention mask and weights
theta = attention_mask_from_cells((hot.tv_id, hot.bin), tv_id_to_idx=row_map, T=T)
params = HyperParams(S0=5.0)

score_val = score(
    t_G=tG,
    hotspot_row=row_map[hot.tv_id],
    hotspot_bin=hot.bin,
    tau_row_to_bins=tau,
    hourly_excess_bool=caches['hourly_excess_bool'],
    slack_per_bin_matrix=caches['slack_per_bin_matrix'],
    params=params,
    xG=xG,
    theta_mask=theta,
    use_soft_eligibility=True,
)
print("Score:", score_val)
```

## Pairwise Diagnostics and Grouping

```python
from parrhesia.metaopt import temporal_overlap, offset_orthogonality, slack_profile, slack_corr, price_gap, score as score_flow
from parrhesia.metaopt.grouping import cluster_flows
import numpy as np

flow_ids = list(flights_by_flow.keys())
T = indexer.num_time_bins

# Precompute xG, tG, and tau for each flow
xG_map, tG_map, tau_map, scores_map = {}, {}, {}, {}
for f in flow_ids:
    xG_map[f] = build_xG_series(flights_by_flow, ctrl_by_flow, f, T)
    tau_map[f] = flow_offsets_from_ctrl(ctrl_by_flow[f], flight_list.tv_id_to_idx, bin_offsets)
    ctrl_row = flight_list.tv_id_to_idx.get(ctrl_by_flow[f], None)
    tG_map[f] = phase_time(ctrl_row, hot, tau_map[f], T)
    scores_map[f] = score_flow(
        t_G=tG_map[f],
        hotspot_row=flight_list.tv_id_to_idx[hot.tv_id],
        hotspot_bin=hot.bin,
        tau_row_to_bins=tau_map[f],
        hourly_excess_bool=caches['hourly_excess_bool'],
        slack_per_bin_matrix=caches['slack_per_bin_matrix'],
        params=HyperParams(S0=5.0),
        xG=xG_map[f],
        theta_mask=attention_mask_from_cells((hot.tv_id, hot.bin), tv_id_to_idx=flight_list.tv_id_to_idx, T=T),
        use_soft_eligibility=True,
    )

pair_feats = {}
window = list(range(-2, 3))
for i in range(len(flow_ids)):
    for j in range(i+1, len(flow_ids)):
        fi, fj = flow_ids[i], flow_ids[j]
        # Align xGj to phase of i
        d = tG_map[fi] - tG_map[fj]
        xj_shift = np.zeros_like(xG_map[fj])
        if d >= 0:
            xj_shift[d:] = xG_map[fj][:T-d]
        else:
            xj_shift[:T+d] = xG_map[fj][-d:]
        W = [tG_map[fi] + k for k in window]
        ov = temporal_overlap(xG_map[fi], xj_shift, window_bins=W)
        orth = offset_orthogonality(flight_list.tv_id_to_idx[hot.tv_id], hot.bin, tau_map[fi], tau_map[fj], caches['hourly_excess_bool'])
        Si = slack_profile(tG_map[fi], tau_map[fi], caches['slack_per_bin_matrix'], window)
        Sj = slack_profile(tG_map[fj], tau_map[fj], caches['slack_per_bin_matrix'], window)
        sc = slack_corr(Si, Sj)
        pg = price_gap(scores_map[fi], scores_map[fj])
        pair_feats[(fi, fj)] = {"overlap": ov, "orth": orth, "slack_corr": sc, "price_gap": pg}

labels = cluster_flows(flow_ids, pair_feats, thresholds={"tau_ov": 0.0, "tau_sl": 0.0, "tau_pr": 0.5, "tau_orth": 0.8})
print("Group labels:", labels)
```

## Build Proposals

```python
from parrhesia.metaopt.planner import make_proposals
from parrhesia.metaopt.types import FlowSpec

flows = [FlowSpec(flow_id=f, control_tv_id=ctrl_by_flow.get(f), flight_ids=[sp['flight_id'] for sp in flights_by_flow.get(f, [])]) for f in flow_ids]
proposals = make_proposals(hot, flows, labels, xG_map, tG_map, ctrl_by_flow)
for p in proposals:
    print(p)
```

## High‑level Runner

```python
from parrhesia.metaopt.runner import rank_flows_and_plan

proposals, diag = rank_flows_and_plan(
    flight_list=flight_list,
    indexer=indexer,
    travel_minutes_map=travel_minutes_map,
    flights_by_flow=flights_by_flow,
    ctrl_by_flow=ctrl_by_flow,
    hotspot=hot,
    capacities_by_tv=capacities_by_tv,
)
print("Proposals:", proposals)
print("Diagnostics keys:", diag.keys())
```

## Tips

- Check that your hotspot TV exists in `flight_list.tv_id_to_idx`.
- Ensure `requested_bin` exists per flight spec in `flights_by_flow`; use `prepare_flow_scheduling_inputs` to build these.
- Time‑bin alignment is integral; `time_bin_minutes` must divide 60.
