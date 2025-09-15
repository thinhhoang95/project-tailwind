# MetaOpt Feature Engineering Overview

This document explains the methodology for the feature engineering pipeline implemented under `src/parrhesia/metaopt`. It covers the conceptual background, inputs/outputs, the per‑flow and pairwise features, and how these feed a grouping and planning stage for regulation proposals.

Sections
- Motivation and scope
- Inputs and shared caches
- Travel offsets and phase alignment
- Per‑flow features: v_G, v_{G→H}, Slack_G, eligibility, score
- Pairwise features: overlap, orthogonality, slack correlation, price gap
- Grouping and proposal planning
- Usage examples (code snippets)

## Motivation and Scope

We aim to evaluate and triage candidate flows to relieve hotspot overloads by computing features that are (a) aligned to travel offsets from putative control volumes, and (b) sensitive to rolling‑hour overloads and slack. These features provide per‑flow "price"/risk and pairwise diagnostics that can guide whether to union multiple flows under a common cap or keep them separate.

The implementation is designed for rapid experimentation and to integrate with the existing SA optimizer stack. It reuses data structures from the project (FlightList, TVTWIndexer, capacity parsing) and deliberately isolates heavy computations behind reusable caches.

## Inputs and Shared Caches

Base inputs and caches are built once per scenario using `build_base_caches`:
- `occ_base` (V·T): baseline per‑TV per‑bin occupancy from all flights.
- `hourly_capacity_matrix` (V×24): hourly capacity per TV.
- `cap_per_bin` (V·T): capacity per bin (hourly capacity divided by bins per hour).
- `hourly_occ_base` (V×24): baseline hourly occupancy per TV.
- `slack_per_bin_matrix` (V×T): per‑bin slack from hourly slack distributed uniformly within the hour.
- `hourly_excess_bool` (V×T): indicator of rolling‑hour overload (forward‑looking window of 60 minutes equivalent in bins).

Here V is the number of TVs, T is time bins per day.

These caches depend on:
- `FlightList` for baseline occupancy: `get_total_occupancy_by_tvtw()` (length V·T)
- `TVTWIndexer` for time bin helpers (`time_bin_minutes`, `num_time_bins`)
- Capacities parsed with `parrhesia.optim.capacity.build_bin_capacities` (tv_id → per‑bin hourly values)

Attention mask θ is built via `attention_mask_from_cells((tv_id, bin), ripple_cells…)` and maps sparse cells `(tv_row, bin) → weight` for target and optional ripple cells.

## Travel Offsets and Phase Alignment

Nominal travel minutes between TVs (e.g., from centroid distances at 475 kts) are converted to bin offsets:
- `minutes_to_bin_offsets(minutes_map, time_bin_minutes) → {src: {dst: bins}}`
- For a flow controlled at TV `s_ctrl`, `flow_offsets_from_ctrl` yields `τ_{G,s}` for all TVs `s` as a mapping `row_index → offset_bins`.
  - Signed option: pass flow context (`flow_flight_ids`, `flight_list`) and `direction_sign_mode="order_vs_ctrl"` to infer signs per TV (+ downstream, − upstream) with geometric fallback (`"vector_centroid"`).
  - Back‑compat: if flow context is omitted, τ magnitudes remain non‑negative (old behavior).

Given a hotspot H = (s*, t*), the flow’s phase time is:
- `t_G = t* − τ_{G,s*}` (aligned index at the flow’s control row). Use `phase_time(hotspot_row, hotspot, tau, T)` where `hotspot_row` is the row index of s*.

## Per‑Flow Features

- v_G(t): Sum of weights over TVs that are overloaded at `t + τ_{G,s}`
  - Implementation: `price_kernel_vG(t_G, tau, hourly_excess_bool, theta_mask, w_sum, w_max)`
  - θ adds weight for attention cells if provided; otherwise the kernel is the count of overloaded TVs times `w_sum`.

- v_{G→H}: Price directed at the hotspot with ripple terms
  - `price_to_hotspot_vGH(h_row, h_bin, tau, hourly_excess_bool, theta_mask, w_sum, w_max, kappa)`
  - Primary term is at (s*, t*); ripple terms sum over TVs after aligning by `τ_{G,s} − τ_{G,s*}` and weighting by κ.

- Slack_G(t): Minimum slack across TVs sampled at `t + τ_{G,s}`
  - `slack_G_at(t, tau, slack_per_bin_matrix)` returns min slack across rows at their aligned indices.

- Eligibility at phase: hard or soft
  - Hard: `1{x_G(t_G) ≥ q0}`; Soft: `σ(γ(x_G(t_G) − q0))`
  - `eligibility_a(xG, t_G, q0, gamma, soft=True|False)`

- Net score: matched‑filter objective
  - `score(t_G, h_row, h_bin, tau, hourly_excess_bool, slack_per_bin_matrix, params, xG, theta)` computes:
    - `α · a_{G→H} · v_{G→H} − β · ρ_{G→H}` with `ρ = [1 − Slack_G(t_G)/S0_eff]_+`.
    - `S0_eff` can be dynamic via `params.S0_mode`:
      - `"x_at_argmin"` (default): `x_G` at the aligned time achieving the min in `Slack_G(t_G)`.
      - `"x_at_control"`: `x_G(t_G)`.
      - `"constant"`: the provided `params.S0`.
  - `lambda_delay` is included in params but not consumed yet (reserved for future delay penalty).

## Pairwise Features

- Temporal overlap: `Overlap_{ij} = ∑ min{x_i(t), x_j(t)}` over a window
  - `temporal_overlap(xGi, xGj, window_bins)`; for cross‑phase comparison, shift one series by `Δ = tGi − tGj` and sum over a window around `tGi`.

- Orthogonality: alignment of overloaded TV sets
  - For each flow, build set `T_i = {s: o_s(t* + τ_{G_i,s} − τ_{G_i,s*})>0}`
  - `offset_orthogonality(h_row, h_bin, tau_i, tau_j, hourly_excess_bool, tv_universe_mask)` returns `1 − |T_i ∩ T_j| / |T_i ∪ T_j|` (or vs a provided universe mask).

- Slack correlation: similarity of slack profiles around phases
  - `slack_profile(t_G, tau, slack_per_bin_matrix, window_bins)` computes `S_G(Δ) = Slack_G(t_G+Δ)` over Δ in a window
  - `slack_corr(profile_i, profile_j)` computes Pearson correlation.

- Price gap: relative difference of prices at phases
  - `price_gap(vGi, vGj, eps)` = `|vGi − vGj| / (vGi + vGj + ε)`

## Grouping and Proposal Planning

Heuristic union/separate decision per pair uses thresholds on the features:
- `decide_union_or_separate({overlap, orth, slack_corr, price_gap}, {tau_*})`
- `cluster_flows(flow_ids, pairwise_feature_map, thresholds)` builds connected components of pairs labeled as “union”.

Planning converts group labels into draft `RegulationProposal`s:
- `choose_active_window(x_G, t_G, overloaded_window_aligned, min_frac_of_peak, max_span)` selects bins around phase where activity remains significant.
- `make_proposals(hotspot, flows, labels, xG_map, tG_map, ctrl_by_flow)` returns:
  - `[{ flow_ids, control_tv_id, active_bins, rate_guess, meta }]` with modal control TV per group and union of member windows.

## Usage Examples

Below are illustrative snippets assuming you already computed flows and have capacities and travel minutes available.

### Build capacities and caches
```python
from parrhesia.optim.capacity import build_bin_capacities
from parrhesia.metaopt import build_base_caches

capacities_by_tv = build_bin_capacities(geojson_path, indexer)
caches = build_base_caches(flight_list, capacities_by_tv, indexer)
```

### Travel offsets per flow
```python
from parrhesia.metaopt import minutes_to_bin_offsets, flow_offsets_from_ctrl

bin_offsets = minutes_to_bin_offsets(travel_minutes_map, indexer.time_bin_minutes)
row_map = flight_list.tv_id_to_idx  # tv_id -> row
# Signed τ (optional):
tau = flow_offsets_from_ctrl(
    control_tv_id,
    row_map,
    bin_offsets,
    flow_flight_ids=[sp['flight_id'] for sp in flights_by_flow[flow_id]],
    flight_list=flight_list,
    hotspots=[hotspot_tv_id],
    trim_policy="earliest_hotspot",
    direction_sign_mode="order_vs_ctrl",
    tv_centroids=tv_centroids,  # optional, for geometric fallback
)
# Back‑compat (unsigned): tau = flow_offsets_from_ctrl(control_tv_id, row_map, bin_offsets)
```

### Per-flow series and score
```python
from parrhesia.metaopt import (
    build_xG_series, phase_time, attention_mask_from_cells, score as score_flow,
)
from parrhesia.metaopt import HyperParams

xG = build_xG_series(flights_by_flow, ctrl_by_flow, flow_id, indexer.num_time_bins)
h_row = flight_list.tv_id_to_idx[h_tv]
tG = phase_time(h_row, Hotspot(h_tv, h_bin), tau, indexer.num_time_bins)
theta = attention_mask_from_cells((h_tv, h_bin), tv_id_to_idx=flight_list.tv_id_to_idx, T=indexer.num_time_bins)
params = HyperParams(w_sum=1.0, w_max=1.0, kappa=0.25, alpha=1.0, beta=1.0, S0=5.0)

s = score_flow(
    t_G=tG,
    hotspot_row=flight_list.tv_id_to_idx[h_tv],
    hotspot_bin=h_bin,
    tau_row_to_bins=tau,
    hourly_excess_bool=caches['hourly_excess_bool'],
    slack_per_bin_matrix=caches['slack_per_bin_matrix'],
    params=params,
    xG=xG,
    theta_mask=theta,
    use_soft_eligibility=True,
)
```

### Pairwise features and grouping
```python
from parrhesia.metaopt import temporal_overlap, offset_orthogonality, slack_profile, slack_corr, price_gap
from parrhesia.metaopt.grouping import cluster_flows

window = list(range(-2, 3))  # around tGi
pair_feats = {}
for i, fi in enumerate(flow_ids):
    for fj in flow_ids[i+1:]:
        d = tG_map[fi] - tG_map[fj]
        xi = xG_map[fi]
        xj = xG_map[fj]
        xj_shift = np.roll(xj, d)
        ov = temporal_overlap(xi, xj_shift, window_bins=[tG_map[fi]+k for k in window])
        orth = offset_orthogonality(h_row, h_bin, tau_map[fi], tau_map[fj], caches['hourly_excess_bool'])
        Si = slack_profile(tG_map[fi], tau_map[fi], caches['slack_per_bin_matrix'], window)
        Sj = slack_profile(tG_map[fj], tau_map[fj], caches['slack_per_bin_matrix'], window)
        sc = slack_corr(Si, Sj)
        pg = price_gap(scores[fi], scores[fj])
        pair_feats[(fi, fj)] = {"overlap": ov, "orth": orth, "slack_corr": sc, "price_gap": pg}

labels = cluster_flows(flow_ids, pair_feats, thresholds={"tau_ov": 0.0, "tau_sl": 0.0, "tau_pr": 0.5, "tau_orth": 0.8})
```

### Build proposals
```python
from parrhesia.metaopt.planner import make_proposals
from parrhesia.metaopt.types import FlowSpec, Hotspot

flows = [FlowSpec(flow_id=f, control_tv_id=ctrl_by_flow.get(f), flight_ids=[sp['flight_id'] for sp in flights_by_flow[f]]) for f in flow_ids]
proposals = make_proposals(Hotspot(h_tv, h_bin), flows, labels, xG_map, tG_map, ctrl_by_flow)
```

## Notes and Extensibility

- The vectorized design allows swapping in richer attention masks or alternative travel‑time models (e.g., wind‑adjusted) by replacing τ and θ providers.
- The SA optimizer integration is intentionally left out of this doc; proposals can be translated to the optimizer’s expected inputs by filtering flows and windows accordingly.
- Hyperparameters in `HyperParams` expose the key trade‑offs; tune based on validation scenarios.
