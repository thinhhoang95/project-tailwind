FlowFeaturesExtractor — Per‑flow Feature Aggregation
===================================================

Overview
- Computes non‑pairwise, per‑flow features over a hotspot period [start_bin, end_bin).
- Aggregates quantities across the period and exposes both sum and average variants where relevant.
- Mirrors the semantics demonstrated in `src/parrhesia/metaopt/usage/flows_features_example*.py`.

Key Features per Flow
- xGH: Sum over bins of x̂_GH = xG(tG).
- tGl, tGu: Min/max phase time tG (control volume time) across the hotspot period.
- DH: Sum over bins of hotspot exceedance magnitude at s* and t*.
- gH:
  - gH (derived): xGH / (xGH + DH) with epsilon guard.
  - gH_sum: Sum of per‑bin g_H values across period.
  - gH_avg: Average of per‑bin g_H values across period.
- v_tilde: Sum over bins of contribution‑weighted unit price ṽ.
- gH_v_tilde: Derived period‑level gH (above) times v_tilde sum.
- Slack sums and argmin rows:
  - Slack_G0/15/30/45: Sum over period of Slack_G(tG + Δ), where Δ ∈ {0, 15, 30, 45} minutes mapped to bin shifts.
  - Slack_G{Δ}_row: Single global argmin row index across the period for each Δ, i.e., the TV row that attains the minimal Slack_G+Δ at any bin in the period.
- rho: Sum over bins of slack penalty ρ.

Inputs
- indexer: Provides `num_time_bins` and `time_bin_minutes`.
- flight_list: Provides occupancy data, TV id/index mappings, and access to flight TV sequences.
- capacities_by_tv: TV id → per‑bin capacity array (hourly capacity replicated across bin(s) within hour).
- travel_minutes_map: Nested mapping of minutes[src][dst] used to build bin offsets.
- params: Optional `HyperParams` (defaults keep `S0_mode="x_at_argmin"`).
- flows_payload: Optional pre‑computed payload from `parrhesia.api.flows.compute_flows` to reuse flows.
- autotrim_from_ctrl_to_hotspot (bool, default False):
  - When False (default), the domain of TVs considered for a flow is the union of all TVs that any flight in the flow reaches ("can touch").
  - When True, per‑flight TV sequences and τ rows are trimmed to the prefix up to and including the first visit to the hotspot.
  - Note: With `S0_mode="x_at_argmin"`, if a TV is "touched" but at a different aligned time bin such that `xG[t̂]=0`, then `rho` remains 0 by design (since `S0_eff<=0`).

API
- Class: `parrhesia.metaopt.feats.FlowFeaturesExtractor`
  - `__init__(indexer, flight_list, capacities_by_tv, travel_minutes_map, params=None)`
  - `compute_for_hotspot(hotspot_tv: str, timebins: Sequence[int], flows_payload=None, direction_opts=None) -> Dict[int, FlowFeatures]`

- Dataclass: `FlowFeatures`
  - Identifiers: `flow_id`, `control_tv_id`
  - Phase bounds: `tGl`, `tGu`
  - Aggregates: `xGH`, `DH`, `gH_sum`, `gH_avg`, `gH` (derived), `v_tilde`, `gH_v_tilde`, `rho`, `bins_count`
  - Slack: `Slack_G0/15/30/45` and corresponding `Slack_G*_row` indices

Notes and Semantics
- Period aggregation is sum‑based for primary quantities (x̂, D, ṽ, slack, ρ). For g_H we expose both sum and average; the derived `gH = xGH/(xGH + DH)` is also included, as requested.
- Slack row indices are single global argmin per Δ across the period. The example also tracks the minimum slack value internally to pick the corresponding row.
- Minutes → bins mapping uses `round(Δ_minutes / time_bin_minutes)` with a minimum of 1 bin if Δ > 0, so Δ=15 maps to 1 bin when `time_bin_minutes=15`.
- When rolling‑hour occupancy and hourly capacity are available (via base caches), slack slices are computed as `capacity − occupancy`. Otherwise, the cached non‑negative slack matrix is used.

Quick Usage
1) Prepare resources (indexer, flight_list, travel_minutes_map, capacities_by_tv), and compute flows for a hotspot and time range.
2) Instantiate the extractor and compute features:

```python
from parrhesia.metaopt.feats import FlowFeaturesExtractor
from parrhesia.metaopt.types import HyperParams

extractor = FlowFeaturesExtractor(
    indexer,
    flight_list,
    capacities_by_tv,
    travel_minutes_map,
    params=HyperParams(S0_mode="x_at_argmin"),
    autotrim_from_ctrl_to_hotspot=False,  # default
)
features_by_flow = extractor.compute_for_hotspot(hotspot_tv="LSGL13W", timebins=[45,46,47,48], flows_payload=flows_payload)

for fid, feats in features_by_flow.items():
    print(fid, feats.gH, feats.v_tilde, feats.Slack_G15, feats.Slack_G15_row)
```

Cross‑checking
- See the runnable example at `src/parrhesia/metaopt/usage/flows_features_extractor_example.py`. It computes features using the extractor and, optionally, recomputes the same aggregates manually (mirroring the original example logic) to show side‑by‑side comparisons.
