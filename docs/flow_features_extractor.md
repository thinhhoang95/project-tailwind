FlowFeaturesExtractor — Per-flow Feature Aggregation
===================================================

Overview
- Computes non-pairwise, per-flow features over a hotspot period `[start_bin, end_bin)`.
- Rebuilds demand and travel offsets per flow, then loops over each requested hotspot bin.
- For every bin it evaluates `phase_time`, mass/price/slack helpers from `per_flow_features.py`, and accumulates period-level summaries.

Processing Outline (per hotspot bin)
1. Ensure the hotspot row exists in the τ map; if missing, recompute τ for the flow with sign inference and guarded fallbacks (warnings emitted), or finally fall back to `0`.
2. Derive the control-volume phase time `tG = phase_time(...)`, clamped to `[0, T-1]`.
3. Compute per-bin quantities: `x̂_GH, D_H, g_H`, `ṽ_{G→H}`, slack shifts, and `ρ_{G→H}`.
4. Track aggregate sums, bin counters, and global argmins for the slack features.

Feature Details

### Identifiers

**flow_id**
- Computation: sourced directly from each entry in the `flows_payload` (cast to `int`).
- Informal explanation: the numeric handle by which the upstream flow optimizer refers to this flow.
- Edge-case behavior:
  - If the payload omits the id or provides a non-integer, `int(...)` raises immediately; the caller needs to ensure the payload is well-formed.

**control_tv_id**
- Computation: copied as-is from `controlled_volume` in the payload; preserved as `None` when absent.
- Informal explanation: the traffic volume that serves as the flow's control point; helps relate τ offsets and slack rows back to real-world TVs.
- Edge-case behavior:
  - Serialized values are stringified; non-string inputs are coerced via `str(...)`.
  - When missing, downstream logic treats the flow as uncontrolled; re-computation of τ for the hotspot falls back to `0` offset.

### Phase-Time Bounds and Counting

**tGl / tGu**
- Computation: on every hotspot bin `b`, compute `tG = phase_time(h_row, Hotspot(tv, b), tau_map[fid], T)`. `tGl` is the minimum `tG` seen, `tGu` the maximum.
- Informal explanation: the earliest and latest control-volume times within the hotspot period. Think of them as the time window, in control space, that the flow interacts with while the hotspot is active.
- Edge-case behavior:
  - `phase_time` subtracts `τ_{G,s*}` if present; if the hotspot row is missing in τ, it pretends the offset is `0`.
  - Results are clamped to `[0, T-1]` so bins outside the planning horizon fold onto the boundary.
  - When `autotrim_from_ctrl_to_hotspot=True`, τ and flight sequences are truncated to the earliest hotspot visit; otherwise every touched TV keeps its τ, which can extend the observed `tGl/tGu` window.

**bins_count**
- Computation: increments once per hotspot bin processed for the flow. The final `FlowFeatures` stores the raw count; downstream averages use it as the divisor.
- Informal explanation: how many hotspot bins actually contributed to the aggregates for this flow.
- Edge-case behavior:
  - When the flow never touches a requested bin, the internal count stays at `0`, but the exported value is forced to `1` to keep average computations finite; paired sums remain `0`, so downstream math still yields `0` averages.

### Mass-Weight Components

**xGH**
- Computation: uses `mass_weight_gH`. For each bin, clamp `tG` to `[0, T-1]` and take `x̂_GH = xG[tG]`. Aggregate by summing across bins.
- Informal explanation: total controlled demand that arrives at the control point when the hotspot is relevant.
- Edge-case behavior:
  - If `xG` is empty, the per-bin contribution is `0`.
  - When `tG` falls outside the array bounds, it is clamped before indexing so no IndexError is possible.
  - Any NaNs already in `xG` pass through; upstream caches are expected to provide clean arrays.

**DH**
- Computation: also from `mass_weight_gH`. At each hotspot bin compute `D_H = max(0, roll_occ - capacity)` when rolling-hour caches are available; otherwise fall back to `max(0, -slack_per_bin_matrix[hotspot_row, bin])`. Sum over bins.
- Informal explanation: the total magnitude of hotspot overload that coincides with the flow across the evaluation period.
- Edge-case behavior:
  - Hourly indexing clamps hour ids to `[0, 23]` and catches exceptions by falling back to the cached slack matrix.
  - If the hotspot row or bin is out of range the deficit contributes `0`.
  - Positive slack (i.e., available capacity) yields `D_H=0`.

**gH_sum / gH_avg / gH**
- Computation:
  1. Per bin compute `g_H = x̂_GH / (x̂_GH + D_H)` unless the denominator is `<= eps`, in which case `g_H=0`.
  2. `gH_sum` adds those per-bin `g_H` values; `gH_avg` divides by `max(1, bins_count)` so empty flows still return `0` instead of NaN.
  3. Period-level `gH` is recomputed as `xGH / (xGH + DH)` with the same epsilon guard.
- Informal explanation: `g_H` is the share of overload that could be removed by this flow; the derived `gH` re-evaluates that share using total mass and deficit.
- Edge-case behavior:
  - If both `x̂_GH` and `D_H` are negligible (`<=eps`), every variant collapses to `0`.
  - `gH_avg` is meaningful only when the flow actually touched bins; when no bins were counted, the guard divisor of `1` still yields `0`.
  - Because the derived `gH` uses sums, it can differ from `gH_avg` whenever `g_H` fluctuates across bins.

### Pricing Components

**v_tilde**
- Computation: `price_contrib_v_tilde` builds a unit-price matrix `P` from `hourly_excess_bool` and an optional `theta_mask`, computes aligned indices `t_hot(s)` and `t_ctl(s)`, then evaluates
  - `x_{s,G} = xG[t_hot(s)]`,
  - `D_s = max(0, roll_occ − capacity)` (fallback: `max(0, -slack)`), and
  - `ω_{s,G|H} = min(1, x_{s,G} / (D_s + eps))` for rows touched by the flow.
  The final value is `P_{s*,t*} ω_{s*,G|H} + κ Σ_{s≠s*} P_{s,t_hot(s)} ω_{s,G|H}`. The extractor sums this value across all hotspot bins.
- Informal explanation: revenue-weighted bang-for-buck; it measures how much priced overload the flow could relieve directly at the hotspot and in nearby sectors.
- Edge-case behavior:
  - All indices are clamped to `[0, T-1]`, so even large τ offsets stay in-bounds.
  - Rows not present in τ get `ω=0` and therefore contribute nothing.
  - If `hourly_excess_bool` reports no overload at either the hotspot or aligned ripple cells, the price matrix is zeroed and `ṽ` collapses to `0`.
  - Any failure while reading rolling occupancy or capacity falls back to the cached slack matrix to keep the value finite.

**gH_v_tilde**
- Computation: computed only once per flow after accumulation as `gH * v_tilde` using the derived `gH` and the period-summed `v_tilde`.
- Informal explanation: combines effectiveness (`gH`) with priced opportunity (`ṽ`) into a single scalar, mirroring the product that appears in Rev1 scores.
- Edge-case behavior:
  - If either factor is zero, the product is zero; no additional guards are applied.
  - Uses the epsilon-guarded `gH`, so it never amplifies noise from near-empty denominators.

### Slack Aggregates

General setup: for each `Δ ∈ {0, 15, 30, 45}` minutes, convert the minute shift to bins via `round(Δ / time_bin_minutes)` with a minimum of `1` bin for any positive `Δ`. Evaluate the slack helper at `t_eval = tG + shift` whenever `t_eval` is inside `[0, T-1]`.

**Slack_GΔ (Δ ∈ {0, 15, 30, 45})**
- Computation: sum `slack_G_at(t_eval, τ, S_mat, ...)` over all hotspot bins. `slack_G_at` restricts to rows touched by τ, shifts each row by its τ, and takes the minimum slack (capacity − occupancy when the rolling caches exist, or the cached slack otherwise).
- Informal explanation: aggregated residual slack observed by the flow at the control time plus the evaluated shift. Larger values mean more spare capacity available when the flow would arrive after the shift.
- Edge-case behavior:
  - When `t_eval` exits the planning horizon, that bin contributes nothing for the corresponding `Δ`.
  - If the flow touches no rows (τ empty) the helper returns `0`, so the sum stays at `0`.
  - Any issue while reading rolling occupancy/capacity gracefully falls back to `S_mat` to avoid NaNs.

**Slack_GΔ_row (Δ ∈ {0, 15, 30, 45})**
- Computation: tracks the single row index that achieves the smallest `Slack_G(t_eval)` over the entire hotspot period. `_slack_min_row` evaluates the same aligned slack slice as `slack_G_at`, then returns the row for the minimal slack value if one exists.
- Informal explanation: points to the tightest sector (after applying Δ) that the flow interacts with, which is often the bottleneck for eligibility.
- Edge-case behavior:
  - Initialized to `None`; remains `None` if every `t_eval` lands outside the horizon or τ is empty.
  - On ties, NumPy’s `argmin` picks the first row in index order; there is no deterministic tie-break beyond row order.
  - Uses the same fallback path as `Slack_GΔ` when rolling occupancy data are unavailable.

**Slack_GΔ_occ / Slack_GΔ_cap (Δ ∈ {0, 15, 30, 45})**
- Computation: stored alongside `Slack_GΔ_row`. Whenever a new global minimum slack is observed for a given Δ, the extractor records the rolling-hour occupancy `rolling_occ_by_bin[row, aligned_bin]` and hourly capacity `hourly_capacity_matrix[row, hour_idx]` from the same aligned bin `t_eval = tG + shift`. These values therefore correspond exactly to the bin/time pair that produced the minimum slack across the requested hotspot bin range. If the rolling caches are unavailable, both fields remain `None` and the slack value falls back to `S_mat`.
- Informal explanation: surfaces the context (demand vs capacity) at the tightest point the flow encounters after applying the Δ shift, making it easy to see how close to capacity the bottleneck was when slack was minimal.
- Edge-case behavior:
  - When the minimum slack comes from a bin outside the rolling-hour cache range (e.g., due to clamping) the occupancy/capacity values are taken from the clamped index; if the cache lookup fails, they revert to `None` while the slack value falls back to the static matrix.
  - If no minimum is recorded (because the flow never yields an in-range `t_eval`), both fields stay `None`.
  - The hour index used for capacity is `floor((t_eval + τ_row) / bins_per_hour)` and is clamped to the matrix width to avoid IndexError.

### Risk Penalty

**rho**
- Computation: sum the per-bin result of `slack_penalty`. The helper:
  1. Restricts to rows touched by τ and evaluates `Slack_G(tG)` using the same occupancy/capacity preference as above.
  2. Finds the row/time pair `(r̂, t̂)` with minimal slack.
  3. Chooses the normalization `S0_eff` based on `HyperParams.S0_mode` (default `"x_at_argmin"` uses `xG[t̂]`; other modes use `xG[tG]` or the constant `S0`).
  4. Returns `max(0, 1 - Slack_G(tG) / S0_eff)`; if `S0_eff <= 0`, returns `0`.
- Informal explanation: a soft penalty for flows that would still exceed slack thresholds even after being applied; larger `ρ` means the flow remains risky.
- Edge-case behavior:
  - If τ is empty or the slack slice is empty after clamping, the penalty is `0`.
  - Any failure while sampling rolling occupancy falls back to cached slack values.
  - Negative or zero `S0_eff` (e.g., because `xG[t̂] = 0`) short-circuits the penalty to `0`.

Inputs
- `indexer`: supplies `num_time_bins` and `time_bin_minutes`, used for bin clamping and Δ shifts.
- `flight_list`: holds TV id/index mappings, occupancy caches, and per-flight TV sequences (needed for τ trimming).
- `capacities_by_tv`: per-TV capacity arrays backing the occupancy/capacity caches.
- `travel_minutes_map`: raw minutes used to build τ bin offsets.
- `params`: optional `HyperParams` for epsilon, pricing weights, and `S0` settings (defaults keep `S0_mode="x_at_argmin"`).
- `flows_payload`: optional reuse of `parrhesia.api.flows.compute_flows`; if omitted the extractor recomputes flows internally.
- `autotrim_from_ctrl_to_hotspot` (default `False`): trims τ rows and per-flight TV sequences to the prefix up to the first hotspot visit when `True`, ensuring all features inspect only the part of the flow before reaching the hotspot.

τ Inference and Fallbacks
- Primary sign mode is `order_vs_ctrl` using per-flight TV sequences. When `autotrim_from_ctrl_to_hotspot=True`, sequences are trimmed to the earliest hotspot visit before sign inference.
- Initial τ construction inside the extractor uses `order_vs_ctrl`.
- If, during per-bin processing, the hotspot row is missing from τ for a given flow, the extractor recomputes τ for that flow with the following preference:
  - If flight IDs are available: use `order_vs_ctrl`.
  - Else if `tv_centroids` are available on the indexer/direction options: use `vector_centroid` (geometric fallback).
  - Else: fall back to magnitude-only τ (non-signed).
- In all fallback cases, a `RuntimeWarning` is emitted to help explain potential anomalies. If the hotspot τ remains missing after recompute or an exception occurs, the extractor warns and uses `τ(hotspot)=0`.
- Note: the extractor does not use any `direction_opts.mode` string to select the sign mode; it only consumes `tv_centroids` if provided to enable the geometric fallback.

API Surface
- Class: `parrhesia.metaopt.feats.FlowFeaturesExtractor`
  - `__init__(indexer, flight_list, capacities_by_tv, travel_minutes_map, params=None, autotrim_from_ctrl_to_hotspot=False)`
  - `compute_for_hotspot(hotspot_tv: str, timebins: Sequence[int], flows_payload=None, direction_opts=None) -> Dict[int, FlowFeatures]`
- Dataclass: `FlowFeatures`
  - fields listed above; each is populated following the computations described in “Feature Details”.

Usage Notes
- Minutes-to-bin conversion uses `round` and enforces a minimum shift of one bin whenever the minute shift is positive, so a 15-minute shift still advances by one bin even if bins are coarser.
- When rolling-hour occupancy or hourly capacity are missing from caches, every helper transparently reverts to the static slack matrix; values remain non-negative because the cache stores post-clipped slack.
- Slack row indices and penalties only consider rows actually touched by the flow after τ trimming, keeping feature semantics aligned with the ripple domain used in pricing.
- The extractor emits `RuntimeWarning`s when it must fall back from order-based signs to geometric signs or to magnitude-only τ, and when it must default `τ(hotspot)=0`.

Example

```python
from parrhesia.metaopt.feats import FlowFeaturesExtractor
from parrhesia.metaopt.types import HyperParams

extractor = FlowFeaturesExtractor(
    indexer,
    flight_list,
    capacities_by_tv,
    travel_minutes_map,
    params=HyperParams(S0_mode="x_at_argmin"),
)

features = extractor.compute_for_hotspot(
    hotspot_tv="LSGL13W",
    timebins=[45, 46, 47, 48],
    flows_payload=flows_payload,
)

for fid, feats in features.items():
    print(fid, feats.gH, feats.v_tilde, feats.Slack_G15, feats.Slack_G15_row)
```

Further Reading
- `src/parrhesia/metaopt/usage/flows_features_extractor_example.py` recomputes each metric manually to illustrate how the helpers align.
