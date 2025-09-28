### Goal
Implement a regulation-proposal “regen” module that, for a single hotspot, ranks candidate flows, bundles 1–3 flows, picks a robust regulation window, translates hotspot exceedance to per-flow hourly rate cuts, performs a tiny local search, predicts improvement via the safe-spill objective using a localized context, and returns k diverse proposals with diagnostics. This plan is an exact, detailed blueprint from `prompts/reg_generator/regen_resp1.md` with no simplifications applied.

### Key upstream APIs you’ll integrate with
- Flow features and per-bin caches:
```64:106:src/parrhesia/metaopt/feats/flow_features.py
class FlowFeaturesExtractor:
    """
    Compute per-flow, non-pairwise features over a hotspot time period.
    ...
    """
    def __init__(
        self,
        indexer: Any,
        flight_list: Any,
        capacities_by_tv: Mapping[str, np.ndarray],
        travel_minutes_map: Mapping[str, Mapping[str, float]],
        params: Optional[HyperParams] = None,
        *,
        autotrim_from_ctrl_to_hotspot: bool = False,
    ) -> None:
        self.indexer = indexer
        self.flight_list = flight_list
        self.capacities_by_tv = capacities_by_tv
        self.travel_minutes_map = travel_minutes_map
        self.params = params or HyperParams()
        self.autotrim_from_ctrl_to_hotspot: bool = bool(autotrim_from_ctrl_to_hotspot)

        # Build caches once
        self.caches: Dict[str, Any] = build_base_caches(
            flight_list=self.flight_list,
            capacities_by_tv=self.capacities_by_tv,
            indexer=self.indexer,
        )
```

```275:281:src/parrhesia/metaopt/feats/flow_features.py
def compute_for_hotspot(
        self,
        hotspot_tv: str,
        timebins: Sequence[int],
        *,
        flows_payload: Optional[Mapping[str, Any]] = None,
        direction_opts: Optional[Mapping[str, Any]] = None,
    ) -> Dict[int, FlowFeatures]:
```

- Safe-spill objective scorer:
```20:28:src/parrhesia/flow_agent/safespill_objective.py
def score_with_context(
    n_f_t: Mapping[Any, Union[Sequence[int], Mapping[int, int]]],
    *,
    flights_by_flow: Mapping[Any, Sequence[Any]],
    capacities_by_tv: Mapping[str, np.ndarray],
    flight_list: Optional[object],
    context: ScoreContext,
    audit_exceedances: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
```

- Indexer helpers:
```187:195:src/project_tailwind/impact_eval/tvtw_indexer.py
def rolling_window_size(self) -> int:
    """
    Return K, the number of bins in a rolling-hour window, i.e.,
    K = 60 / time_bin_minutes. Raises if 60 is not divisible by bin size.
    """
    if 60 % self.time_bin_minutes != 0:
        raise ValueError("60 is not divisible by time_bin_minutes; cannot form hour window")
    return 60 // self.time_bin_minutes
```

### New package layout (to add)
- `src/parrhesia/flow_agent35/regen/__init__.py`
- `src/parrhesia/flow_agent35/regen/types.py`
- `src/parrhesia/flow_agent35/regen/config.py`
- `src/parrhesia/flow_agent35/regen/exceedance.py`
- `src/parrhesia/flow_agent35/regen/features_bridge.py`
- `src/parrhesia/flow_agent35/regen/scoring.py`
- `src/parrhesia/flow_agent35/regen/bundles.py`
- `src/parrhesia/flow_agent35/regen/window.py`
- `src/parrhesia/flow_agent35/regen/rates.py`
- `src/parrhesia/flow_agent35/regen/search.py`
- `src/parrhesia/flow_agent35/regen/predict.py`
- `src/parrhesia/flow_agent35/regen/engine.py`
- Optional CLI: `src/parrhesia/flow_agent35/regen/cli.py`

### Public API
- `propose_regulations_for_hotspot(...) -> List[Proposal]` in `engine.py`
  - Minimal inputs (strict):
    - `indexer: TVTWIndexer`
    - `flight_list: FlightList-like`
    - `capacities_by_tv: Mapping[str, np.ndarray]`
    - `travel_minutes_map: Mapping[str, Mapping[str, float]]`
    - `hotspot_tv: str`
    - `timebins_h: Sequence[int]` (inclusive bin indices of hotspot)
    - `flows_payload: Mapping[str, Any]` OR `flow_to_flights: Mapping[int|str, Sequence[str]]`
    - `weights: FlowScoreWeights | None`
    - `config: RegenConfig | None`
  - Returns `List[Proposal]` sorted by predicted improvement with diversity.

### Data models (in `types.py`)
- `FlowScoreWeights`: fields `w1..w7` with defaults from spec.
- `RegenConfig`:
  - `g_min=0.1, rho_max=0.7, slack_min=2`
  - `window_margin_hours=0.25, min_window_hours=1.0`
  - `e_target_mode="q95"`; `fallback_to_peak=True`
  - `convert_occupancy_to_entrance=True|False`, `dwell_minutes: Optional[float]`, `alpha_occupancy_to_entrance: float=1.0` (when no dwell)
  - `local_search_steps=(±1, ±2)`, `max_variants_per_bundle=8`
  - `diversity_alpha=0.2`, `k_proposals=5`
  - `max_bundle_size=3`
  - `distinct_controls_required=True`
  - `autotrim_from_ctrl_to_hotspot=False`
  - `raise_on_edge_cases=True`
- `FlowDiagnostics`: fields for gH, gH_v_tilde, v_tilde, ρ, slack15/30/45, coverage, r0_i, xGH, DH, tGl, tGu, bins_count.
- `FlowScore`: `flow_id, control_tv_id, score, diagnostics: FlowDiagnostics`
- `Bundle`: `flows: List[FlowScore]`, `weights_by_flow: Dict[flow_id, float]`
- `Window`: `start_bin: int, end_bin: int  # inclusive`
- `RateCut`: `flow_id, baseline_rate_r0, cut_per_hour_lambda, allowed_rate_R`
- `BundleVariant`: `bundle, window, rates: List[RateCut]`
- `PredictedImprovement`: `delta_deficit_per_hour: float, delta_objective_score: float`
- `Proposal`: 
  - `hotspot_id: str`, `controlled_volume: str`
  - `window: Window`
  - `flows_info: List[{flow_id, control_tv_id, R_i, r0_i, lambda_cut_i}]`
  - `predicted_improvement: PredictedImprovement`
  - `diagnostics: {E_target, per-flow metrics, coverage_i, weights_used}`

### Detailed algorithm and module responsibilities

- `exceedance.py`
  - `compute_hotspot_exceedance(indexer, flight_list, capacities_by_tv, hotspot_tv, timebins_h) -> Dict[str, Any]`
    - Use base caches from `FlowFeaturesExtractor` or a minimal cache builder to reuse rolling occupancy and capacity per bin.
    - Compute per-bin exceedance: `D_b = max(0, rolling_occ[tv][b] − hourly_cap[tv][b])` for each `b ∈ timebins_h`.
    - Return `D_vec, D_total, D_peak, D_q95`.
  - Notes:
    - Prefer reusing `build_base_caches(...)` via `features_bridge` to get `rolling_occ_by_bin` and per-hour capacities aligned per bin, avoiding recomputation.

- `features_bridge.py`
  - `extract_features_for_flows(indexer, flight_list, capacities_by_tv, travel_minutes_map, hotspot_tv, timebins_h, flows_payload, autotrim_from_ctrl_to_hotspot) -> Dict[int, FlowFeatures]`
    - Thin adapter over `FlowFeaturesExtractor(...).compute_for_hotspot(...)`, returning the `FlowFeatures` per flow.
  - `baseline_demand_by_flow_from_payload(...) -> Dict[flow_id, np.ndarray]`
    - If `flows_payload` or `HotspotInventory.metadata.flow_proxies` provide entrant histograms aligned to `timebins_h`, normalize them to length `T` and produce demand vectors.
    - Else, reconstruct from `flight_list` by placing each flight’s `requested_bin` (via upstream scheduling inputs) into `d_by_flow[f][t] += 1`.
  - `coverage_i(...)` helper from `tGl, tGu` and `timebins_h`.

- `scoring.py`
  - `prune_flows(features: Dict[int, FlowFeatures], demand_by_flow: Dict[int, np.ndarray], config: RegenConfig) -> List[int]`
    - Drop flows with xGH≈0 or r0_i≈0, `gH ≤ g_min`, `ρ ≥ rho_max`, `Slack_G15 ≤ slack_min` (with optional substitution using Slack_G30/45).
  - `score_flows(eligible_flows, features, demand_by_flow, weights: FlowScoreWeights, indexer) -> List[FlowScore]`
    - `avg_bin_demand_i = xGH / max(1, bins_count)`.
    - `r0_i = avg_bin_demand_i × bins_per_hour`.
    - `coverage_i = (min(T-1,tGu) − max(0,tGl) + 1) / len(timebins_h) clamped [0,1]`.
    - `S_i = w1*gH + w2*gH_v_tilde + w3*v_tilde + w4*Slack_G15 + w5*Slack_G30 − w6*ρ + w7*coverage_i`, clamp to `max(0, S_i)`.

- `bundles.py`
  - `build_candidate_bundles(scored: List[FlowScore], max_bundle_size=3, distinct_controls_required=True) -> List[Bundle]`
    - Top-1, top-2, top-3 by `score`, skipping combinations with duplicate `control_tv_id` if required.
  - Optional: Use deterministic dedup key: frozenset of `flow_id`s.

- `window.py`
  - `select_window_for_bundle(bundle: Bundle, features, indexer, timebins_h, margin_hours=0.25, min_window_hours=1.0) -> Window`
    - `start = min_i tGl_i`, `end = max_i tGu_i`, expand by `ceil(margin * bins_per_hour)` on both sides with backward bias (as per spec).
    - Clip to `[0, T-1]`; enforce `end - start + 1 ≥ bins_per_hour`.

- `rates.py`
  - `compute_e_target(D_vec, mode="q95", fallback_to_peak=True) -> float`
  - `occupancy_to_entrance(E_target_occ, dwell_minutes, alpha)` 
    - If `convert_occupancy_to_entrance`: `rate_cut_per_hour ≈ (60 / dwell_minutes) × occupancy_reduction`; otherwise multiply by `alpha` (defaults to 1.0).
  - `rate_cuts_for_bundle(bundle, E_target, features, demand_by_flow, indexer) -> List[RateCut]`
    - Weights `w_i = S_i / sum_j S_j` (uniform if all zero).
    - `λcut_i = round_to_int(w_i * E_target)`, clamp `≤ r0_i`, enforce floor of `1` if `λcut_i>0`.
    - `R_i = max(0, r0_i − λcut_i)`.
  - `distribute_hourly_rate_to_bins(R_i, indexer, start_bin, end_bin) -> np.ndarray`
    - Let `bph = bins_per_hour`, `R_i = q*bph + r`; build per-bin allowance as `q` plus `1` in `r` bins inside each hour segment; clamp by demand later.

- `search.py`
  - `local_search_variants(bundle, window, base_cuts, steps=(±1,±2), max_variants=8) -> List[List[RateCut]]`
    - Generate small integer perturbations per flow, clamp `[0, r0_i]`, limit variant count. Two modes:
      - independent per-flow tweaks,
      - coupled tweaks that approximately preserve total cut (configurable).
  - De-duplicate and keep the best `max_variants` by heuristic (e.g., balance across flows).

- `predict.py`
  - `build_local_context(indexer, flight_list, capacities_by_tv, target_cells, flights_by_flow, weights) -> ScoreContext`
    - Use `parrhesia.optim.objective.build_score_context(...)` to prepare a localized `ScoreContext` with `target_cells` the hotspot TV(s), and optionally a tv-filter (union of hotspot and ripple TVs).
  - `baseline_schedule_from_context(context) -> Dict[flow, np.ndarray]`
    - `n0_f_t = context.d_by_flow` (copy).
  - `apply_regulation_to_schedule(n0_f_t, demand_by_flow, rates_per_flow, indexer, window) -> Dict[flow, np.ndarray]`
    - For each bundle flow and for bins in `[start, end]`, clamp `n[f][t] = min(demand[f][t], allowed[t])`, where `allowed[t]` comes from `distribute_hourly_rate_to_bins`.
  - `score_pair(n0, n_reg, context, flights_by_flow, capacities_by_tv, flight_list) -> (J_before, J_after, occ_before, occ_after)`
    - Call `score_with_context(...)` twice with identical `context` and `audit_exceedances=True`.
  - `compute_delta_deficit_per_hour(occ_pre, occ_post, capacities_by_tv, target_cells, indexer, window) -> float`
    - For each target TV and each bin `t` intersecting `[start, end]`, compute rolling forward sums over width `K=bins_per_hour`: `ex_t = max(0, sum_{τ=t..t+K-1} occ[τ] − sum_{τ=t..t+K-1} cap[τ])`.
    - Average `ex_t` over bins in window; sum across target TVs. Return pre−post.

- `engine.py`
  - `propose_regulations_for_hotspot(...)`
    - Assemble timebins_h; compute `D_vec, E_target`.
    - Extract features and baseline demand; prune; score; build bundles.
    - For each bundle:
      - Pick window.
      - Convert exceedance to entrance-rate target (honoring dwell or `alpha`).
      - Compute base cuts and small-variant grids.
      - Build one context per hotspot (reuse across bundle variants).
      - For each variant: build `n_reg`, score both schedules, compute `delta_deficit_per_hour` and `delta_objective_score`.
      - Keep best variant per bundle.
    - Rank proposals by predicted improvement − diversity penalty; return top `k`.
  - Diversity penalty:
    - `overlap(B1, B2) = 0.5*Jaccard(flow_ids) + 0.5*time_overlap_ratio`, or as spec: Jaccard on flow sets plus window overlap ratio; `div_penalty = α * max_overlap_with_selected`.

### Assumptions and non-simplifications
- We will implement both occupancy→entrance conversion modes:
  - exact dwell-based conversion when `dwell_minutes` is provided,
  - scalar `alpha_occupancy_to_entrance` fallback (default 1.0) when not calibrated.
- We will compute rolling-hour exceedances via forward sums consistent with caches and indexer `rolling_window_size()`. No approximation.
- Local search will implement both independent and coupled modes; default independent to match spec; coupled mode guarded by config.
- We will raise explicit exceptions for malformed inputs (missing flows, time-bin mismatches, unknown TVs) per “Important Remarks”.

### Edge cases to handle (with explicit behavior)
- All `S_i == 0`: use uniform weights in cut allocation; if resulting `sum λcut_i == 0`, increase `E_target` by one unit step or drop bundle (configurable; default: drop).
- `D_q95 == 0` and `fallback_to_peak=True`: use `D_peak`. If still `0`, drop bundle.
- `r0_i == 0`, `xGH == 0`, `bins_count == 0`: flow pruned.
- Windows clipped at day bounds; enforce minimum 1 hour even if `tGl..tGu` is shorter.
- Distinct controls constraint: enforced for 2- and 3-flow bundles; one-flow bundle always allowed.
- Demand clamping when distributing allowed per-bin rates.
- If feature extractor must recompute τ signs, propagate warnings; do not silently ignore.

### Configuration defaults (exactly as spec)
- g_min=0.1, rho_max=0.7, slack_min=2 flights
- Window margin=0.25 hour each side, min window=1 hour
- E_target=D_q95; fallback D_peak if D_q95=0
- Local search: ±1 and ±2 flights/h per selected flow; ≤8 variants
- Diversity α=0.2

### Performance & caching
- Build a single `ScoreContext` per hotspot and reuse across bundles/variants.
- Reuse `FlowFeaturesExtractor.caches` to avoid recomputing rolling occupancy and per-bin capacity for exceedance calculation.
- Precompute and cache:
  - `tvs_of_interest` (hotspot + flows’ footprints), 
  - `baseline n0` and `demand_by_flow`,
  - `base_occ_all_by_tv` and optionally `base_occ_sched_zero_by_tv` if needed for fast deltas.
- Limit variants to `max_variants_per_bundle` to keep evaluation O(bundles × variants).

### Example usage (engine)
```python
from parrhesia.flow_agent35.regen.engine import propose_regulations_for_hotspot
from parrhesia.flow_agent35.regen.config import RegenConfig, FlowScoreWeights

proposals = propose_regulations_for_hotspot(
    indexer=idx,
    flight_list=fl,
    capacities_by_tv=capacities_by_tv,
    travel_minutes_map=travel_minutes_map,
    hotspot_tv="LSGL13W",
    timebins_h=list(range(t0, t1+1)),
    flows_payload=hotspot_descriptor.metadata.get("flow_to_flights"),
    weights=FlowScoreWeights(),         # defaults per spec
    config=RegenConfig(                 # defaults per spec
        convert_occupancy_to_entrance=False,  # set True when dwell calibrated
        alpha_occupancy_to_entrance=1.0,
    ),
)
```

### Implementation notes tying to the spec (no simplifications)
- Exceedance target and dwell conversion: implemented verbatim with both modes; default `alpha=1.0` unless `dwell_minutes` is provided.
- Demand and r0_i: use observed entrants over the hotspot window (flow proxies if available); `r0_i` is per-hour baseline when hotspot active, `avg_bin_demand × bins_per_hour`.
- Window selection biases start backward by margin per the spec; clipping and minimum duration enforced.
- Allowed-rate discretization distributes remainder `r` bins within each hour; clamp by per-bin demand before scoring.
- Prediction uses `score_with_context` (safe spill), identical context for before/after, and computes rolling exceedance deltas directly from returned occupancy to produce `delta_deficit_per_hour`.


- Final proposals return exactly:
  - `hotspot_id`, `controlled_volume`
  - `window: [start_bin, end_bin]`
  - `flow list: [{flow_id, control_tv_id, R_i, r0_i, λcut_i}]`
  - `predicted_improvement: {delta_deficit_per_hour, delta_objective_score}`
  - `diagnostics: {E_target, per-flow metrics, coverage_i, weights_used}`

- Assumptions explicitly called:
  - Caller supplies `capacities_by_tv` consistent with `flight_list` and `indexer`.
  - Caller supplies `flows_payload` or `flow_to_flights`; otherwise, regen cannot proceed and will raise.
  - Day-bound windows only; multi-day windows would require additional handling and are out of scope unless requested.
