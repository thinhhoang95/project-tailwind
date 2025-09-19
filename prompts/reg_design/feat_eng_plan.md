## 1) Inputs inventory for each expression

Context:
- Let H = (s*, t*) be the hotspot TV/time-bin.
- Let G denote a candidate flow with a control volume s_ctrl and a flow activity time series x_G(t) at s_ctrl.
- Time is in TVTW bins; per-hour quantities are distributed across `bins_per_hour = 60 / time_bin_minutes`.

From the doc:

- v_G(t) = sum_s [w_sum + w_max θ_{s, t + τ_{G,s}}] · 1{o_s(t + τ_{G,s}) > 0}
  - Inputs:
    - Set of TVs s (the TV universe in the indexer).
    - Weights: w_sum, w_max.
    - θ_{s,t}: indicator/weight for attention cells; minimally 1 for (s*, t*) and 0 elsewhere; optionally nonzero for ripple cells.
    - o_s(t): overload indicator. Use hourly-excess > 0 (rolling-hour occupancy − hourly capacity) at (s,t).
    - τ_{G,s}: bin offset from the flow’s control volume time to TV s. Derived from TV-to-TV travel minutes matrix and time_bin_minutes.
    - time_bin_minutes, bins_per_hour.

- Slack_G(t) = min_s s_s(t + τ_{G,s})
  - Inputs:
    - s_s(t): slack per TV and bin = max(capacity_per_bin − occupancy_per_bin, 0), where capacity_per_bin repeats hourly throughput within the hour; occupancy is the base (pre-regulation) occupancy per bin.
    - τ_{G,s} as above.

- Phase alignment: t_G := t* − τ_{G,s*}
  - Inputs:
    - τ_{G,s*} and hotspot time t*.

- v_{G→H}:
  - v_{G→H} = [w_sum + w_max θ_{s*,t*}] 1{o_{s*}(t*)>0}
    + κ ∑_{s≠s*} [w_sum + w_max θ_{s, t* + τ_{G,s} − τ_{G,s*}}] 1{o_s(t* + τ_{G,s} − τ_{G,s*})>0}
  - Inputs:
    - Same as v_G plus κ ∈ [0,1] and the hotspot H = (s*,t*).

- Eligibility a_{G→H}:
  - Hard: 1{x_G(t_G) ≥ q0}, or Soft: σ(γ(x_G(t_G) − q0))
  - Inputs:
    - x_G(t): flow activity at the control volume (counts at s_ctrl per bin).
    - Thresholds q0, γ; sigmoid σ(·).
    - t_G.

- Slack-risk penalty:
  - ρ_{G→H} = [1 − Slack_G(t_G)/S0]_+
  - Inputs:
    - Slack_G(t_G), normalizer S0 (e.g., a slack unit or percentile of slack distribution).

- Matched-filter net score:
  - Score(G | H) = α · a_{G→H} · v_{G→H} − β · ρ_{G→H} − λ_delay (optional per-minute penalty)
  - Inputs:
    - α, β, λ_delay.

Union-cap vs separate-caps features:

- Overlap_{ij} = ∑_{t∈W} min{x_i(t), x_j(t)}
  - Inputs:
    - Per-flow activity time series x_i(t) and x_j(t) at each flow’s control volume. Window W around the phases (config).

- Orth_{ij} =
  1 − |{s: o_s(t* + τ_{G_i,s} − τ_{G_i,s*})>0} ∩ {s: o_s(t* + τ_{G_j,s} − τ_{G_j,s*})>0}| / |{s: o_s(·)>0}|
  - Inputs:
    - Overloaded-TV indicator set at times aligned to each flow’s phase; universe of TVs that ever overload (or in a window around t*).

- SlackCorr_{ij} = corr(Slack_{G_i}(·), Slack_{G_j}(·)) on a window around phases
  - Inputs:
    - Slack_{G}(t) profiles versus t around t_G for each flow.

- PriceGap_{ij} = |v_{G_i}(t_{G_i}) − v_{G_j}(t_{G_j})| / (v_{G_i}(t_{G_i}) + v_{G_j}(t_{G_j}) + ε)
  - Inputs:
    - v_{G}(t_G) for each flow, ε.

- Decision thresholds: τ_ov, τ_sl, τ_pr, τ_orth (hyperparameters).

Shared base inputs we will precompute:
- TVTW indexer (TV list, num_time_bins, tv_row_of_tvtw, hour_of_tvtw).
- Base occupancy per TVTW: occ_base (counts).
- Hourly capacity map and per-bin capacity vector: cap_per_bin; hourly capacity matrix [num_tvs, 24].
- Rolling-hour occupancy per TV row and bin to compute hourly_excess (or compute hourly_excess via aggregation ops mirroring evaluator).
- Slack vector per TVTW: slack = max(cap_per_bin − occ_base, 0).
- Overload indicator per TVTW: o = (hourly_excess > 0).
- TV-to-TV travel minutes matrix, turned into bin offsets via time_bin_minutes.
- Flow groups/mapping and each flow’s control volume and requested bins at control volume: from `prepare_flow_scheduling_inputs` output, i.e., `flights_by_flow`, `ctrl_by_flow` (per `src/parrhesia/pipeline.py`).

## 2) Computation plan (reusing existing code or proposing minimal additions)

Precomputation layer (global, once per dataset):
- Base occupancy, capacity, slack, hourly excess, and mapping arrays:
  - Mirror the vectorized cache pattern in `src/project_tailwind/flow_x/cache_flow_extract_debug1.py`:
    - occ_base = FlightList.get_total_occupancy_by_tvtw().
    - num_tvtws, num_tvs, num_time_bins_per_tv, bins_per_hour.
    - tv_row_of_tvtw, hour_of_tvtw arrays.
    - cap_per_bin: from hourly capacity distributed across bins.
    - hourly_capacity_matrix [num_tvs, 24].
    - rolling-hour occupancy per TV row; hourly_excess = max(rolling − hourly_capacity_by_hour, 0). This reproduces o_s(t) at hour granularity.
  - Slack vector: slack = max(cap_per_bin − occ_base, 0).
  - Overload indicator: o_bool = (hourly_excess > 0).
  - Attention mask θ_{s,t}: default θ=1 for the hotspot cell (s*, t*); optionally add ripple weights if a ripple set is provided.

- TV-to-TV travel offsets:
  - Build and cache pairwise travel minutes once (see `prompts/slack_api/slack_planning.md`).
  - minutes_to_bins = round(minutes / time_bin_minutes).
  - For each flow’s control volume `s_ctrl`, derive τ_{G,s} for all TVs s using that row of the matrix (positive or negative depending on plus/minus convention; for ground delay use forward shifts from s_ctrl to s).

Per-flow summaries:
- Flow occupancy vector g0:
  - Sum per-flight occupancy vectors for flights in flow G (vectorized).
- Flow control-volume activity x_G(t):
  - From `flights_by_flow[G]`’s `requested_bin` at `ctrl_by_flow[G]`, histogram into a per-bin count array x_G(t) of length `num_time_bins_per_tv` for the control row.

Per-flow per-hotspot features:
- Phase time t_G:
  - t_G = t* − τ_{G,s*}.
- v_G(t_G):
  - For each TV s, consider time index u = t_G + τ_{G,s}. Contribution is (w_sum + w_max θ_{s,u}) if o_s(u) > 0, else 0.
  - Implementation: use τ_{G,*} to map a boolean vector of overloaded bins per TV to the phase-shifted u indices and sum the weighted mask quickly (vectorized gather with bounds checks). θ is looked up for (s,u).
- v_{G→H}:
  - Local term: [w_sum + w_max θ_{s*,t*}] if o_{s*}(t*) > 0.
  - Collateral term: κ times the same sum as v_G but evaluated at u' = t* + τ_{G,s} − τ_{G,s*}, s ≠ s*.
- Slack_G(t_G):
  - For each TV s, look up slack at u = t_G + τ_{G,s}; take min over s. Vectorized gather using τ offsets.
- a_{G→H}:
  - Hard: 1{x_G(t_G) ≥ q0}, or
  - Soft: sigmoid(γ(x_G(t_G) − q0)).
- ρ_{G→H}:
  - ρ = max(1 − Slack_G(t_G)/S0, 0).
- Score(G | H):
  - α a v − β ρ − λ_delay, with λ_delay optional (e.g., per-minute penalty per unit removal).

Top-k flow selection:
- Rank flows G by Score(G|H). Keep top K per hotspot.

Union-cap vs separate-caps features:
- Overlap_{ij}:
  - Over a window W around each flow’s phase (e.g., t ∈ [t_G − w_left, t_G + w_right] intersected with valid range), compute ∑ min{x_i(t), x_j(t)}.
- Orth_{ij}:
  - Build two TV sets using o_s at times aligned to each flow’s phase at hotspot: T_i = {s: o_s(t* + τ_{G_i,s} − τ_{G_i,s*}) > 0}, T_j similarly. Orth = 1 − |T_i ∩ T_j| / |T_all_overloaded|. Use a consistent TV universe, possibly TVs that overload within the same broader window around t*.
- SlackCorr_{ij}:
  - For each flow, compute a slack profile S_G(Δ) = Slack_G(t_G + Δ) for Δ in a small window. Compute Pearson corr(S_Gi, S_Gj).
- PriceGap_{ij}:
  - Using v_{G}(t_G) from above.
- Decision:
  - Union if Overlap ≤ τ_ov and SlackCorr ≥ τ_sl and PriceGap ≤ τ_pr.
  - Separate if Orth ≥ τ_orth or PriceGap ≥ τ_pr.
  - Tie-break by price/waste difference using precomputed v_G.

Proposing regulation plan to feed the rate optimizer:
- For each hotspot H:
  - Select top flows and apply the union/separate rule (cluster if >2 flows).
  - For each chosen group:
    - Control volume: use `ctrl_by_flow` if separate caps; for union, pick the modal control volume among group members or explicitly designate the hotspot’s upstream “best” control (simple rule: the control volume with the largest x_G peak near t_G).
    - Active time windows:
      - Center at t_G; select a contiguous window W where x_G(t) exceeds a fraction of its peak or covers the hotspot overloaded window mapped to the control row (e.g., bins covering the hotspot’s overloaded segment translated by τ_{G,s*}).
      - Materialize as a list of time bin indices for the rate optimizer to use when building `requested_bin` constraints; if your rate optimizer uses per-flight requested bins (as in `prepare_flow_scheduling_inputs`), filter flights to those with `requested_bin` ∈ W.
    - Initial rate guesses:
      - Simple heuristic: min(hourly_capacity_at_hotspot, current hourly_occ_at_hotspot_aligned) or a small grid (like in Tabu initializer).
  - Output an ordered list of `RegulationProposal` items: {flows (one or many for union), control_volume_id, active_windows, initial_rate_guess} in the schema your rate optimizer expects.

Notes on reusing existing code:
- Base caches: mirror/port `precompute_cache_context` in `src/project_tailwind/flow_x/cache_flow_extract_debug1.py` and the rolling-hour logic from `src/server_tailwind/airspace/network_evaluator_for_api.py` to compute hourly_excess and cap_per_bin.
- Flow x_G(t) and g0: follow `prompts/flow_x/plan_for_flow_dumping.md` aggregation: build g0 per flow once, then shift/gather for time windows.
- Travel offsets: implement per `prompts/slack_api/slack_planning.md` (persisted JSON with travel minutes, converted to bin shifts).
- Flows and control volumes: reuse `src/parrhesia/pipeline.py` outputs `flights_by_flow` and `ctrl_by_flow` (already available after `prepare_flow_scheduling_inputs`).

## 3) Revised modular design (helpers and modules)

Proposed module namespace: `src/parrhesia/metaopt/`

- `types.py`
  - Data classes:
    - Hotspot: {tv_id: str, bin: int}
    - FlowSpec: {flow_id: Hashable, control_tv_id: str, flight_ids: list[str]}
    - RegulationProposal: {flow_ids: list, control_tv_id: str, active_bins: list[int], rate_guess: int}
    - HyperParams: {w_sum, w_max, κ, α, β, λ_delay, q0, γ, ε, thresholds...}

- `base_caches.py`
  - build_base_caches(flight_list, evaluator_or_caps, indexer) -> {
    occ_base: np.ndarray,
    cap_per_bin: np.ndarray,
    hourly_capacity_matrix: np.ndarray,
    tv_row_of_tvtw: np.ndarray,
    hour_of_tvtw: np.ndarray,
    bins_per_hour: int,
    num_tvs: int,
    num_time_bins_per_tv: int,
    slack_vector: np.ndarray,
    hourly_excess_bool: np.ndarray [num_tvs, num_time_bins_per_tv] or compatible,
  }
  - get_attention_mask(hotspot, ripple_cells) -> θ lookup.

- `travel_offsets.py`
  - build_tv_travel_minutes(tv_gdf, speed_kts=475) -> dict[tv_id][tv_id] = minutes
  - minutes_to_bin_offsets(minutes_map, time_bin_minutes) -> dict[tv_id][tv_id] = int bins
  - flow_offsets(flow: FlowSpec, bin_offsets_map) -> τ_{G,s} vector over TVs.

- `flow_signals.py`
  - build_flow_g0(flow: FlowSpec, flight_list) -> g0 (1D per TVTW) and masks per TV row.
  - build_xG_series(flow: FlowSpec, flights_by_flow, ctrl_by_flow, num_time_bins_per_tv) -> x_G(t) for control row.

- `per_flow_features.py`
  - phase_time(flow, hotspot, τ) -> t_G
  - price_kernel_vG(flow, t_G, τ, base_caches, θ) -> v_G(t_G)
  - price_to_hotspot_vGH(flow, hotspot, τ, base_caches, θ, κ) -> v_{G→H}
  - slack_G_at(flow, t, τ, base_caches) -> Slack_G(t)
  - eligibility_a(flow, xG, t_G, q0, γ, soft: bool) -> a
  - slack_penalty(flow, t_G, τ, base_caches, S0) -> ρ
  - score(flow, hotspot, params, caches, τ, xG) -> scalar

- `pairwise_features.py`
  - temporal_overlap(xGi, xGj, window_bins) -> Overlap_{ij}
  - offset_orthogonality(flow_i, flow_j, hotspot, τ_i, τ_j, hourly_excess_bool, tv_universe_mask) -> Orth_{ij}
  - slack_profile(flow, t_G, τ, base_caches, window_bins) -> array
  - slack_corr(profile_i, profile_j) -> SlackCorr_{ij}
  - price_gap(vGi, vGj, ε) -> PriceGap_{ij}

- `grouping.py`
  - decide_union_or_separate(features_{ij}, thresholds) -> label
  - cluster_flows(flows, pairwise_features, thresholds) -> groups

- `planner.py`
  - choose_active_window(x_G, t_G, overloaded_window_at_hotspot_aligned, min_frac_of_peak, max_span) -> list[int]
  - make_proposals(hotspot, flows, group_labels, xG_map, τ_map, caches, params) -> list[RegulationProposal]
  - to_rate_optimizer_inputs(proposals, flights_by_flow, ctrl_by_flow) -> filtered mappings usable by `run_sa`.

- `runner.py`
  - rank_flows_and_plan(flight_list, indexer, tv_gdf, flights_by_flow, ctrl_by_flow, hotspot: Hotspot, params) -> proposals + diagnostics.

Where to reuse code:
- Base caches and aggregation mirror `src/project_tailwind/flow_x/cache_flow_extract_debug1.py` (occ_base, cap_per_bin, mapping arrays).
- Slack/travel-minute design mirrors `prompts/slack_api/slack_planning.md`.
- Flows and control volumes: already available via `prepare_flow_scheduling_inputs` in `src/parrhesia/pipeline.py`.

Minimal new computations:
- Price kernels v_G and v_{G→H} via vectorized gathers using τ offsets.
- Slack_G via min over τ-shifted slack.
- Pairwise diagnostics via precomputed profiles and masks.
- Proposal window selection using x_G around t_G and the hotspot’s overloaded segment mapped to the control row.

Key hyperparameters to surface:
- Matched-filter: w_sum, w_max, κ, α, β, λ_delay, S0.
- Eligibility: q0, γ, hard/soft toggle.
- Windows: width around t_G for features and proposals.
- Classifier thresholds: τ_ov, τ_sl, τ_pr, τ_orth.

Deliverable to rate-optimizer:
- A narrowed set of flows and control volumes with active time windows and optional initial rate guesses, ready to pass to the existing SA (`run_sa`) after filtering `flights_by_flow` by selected `active_bins` per regulation.

- Status: I mapped every symbol in the doc to concrete inputs and showed how to compute each from existing components. Next, I can stub the `metaopt` module skeleton and wire it to `src/parrhesia/pipeline.py`’s existing flow-prep if you want me to proceed.

- I analyzed `feat_eng.md` and the current codebase to identify reuse points (flow caches, slack/excess computation, travel-time offsets).
- I specified the required inputs for every expression in the document.
- I proposed a computation plan for each quantity, reusing existing vectorized cache logic and outlining minimal additions (price kernels, τ-based gathers).
- I modularized the design into helper modules under `src/parrhesia/metaopt/` with clear responsibilities and interfaces.