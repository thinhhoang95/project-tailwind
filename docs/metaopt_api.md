# MetaOpt API Reference

This reference documents public functions and classes under `parrhesia.metaopt`.

## Types

- `Hotspot(tv_id: str, bin: int)`
  - Identifies a target cell `(s*, t*)` by TV id and time‑bin index.

- `FlowSpec(flow_id, control_tv_id: Optional[str], flight_ids: Sequence[str])`
  - Describes a flow with its chosen control volume (if any) and member flights.

- `RegulationProposal(flow_ids: list, control_tv_id: str, active_bins: list[int], rate_guess: int, meta: dict|None)`
  - Structured proposal for rate optimization.

- `HyperParams`
  - Fields: `w_sum, w_max, kappa, alpha, beta, lambda_delay, q0, gamma, S0, eps, window_left, window_right`.

## Base Caches and Attention

- `build_base_caches(flight_list, capacities_by_tv, indexer) -> dict`
  - Returns a dict with keys:
    - `occ_base: np.ndarray (V·T)`
    - `hourly_capacity_matrix: np.ndarray (V×24)`
    - `cap_per_bin: np.ndarray (V·T)`
    - `hourly_occ_base: np.ndarray (V×24)`
    - `slack_per_bin: np.ndarray (V·T)`
    - `slack_per_bin_matrix: np.ndarray (V×T)`
    - `hourly_excess_bool: np.ndarray (V×T)`
    - `tv_row_of_tvtw: np.ndarray (V·T, int32)`
    - `hour_of_tvtw: np.ndarray (V·T, int32)`
    - `bins_per_hour: int`, `num_tvs: int`, `T: int`

- `attention_mask_from_cells(hotspot, ripple_cells=None, weights=None, tv_id_to_idx=None, T=None) -> dict[(row, bin) -> weight]`

## Travel Offsets

- `minutes_to_bin_offsets(travel_minutes: dict[str, dict[str, float]], time_bin_minutes: int) -> dict[str, dict[str, int]]`
  - Rounds minutes to nearest integer bins.

- `flow_offsets_from_ctrl(control_tv_id: Optional[str], tv_id_to_idx: Mapping[str, int], bin_offsets) -> dict[int, int] | None`
  - Returns `τ_{G,s}` as a mapping `row -> Δbins`.

## Flow Signals

- `build_flow_g0(flight_list, flight_ids: Sequence[str]) -> np.ndarray`
  - Total occupancy of all flights in the flow.

- `build_xG_series(flights_by_flow, ctrl_by_flow, flow_id, num_time_bins_per_tv) -> np.ndarray`
  - Histogram of `requested_bin` values at the flow’s control volume.

## Per‑flow Features

- `phase_time(hotspot_row: Optional[int], hotspot: Hotspot, tau_row_to_bins: Mapping[int, int]|None, T: int) -> int`
- `price_kernel_vG(t_G: int, tau_row_to_bins, hourly_excess_bool, theta_mask=None, w_sum=1.0, w_max=1.0) -> float`
- `price_to_hotspot_vGH(hotspot_row: int, hotspot_bin: int, tau_row_to_bins, hourly_excess_bool, theta_mask=None, w_sum=1.0, w_max=1.0, kappa=0.25) -> float`
- `slack_G_at(t: int, tau_row_to_bins, slack_per_bin_matrix) -> float`
- `eligibility_a(xG: np.ndarray, t_G: int, q0: float, gamma: float, soft=False) -> float`
- `slack_penalty(t_G: int, tau_row_to_bins, slack_per_bin_matrix, S0: float) -> float`
- `score(t_G: int, hotspot_row: int, hotspot_bin: int, tau_row_to_bins, hourly_excess_bool, slack_per_bin_matrix, params: HyperParams, xG=None, theta_mask=None, use_soft_eligibility=False) -> float`

## Pairwise Features

- `temporal_overlap(xGi: np.ndarray, xGj: np.ndarray, window_bins: Optional[Sequence[int]] = None) -> float`
- `offset_orthogonality(hotspot_row: int, hotspot_bin: int, tau_i, tau_j, hourly_excess_bool, tv_universe_mask=None) -> float`
- `slack_profile(t_G: int, tau_row_to_bins, slack_per_bin_matrix, window_bins: Sequence[int]) -> np.ndarray`
- `slack_corr(profile_i: np.ndarray, profile_j: np.ndarray) -> float`
- `price_gap(vGi: float, vGj: float, eps: float = 1e-6) -> float`

## Grouping and Planning

- `grouping.decide_union_or_separate(features_ij: Mapping[str, float], thresholds: Mapping[str, float]) -> str`
- `grouping.cluster_flows(flow_ids: Sequence, pairwise_feature_map: Mapping[tuple, Mapping[str, float]], thresholds) -> dict[flow_id -> group_id]`
- `planner.choose_active_window(x_G: np.ndarray, t_G: int, overloaded_window_at_hotspot_aligned=None, min_frac_of_peak=0.5, max_span=12) -> list[int]`
- `planner.make_proposals(hotspot: Hotspot, flows: Sequence[FlowSpec], group_labels, xG_map, tG_map, ctrl_by_flow, default_rate_guess=1) -> list[RegulationProposal]`
- `planner.to_rate_optimizer_inputs(proposals, flights_by_flow, ctrl_by_flow) -> (flights_by_flow_filtered, ctrl_by_flow_filtered)`

## Runner

- `runner.rank_flows_and_plan(flight_list, indexer, travel_minutes_map, flights_by_flow, ctrl_by_flow, hotspot: Hotspot, params: HyperParams|None, capacities_by_tv) -> (proposals, diagnostics)`
  - Diagnostics include `scores_by_flow`, `pairwise_features`, and `labels`.
