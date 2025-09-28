# Regulation Proposal High-Level Plan

# High-level idea
- Rank flows by a weighted combination of “can fix the hotspot” (gH, xGH) and “won’t cause collateral problems” (slack, low ρ).
- For a chosen hotspot window, compute an exceedance target (how many flights/hour must be removed).
- Allocate that cut across 1–3 top flows and translate it into hourly rates at their controls.
- Do a tiny local search around those rates.
- Predict improvement with a fast proxy that reuses the per-flow features and the allocated rate cuts.

# Inputs required
- HotspotInventory: provides hotspots and their rolling-hour occupancy excess per bin.
- FlowFeaturesExtractor (already documented).
- The hotspot to propose regulations.
- Weights for a linear scoring function over features.

## Key simplification on units
- D_H, roll_occ, and hourly_capacity are compared directly in per_flow_features.py, and *exceedances* are “occupancy per rolling hour” too. Regulation “rate” is also flights per hour. You can work 1:1 without a dwell conversion.
- We will need to convert from rolling occupancy count to **entrance rate count** (the regulation "rate" is actually defined in entrance rate) with a dwell factor: rate_cut_per_hour ≈ (60 / dwell_minutes) × occupancy_reduction. More later.

# Algorithm

1) Precompute for the designated input hotspot h:
  - timebins_h: the bins comprising the hotspot period
  - D_b: per-bin exceedance = max(0, roll_occ − capacity)
  - D_total = sum_b D_b
  - D_peak = max_b D_b
  - D_q95 = 95th percentile of D_b (often more stable than D_peak)
- For each candidate flow i that touches h:
  - Use FlowFeaturesExtractor over timebins_h to read:
    - gH, gH_v_tilde, v_tilde, xGH, DH, ρ, Slack_G15/30/45, tGl, tGu, bins_count
  - Derived estimates you’ll use:
    - avg_bin_demand_i = xGH / max(1, bins_count)
    - per-hour baseline rate at control when the hotspot is active:
      r0_i ≈ avg_bin_demand_i × bins_per_hour
    - coverage_i = (min(T-1, tGu) − max(0, tGl) + 1) / len(timebins_h), clamped to [0,1]

2) Flow eligibility filter (cheap pruning)
- Drop flows with any of:
  - xGH ≈ 0 or r0_i ≈ 0
  - gH ≤ g_min (e.g., 0.05–0.1)
  - ρ ≥ rho_max (e.g., 0.7)
  - Slack_G15 ≤ slack_min (e.g., 1–2 flights), unless you intend to push ≥30–45 minutes, in which case check Slack_G30/45 instead.
This keeps only flows that have demand during the hotspot, can remove a meaningful share of deficit, and are less likely to recreate hotspots elsewhere.

3) Flow scoring (linear, fast, user-weighted)
- Define a weighted score S_i for each eligible flow:
  S_i = w1*gH + w2*gH_v_tilde + w3*v_tilde + w4*Slack_G15 + w5*Slack_G30 − w6*ρ + w7*coverage_i
- Normalize scores to non-negative with max(0, S_i).
- The user’s “reduce overload vs avoid collateral issues” preference is expressed via the weights. Defaults if you need them:
  - w1=1.0, w2=0.5, w3=0.25, w4=0.25, w5=0.15, w6=0.5, w7=0.1

4) Build candidate regulation bundles
- build candidate bundles:
  - 1-flow bundle: top-1 S_i
  - 2-flow bundle: top-2 S_i (if distinct controls)
  - 3-flow bundle: top-3 S_i (if distinct controls)
- Maintain diversity: penalize or skip bundles whose flow set and window overlaps ≥70% with an already selected proposal (see step 8).

5) Pick a regulation window (control-time)
- For a bundle B, define:
  - start_bin = min_i tGl_i
  - end_bin = max_i tGu_i
  - Expand by a small margin for robustness: start_bin -= ceil(0.25*bins_per_hour); end_bin += ceil(0.25*bins_per_hour). We bias towards the past a little bit because the rolling-hour occupancy could involve flights up to one hour later than the current time bin.
  - Clip to [0, T-1]. Enforce a minimum duration (e.g., ≥ bins_per_hour).
- This ensures control actions are in effect when those flights would impact the hotspot.

6) Translate (entrance) exceedance to per-flow rate cuts.

*Note:* this is entrance exceedance. What we directly compute from HotspotInventory is the occupancy exceedance, which would require conversion to entrance exceedance (see below).

- Target exceedance to remove during the window:
  - E_target = D_q95 (simple, robust), optionally cap by D_peak. Pick which target being less pathological (in edge cases).
- Compute bundle weights from scores:
  - w_i = S_i / sum_{j∈B} S_j (if all S_i=0, use uniform weights)
- Per-flow rate cut (flights per hour):
  - λcut_i = round_to_int(w_i * E_target)
  - Cap by baseline: λcut_i ≤ r0_i (cannot cut below 0 rate)
  - Minimum granularity: you enforce a floor of 1 flight/h if λcut_i>0
- New allowed rate per flow:
  - R_i = max(0, r0_i − λcut_i)
- Sanity: if sum_i λcut_i == 0, either increase E_target or drop the bundle.

7) Tiny local search around rates (budgeted)
- For each bundle:
  - Explore a handful of integer perturbations around λcut_i: ±2, ±4 (clamped to [0, r0_i]).
  - Keep total variants small (e.g., ≤ 8 per bundle); compute predicted improvement (step 9); keep the best.
- This costs O(bundles × variants), all constant-time per variant.

8) Diversity across final proposals
- When assembling the final k_proposals, include a diversity penalty in ranking:
  - div_penalty(B) = α × max_overlap_with_already_selected, α ∈ [0, 0.3]
  - Overlap metric: Jaccard on flow sets plus time overlap ratio on windows.
- Rank by predicted improvement − div_penalty and take the top k distinct proposals.

9) Predict objective change 

  - Use the scorer in `src/parrhesia/flow_agent/safespill_objective.py` (`score_with_context`) to evaluate before/after with safe-spill delays, while restricting computation to only TVs touched by flights in the proposal.
  - Build a localized context once per hotspot (and reuse across bundles/variants):
    - `tvs_of_interest`: union of the hotspot TV(s) and every TV visited by any flight in the bundle’s flows within the window (optionally add ripple cells).
    - `sched_fids`: all flights in the bundle’s flows whose scheduled control-time falls in the proposal window.
    - `base_occ_all_by_tv`: occupancy for all flights with zero delays over `tvs_of_interest` (cache).
    - `base_occ_sched_zero_by_tv`: occupancy for just `sched_fids` with zero delays over `tvs_of_interest` (cache).
    - Reuse existing fields already used by the scorer: `indexer`, `flights_sorted_by_flow`, `d_by_flow`, `beta_gamma_by_flow`, `alpha_by_tv`, `weights`, `target_cells`, `ripple_cells`.
  - For each bundle variant:
    - Construct allowed-rate maps `n_f_t` (per-flow, per-bin):
      - Before (no regulation): `n0_f_t[f,t] = d_by_flow[f][t]` for all flows/bins.
      - After (with regulation): identical to `n0_f_t` except for the bundle’s flows inside `[start_bin, end_bin]`, where you cap to the chosen hourly rate `R_i`:
        - Let `bph = indexer.bins_per_hour`. With `(q, r) = divmod(R_i, bph)`, assign per-bin allowance as `q` plus one for `r` bins (distribute within each hour), and clamp by demand: `n[f][t] = min(d_by_flow[f][t], allowed)`.
    - Score both states using the same context:
      - `J_before, comps_before, art_before = score_with_context(n0_f_t, ..., audit_exceedances=True)`
      - `J_after, comps_after, art_after = score_with_context(n_reg_f_t, ..., audit_exceedances=True)`
    - Objective delta: `delta_objective_score = J_before − J_after`.
    - Deficit change over the proposal window on `target_cells` only, using returned occupancy and `K = indexer.rolling_window_size()`:
      - For each `tv` in `target_cells`, compute rolling exceedance `ex_t = max(0, sum_{τ=t−K+1..t} occ[tv][τ] − sum_{τ=t−K+1..t} cap[tv][τ])`.
      - Average `ex_t` over bins intersecting `[start_bin, end_bin]` and sum across `target_cells` to get `deficit_per_hour`. The difference between before/after gives `delta_deficit_per_hour`.
  - Practical defaults and guards:
    - Keep `target_cells` = the hotspot’s TV(s); place other affected TVs in `ripple_cells`.
    - Skip variants with `sum_i λcut_i == 0` or zero baseline deficit in the window.
  - Output per proposal:
    - `predicted_improvement: {delta_deficit_per_hour, delta_objective_score}`.


Suggested defaults
- g_min = 0.1, rho_max = 0.7, slack_min = 2 flights
- Window margin = 0.25 hour on each side, min window = 1 hour
- E_target = D_q95, fallback D_peak if D_q95=0
- Local search: ±1 and ±2 flights/h per selected flow
- Diversity α = 0.2

Complexity
- Feature extraction: already O(#flows × #hotspot_bins)
- Everything else is linear in #flows for scoring and constant-time per bundle/variant.
- This will run in milliseconds to low tens of milliseconds per hotspot for typical sizes.

What each proposal returns
- hotspot_id and controlled_volume - usually coincide
- window: [start_bin, end_bin] in control time
- flow list: [{flow_id, control_tv_id, allowed_rate_per_hour R_i, baseline_rate r0_i, assigned_cut λcut_i}]
- predicted_improvement: {delta_deficit_per_hour, delta_objective_score}
- diagnostics: {E_target, gH_i, v_tilde_i, ρ_i, Slack_G15/30/45_i, coverage_i, weights_used}

About dwell ratio
- Let τ_dwell be average minutes an aircraft contributes to the hotspot’s occupancy.
  - Estimate τ_dwell from travel_minutes_map or historical mean for that sector/route.
- Then convert occupancy-reduction to rate-cut:
  - rate_cut_per_hour ≈ (60 / τ_dwell) × occupancy_reduction
- In our pipeline, prefer α=1.0 (no conversion) unless a quick calibration shows otherwise.


# Important Remarks
- Do not overhandle edge cases, raise Exception so that potentially wrong behaviors could surface instead of falling back to some defaults.
