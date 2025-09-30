### Automatic rate adjustment (simulated annealing) — objective used

- **Objective components**
  - **Capacity exceedance (J_cap)**: rolling-hour exceedance weighted by per-cell alphas; summed over TVs-of-interest and bins.
  - **Delay (J_delay)**: sum of per-flight delays (minutes) scaled by `lambda_delay`.
  - **Rate deviation (J_reg)**: L1 distance between schedule `n_f(t)` and demand `d_f(t)`, per flow and bin with class-weighted betas.
  - **Smoothness (J_tv)**: total variation of `n_f(t)` with class-weighted gammas.
  - **Optional**: fairness (J_share) and spill (J_spill) only if `theta_share > 0` or `eta_spill > 0`.
  
```6:15:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
  - J_cap: weighted rolling-hour capacity exceedance across (v, t) cells
  - J_delay: sum of pushback delays (in minutes)
  - J_reg: per-flow rate deviation from baseline demand d_f(t)
  - J_tv: per-flow total variation (temporal smoothness)
  - J_share (optional): deviation of per-bin flow shares from demand shares
  - J_spill (optional): penalty on overflow releases n_f(T)
A top-level function `score(...)` orchestrates
  n_f_t  ->  FIFO scheduling  ->  occupancy/exceedance  ->  J-components.
```

- **Scope: TVs and bins considered**
  - TVs used in scoring are restricted to the union of target and ripple TVs via `tv_filter`. This limits occupancy/J_cap to those TVs only.
  - Alpha weights assign target/ripple/context per exact (tv, bin) cells; classification for J_reg/J_tv uses median offsets and ±w tolerance over the same TVs.
  - Schedules are length T+1 with an overflow bin at index T; all other components operate on bins 0..T-1.
  
```328:334:/mnt/d/project-tailwind/src/parrhesia/api/automatic_rate_adjustment.py
    # Restrict scoring to TVs of interest only (targets ∪ ripples)
    tv_filter = set(target_tv_ids) | set(ripple_tv_ids)
```

```146:157:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
    # TVs of interest
    if tv_filter is not None:
        tvs_of_interest = set(str(tv) for tv in tv_filter)
    else:
        tvs_of_interest = set(capacities_by_tv.keys())
        ...
    if not tvs_of_interest:
        tvs_of_interest = set(indexer.tv_id_to_idx.keys())
```

- **Rolling-hour vs non-rolling**
  - **J_cap is rolling-hour**: uses `K = indexer.rolling_window_size()` and computes forward-looking rolling sums RH[t] = sum_{u=t}^{t+K-1} occupancy.
  - **J_reg, J_tv, J_share** are non-rolling per-bin computations; **J_delay** is a scalar sum.
  
```1236:1239:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
    T = int(indexer.num_time_bins)
    K = int(indexer.rolling_window_size())
```

```1010:1042:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
def _compute_J_cap_fast(..., K: int) -> float:
    ...
    RH = _rolling_hour_sum_2d(O, int(K)).astype(np.float32, copy=False)
    exceed = RH - C
    np.maximum(exceed, 0.0, out=exceed)
    J_cap = float(np.sum(exceed * A, dtype=np.float64))
```

```110:130:/mnt/d/project-tailwind/src/parrhesia/optim/capacity.py
def rolling_hour_sum(occ_by_bin: np.ndarray, K: int) -> np.ndarray:
    ...
    idx_end = np.minimum(idx_end + K, T)
    s_end = np.take(csum, idx_end, axis=-1)
    s_start = csum[..., :T]
    return s_end - s_start
```

- **What SA actually calls**
  - Baseline objective `J0` is computed with `score(...)` using the same `tv_filter`, weights, spill mode, and cells.
  - SA uses `score_with_context(...)` (same components and scope) throughout the annealing loop; it reuses a context with the same `tv_filter` and cell sets.

```331:347:/mnt/d/project-tailwind/src/parrhesia/api/automatic_rate_adjustment.py
    weights = ObjectiveWeights(**(payload.get("weights") or {}))
    spill_mode = str(payload.get("spill_mode", "dump_to_next_bin") or "dump_to_next_bin")
    release_rate_for_spills = payload.get("release_rate_for_spills")
    J0, comps0, _arts0 = score(
        n0,
        flights_by_flow=flights_by_flow,
        indexer=idx,
        capacities_by_tv=capacities_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        flight_list=fl,
        weights=weights,
        tv_filter=tv_filter,
        spill_mode=spill_mode,
        release_rate_for_spills=release_rate_for_spills,
    )
```

```349:365:/mnt/d/project-tailwind/src/parrhesia/api/automatic_rate_adjustment.py
    params = SAParams(**{k: v for k, v in sa_kwargs.items() if k in SAParams.__dataclass_fields__})
    n_best, J_star, comps_star, arts_star = run_sa(
        flights_by_flow=flights_by_flow,
        flight_list=fl,
        indexer=idx,
        capacities_by_tv=capacities_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        weights=weights,
        params=params,
        tv_filter=tv_filter,
        spill_mode=spill_mode,
        release_rate_for_spills=release_rate_for_spills,
    )
```

```681:706:/mnt/d/project-tailwind/src/parrhesia/optim/sa_optimizer.py
    context = build_score_context(
        flights_by_flow,
        indexer=indexer,
        capacities_by_tv=capacities_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        flight_list=flight_list,
        weights=weights,
        tv_filter=tv_filter,
    )
    J_best, comps_best, arts_best = score_with_context(
        n_by_flow,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=capacities_by_tv,
        flight_list=flight_list,
        context=context,
        spill_mode=spill_mode,
        release_rate_for_spills=release_rate_for_spills,
    )
```

- **Comparability cautions (why numbers can differ elsewhere)**
  - **TV scope restricted**: Only target∪ripple TVs contribute to J_cap/occupancy; a global-objective run over all TVs will not match.
  - **Capacity policy differences**: When capacities come from app resources, zeros are turned into “9999” (effectively no limit), but the file-based builder leaves zeros as 0. This can dramatically change J_cap.
  
```257:269:/mnt/d/project-tailwind/src/parrhesia/api/automatic_rate_adjustment.py
            mat = _res.capacity_per_bin_matrix  # shape: [num_tvs, T]
            if mat is not None:
                capacities_by_tv = {}
                for tv_id, row_idx in _res.flight_list.tv_id_to_idx.items():
                    arr = mat[int(row_idx), :]
                    capacities_by_tv[str(tv_id)] = (arr * (arr >= 0.0)).astype(int)
                    capacities_by_tv[str(tv_id)][capacities_by_tv[str(tv_id)] == 0] = 9999
        except Exception:
            capacities_by_tv = None
```

  - **Rolling-hour vs per-bin**: J_cap is rolling-hour; if compared to a per-bin exceedance objective, results won’t align.
  - **Spill handling default differs**: This module defaults to `spill_mode="dump_to_next_bin"`, not `"overflow_bin"`. Different spill semantics alter delays, occupancy, and J_reg/J_tv/J_cap coupling.
  
```331:335:/mnt/d/project-tailwind/src/parrhesia/api/automatic_rate_adjustment.py
    spill_mode = str(payload.get("spill_mode", "dump_to_next_bin") or "dump_to_next_bin")
```

  - **Occupancy includes unscheduled traffic**: SA scoring composes occupancy as “all-flights base + scheduled-current − scheduled-zero,” so non-optimized traffic is present. If another computation measures only scheduled flights, J_cap will differ.

```318:331:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
        for tv in context.tvs_of_interest:
            ...
            occ_by_tv[tvs] = (
                base_all.astype(np.int64)
                + sched_cur.astype(np.int64)
                - base_sched_zero.astype(np.int64)
            )
```

  - **Alpha vs classification windows**: Alpha weights apply only to exact cells; classification for J_reg/J_tv uses per-TV median offsets and ±w dilation (`class_tolerance_w`, default 1). If another run dilates alpha cells or uses different offsets/tolerance, components differ.

```536:556:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
class ObjectiveWeights:
    ...
    alpha_gt: float = 10.0
    alpha_rip: float = 3.0
    alpha_ctx: float = 0.5
    ...
    class_tolerance_w: int = 1
```

- **Defaults used unless overridden**
  - Weights default to `alpha_gt=10, alpha_rip=3, alpha_ctx=0.5; beta_ctx=1, beta_rip=0.5, beta_gt=0.1; gamma_ctx=0.5, gamma_rip=0.25, gamma_gt=0.1; lambda_delay=0.1; theta_share=0; eta_spill=0`.

```536:556:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
    alpha_gt: float = 10.0
    alpha_rip: float = 3.0
    alpha_ctx: float = 0.5
    ...
    beta_ctx: float = 1.0
    ...
    gamma_ctx: float = 0.5
    ...
    lambda_delay: float = 0.1
    ...
    theta_share: float = 0.0
    eta_spill: float = 0.0
```

- **Where it’s wired in this module**
  - Baseline and SA both use the same objective and `tv_filter` of target∪ripple; SA iterates using that same scoring context.

```335:347:/mnt/d/project-tailwind/src/parrhesia/api/automatic_rate_adjustment.py
    J0, comps0, _arts0 = score(
        n0,
        flights_by_flow=flights_by_flow,
        indexer=idx,
        capacities_by_tv=capacities_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        flight_list=fl,
        weights=weights,
        tv_filter=tv_filter,
        spill_mode=spill_mode,
        release_rate_for_spills=release_rate_for_spills,
    )
```

```354:365:/mnt/d/project-tailwind/src/parrhesia/api/automatic_rate_adjustment.py
    n_best, J_star, comps_star, arts_star = run_sa(
        flights_by_flow=flights_by_flow,
        flight_list=fl,
        indexer=idx,
        capacities_by_tv=capacities_by_tv,
        target_cells=target_cells,
        ripple_cells=ripple_cells,
        weights=weights,
        params=params,
        tv_filter=tv_filter,
        spill_mode=spill_mode,
        release_rate_for_spills=release_rate_for_spills,
    )
```

Summary
- Uses a 5-term objective: rolling-hour exceedance (J_cap), delay, rate deviation, smoothness, with optional fairness/spill; weights default as above.
- Scope is limited to target∪ripple TVs; alpha applies to exact cells; classification (for betas/gammas) uses median-offset mapping with ±1 bin tolerance.
- Rolling-hour only affects J_cap; other terms are per-bin.
- Differences that can break comparability: TV restriction, rolling-hour vs per-bin exceedance, spill mode default (“dump_to_next_bin”), occupancy including unscheduled traffic, and capacity zeros → 9999 when sourced from app resources.