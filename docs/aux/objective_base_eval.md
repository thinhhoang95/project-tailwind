### 1) In `src/parrhesia/api/base_evaluation.py`
- Scope of TVs:
  - **What’s included**: scoring is restricted to TVs in `targets ∪ ripples`.
  - **Where set**:
    ```383:396:/mnt/d/project-tailwind/src/parrhesia/api/base_evaluation.py
    # Restrict scoring to TVs of interest (targets ∪ ripples) to align with /automatic_rate_adjustment
    tv_filter = set(target_tv_ids) | set(ripple_tv_ids)
    J, components, _arts = score(
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
  - **Auto-ripple option**: when `auto_ripple_time_bins > 0`, ripple cells become the union of all flow flight footprints (dilated ±w), which can greatly expand the TV set:
    ```286:294:/mnt/d/project-tailwind/src/parrhesia/api/base_evaluation.py
    try:
        _auto_w = int(payload.get("auto_ripple_time_bins", 0))
    except Exception:
        _auto_w = 0
    if _auto_w > 0:
        ripple_cells = _auto_ripple_cells_from_flows(idx, fl, flow_map.keys(), _auto_w)
    ```

- Scope of time bins:
  - **What’s included**: all time bins \(t = 0..T−1\) for each TV in `tv_filter`. Target/ripple cells only alter α-weights at those bins; non-target/ripple bins on those TVs still contribute with context weight.
  - **Evidence (occupancy computed for all bins on the TVs of interest)**:
    ```333:338:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
    occ_by_tv = compute_occupancy(
        flight_list if flight_list is not None else type("_Dummy", (), {"flight_metadata": {}})(),
        delays_min,
        indexer,
        tv_filter=context.tvs_of_interest,
    )
    ```
  - `T` comes from the indexer:
    ```269:270:/mnt/d/project-tailwind/src/parrhesia/api/base_evaluation.py
    T = int(idx.num_time_bins)
    ```

- Rolling-hour vs non-rolling-hour:
  - **Rolling-hour occupancy window** \(width \(K\) = `rolling_window_size`\) is used in capacity exceedance:
    ```343:351:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
    K = int(indexer.rolling_window_size())
    J_cap = _compute_J_cap(
        occ_by_tv,
        capacities_by_tv,
        context.alpha_by_tv,
        K,
        audit_exceedances=audit_exceedances,
        indexer=indexer,
        target_cells=context.target_cells,
        ripple_cells=context.ripple_cells,
    )
    ```
  - **How exceedance is computed**: rolling sum minus per-bin capacity, then clamped:
    ```1070:1079:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
    T = occ.size
    cap = np.asarray(capacities_by_tv.get(tv_id, np.zeros(T, dtype=np.int64)), dtype=np.float64)
    ...
    rh = rolling_hour_sum(occ.astype(np.int64, copy=False), int(K)).astype(np.float64)
    exceed = np.maximum(0.0, rh - cap)
    ```
  - **Capacity units** in base evaluation:
    - If using project GeoJSON: capacity is repeated across all bins within the hour (not divided by bins per hour):
      ```52:69:/mnt/d/project-tailwind/src/parrhesia/optim/capacity.py
      def build_bin_capacities(...):
          ...
          # arr[start_bin:end_bin] = v
      ```
      ```102:107:/mnt/d/project-tailwind/src/parrhesia/optim/capacity.py
      if end_bin <= start_bin:
          continue
      arr[start_bin:end_bin] = v
      ```
    - If using server-shared resources, zeros and missing are made effectively “unbounded” (9999) which suppresses exceedance:
      ```203:213:/mnt/d/project-tailwind/src/parrhesia/api/base_evaluation.py
      mat = _res.capacity_per_bin_matrix  # shape: [num_tvs, T]
      ...
      capacities_by_tv[str(tv_id)] = (arr * (arr >= 0.0)).astype(int)
      capacities_by_tv[str(tv_id)][capacities_by_tv[str(tv_id)] == 0] = 9999
      ```
      Fallback (if no server resources) reverts to the GeoJSON behavior:
      ```216:219:/mnt/d/project-tailwind/src/parrhesia/api/base_evaluation.py
      if capacities_by_tv is None:
          # Fallback to project default path
          capacities_by_tv = build_bin_capacities(str(cap_path_default), idx)
      ```

- Objective components and settings:
  - **Components included**: `J_cap + J_delay + J_reg + J_tv (+ J_share + J_spill if weights>0)`.
    ```1351:1363:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
    J_total = J_cap + J_delay + J_reg + J_tv + J_share + J_spill
    components: Dict[str, float] = {
        "J_cap": float(J_cap),
        "J_delay": float(J_delay),
        "J_reg": float(J_reg),
        "J_tv": float(J_tv),
    }
    ```
  - **Spill semantics**: base evaluation passes `spill_mode` (defaulting to “dump_to_next_bin”), overriding the generic scorer’s default of “overflow_bin”.
    ```1132:1146:/mnt/d/project-tailwind/src/parrhesia/optim/objective.py
    def score(..., spill_mode: SpillMode = "overflow_bin", ...)
    ```
    ```380:395:/mnt/d/project-tailwind/src/parrhesia/api/base_evaluation.py
    spill_mode = str(payload.get("spill_mode", "dump_to_next_bin") or "dump_to_next_bin")
    release_rate_for_spills = payload.get("release_rate_for_spills")
    ...
    score(..., spill_mode=spill_mode, release_rate_for_spills=release_rate_for_spills)
    ```
  - **Weights**: `ObjectiveWeights` may be overridden via payload. α-weights emphasize target bins (and optionally ripple bins) but do not limit the time horizon.

- Differences that can render results non-comparable to other places (e.g., regen safe-spill/predict):
  - **TV/time scope**: base evaluation scores over all bins across `targets ∪ ripples` TVs, not just in a window. Other flows (e.g., regen) may include additional TVs like control TVs and all traversed TVs, changing scope.
  - **Capacity baseline**: using server resources turns zero/missing capacity into 9999, effectively disabling exceedance on those bins; GeoJSON fallback uses literal zeros. This drastically changes `J_cap`.
  - **Spill behavior**: base evaluation uses “dump_to_next_bin” by default but is configurable; other places may hard-code it or use different defaults.
  - **Scaling differences vs safe-spill scorer**: if you compare against the safe-spill variant, note it scales `J_cap` by 100 in its total:
    ```120:121:/mnt/d/project-tailwind/src/parrhesia/flow_agent/safespill_objective.py
    J_total = 100.0 * J_cap + J_delay + J_reg + J_tv + J_share + J_spill
    ```
    Base evaluation uses the generic scorer (no ×100 scaling).
  - **Rolling-hour consistency**: base evaluation’s `J_cap` subtracts a single hourly capacity per bin of the rolling window (with per-bin capacities being hourly values repeated across bins). If you compare to any metric that sums capacity across K bins for rolling windows, the numbers won’t align.

Summary
- Base evaluation scopes scoring to TVs in `targets ∪ ripples` and to all time bins in the horizon; target/ripple bins are emphasized via α-weights.
- It uses rolling-hour occupancy and subtracts a single hourly capacity value per bin (capacity arrays repeat the hourly cap across bins).
- Objective includes capacity, delay, regularization, TV smoothness (plus optional share/spill); default spill is “dump_to_next_bin”.
- Results can be non-comparable if other places: use broader/narrower TV sets or only a window, scale `J_cap` (safe-spill ×100), or alter capacity baselines (e.g., zeros → 9999 via server resources).