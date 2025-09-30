### What `run_regulation_plan_simulation` computes
- The method returns two different objectives:
  - Legacy (hour-based) objective from the underlying evaluator.
  - A new sliding rolling-hour objective computed directly in this method.

Code references:
```1049:1064:/mnt/d/project-tailwind/src/server_tailwind/airspace/airspace_api_wrapper.py
            # 5) Rolling-hour (forward) sums per bin: sum over bins [i, i+W-1] (clamped at end)
            time_start = time.time()
            def rolling_forward_sum_full(mat: np.ndarray, window: int) -> np.ndarray:
                # pad zeros at end to keep length the same (forward-looking window)
                pad = [(0, 0), (0, window - 1)]
                padded = np.pad(mat, pad, mode="constant", constant_values=0.0)
                cs = np.cumsum(padded, axis=1, dtype=np.float64)
                # out[i] = cs[i+W-1] - cs[i-1]; implement vectorized with shifted cs
                left = np.concatenate([np.zeros((mat.shape[0], 1), dtype=np.float64), cs[:, :-window]], axis=1)
                right = cs[:, window - 1 : window - 1 + mat.shape[1]]
                return (right - left).astype(np.float32, copy=False)

            pre_roll = rolling_forward_sum_full(pre_by_tv, bins_per_hour)
            post_roll = rolling_forward_sum_full(post_by_tv, bins_per_hour)
            time_end = time.time()
            print(f"Rolling-hour sums computation took {time_end - time_start} seconds")
```

```1066:1082:/mnt/d/project-tailwind/src/server_tailwind/airspace/airspace_api_wrapper.py
            # 6) Capacity per bin for each TV (repeat hourly capacity across bins in that hour)
            # Reuse the cached evaluator for consistent capacity parsing
            cap_by_tv = {}
            for tv_id, row in tv_items:
                hourly_caps = self._evaluator.hourly_capacity_by_tv.get(tv_id, {})
                if not hourly_caps:
                    # mark as missing capacity with -1.0 per bin
                    cap_by_tv[tv_id] = np.full(bins_per_tv, -1.0, dtype=np.float32)
                    continue
                arr = np.full(bins_per_tv, -1.0, dtype=np.float32)
                for h, c in hourly_caps.items():
                    if 0 <= int(h) < (bins_per_tv // bins_per_hour):
                        start = int(h) * bins_per_hour
                        end = start + bins_per_hour
                        arr[start:end] = float(c)
                cap_by_tv[tv_id] = arr
```

```1083:1102:/mnt/d/project-tailwind/src/server_tailwind/airspace/airspace_api_wrapper.py
            # Compute post-regulation capacity exceedance (J_cap) and total delay minutes (J_delay)
            j_cap_total = 0.0
            z_max_roll = 0.0
            for tv_id, row in tv_items:
                cap_arr = cap_by_tv.get(tv_id)
                if cap_arr is None or cap_arr.size == 0:
                    continue
                pr = post_roll[row, :]
                valid = cap_arr >= 0.0
                if not np.any(valid):
                    continue
                exceed = pr - cap_arr
                exceed = np.where(valid, exceed, 0.0)
                pos = np.maximum(exceed, 0.0)
                j_cap_total += float(np.sum(pos, dtype=np.float64))
                try:
                    z_max_roll = max(z_max_roll, float(np.max(pos)))
                except Exception:
                    pass
```

```1106:1122:/mnt/d/project-tailwind/src/server_tailwind/airspace/airspace_api_wrapper.py
            # Rolling-hour objective (network-level) with weights; keep hourly (legacy) separately
            # Defaults mirror PlanEvaluator defaults
            default_weights = {"alpha": 1.0, "beta": 2.0, "gamma": 0.1, "delta": 25.0}
            if weights:
                try:
                    w = {**default_weights, **weights}
                except Exception:
                    w = default_weights
            else:
                w = default_weights
            num_regs = len(network_plan.regulations)
            objective_new = (
                float(w["alpha"]) * float(j_cap_total)
                + float(w["beta"]) * float(z_max_roll)
                + float(w["gamma"]) * float(j_delay_min)
                + float(w["delta"]) * float(num_regs)
            )
```

```1264:1273:/mnt/d/project-tailwind/src/server_tailwind/airspace/airspace_api_wrapper.py
                # Preserve hourly objective as legacy
                "legacy_objective": float(plan_result.get("objective", 0.0)),
                "legacy_objective_components": plan_result.get("objective_components", {}),
                "pre_flight_context": pre_flight_context,
                # New key with changed TVs only
                "rolling_changed_tvs": rolling_changed_tvs,
                # Backward-compatibility alias for one release
                "rolling_top_tvs": rolling_changed_tvs,
```

### Legacy (hour-based) objective it compares against
- Produced by `PlanEvaluator.evaluate_plan(...)`, which itself calls an hourly, non-rolling overload metric and then builds the scalar objective:
```124:142:/mnt/d/project-tailwind/src/project_tailwind/optimize/eval/plan_evaluator.py
        evaluator = NetworkEvaluator(self.traffic_volumes_gdf, delta_view)
        excess_vector = evaluator.compute_excess_traffic_vector()
        delay_stats = evaluator.compute_delay_stats()

        # Compute scalar objective and components using default or provided weights
        objective, components = self.compute_objective(
            excess_vector=excess_vector,
            delay_stats=delay_stats,
            num_regs=len(network_plan.regulations),
            weights=weights,
        )

        return {
            "delays_by_flight": delays_by_flight,
            "delta_view": delta_view,
            "excess_vector": excess_vector,
            "delay_stats": delay_stats,
            "objective": objective,
            "objective_components": components,
        }
```

```145:199:/mnt/d/project-tailwind/src/project_tailwind/optimize/eval/plan_evaluator.py
    def compute_objective(
        self,
        *,
        excess_vector: Any,
        delay_stats: Dict[str, float],
        num_regs: int,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute scalar objective as a weighted combination of overload and delay.

        Objective: alpha*z_sum + beta*z_max + gamma*delay_min + delta*num_regs
        """
        default_weights = {"alpha": 1.0, "beta": 2.0, "gamma": 0.1, "delta": 25.0}
        if weights:
            w = {**default_weights, **weights}
        else:
            w = default_weights

        ev = np.asarray(excess_vector, dtype=float)
        z_sum = float(np.sum(ev)) if ev.size > 0 else 0.0
        z_max = float(np.max(ev)) if ev.size > 0 else 0.0
        delay_min = float(delay_stats.get("total_delay_seconds", 0.0)) / 60.0

        objective = (
            w["alpha"] * z_sum
            + w["beta"] * z_max
            + w["gamma"] * delay_min
            + w["delta"] * float(num_regs)
        )
```

### Comparison: scope and time-window semantics
- Scope of TVs:
  - Both objectives aggregate across the entire network (all traffic volumes).
  - The new rolling objective still computes per-TV arrays but sums/maxes across all TVs; legacy does the same via the global `excess_vector`.

- Scope of time windows/bins:
  - New rolling objective: sliding window over bins, window length = 60 minutes, stride = 1 bin; it computes forward-looking rolling sums per bin, i.e., windows can cross hour boundaries.
  - Legacy objective: non-rolling, whole-hour aggregation; occupancy is summed per hour (aligned to clock hours) and compared to hourly capacity; excess is then distributed back into the constituent bins of that hour to form `excess_vector`.

- Capacity alignment:
  - New rolling objective: compares each rolling 60-minute demand to a per-bin capacity that repeats the hour’s capacity across all bins in that hour. For windows straddling hours, demand spans two hours but capacity is taken from the hour of the bin being evaluated, which is a mismatch for cross-hour windows.
  - Legacy objective: compares hour-summed demand to the hour’s capacity within that hour only; no cross-hour windows.

### Comparison: component definitions (non-comparabilities)
- Overload “sum” component:
  - New rolling objective uses J_cap = sum over all bins of positive (rolling_60min_demand − hourly_capacity_at_bin). Because windows overlap, the same demand contributes to many overlapping windows. This double-counts demand relative to hour-bucketed excess and will scale roughly with bins_per_hour; not directly comparable to an hour-based z_sum.
  - Legacy objective uses z_sum = sum(excess_vector), where excess is computed once per hour and then apportioned to bins; no overlap duplication.

- Peak component:
  - New rolling objective uses z_max_roll = max over bins of positive (rolling_60min_demand − hourly_capacity_at_bin); this finds the worst sliding 60-minute exceedance.
  - Legacy objective uses z_max = max(excess_vector) after distributing an hour’s excess to its bins; this peak is bounded by the hour’s total excess and depends on intra-hour distribution weights. It is not the same as the peak rolling-hour exceedance.

- Delay and regularization:
  - Both use the same delay_min (total_delay_seconds/60 from the same `plan_result`) and the same num_regs and weights.

### Practical implications
- The two objectives can diverge materially:
  - Sliding windows vs fixed hour buckets.
  - Cross-hour rolling windows compared against a single hour’s capacity.
  - Overlap-induced inflation in J_cap versus hour-based z_sum.
  - Peak measured on rolling exceedances (z_max_roll) versus per-bin share of an hour’s excess (z_max).

- Therefore the “objective” returned by `run_regulation_plan_simulation` and its “legacy_objective” are not directly comparable; alignment (e.g., using the same windowing and capacity treatment) would be required to compare or combine them meaningfully.

- The method explicitly returns both for visibility:
```1264:1270:/mnt/d/project-tailwind/src/server_tailwind/airspace/airspace_api_wrapper.py
                "legacy_objective": float(plan_result.get("objective", 0.0)),
                "legacy_objective_components": plan_result.get("objective_components", {}),
                "pre_flight_context": pre_flight_context,
                # New key with changed TVs only
                "rolling_changed_tvs": rolling_changed_tvs,
```

- Note: The hour-based legacy pipeline computes excess at whole-hour resolution; rolling-hour overloads can be higher than any single hour’s excess because they align windows to the peaks.

- If you need apples-to-apples:
  - Either convert the rolling objective to hour-aligned windows (drop overlap), or
  - Recompute the legacy excess on rolling windows (and adjust capacity treatment for cross-hour windows).

- Weight defaults are identical in both places, so differences stem solely from the overload metrics.

- Short checklist of differences
  - Scope of time: sliding 60-minute vs fixed clock-hour.
  - Capacity applied: per-bin repeated hour capacity vs per-hour capacity (no cross-hour).
  - Aggregation: overlapping window sum (J_cap) vs non-overlapping hour excess (z_sum).
  - Peak: rolling-window peak vs per-bin share of hourly excess.
  - Delay, num_regs, weights: identical.

- Caveat called out in evaluator docs (hour-level exceedances only) reinforces the non-comparability with rolling-hour measures.

- If desired, I can provide a drop-in switch to compute both objectives on the same rolling-hour basis for fair comparison.

- Updated analysis done.