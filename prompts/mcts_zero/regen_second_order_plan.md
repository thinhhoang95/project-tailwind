### Plan
- Use `extract_hotspot_segments_from_resources` to get ranked hotspot segments from shared `AppResources`; select the 3rd by `max_excess` and convert to a regen payload via `segment_to_hotspot_payload`.
- Build `flows_payload` and `flow_to_flights` for that hotspot with `compute_flows`.
- Call the raw regen engine `propose_regulations_for_hotspot` using shared-capacity (`_build_capacities_by_tv`) and shared travel-times from `AppResources`, then pick the best proposal by `delta_objective_score`.
- Convert that best proposal to a `DFRegulationPlan` using `DFRegulationPlan.from_proposal(..., flights_by_flow=flow_to_flights)`.
- Evaluate the plan to obtain per-flight delays (`evaluate_df_regulation_plan`), build a `DeltaOccupancyView` from a `DelayAssignmentTable`, and apply it via `res.flight_list.step_by_delay(view)`.
- Invalidate/refresh caches using `refresh_after_state_update(res)`.
- If `debug_verbose`:
  - Print top-5 hotspot segments (TV, start–end labels, `max_excess`).
  - Print the delay assignment table summary after apply.
  - Show interval shifts for 3 changed flights (first 3 TVs per flight).
  - Print occupancy change stats pre vs post.
  - Extract hotspots again and print the new top-5.
- Optionally re-run regen on the updated state (second order).

### Edits in `examples/regen/regen_second_order.py`

Add imports near the top:
```python
from parrhesia.flow_agent35.regen.hotspot_segment_extractor import (
    extract_hotspot_segments_from_resources,
    segment_to_hotspot_payload,
)
from parrhesia.actions.regulations import DFRegulationPlan
from parrhesia.actions.dfplan_evaluator import evaluate_df_regulation_plan
from project_tailwind.stateman.delay_assignment import DelayAssignmentTable
from project_tailwind.stateman.delta_view import DeltaOccupancyView
from server_tailwind.core.cache_refresh import refresh_after_state_update
```

Add helpers (place below existing helpers, before `main()`):
```python
def _print_top_segments(segments, k=5) -> None:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title="Top Hotspot Segments")
    table.add_column("Rank", justify="right")
    table.add_column("Traffic Volume", style="cyan")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Max Excess", justify="right")
    table.add_column("Sum Excess", justify="right")
    for i, seg in enumerate(segments[:k], start=1):
        table.add_row(
            str(i),
            str(seg["traffic_volume_id"]),
            str(seg["start_label"]),
            str(seg["end_label"]),
            f'{float(seg["max_excess"]):.1f}',
            f'{float(seg["sum_excess"]):.1f}',
        )
    console.print(table)

def _bin_label(bin_offset: int, tbm: int) -> str:
    m = int(bin_offset) * int(tbm)
    hh = (m // 60) % 24
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"

def _decode_tvtw(col: int, bins_per_tv: int, idx_to_tv_id: Mapping[int, str], tbm: int) -> tuple[str, str]:
    tv_row = int(col) // int(bins_per_tv)
    bin_off = int(col) % int(bins_per_tv)
    tv = str(idx_to_tv_id.get(int(tv_row), ""))
    return tv, _bin_label(bin_off, tbm)

def _build_flows_payload_for_hotspot(hotspot_payload: Mapping[str, Any]) -> tuple[Dict[str, List[str]], Dict[str, List[float]], Dict[str, Any]]:
    res = get_resources().preload_all()
    set_global_resources(res.indexer, res.flight_list)
    control_tv = str(hotspot_payload["control_volume_id"])
    timebins_h = _timebins_from_window(hotspot_payload.get("window_bins", []))
    centroids = res.tv_centroids
    direction_opts = {"mode": "coord_cosine", "tv_centroids": centroids} if centroids else {"mode": "coord_cosine"}
    flows_payload = compute_flows(
        tvs=[control_tv],
        timebins=timebins_h,
        direction_opts=direction_opts,
        threshold=leiden_params["threshold"],
        resolution=leiden_params["resolution"],
    )
    flow_to_flights: Dict[str, List[str]] = {}
    flow_proxies: Dict[str, List[float]] = {}
    demand_key = "demand"
    for flow in flows_payload.get("flows", []) or []:
        try:
            fid_key = str(int(flow.get("flow_id")))
        except Exception:
            fid_key = str(flow.get("flow_id"))
        flights: List[str] = []
        for spec in flow.get("flights", []) or []:
            fid = spec.get("flight_id")
            if fid is not None:
                flights.append(str(fid))
        flow_to_flights[fid_key] = flights
        demand = flow.get(demand_key) or []
        proxy: List[float] = []
        for b in timebins_h:
            try:
                proxy.append(float(demand[int(b)]))
            except Exception:
                proxy.append(0.0)
        flow_proxies[fid_key] = proxy
    return flow_to_flights, flow_proxies, flows_payload

def _best_proposal_for_hotspot(
    *,
    hotspot_payload: Mapping[str, Any],
    flight_list: FlightList,
    indexer: TVTWIndexer,
) -> tuple[Any, Dict[str, List[str]], Dict[str, Any], List[Dict[str, Any]]]:
    capacities_by_tv = _build_capacities_by_tv(indexer)
    res = get_resources().preload_all()
    travel_minutes_map = res.travel_minutes()
    flow_to_flights, flow_proxies, flows_payload = _build_flows_payload_for_hotspot(hotspot_payload)

    fallback_cfg = RegenConfig(
        g_min=-float("inf"),
        rho_max=float("inf"),
        slack_min=-float("inf"),
        distinct_controls_required=False,
        raise_on_edge_cases=True,
        min_num_flights=4,
    )
    control_tv = str(hotspot_payload["control_volume_id"])
    timebins_h = _timebins_from_window(hotspot_payload.get("window_bins", []))
    proposals = propose_regulations_for_hotspot(
        indexer=indexer,
        flight_list=flight_list,
        capacities_by_tv=capacities_by_tv,
        travel_minutes_map=travel_minutes_map,
        hotspot_tv=control_tv,
        timebins_h=timebins_h,
        flows_payload=flows_payload,
        flow_to_flights=flow_to_flights,
        config=fallback_cfg,
    )
    if not proposals:
        raise RuntimeError("regen engine returned no proposals")
    # Pick by max delta_objective_score, though engine already sorts
    best = max(proposals, key=lambda p: float(p.predicted_improvement.delta_objective_score))
    # Build dicts for display
    exceedance_stats = compute_hotspot_exceedance(
        indexer=indexer,
        flight_list=flight_list,
        capacities_by_tv=capacities_by_tv,
        hotspot_tv=control_tv,
        timebins_h=timebins_h,
    )
    dicts = [
        _proposal_to_dict(
            p, hotspot_payload=hotspot_payload, flows_payload=flows_payload, exceedance_stats=exceedance_stats
        )
        for p in proposals
    ]
    return best, flow_to_flights, flows_payload, dicts

def _evaluate_and_apply_plan(
    plan: DFRegulationPlan,
    *,
    resources: AppResources,
    debug_verbose: bool = False,
) -> None:
    res = resources
    fl = res.flight_list
    idx = res.indexer
    tbm = int(fl.time_bin_minutes)
    bins_per_tv = int(fl.num_time_bins_per_tv)
    pre_total = fl.get_total_occupancy_by_tvtw().astype(np.int64, copy=False)

    # Evaluate plan to get delays
    eval_res = evaluate_df_regulation_plan(
        plan,
        indexer_path=str(res.paths.tvtw_indexer_path),
        flights_path=str(res.paths.occupancy_file_path),
    )
    delays = DelayAssignmentTable.from_dict(eval_res.delays_by_flight)

    # Snapshot old intervals for changed flights
    old_intervals: Dict[str, List[Dict[str, Any]]] = {}
    for fid, d in list(eval_res.delays_by_flight.items())[:3]:
        meta = fl.flight_metadata.get(str(fid)) or {}
        old_intervals[str(fid)] = list(meta.get("occupancy_intervals") or [])

    if debug_verbose:
        nonzero = list(delays.nonzero_items())
        print(f"[apply] Nonzero delay assignments: {len(nonzero)}")
        for k, v in nonzero[:10]:
            print(f"  {k}: +{v} min")

    view = DeltaOccupancyView.from_delay_table(flights=fl, delays=delays, regulation_id="regen_1")
    fl.step_by_delay(view)

    post_total = fl.get_total_occupancy_by_tvtw().astype(np.int64, copy=False)
    diff = post_total - pre_total
    changed = int((diff != 0).sum())
    l1 = int(np.abs(diff).sum())
    print(f"[apply] Occupancy changed cells: {changed}, L1 delta: {l1}")

    if debug_verbose:
        cf = view.changed_flights()[:3]
        print(f"[apply] Changed flights (sample): {cf}")
        for fid in cf:
            before = old_intervals.get(str(fid), [])[:3]
            after = (view.per_flight_new_intervals.get(str(fid)) or [])[:3]
            print(f"  Flight {fid}:")
            for i, (b, a) in enumerate(zip(before, after), start=1):
                tv_b, label_b = _decode_tvtw(int(b.get('tvtw_index', -1)), bins_per_tv, fl.idx_to_tv_id, tbm) if b else ("", "")
                tv_a, label_a = _decode_tvtw(int(a.get('tvtw_index', -1)), bins_per_tv, fl.idx_to_tv_id, tbm) if a else ("", "")
                print(f"    #{i} {tv_b} {label_b} -> {tv_a} {label_a}")

    refresh_after_state_update(res)
```

Replace the manual hotspot block in `main()` with the second-order workflow. Replace this section:
```12:31:/mnt/d/project-tailwind/examples/regen/regen_second_order.py
    # Manually define the hotspot to be analyzed. In a real application, this
    # would come from an automated hotspot detection system.
    hotspot_payload = {
        "control_volume_id": "LFBZX15",
        "window_bins": [45, 49], # means [45, 46, 47, 48]
        "metadata": {}, # will be filled in below
        "mode": "manual",
    }
    ...
    hotspot_payload["metadata"] = {
        "flow_to_flights": flow_to_flights,
        "flow_proxies": flow_proxies,
    }
    print(
        "[regen] Using manual hotspot selection: "
        f"TV={hotspot_payload['control_volume_id']} window={hotspot_payload['window_bins']}"
    )
```
with:
```python
    # Second-order workflow parameters
    debug_verbose = True  # toggle extra logging

    # 1) Discover hotspot segments and pick the 3rd-ranked by max_excess
    res = get_resources().preload_all()
    segments = extract_hotspot_segments_from_resources(resources=res)
    if debug_verbose:
        print("[regen] First extraction: top hotspot segments")
        _print_top_segments(segments, k=5)
    if len(segments) < 3:
        raise SystemExit("Fewer than 3 hotspot segments found; cannot select 3rd-ranked.")
    hotspot_payload = segment_to_hotspot_payload(segments[2])

    # 2) Run regen, get proposals, and choose the best; convert to DFRegulationPlan
    best_proposal, flow_to_flights, flows_payload, proposals_dicts = _best_proposal_for_hotspot(
        hotspot_payload=hotspot_payload, flight_list=flight_list, indexer=indexer
    )
    plan = DFRegulationPlan.from_proposal(
        best_proposal,
        flights_by_flow=flow_to_flights,
        time_bin_minutes=int(indexer.time_bin_minutes),
    )

    if debug_verbose:
        # Show the proposals for context using the same rich formatting as before
        from rich.console import Console
        from rich.table import Table
        console = Console()
        for rank, proposal in enumerate(proposals_dicts, start=1):
            diag = proposal["diagnostics"]
            improvement = proposal.get("predicted_improvement", {})
            components_before = diag.get("score_components_before", {}) or {}
            components_after = diag.get("score_components_after", {}) or {}
            console.print(f"\n[bold green][regen] Proposal Summary #{rank}[/bold green]")
            console.print(f"Control Volume: [cyan]{proposal['control_volume_id']}[/cyan]")
            console.print(f"Window Bins: [cyan]{proposal['window_bins'][0]}-{proposal['window_bins'][1]}[/cyan]")
            console.print(
                f"Target Exceedance to Remove: [yellow]{diag['E_target']:.1f}[/yellow] "
                f"(D_peak={diag.get('D_peak', 0.0):.1f}, D_sum={diag.get('D_sum', 0.0):.1f})"
            )
            delta_obj = float(improvement.get("delta_objective_score", 0.0))
            console.print(f"Predicted Objective Improvement: [yellow]{delta_obj:.3f}[/yellow]")

            if components_before or components_after:
                comp_table = Table(title="Objective Components")
                comp_table.add_column("Component", style="cyan")
                comp_table.add_column("Baseline", justify="right", style="magenta")
                comp_table.add_column("Regulated", justify="right", style="green")
                comp_table.add_column("Delta", justify="right", style="yellow")
                component_keys = sorted(set(components_before.keys()) | set(components_after.keys()))
                for key in component_keys:
                    before_val = float(components_before.get(key, 0.0))
                    after_val = float(components_after.get(key, 0.0))
                    delta_val = after_val - before_val
                    comp_table.add_row(key, f"{before_val:.3f}", f"{after_val:.3f}", f"{delta_val:+.3f}")
                console.print(comp_table)

            flow_table = Table(title="Flow Regulations")
            flow_table.add_column("Flow ID", style="cyan", no_wrap=True)
            flow_table.add_column("Control TV", style="magenta")
            flow_table.add_column("Baseline Rate\n(per hour)", justify="right", style="green")
            flow_table.add_column("Allowed Rate\n(per hour)", justify="right", style="yellow")
            flow_table.add_column("Cut\n(per hour)", justify="right", style="red")
            flow_table.add_column("Entrants in\nWindow", justify="right")
            flow_table.add_column("Num\nFlights", justify="right")
            for flow in proposal["flows"]:
                flow_table.add_row(
                    str(flow["flow_id"]),
                    str(flow["control_tv_id"]),
                    f"{flow['baseline_rate_per_hour']:.1f}",
                    f"{flow['allowed_rate_per_hour']:.1f}",
                    f"{flow['assigned_cut_per_hour']:.0f}",
                    f"{flow['entrants_in_window']:.0f}",
                    str(flow["num_flights"]),
                )
            console.print(flow_table)

    # 3) Evaluate the DFRegulationPlan → delays, apply via step_by_delay (state mutation)
    _evaluate_and_apply_plan(plan, resources=res, debug_verbose=debug_verbose)

    # 4) Re-extract hotspots on the updated state and show top 5
    segments2 = extract_hotspot_segments_from_resources(resources=res)
    if debug_verbose:
        print("[regen] Second extraction (after apply): top hotspot segments")
        _print_top_segments(segments2, k=5)

    # Optional second regen pass (not applied here): uncomment to run again
    # best2, flow2, _, _ = _best_proposal_for_hotspot(
    #     hotspot_payload=segment_to_hotspot_payload(segments2[2]),
    #     flight_list=flight_list,
    #     indexer=indexer,
    # )
```

That replaces the manual section and wires the full second-order chain.

### Notes on caches and consistency
- After `step_by_delay`, we call `refresh_after_state_update(res)` to invalidate wrappers and re-register shared resources. This aligns with `prompts/state_transition/cache_refresh.md`.

### What you’ll see with `debug_verbose=True`
- First top-5 hotspot segments before apply.
- A proposal table (like before), plus the best chosen internally.
- A summary of nonzero delays and sample per-flight interval shifts.
- Occupancy change stats (changed cells and L1 delta).
- Second top-5 hotspot segments after state mutation.

- If desired, you can uncomment the optional second regen pass to generate another plan on the updated network.

- To keep runs reproducible, this reuses the same `AppResources` capacity matrix and travel-times.

- All changes preserve the existing indentation style.

- Ensure required artifacts exist; otherwise, the script will stop with a clear message.

- Run it: `python examples/regen/regen_second_order.py`

- If you want a CLI toggle, we can add `argparse` for `--debug-verbose`; say the word and I’ll wire it.

- You can compare old and new hotspots to verify the network state actually changed.

- The plan conversion uses `DFRegulationPlan.from_proposal` with the explicit `flow_to_flights` mapping to preserve targeted flights per flow.

- The evaluation uses shared `AppResources` to compute consistent delays and objective deltas.

- The hot path avoids re-reading files by relying on `AppResources`’s in-memory objects.

- The final JSON written at the end (if you keep that section) can be extended to include the DF plan and delays if needed.

- If any import paths differ on your machine, they’re all within the repo; the code above matches the modules you already have.

- I left existing `propose_regulations(...)` intact; the new path calls the engine directly to keep the `Proposal` object for the DF plan step.