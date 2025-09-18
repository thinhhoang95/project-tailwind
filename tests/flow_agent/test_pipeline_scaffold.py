from __future__ import annotations

import numpy as np

from parrhesia.flows.flow_pipeline import build_global_flows, collect_hotspot_flights
from parrhesia.optim.objective import ObjectiveWeights, score
from parrhesia.optim.sa_optimizer import SAParams, prepare_flow_scheduling_inputs, run_sa


def _uniform_capacities(indexer, tv_ids, hourly_cap=120.0):
    """Local helper mirroring the conftest version for test determinism."""
    T = indexer.num_time_bins
    proto = np.full(T, hourly_cap, dtype=np.float32)
    return {tv: proto.copy() for tv in tv_ids}


def test_hotspot_sampling_consistency(tailwind_sample):
    idx = tailwind_sample["indexer"]
    hotspots = tailwind_sample["hotspots"]
    windows = tailwind_sample["active_windows"]

    assert hotspots, "expected at least one hotspot in sample"
    for tv in hotspots:
        assert tv in idx.tv_id_to_idx
        assert windows.get(tv), f"no active windows recorded for {tv}"


def test_collect_hotspot_flights_round_trip(tailwind_sample):
    fl = tailwind_sample["flight_list"]
    hotspots = tailwind_sample["hotspots"]
    windows = tailwind_sample["active_windows"]

    union_ids, meta = collect_hotspot_flights(fl, hotspots, active_windows=windows)

    assert union_ids, "hotspot flight union should not be empty"
    assert set(union_ids).issubset(set(fl.flight_ids))
    assert meta, "metadata payload should not be empty"

    sample_ids = union_ids[: min(5, len(union_ids))]
    for fid in sample_ids:
        info = meta[fid]
        assert info.get("first_crossings"), "expected first_crossings entry"
        fg = info.get("first_global")
        assert fg and fg.get("hotspot_id") in hotspots


def test_build_global_flows_partition(tailwind_sample):
    fl = tailwind_sample["flight_list"]
    hotspots = tailwind_sample["hotspots"]
    windows = tailwind_sample["active_windows"]

    union_ids, _ = collect_hotspot_flights(fl, hotspots, active_windows=windows)
    flow_map = build_global_flows(
        fl,
        union_ids,
        hotspots=hotspots,
        trim_policy="earliest_hotspot",
        leiden_params={"threshold": 0.1, "resolution": 1.0, "seed": 0},
        direction_opts={"mode": "none"},
    )

    assert flow_map, "flow clustering returned no assignments"
    assert set(flow_map) == set(union_ids)
    assert min(flow_map.values()) == 0
    assert max(flow_map.values()) >= 0


def test_baseline_objective_matches_live_pipeline(tailwind_sample):
    fl = tailwind_sample["flight_list"]
    idx = tailwind_sample["indexer"]
    hotspots = tailwind_sample["hotspots"]
    windows = tailwind_sample["active_windows"]

    union_ids, _ = collect_hotspot_flights(fl, hotspots, active_windows=windows)
    flow_map = build_global_flows(
        fl,
        union_ids,
        hotspots=hotspots,
        trim_policy="earliest_hotspot",
        leiden_params={"threshold": 0.1, "resolution": 1.0, "seed": 0},
        direction_opts={"mode": "none"},
    )
    flights_by_flow, ctrl_by_flow = prepare_flow_scheduling_inputs(
        flight_list=fl,
        flow_map=flow_map,
        hotspot_ids=hotspots,
    )

    assert flights_by_flow, "no per-flow request series generated"

    T = idx.num_time_bins
    baseline = {}
    for flow_id, specs in flights_by_flow.items():
        counts = [0] * (T + 1)
        for spec in specs:
            rb = int(spec["requested_bin"])
            if 0 <= rb <= T:
                counts[rb] += 1
        baseline[flow_id] = counts

    tvs_for_caps = set(hotspots)
    tvs_for_caps.update(tv for tv in ctrl_by_flow.values() if tv)
    capacities = _uniform_capacities(idx, sorted(tvs_for_caps))

    target_cells = [(tv, win) for tv, wins in windows.items() for win in wins]
    weights = ObjectiveWeights()

    J0, comps0, _ = score(
        baseline,
        flights_by_flow=flights_by_flow,
        indexer=idx,
        capacities_by_tv=capacities,
        target_cells=target_cells,
        ripple_cells=[],
        flight_list=fl,
        weights=weights,
    )

    assert np.isfinite(J0), "baseline objective must be finite"
    assert "J_cap" in comps0 and "J_delay" in comps0


def test_sa_smoke_improves_or_matches_baseline(tailwind_sample):
    fl = tailwind_sample["flight_list"]
    idx = tailwind_sample["indexer"]
    hotspots = tailwind_sample["hotspots"]
    windows = tailwind_sample["active_windows"]

    union_ids, _ = collect_hotspot_flights(fl, hotspots, active_windows=windows)
    flow_map = build_global_flows(
        fl,
        union_ids,
        hotspots=hotspots,
        trim_policy="earliest_hotspot",
        leiden_params={"threshold": 0.1, "resolution": 1.0, "seed": 0},
        direction_opts={"mode": "none"},
    )
    flights_by_flow, ctrl_by_flow = prepare_flow_scheduling_inputs(
        flight_list=fl,
        flow_map=flow_map,
        hotspot_ids=hotspots,
    )

    T = idx.num_time_bins
    baseline = {}
    for flow_id, specs in flights_by_flow.items():
        counts = [0] * (T + 1)
        for spec in specs:
            rb = int(spec["requested_bin"])
            if 0 <= rb <= T:
                counts[rb] += 1
        baseline[flow_id] = counts

    tvs_for_caps = set(hotspots)
    tvs_for_caps.update(tv for tv in ctrl_by_flow.values() if tv)
    capacities = _uniform_capacities(idx, sorted(tvs_for_caps))

    target_cells = [(tv, win) for tv, wins in windows.items() for win in wins]
    weights = ObjectiveWeights()

    J0, _, _ = score(
        baseline,
        flights_by_flow=flights_by_flow,
        indexer=idx,
        capacities_by_tv=capacities,
        target_cells=target_cells,
        ripple_cells=[],
        flight_list=fl,
        weights=weights,
    )

    params = SAParams(iterations=30, warmup_moves=10, seed=0, verbose=False)
    _, J_best, _, _ = run_sa(
        flights_by_flow=flights_by_flow,
        flight_list=fl,
        indexer=idx,
        capacities_by_tv=capacities,
        target_cells=target_cells,
        ripple_cells=[],
        weights=weights,
        params=params,
    )

    assert J_best <= J0 + 1e-6, "SA should not regress compared to baseline"
