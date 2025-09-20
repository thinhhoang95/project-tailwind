from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pytest
from scipy import sparse

from parrhesia.flow_agent import (
    PlanState,
    NewRegulation,
    PickHotspot,
    CheapTransition,
    MCTS,
    MCTSConfig,
    RateFinder,
    RateFinderConfig,
)
from parrhesia.flows.flow_pipeline import build_global_flows, collect_hotspot_flights
from parrhesia.optim.sa_optimizer import prepare_flow_scheduling_inputs
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator


def _build_proxies_with_rfinder(
    rf: RateFinder,
    control_volume_id: str,
    window_bins: Tuple[int, int],
    flows: Dict[str, Sequence[str]],
) -> Dict[str, np.ndarray]:
    window_start, window_end = int(window_bins[0]), int(window_bins[1])
    active = list(range(window_start, window_end))
    entrants = rf._compute_entrants(control_volume_id, active, flows)
    win_len = max(1, window_end - window_start)
    out: Dict[str, np.ndarray] = {}
    for fid in flows.keys():
        bins = np.zeros(win_len, dtype=float)
        for _fl, _dt, t in entrants.get(fid, []):
            tt = int(t) - window_start
            if 0 <= tt < win_len:
                bins[tt] += 1.0
        out[str(fid)] = bins
    return out


@pytest.mark.slow
def test_mcts_real_data_smoke():
    project_root = Path(__file__).resolve().parents[2]
    occupancy_candidates = [
        project_root / "output" / "so6_occupancy_matrix_with_times.json",
        project_root / "data" / "tailwind" / "so6_occupancy_matrix_with_times.json",
    ]
    indexer_candidates = [
        project_root / "output" / "tvtw_indexer.json",
        project_root / "data" / "tailwind" / "tvtw_indexer.json",
    ]
    caps_candidates = [
        Path("/mnt/d/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"),
        Path("D:/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"),
        Path("/Volumes/CrucialX/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"),
        project_root / "data" / "cirrus" / "wxm_sm_ih_maxpool.geojson",
    ]

    def _pick_path(candidates):
        for path in candidates:
            if path.exists():
                return path
        return None

    occupancy_path = _pick_path(occupancy_candidates)
    indexer_path = _pick_path(indexer_candidates)
    caps_path = _pick_path(caps_candidates)

    if not occupancy_path or not indexer_path or not caps_path:
        pytest.skip("Real-data artifacts for MCTS test are unavailable")

    indexer = TVTWIndexer.load(str(indexer_path))
    with open(occupancy_path, "r", encoding="utf-8") as handle:
        _ = json.load(handle)

    flight_list = FlightList(str(occupancy_path), str(indexer_path))
    # Ensure matrix width matches expected TVTW space
    expected_tvtws = len(indexer.tv_id_to_idx) * indexer.num_time_bins
    if flight_list.num_tvtws < expected_tvtws:
        pad_cols = expected_tvtws - flight_list.num_tvtws
        pad_matrix = sparse.lil_matrix((flight_list.num_flights, pad_cols))
        flight_list._occupancy_matrix_lil = sparse.hstack(
            [flight_list._occupancy_matrix_lil, pad_matrix], format="lil"
        )
        flight_list.num_tvtws = expected_tvtws
        flight_list._temp_occupancy_buffer = np.zeros(expected_tvtws, dtype=np.float32)
        flight_list._lil_matrix_dirty = True
        flight_list._sync_occupancy_matrix()

    caps_gdf = gpd.read_file(str(caps_path))
    if caps_gdf.empty:
        pytest.skip("Traffic volume capacity file did not contain any records")

    full_evaluator = NetworkEvaluator(caps_gdf, flight_list)
    hotspot_segments = full_evaluator.get_hotspot_segments(threshold=0.0)
    if not hotspot_segments:
        pytest.skip("No overloaded hotspot segments found")

    primary_segment: Dict[str, Any] | None = None
    for seg in hotspot_segments:
        tv_id = str(seg.get("traffic_volume_id"))
        if tv_id in indexer.tv_id_to_idx:
            primary_segment = seg
            break
    if not primary_segment:
        pytest.skip("No controllable volume in hotspot segments")

    hotspot_tv = str(primary_segment.get("traffic_volume_id"))
    start_bin = int(primary_segment.get("start_bin", 0))
    end_bin = int(primary_segment.get("end_bin", start_bin))
    if end_bin < start_bin:
        end_bin = start_bin
    active_windows: Dict[str, List[int]] = {hotspot_tv: list(range(start_bin, end_bin + 1))}
    if not active_windows[hotspot_tv]:
        pytest.skip("No contiguous bins for hotspot")

    # Collect flights touching the hotspot and cluster to flows
    union_ids, _meta = collect_hotspot_flights(flight_list, [hotspot_tv], active_windows=active_windows)
    if not union_ids:
        pytest.skip("Hotspot flight union empty")

    flow_map = build_global_flows(
        flight_list,
        union_ids,
        hotspots=[hotspot_tv],
        trim_policy="earliest_hotspot",
        leiden_params={"threshold": 0.3, "resolution": 1.0, "seed": 0},
        direction_opts={"mode": "none"},
    )
    if not flow_map:
        pytest.skip("Global flow clustering returned no assignments")

    flights_by_flow, ctrl_by_flow = prepare_flow_scheduling_inputs(
        flight_list=flight_list,
        flow_map=flow_map,
        hotspot_ids=[hotspot_tv],
    )

    flows_by_ctrl: Dict[str, Dict[str, List[str]]] = {}
    for flow_id, ctrl in ctrl_by_flow.items():
        if ctrl is None or ctrl not in active_windows:
            continue
        specs = flights_by_flow.get(flow_id)
        if not specs:
            continue
        flows_by_ctrl.setdefault(str(ctrl), {})[str(flow_id)] = [spec["flight_id"] for spec in specs]
    if not flows_by_ctrl:
        pytest.skip("No flows associated with controllable volume")

    # Pick the control volume with most flights
    def _score(mapping: Dict[str, List[str]]) -> int:
        return sum(len(fids) for fids in mapping.values())

    selected_ctrl, selected_flow_map = max(flows_by_ctrl.items(), key=lambda item: _score(item[1]))

    # Trim flows: top 3 flows, up to 15 flights per flow
    trimmed_flows: Dict[str, Tuple[str, ...]] = {}
    for flow_id in sorted(selected_flow_map.keys(), key=lambda x: int(x)):
        flights = selected_flow_map[flow_id][:15]
        if flights:
            trimmed_flows[flow_id] = tuple(flights)
        if len(trimmed_flows) >= 3:
            break
    if not trimmed_flows:
        pytest.skip("No viable flows after trimming")

    bins_for_ctrl = active_windows.get(selected_ctrl)
    if not bins_for_ctrl:
        pytest.skip("Selected ctrl missing bins")
    window_start = int(min(bins_for_ctrl))
    window_end = int(max(bins_for_ctrl)) + 1
    if window_end <= window_start:
        window_end = window_start + 1
    window_bins = (window_start, window_end)

    relevant_tvs = {selected_ctrl}
    for meta in flight_list.flight_metadata.values():
        for interval in meta.get("occupancy_intervals", []):
            try:
                idx_val = int(interval["tvtw_index"])
            except Exception:
                continue
            decoded = indexer.get_tvtw_from_index(idx_val)
            if decoded:
                relevant_tvs.add(str(decoded[0]))

    caps_subset = gpd.GeoDataFrame(caps_gdf[caps_gdf["traffic_volume_id"].isin(relevant_tvs)].copy())
    if caps_subset.empty:
        pytest.skip("Caps subset empty for selected TVs")

    evaluator = NetworkEvaluator(caps_subset, flight_list)
    rf = RateFinder(
        evaluator=evaluator,
        flight_list=flight_list,
        indexer=indexer,
        config=RateFinderConfig(rate_grid=tuple(), passes=1, max_eval_calls=128, use_adaptive_grid=True),
    )

    proxies = _build_proxies_with_rfinder(rf, selected_ctrl, window_bins, trimmed_flows)

    # Seed plan state and cheap transition with proxies
    transition = CheapTransition(flow_proxies=proxies, clip_value=250.0)
    state = PlanState()
    state, _, _ = transition.step(state, NewRegulation())
    state, _, _ = transition.step(
        state,
        PickHotspot(
            control_volume_id=str(selected_ctrl),
            window_bins=window_bins,
            candidate_flow_ids=tuple(sorted(trimmed_flows.keys(), key=lambda x: int(x))),
            metadata={
                "flow_to_flights": trimmed_flows,
                "flow_proxies": {k: v.tolist() for k, v in proxies.items()},
            },
        ),
    )

    # Run MCTS with small budgets
    mcts = MCTS(transition=transition, rate_finder=rf, config=MCTSConfig(max_sims=12, commit_depth=1, commit_eval_limit=3, seed=0))
    commit_action = mcts.run(state)

    # Assert we found a useful commit
    assert isinstance(commit_action.committed_rates, (dict, int))
    info = (commit_action.diagnostics or {}).get("rate_finder", {})
    delta_j = float(info.get("delta_j", 0.0))
    assert delta_j < 0.0  # improvement achieved

