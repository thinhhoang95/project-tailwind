from __future__ import annotations

import math
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from parrhesia.flow_agent.rate_finder import RateFinder, RateFinderConfig
from parrhesia.flow_agent.state import PlanState
from parrhesia.flows.flow_pipeline import build_global_flows, collect_hotspot_flights
from parrhesia.optim.sa_optimizer import prepare_flow_scheduling_inputs
from parrhesia.optim.objective import ObjectiveWeights, score_with_context
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator



# Profiling helpers ===
from contextlib import contextmanager
from collections import defaultdict
import time, atexit

_stats = defaultdict(lambda: [0, 0.0])  # name -> [calls, total_seconds]

@contextmanager
def timed(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _stats[name][0] += 1
        _stats[name][1] += dt

@atexit.register
def _report_timings():
    if not _stats:
        return
    total = sum(sec for _, sec in _stats.values())
    width = max(len(k) for k in _stats)
    print("\n=== Timing summary (wall time) ===")
    for name, (calls, sec) in sorted(_stats.items(), key=lambda kv: kv[1][1], reverse=True):
        avg = sec / calls
        share = (sec / total) if total else 0
        print(f"{name:<{width}}  total {sec*1000:8.1f} ms  avg {avg*1000:7.1f} ms  calls {calls:5d}  share {share:5.1%}")
# End profiling helpers ===



class DummyFlightList:
    """Lightweight FlightList substitute for rate finder tests."""

    def __init__(self, *, indexer: TVTWIndexer, flight_metadata: Dict[str, Dict[str, object]]):
        self.indexer = indexer
        self.time_bin_minutes = indexer.time_bin_minutes
        self.tv_id_to_idx = indexer.tv_id_to_idx
        self.num_tvtws = len(self.tv_id_to_idx) * indexer.num_time_bins
        self.flight_metadata = flight_metadata
        self.flight_ids = list(flight_metadata.keys())
        self.num_flights = len(self.flight_ids)
        self.occupancy_matrix = self._build_sparse()

    def _build_sparse(self):
        rows = []
        cols = []
        data = []
        for row, fid in enumerate(self.flight_ids):
            for interval in self.flight_metadata[fid]["occupancy_intervals"]:
                cols.append(int(interval["tvtw_index"]))
                rows.append(row)
                data.append(1.0)
        shape = (self.num_flights, self.num_tvtws)
        return sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)

    def get_occupancy_vector(self, flight_id: str) -> np.ndarray:
        row = self.flight_ids.index(flight_id)
        return self.occupancy_matrix.getrow(row).toarray().ravel()

    def shift_flight_occupancy(self, flight_id: str, delay_minutes: int) -> np.ndarray:
        bins = int(math.ceil(max(delay_minutes, 0) / self.time_bin_minutes))
        vec = self.get_occupancy_vector(flight_id)
        if bins <= 0:
            return vec.copy()
        shifted = np.zeros_like(vec)
        if bins < len(vec):
            shifted[bins:] = vec[:-bins]
        return shifted

    def get_total_occupancy_by_tvtw(self) -> np.ndarray:
        return np.asarray(self.occupancy_matrix.sum(axis=0)).ravel()

    def iter_hotspot_crossings(self, hotspot_ids, active_windows=None):
        window_map = None
        if isinstance(active_windows, dict):
            window_map = {k: set(int(x) for x in v) for k, v in active_windows.items()}
        hotspots = set(str(h) for h in hotspot_ids)
        for fid, meta in self.flight_metadata.items():
            takeoff = meta["takeoff_time"]
            for interval in meta["occupancy_intervals"]:
                idx = int(interval["tvtw_index"])
                decoded = self.indexer.get_tvtw_from_index(idx)
                if not decoded:
                    continue
                tv_id, time_idx = decoded
                if tv_id not in hotspots:
                    continue
                if window_map is not None and int(time_idx) not in window_map.get(tv_id, set()):
                    continue
                entry_s = float(interval.get("entry_time_s", 0.0))
                entry_dt = takeoff + timedelta(seconds=entry_s)
                yield (fid, tv_id, entry_dt, int(time_idx))

    def copy(self):
        # NetworkEvaluator snapshots the original baseline via copy(); tests do not mutate.
        return self


@pytest.fixture(scope="module")
def rate_finder_env() -> Dict[str, object]:
    indexer = TVTWIndexer(time_bin_minutes=30)
    indexer._tv_id_to_idx = {"TV1": 0}
    indexer._idx_to_tv_id = {0: "TV1"}
    indexer._populate_tvtw_mappings()

    window_bins = (16, 18)  # covers 08:00-09:00
    tv_index_a = indexer.get_tvtw_index("TV1", window_bins[0])
    tv_index_b = indexer.get_tvtw_index("TV1", window_bins[0] + 1)

    t0 = datetime(2024, 1, 1, 8, 0, 0)
    flights = {
        "F1": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": int(tv_index_a), "entry_time_s": 0, "exit_time_s": 300}
            ],
        },
        "F2": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": int(tv_index_a), "entry_time_s": 120, "exit_time_s": 420}
            ],
        },
        "F3": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": int(tv_index_b), "entry_time_s": 60, "exit_time_s": 360}
            ],
        },
    }

    flight_list = DummyFlightList(indexer=indexer, flight_metadata=flights)

    gdf = gpd.GeoDataFrame(
        {
            "traffic_volume_id": ["TV1"],
            "capacity": [{"08:00-09:00": 1}],
            "geometry": [None],
        }
    )

    evaluator = NetworkEvaluator(gdf, flight_list)
    config = RateFinderConfig(rate_grid=(math.inf, 3.0, 2.0, 1.0), passes=2, max_eval_calls=20)
    rf = RateFinder(evaluator=evaluator, flight_list=flight_list, indexer=indexer, config=config)

    flows = {
        "A": ("F1", "F2"),
        "B": ("F3",),
    }
    plan_state = PlanState()
    return {
        "rate_finder": rf,
        "plan_state": plan_state,
        "control_volume": "TV1",
        "window_bins": window_bins,
        "flows": flows,
        "config": config,
        "flight_list": flight_list,
        "indexer": indexer,
        "evaluator": evaluator,
    }


def test_rate_finder_per_flow_improves_or_matches_baseline(rate_finder_env):
    rf = rate_finder_env["rate_finder"]
    plan_state = rate_finder_env["plan_state"]
    rates, delta_j, info = rf.find_rates(
        plan_state=plan_state,
        control_volume_id=rate_finder_env["control_volume"],
        window_bins=rate_finder_env["window_bins"],
        flows=rate_finder_env["flows"],
        mode="per_flow",
    )
    assert set(rates.keys()) == set(rate_finder_env["flows"].keys())
    assert delta_j <= 1e-6  # no regression
    assert info["eval_calls"] <= rate_finder_env["config"].max_eval_calls
    assert info["timing_seconds"] < 0.5
    assert info["passes_ran"] >= 1
    assert math.isclose(info["final_objective"], info["baseline_objective"] + delta_j, rel_tol=1e-6, abs_tol=1e-6)
    assert info["baseline_components"]["J_reg"] == pytest.approx(0.0)
    assert all(delay == 0 for delay in info["baseline_delays_min"].values())
    assert info["aggregate_delays_size"] == len(info["final_delays_min"])  # sanity
    assert set(info["final_components"].keys()) >= {"J_cap", "J_delay", "J_reg", "J_tv"}


def test_rate_finder_finite_rate_tradeoff():
    indexer = TVTWIndexer(time_bin_minutes=30)
    indexer._tv_id_to_idx = {"TV1": 0}
    indexer._idx_to_tv_id = {0: "TV1"}
    indexer._populate_tvtw_mappings()

    window_bins = (16, 18)
    tv_index = indexer.get_tvtw_index("TV1", window_bins[0])
    t0 = datetime(2024, 1, 1, 8, 0, 0)

    flights = {
        "F1": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": int(tv_index), "entry_time_s": 0, "exit_time_s": 300}
            ],
        },
        "F2": {
            "takeoff_time": t0,
            "occupancy_intervals": [
                {"tvtw_index": int(tv_index), "entry_time_s": 30, "exit_time_s": 330}
            ],
        },
    }

    flight_list = DummyFlightList(indexer=indexer, flight_metadata=flights)
    gdf = gpd.GeoDataFrame(
        {
            "traffic_volume_id": ["TV1"],
            "capacity": [{"08:00-09:00": 1}],
            "geometry": [None],
        }
    )

    evaluator = NetworkEvaluator(gdf, flight_list)
    rf = RateFinder(
        evaluator=evaluator,
        flight_list=flight_list,
        indexer=indexer,
        config=RateFinderConfig(rate_grid=(math.inf, 2.0), passes=1, max_eval_calls=8),
    )

    flows = {"A": ("F1", "F2")}
    entrants = rf._compute_entrants("TV1", list(range(*window_bins)), flows)
    flights_by_flow = rf._build_flights_by_flow(
        control_volume_id="TV1",
        active_windows=list(range(*window_bins)),
        flow_map=flows,
        entrants=entrants,
        mode="per_flow",
    )
    capacities = rf._build_capacities_for_tv("TV1")
    context_key = rf._context_key(
        plan_key="tradeoff",
        control_volume_id="TV1",
        window_bins=window_bins,
        flow_ids=list(flows.keys()),
    )
    weights = ObjectiveWeights()
    context, baseline = rf._ensure_context_and_baseline(
        context_key=context_key,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=capacities,
        target_cells={("TV1", t) for t in range(*window_bins)},
        weights=weights,
        tv_filter=["TV1"],
    )

    schedule = rf._build_schedule_from_rates(
        rates_map={"A": 2.0},
        context=context,
        active_bins=list(range(*window_bins)),
        bin_minutes=int(indexer.time_bin_minutes),
    )
    objective, components, artifacts = score_with_context(
        schedule,
        flights_by_flow=flights_by_flow,
        capacities_by_tv=capacities,
        flight_list=flight_list,
        context=context,
    )

    assert components["J_cap"] < baseline.components["J_cap"]
    assert components["J_delay"] > baseline.components["J_delay"]
    assert components["J_reg"] >= baseline.components["J_reg"]
    assert artifacts["delays_min"]["F2"] > 0


def test_rate_finder_reuses_caches(rate_finder_env):
    rf = rate_finder_env["rate_finder"]
    plan_state = rate_finder_env["plan_state"]

    # First run to populate caches
    rf.find_rates(
        plan_state=plan_state,
        control_volume_id=rate_finder_env["control_volume"],
        window_bins=rate_finder_env["window_bins"],
        flows=rate_finder_env["flows"],
        mode="per_flow",
    )

    rates, delta_j, info = rf.find_rates(
        plan_state=plan_state,
        control_volume_id=rate_finder_env["control_volume"],
        window_bins=rate_finder_env["window_bins"],
        flows=rate_finder_env["flows"],
        mode="per_flow",
    )
    assert set(rates.keys()) == set(rate_finder_env["flows"].keys())
    assert info["cache_hits"] > 0
    assert info["eval_calls"] == 0
    assert math.isclose(delta_j, info["delta_j"])  # consistency check


def test_rate_finder_blanket_mode_deterministic(rate_finder_env):
    env = rate_finder_env
    config = RateFinderConfig(rate_grid=(math.inf, 2.0, 1.0), passes=2, max_eval_calls=20)
    rf = RateFinder(
        evaluator=env["evaluator"],
        flight_list=env["flight_list"],
        indexer=env["indexer"],
        config=config,
    )
    plan_state = PlanState()
    result_one = rf.find_rates(
        plan_state=plan_state,
        control_volume_id=env["control_volume"],
        window_bins=env["window_bins"],
        flows=env["flows"],
        mode="blanket",
    )
    result_two = rf.find_rates(
        plan_state=plan_state,
        control_volume_id=env["control_volume"],
        window_bins=env["window_bins"],
        flows=env["flows"],
        mode="blanket",
    )

    rates_one, delta_one, info_one = result_one
    rates_two, delta_two, info_two = result_two

    assert math.isclose(rates_one, rates_two)
    assert math.isclose(delta_one, delta_two)
    assert info_one["final_components"] == info_two["final_components"]


def test_rate_finder_adaptive_grid_generation(rate_finder_env):
    env = rate_finder_env
    rf: RateFinder = env["rate_finder"]
    rf.config.use_adaptive_grid = True
    rf.config.rate_grid = tuple()

    rates, _, info = rf.find_rates(
        plan_state=env["plan_state"],
        control_volume_id=env["control_volume"],
        window_bins=env["window_bins"],
        flows=env["flows"],
        mode="per_flow",
    )

    grid = info["rate_grid"]
    assert math.isinf(grid[0])
    assert len(grid) <= rf.config.max_adaptive_candidates
    assert any(rate >= 1 for rate in grid[1:])
    assert rates  # should still return per-flow rates dictionary


def test_rate_finder_real_data_smoke(tmp_path):
    """Run RateFinder on a real-data sample to ensure end-to-end compatibility."""

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
        # raise ValueError("Real-data artifacts for rate finder test are unavailable")
        pytest.skip("Real-data artifacts for rate finder test are unavailable")
    else:
        print("Occupancy path: ", occupancy_path)
        print("Indexer path: ", indexer_path)
        print("Caps path: ", caps_path)

    with timed("TVTWIndexer.load"):
        indexer = TVTWIndexer.load(str(indexer_path))

    with timed("json.load"):
        with open(occupancy_path, "r", encoding="utf-8") as handle:
            sample = json.load(handle)

    flight_list = FlightList(str(occupancy_path), str(indexer_path))
    print(f"Loaded {flight_list.num_flights} flights from {len(sample)} in sample JSON.")

    with timed("FlightList"):
        expected_tvtws = len(indexer.tv_id_to_idx) * indexer.num_time_bins
        if flight_list.num_tvtws < expected_tvtws:
            pad_cols = expected_tvtws - flight_list.num_tvtws
            pad_matrix = sparse.lil_matrix((flight_list.num_flights, pad_cols), dtype=np.float32)
            flight_list._occupancy_matrix_lil = sparse.hstack(
                [flight_list._occupancy_matrix_lil, pad_matrix], format="lil"
            )
            flight_list.num_tvtws = expected_tvtws
            flight_list._temp_occupancy_buffer = np.zeros(expected_tvtws, dtype=np.float32)
            flight_list._lil_matrix_dirty = True
            flight_list._sync_occupancy_matrix()

    with timed("gpd.read_file"):
        caps_gdf = gpd.read_file(str(caps_path))
    if caps_gdf.empty:
        pytest.skip("Traffic volume capacity file did not contain any records")

    full_evaluator = NetworkEvaluator(caps_gdf, flight_list)
    with timed("full_evaluator.get_hotspot_segments"):
        hotspot_segments = full_evaluator.get_hotspot_segments(threshold=0.0)

    if not hotspot_segments:
        pytest.skip("Rolling-hour hotspot detection found no overloaded segments")

    primary_segment: Dict[str, Any] | None = None
    for seg in hotspot_segments:
        tv_id = str(seg.get("traffic_volume_id"))
        if tv_id in indexer.tv_id_to_idx:
            primary_segment = seg
            break

    if not primary_segment:
        pytest.skip("Rolling-hour hotspot detection did not yield a controllable volume")

    hotspot_tv = str(primary_segment.get("traffic_volume_id"))
    hotspots: List[str] = [hotspot_tv]
    print(f"Selected hotspot: {hotspot_tv}")

    start_bin = int(primary_segment.get("start_bin", 0))
    end_bin = int(primary_segment.get("end_bin", start_bin))
    if end_bin < start_bin:
        end_bin = start_bin

    active_windows: Dict[str, List[int]] = {
        hotspot_tv: list(range(start_bin, end_bin + 1))
    }

    if not active_windows[hotspot_tv]:
        pytest.skip("Hotspot segment did not provide any contiguous bins")

    print(f"Active windows: {active_windows}")

    with timed("collect_hotspot_flights"):
        union_ids, _ = collect_hotspot_flights(flight_list, hotspots, active_windows=active_windows)
    if not union_ids:
        pytest.skip("Hotspot flight union was empty for real-data sample")

    with timed("build_global_flows"):
        flow_map = build_global_flows(
            flight_list,
            union_ids,
            hotspots=hotspots,
            trim_policy="earliest_hotspot",
            leiden_params={"threshold": 0.3, "resolution": 1.0, "seed": 0},
            direction_opts={"mode": "none"},
        )
    if not flow_map:
        pytest.skip("Global flow clustering returned no assignments")

    with timed("prepare_flow_scheduling_inputs"):
        flights_by_flow, ctrl_by_flow = prepare_flow_scheduling_inputs(
        flight_list=flight_list,
        flow_map=flow_map,
        hotspot_ids=hotspots,
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
        pytest.skip("No flows were associated with a controllable volume")

    def _score(mapping: Dict[str, List[str]]) -> int:
        return sum(len(fids) for fids in mapping.values())

    selected_ctrl, selected_flow_map = max(flows_by_ctrl.items(), key=lambda item: _score(item[1]))

    trimmed_flows: Dict[str, tuple] = {}
    for flow_id in sorted(selected_flow_map.keys(), key=lambda x: int(x)):
        flights = selected_flow_map[flow_id][:15]
        if flights:
            trimmed_flows[flow_id] = tuple(flights)
        if len(trimmed_flows) >= 3:
            break

    if not trimmed_flows:
        pytest.skip("Selected control volume did not have any viable flows")

    print(f"Selected control volume for rate finding: {selected_ctrl}")
    print("Flows for rate finding:")
    for flow_id, flights in trimmed_flows.items():
        print(f"  - Flow {flow_id}: {len(flights)} flights")

    bins_for_ctrl = active_windows.get(selected_ctrl)
    if not bins_for_ctrl:
        pytest.skip("Selected control volume is missing active windows")

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

    caps_subset = caps_gdf[caps_gdf["traffic_volume_id"].isin(relevant_tvs)].copy()
    if caps_subset.empty:
        pytest.skip("Capacities GeoJSON did not contain the sampled traffic volumes")

    with timed("NetworkEvaluator"):
        evaluator = NetworkEvaluator(caps_subset, flight_list)

    config = RateFinderConfig(
        rate_grid=tuple(),
        passes=1,
        max_eval_calls=128,
        use_adaptive_grid=True,
    )
    rf = RateFinder(evaluator=evaluator, flight_list=flight_list, indexer=indexer, config=config)

    plan_state = PlanState()
    with timed("RateFinder.find_rates"):
        rates, delta_j, info = rf.find_rates(
            plan_state=plan_state,
            control_volume_id=selected_ctrl,
            window_bins=window_bins,
            flows=trimmed_flows,
            mode="per_flow",
        )

    baseline_objective = info.get("baseline_objective")
    post_optimized_objective = "N/A"
    if baseline_objective is not None:
        post_optimized_objective = baseline_objective + delta_j

    print("\n--- Rate Finder Results ---")
    print(f"Baseline objective: {baseline_objective}")
    print(f"Found rates: {rates}")
    print(f"Post-optimized objective: {post_optimized_objective}")
    print(f"Objective delta: {delta_j}")
    print(f"Additional info: {info}")
    print("--------------------------\n")

    assert isinstance(rates, dict)
    assert set(rates.keys()) == set(trimmed_flows.keys())
    assert info["eval_calls"] > 0
    assert info["baseline_objective"] is not None
    assert delta_j <= 1e-6
