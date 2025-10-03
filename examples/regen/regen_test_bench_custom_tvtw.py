from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np


# Ensure 'src' is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Core data structures
from project_tailwind.impact_eval.distance_computation import haversine_vectorized
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator

# Flow regeneration engine
from parrhesia.api.flows import compute_flows
from parrhesia.api.resources import set_global_resources
from parrhesia.flow_agent35.regen.engine import propose_regulations_for_hotspot
from parrhesia.flow_agent35.regen.exceedance import compute_hotspot_exceedance
from parrhesia.flow_agent35.regen.rates import compute_e_target
from parrhesia.flow_agent35.regen.types import RegenConfig
from parrhesia.optim.capacity import normalize_capacities


# Time Profiling helpers ===
from contextlib import contextmanager
from collections import Counter, defaultdict, deque
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

leiden_params = {
    "threshold": 0.64,
    "resolution": 1.0,
    "seed": 0,
}

def _pick_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_artifacts() -> Tuple[Path, Path, Path]:
    """
    Resolve and validate artifact paths (occupancy, indexer, capacities).
    """
    project_root = REPO_ROOT

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
        project_root / "output" / "wxm_sm_ih_maxpool.geojson",
    ]

    occupancy_path = _pick_existing_path(occupancy_candidates)
    indexer_path = _pick_existing_path(indexer_candidates)
    caps_path = _pick_existing_path(caps_candidates)

    if not occupancy_path or not indexer_path or not caps_path:
        raise SystemExit(
            "Required artifacts not found. Ensure occupancy, indexer, and capacity GeoJSON exist."
        )

    return occupancy_path, indexer_path, caps_path


def build_data(occupancy_path: Path, indexer_path: Path, caps_path: Path) -> Tuple[FlightList, TVTWIndexer, NetworkEvaluator]:
    # Load indexer and occupancy
    indexer = TVTWIndexer.load(str(indexer_path))

    # Touch occupancy JSON to validate read
    with open(occupancy_path, "r", encoding="utf-8") as fh:
        _ = json.load(fh)

    flight_list = FlightList(str(occupancy_path), str(indexer_path))

    # Align occupancy matrix width (defensive, mirroring examples/flow_agent/run_agent.py)
    expected_tvtws = len(indexer.tv_id_to_idx) * indexer.num_time_bins
    if getattr(flight_list, "num_tvtws", 0) < expected_tvtws:
        from scipy import sparse  # type: ignore

        pad_cols = expected_tvtws - int(flight_list.num_tvtws)
        pad_matrix = sparse.lil_matrix((int(flight_list.num_flights), pad_cols))
        flight_list._occupancy_matrix_lil = sparse.hstack(  # type: ignore[attr-defined]
            [flight_list._occupancy_matrix_lil, pad_matrix], format="lil"  # type: ignore[attr-defined]
        )
        flight_list.num_tvtws = expected_tvtws  # type: ignore[assignment]
        flight_list._temp_occupancy_buffer = np.zeros(expected_tvtws, dtype=np.float32)  # type: ignore[attr-defined]
        flight_list._lil_matrix_dirty = True  # type: ignore[attr-defined]
        flight_list._sync_occupancy_matrix()  # type: ignore[attr-defined]

    # Capacities
    caps_gdf = gpd.read_file(str(caps_path))
    if caps_gdf.empty:
        raise SystemExit("Traffic volume capacity file is empty; cannot proceed.")

    evaluator = NetworkEvaluator(caps_gdf, flight_list)
    return flight_list, indexer, evaluator


def _timebins_from_window(window_bins: Iterable[int]) -> List[int]:
    wb = list(int(b) for b in window_bins)
    if not wb:
        return []
    if len(wb) == 1:
        return [wb[0]]
    start = wb[0]
    end_exclusive = wb[1]
    if end_exclusive <= start:
        end_exclusive = start + 1
    return list(range(int(start), int(end_exclusive)))

# INSTRUCTIONS FOR CODING AGENTS AND DEVELOPERS: This function could be a source of inconsistency
# since it replicates regen_api_wrapper's _build_capacities_by_tv function
# If the objective function seem very high, one common culprit is the normalization of missing capacity values
def _build_capacities_by_tv(
    evaluator: NetworkEvaluator,
    indexer: TVTWIndexer,
) -> Dict[str, np.ndarray]:
    """
    Build per-bin capacities for each traffic volume and normalize them,
    mirroring the approach used by the server-side wrapper.
    """
    T = int(indexer.num_time_bins)
    bins_per_hour = int(indexer.rolling_window_size())

    # Construct raw per-bin capacities from evaluator's hourly map
    per_hour = getattr(evaluator, "hourly_capacity_by_tv", {}) or {}
    raw_capacities: Dict[str, np.ndarray] = {}
    for tv_id in indexer.tv_id_to_idx.keys():
        arr = np.zeros(T, dtype=np.float64)
        hours = per_hour.get(tv_id, {}) or {}
        for h, cap in hours.items():
            try:
                hour = int(h)
            except Exception:
                continue
            start = hour * bins_per_hour
            if start >= T:
                continue
            end = min(start + bins_per_hour, T)
            arr[start:end] = float(cap)
        raw_capacities[str(tv_id)] = arr

    if raw_capacities:
        non_positive = [tv for tv, arr in raw_capacities.items() if float(np.max(arr)) <= 0.0]
        if non_positive:
            print(
                f"Regen: capacity rows are non-positive for {len(non_positive)}/{len(raw_capacities)} TVs; sample="
                + ",".join([str(x) for x in non_positive[:5]])
            )

    # Normalize: treat missing/zero bins as unconstrained
    capacities_by_tv = normalize_capacities(raw_capacities)

    if capacities_by_tv:
        sample_items = list(capacities_by_tv.items())[:5]
        sample_stats = []
        for tv, arr in sample_items:
            arr_np = np.asarray(arr, dtype=np.float64)
            if arr_np.size == 0:
                sample_stats.append(f"{tv}:empty")
                continue
            sample_stats.append(
                f"{tv}:min={float(arr_np.min()):.1f},max={float(arr_np.max()):.1f}"
            )
        print(
            f"Regen: normalized capacities ready for {len(capacities_by_tv)} TVs; samples: "
            + "; ".join(sample_stats)
        )

    return capacities_by_tv


def _tv_centroids(
    gdf: gpd.GeoDataFrame,
    indexer: TVTWIndexer,
) -> Dict[str, Tuple[float, float]]:
    try:
        geo = gdf.to_crs(epsg=4326) if gdf.crs and "4326" not in str(gdf.crs) else gdf
    except Exception:
        geo = gdf
    tv_ids = set(str(tv) for tv in indexer.tv_id_to_idx.keys())
    centroids: Dict[str, Tuple[float, float]] = {}
    for _, row in geo.iterrows():
        tv_id = row.get("traffic_volume_id")
        if tv_id is None:
            continue
        tv_key = str(tv_id)
        if tv_key not in tv_ids:
            continue
        geom = row.get("geometry")
        if geom is None or geom.is_empty:
            continue
        try:
            c = geom.centroid
            centroids[tv_key] = (float(c.y), float(c.x))
        except Exception:
            continue
    return centroids


def _travel_minutes_from_centroids(
    centroids: Mapping[str, Tuple[float, float]],
    *,
    speed_kts: float = 475.0,
) -> Dict[str, Dict[str, float]]:
    if not centroids:
        return {}
    ids = sorted(centroids.keys())
    lat_arr = np.asarray([centroids[i][0] for i in ids], dtype=np.float64)
    lon_arr = np.asarray([centroids[i][1] for i in ids], dtype=np.float64)
    dist_nm = haversine_vectorized(lat_arr[:, None], lon_arr[:, None], lat_arr[None, :], lon_arr[None, :])
    minutes = (dist_nm / float(speed_kts)) * 60.0
    out: Dict[str, Dict[str, float]] = {}
    for i, src in enumerate(ids):
        out[src] = {ids[j]: float(minutes[i, j]) for j in range(len(ids))}
    return out


def _lookup_flow_payload(flows_payload: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    lookup: Dict[int, Mapping[str, Any]] = {}
    for flow in flows_payload.get("flows", []) or []:
        try:
            fid = int(flow.get("flow_id"))
        except Exception:
            continue
        lookup[fid] = flow
    return lookup


def _build_flow_summary(
    proposal,
    *,
    hotspot_payload: Mapping[str, Any],
    flows_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    window_bins = list(hotspot_payload.get("window_bins", []))
    timebins_h = _timebins_from_window(window_bins)
    start = timebins_h[0] if timebins_h else 0
    end_exclusive = timebins_h[-1] + 1 if timebins_h else 0
    meta = hotspot_payload.get("metadata", {}) or {}
    flow_to_flights = meta.get("flow_to_flights", {}) or {}
    proxies = meta.get("flow_proxies", {}) or {}
    payload_lookup = _lookup_flow_payload(flows_payload)
    summary: List[Dict[str, Any]] = []
    for flow in proposal.flows_info:
        fid = int(flow.get("flow_id"))
        fid_key = str(fid)
        payload_entry = payload_lookup.get(fid, {})
        demand = payload_entry.get("demand") or []
        entrants = 0.0
        if demand and end_exclusive > start:
            entrants = float(sum(float(demand[t]) for t in range(start, min(end_exclusive, len(demand)))))
        elif proxies.get(fid_key):
            entrants = float(sum(float(x) for x in proxies[fid_key]))
        flights = flow_to_flights.get(fid_key) or []
        summary.append(
            {
                "flow_id": fid,
                "control_tv_id": flow.get("control_tv_id"),
                "baseline_rate_per_hour": float(flow.get("r0_i", 0.0)),
                "allowed_rate_per_hour": float(flow.get("R_i", 0.0)),
                "assigned_cut_per_hour": float(flow.get("lambda_cut_i", 0.0)),
                "entrants_in_window": entrants,
                "num_flights": int(flow.get("num_flights", len(flights))),
            }
        )
    summary.sort(key=lambda item: item["flow_id"])
    return summary


def _proposal_to_dict(
    proposal,
    *,
    hotspot_payload: Mapping[str, Any],
    flows_payload: Mapping[str, Any],
    exceedance_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    flows = _build_flow_summary(proposal, hotspot_payload=hotspot_payload, flows_payload=flows_payload)
    diag = dict(proposal.diagnostics)
    diag.setdefault("D_peak", float(exceedance_stats.get("D_peak", 0.0)))
    diag.setdefault("D_sum", float(exceedance_stats.get("D_total", 0.0)))
    diag.setdefault("window_bins_input", list(hotspot_payload.get("window_bins", [])))
    result = {
        "hotspot_id": proposal.hotspot_id,
        "control_volume_id": proposal.controlled_volume,
        "window_bins": [int(proposal.window.start_bin), int(proposal.window.end_bin)],
        "flows": flows,
        "predicted_improvement": {
            "delta_deficit_per_hour": float(proposal.predicted_improvement.delta_deficit_per_hour),
            "delta_objective_score": float(proposal.predicted_improvement.delta_objective_score),
        },
        "diagnostics": diag,
    }
    return result


def propose_regulations(
    *,
    hotspot_payload: Dict[str, Any],
    evaluator: NetworkEvaluator,
    indexer: TVTWIndexer,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    control_tv = str(hotspot_payload.get("control_volume_id"))
    if not control_tv:
        raise ValueError("hotspot payload missing control volume id")
    window_bins = hotspot_payload.get("window_bins") or []
    timebins_h = _timebins_from_window(window_bins)
    if not timebins_h:
        raise ValueError("hotspot payload missing time window bins")

    flight_list = getattr(evaluator, "flight_list", None)
    if flight_list is None:
        raise ValueError("network evaluator missing flight_list reference")

    # Prepare shared inputs
    capacities_by_tv = _build_capacities_by_tv(evaluator, indexer)
    centroids = _tv_centroids(evaluator.traffic_volumes_gdf, indexer)
    travel_minutes_map = _travel_minutes_from_centroids(centroids)

    # Flow payload via API (reuse in-memory artifacts)
    set_global_resources(indexer, flight_list)
    direction_opts = {"mode": "coord_cosine", "tv_centroids": centroids} if centroids else {"mode": "coord_cosine"}
    flows_payload = compute_flows(tvs=[control_tv], timebins=timebins_h, direction_opts=direction_opts, threshold=leiden_params["threshold"], resolution=leiden_params["resolution"])

    # Flow-to-flights metadata from hotspot discovery (string keys)
    meta = hotspot_payload.get("metadata", {}) or {}
    flow_to_flights_in = meta.get("flow_to_flights", {}) or {}
    flow_to_flights: Dict[str, Sequence[str]] = {
        str(k): tuple(v or []) for k, v in flow_to_flights_in.items()
    }

    proposals: List[Dict[str, Any]] = []
    try:
        fallback_cfg = RegenConfig(
            g_min=-float("inf"),
            rho_max=float("inf"),
            slack_min=-float("inf"),
            distinct_controls_required=False,
            raise_on_edge_cases=True,
            min_num_flights=4
        )


        
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

    except ValueError as exc:
        print(f"[regen] Primary proposal attempt failed: {exc}")
    if not proposals:
        raise RuntimeError("regen engine returned no proposals even with relaxed config")

    exceedance_stats = compute_hotspot_exceedance(
        indexer=indexer,
        flight_list=flight_list,
        capacities_by_tv=capacities_by_tv,
        hotspot_tv=control_tv,
        timebins_h=timebins_h,
    )
    proposals_to_emit: List[Dict[str, Any]] = []
    for idx, proposal in enumerate(proposals):
        if top_k is not None and idx >= int(top_k):
            break
        proposals_to_emit.append(
            _proposal_to_dict(
                proposal,
                hotspot_payload=hotspot_payload,
                flows_payload=flows_payload,
                exceedance_stats=exceedance_stats,
            )
        )

    if not proposals_to_emit:
        raise RuntimeError("Requested top_k resulted in zero proposals being returned")

    return proposals_to_emit


def propose_top_regulations(
    *,
    hotspot_payload: Dict[str, Any],
    evaluator: NetworkEvaluator,
    indexer: TVTWIndexer,
    top_k: int = 8,
) -> Dict[str, Any]:
    return propose_regulations(
        hotspot_payload=hotspot_payload,
        evaluator=evaluator,
        indexer=indexer,
        top_k=top_k,
    )[0]


def main() -> None:
    print("[regen] Resolving artifacts ...")
    occ_path, idx_path, caps_path = load_artifacts()
    print(f"[regen] occupancy: {occ_path}")
    print(f"[regen] indexer:   {idx_path}")
    print(f"[regen] capacities:{caps_path}")

    print("[regen] Loading data ...")
    flight_list, indexer, evaluator = build_data(occ_path, idx_path, caps_path)

    hotspot_payload = {
        "control_volume_id": "LFBZX15",
        "window_bins": [45, 49], # means [45, 46, 47, 48]
        "metadata": {}, # will be filled in below
        "mode": "manual",
    }

    # Build metadata in the same schema as HotspotInventory descriptors
    # - flow_to_flights: { flow_id(str): [flight_id(str), ...] }
    # - flow_proxies: { flow_id(str): [entrants per bin within window] }
    control_tv = str(hotspot_payload["control_volume_id"])
    timebins_h = _timebins_from_window(hotspot_payload.get("window_bins", []))
    set_global_resources(indexer, flight_list)
    centroids = _tv_centroids(evaluator.traffic_volumes_gdf, indexer)
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
    for flow in flows_payload.get("flows", []) or []:
        # Normalize flow id to string
        try:
            fid_key = str(int(flow.get("flow_id")))
        except Exception:
            fid_key = str(flow.get("flow_id"))

        # Flights for this flow
        flights: List[str] = []
        for spec in flow.get("flights", []) or []:
            fid = spec.get("flight_id")
            if fid is not None:
                flights.append(str(fid))
        flow_to_flights[fid_key] = flights

        # Proxies: entrants per bin within [t0, t1)
        demand = flow.get("demand") or []
        proxy: List[float] = []
        for b in timebins_h:
            try:
                proxy.append(float(demand[int(b)]))
            except Exception:
                proxy.append(0.0)
        flow_proxies[fid_key] = proxy

    hotspot_payload["metadata"] = {
        "flow_to_flights": flow_to_flights,
        "flow_proxies": flow_proxies,
    }
    print(
        "[regen] Using manual hotspot selection: "
        f"TV={hotspot_payload['control_volume_id']} window={hotspot_payload['window_bins']}"
    )

    print("[regen] Proposing regulations (per-flow) ...")
    proposals = propose_regulations(
        hotspot_payload=hotspot_payload,
        evaluator=evaluator,
        indexer=indexer,
    )
    print(f"[regen] Retrieved {len(proposals)} regulation candidate(s).")

    # Print proposal summary with rich formatting
    from rich.console import Console
    from rich.table import Table

    console = Console()

    for rank, proposal in enumerate(proposals, start=1):
        # Header info mirrors previous single-proposal view
        diag = proposal["diagnostics"]
        improvement = proposal.get("predicted_improvement", {})
        components_before = diag.get("score_components_before", {}) or {}
        components_after = diag.get("score_components_after", {}) or {}
        console.print(f"\n[bold green][regen] Proposal Summary #{rank}[/bold green]")
        console.print(f"Control Volume: [cyan]{proposal['control_volume_id']}[/cyan]")
        console.print(
            f"Window Bins: [cyan]{proposal['window_bins'][0]}-{proposal['window_bins'][1]}[/cyan]"
        )
        console.print(
            f"Target Exceedance to Remove: [yellow]{diag['E_target']:.1f}[/yellow] "
            f"(D_peak={diag['D_peak']:.1f}, D_sum={diag['D_sum']:.1f})"
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
                comp_table.add_row(
                    key,
                    f"{before_val:.3f}",
                    f"{after_val:.3f}",
                    f"{delta_val:+.3f}",
                )

            console.print(comp_table)

        # Flows table per proposal
        table = Table(title="Flow Regulations")
        table.add_column("Flow ID", style="cyan", no_wrap=True)
        table.add_column("Control TV", style="magenta")
        table.add_column("Baseline Rate\n(per hour)", justify="right", style="green")
        table.add_column("Allowed Rate\n(per hour)", justify="right", style="yellow")
        table.add_column("Cut\n(per hour)", justify="right", style="red")
        table.add_column("Entrants in\nWindow", justify="right")
        table.add_column("Num\nFlights", justify="right")

        for flow in proposal["flows"]:
            table.add_row(
                str(flow["flow_id"]),
                str(flow["control_tv_id"]),
                f"{flow['baseline_rate_per_hour']:.1f}",
                f"{flow['allowed_rate_per_hour']:.1f}",
                f"{flow['assigned_cut_per_hour']:.0f}",
                f"{flow['entrants_in_window']:.0f}",
                str(flow["num_flights"]),
            )

        console.print(table)

    # Optional: write to disk next to artifacts
    out_dir = REPO_ROOT / "agent_runs" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regen_proposal.json"
    output_payload = {
        "top_hotspot": hotspot_payload,
        "proposals": proposals,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2)
    print(f"[regen] Saved {len(proposals)} proposal(s): {out_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np


# Ensure 'src' is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Core data structures
from project_tailwind.impact_eval.distance_computation import haversine_vectorized
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator

# Flow regeneration engine
from parrhesia.api.flows import compute_flows
from parrhesia.api.resources import set_global_resources
from parrhesia.flow_agent35.regen.engine import propose_regulations_for_hotspot
from parrhesia.flow_agent35.regen.exceedance import compute_hotspot_exceedance
from parrhesia.flow_agent35.regen.rates import compute_e_target
from parrhesia.flow_agent35.regen.types import RegenConfig
from parrhesia.optim.capacity import build_bin_capacities, normalize_capacities
from server_tailwind.core.resources import AppResources, ResourcePaths  # NEW


# Time Profiling helpers ===
from contextlib import contextmanager
from collections import Counter, defaultdict, deque
import time, atexit

# This script serves as a test bench for the regulation proposal (regen) engine.
# It simulates the process of identifying a traffic hotspot and then calling the
# regen engine to generate and evaluate potential mitigation strategies in the form
# of flow regulations. It loads all necessary data artifacts, prepares inputs
# in the format expected by the backend APIs, invokes the core proposal logic,
# and then presents the results in a human-readable summary table.
#
# This is useful for debugging the regen engine's logic, tuning its parameters,
# and verifying that its outputs are sensible without needing to run the full
# backend server.

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

leiden_params = {
    "threshold": 0.64,
    "resolution": 1.0,
    "seed": 0,
}

def _pick_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_artifacts() -> Tuple[Path, Path, Path]:
    """
    Resolve and validate artifact paths (occupancy, indexer, capacities).

    This function searches for the required data files in a list of candidate
    locations, which can be in the project's 'output' or 'data' directories,
    or in system-specific paths for larger files like the capacity GeoJSON.
    It ensures that all necessary files are present before proceeding.

    Returns:
        A tuple containing the resolved Paths for the occupancy matrix,
        the TVTW indexer, and the capacity data file.

    Raises:
        SystemExit: If any of the required artifacts cannot be found.
    """
    project_root = REPO_ROOT

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
        project_root / "output" / "wxm_sm_ih_maxpool.geojson",
    ]

    occupancy_path = _pick_existing_path(occupancy_candidates)
    indexer_path = _pick_existing_path(indexer_candidates)
    caps_path = _pick_existing_path(caps_candidates)

    if not occupancy_path or not indexer_path or not caps_path:
        raise SystemExit(
            "Required artifacts not found. Ensure occupancy, indexer, and capacity GeoJSON exist."
        )

    return occupancy_path, indexer_path, caps_path


def build_data(occupancy_path: Path, indexer_path: Path, caps_path: Path) -> Tuple[FlightList, TVTWIndexer, NetworkEvaluator]:
    """
    Loads core data structures from artifact paths.

    This function initializes the main objects required for network evaluation
    and regulation proposals:
    - TVTWIndexer: Maps traffic volumes and time bins to matrix indices.
    - FlightList: Manages flight data and their occupancy over TVTWs.
    - NetworkEvaluator: Handles capacity evaluation and objective scoring.

    It also performs a defensive padding of the occupancy matrix to ensure its
    dimensions are consistent with the indexer, which is a step mirrored from

    Args:
        occupancy_path: Path to the flight occupancy matrix JSON file.
        indexer_path: Path to the TVTW indexer JSON file.
        caps_path: Path to the traffic volume capacities GeoJSON file.

    Returns:
        A tuple containing the initialized FlightList, TVTWIndexer, and
        NetworkEvaluator objects.
    """
    # Use AppResources so flows/regen share the same flight list/indexer as the server
    res = AppResources(
        ResourcePaths(
            occupancy_file_path=occupancy_path,
            tvtw_indexer_path=indexer_path,
            traffic_volumes_path=caps_path,
        )
    ).preload_all()

    indexer = res.indexer
    flight_list = res.flight_list

    # Align occupancy matrix width (defensive, mirroring examples/flow_agent/run_agent.py)
    # This ensures that the occupancy matrix has a column for every TVTW defined
    # in the indexer. If the matrix was generated with a slightly different set
    # of TVTWs, this padding prevents index-out-of-bounds errors.
    expected_tvtws = len(indexer.tv_id_to_idx) * indexer.num_time_bins
    if getattr(flight_list, "num_tvtws", 0) < expected_tvtws:
        from scipy import sparse  # type: ignore

        pad_cols = expected_tvtws - int(flight_list.num_tvtws)
        pad_matrix = sparse.lil_matrix((int(flight_list.num_flights), pad_cols))
        flight_list._occupancy_matrix_lil = sparse.hstack(  # type: ignore[attr-defined]
            [flight_list._occupancy_matrix_lil, pad_matrix], format="lil"  # type: ignore[attr-defined]
        )
        flight_list.num_tvtws = expected_tvtws  # type: ignore[assignment]
        flight_list._temp_occupancy_buffer = np.zeros(expected_tvtws, dtype=np.float32)  # type: ignore[attr-defined]
        flight_list._lil_matrix_dirty = True  # type: ignore[attr-defined]
        flight_list._sync_occupancy_matrix()  # type: ignore[attr-defined]

    # Capacities (reuse resources traffic volumes for consistency)
    caps_gdf = res.traffic_volumes_gdf
    if caps_gdf.empty:
        raise SystemExit("Traffic volume capacity file is empty; cannot proceed.")

    evaluator = NetworkEvaluator(caps_gdf, flight_list)
    # Preserve capacities path for later use by _build_capacities_by_tv.
    # This allows the more robust capacity builder in `parrhesia` to be used,
    # which directly reads and processes the GeoJSON file.
    try:
        evaluator._capacities_path = str(caps_path)  # type: ignore[attr-defined]
    except Exception:
        print(f"[warning] Failed to set capacities path on evaluator")
        pass
    # Attach resources for downstream reuse (centroids, travel minutes)
    try:
        setattr(evaluator, "_resources", res)
    except Exception:
        pass
    return flight_list, indexer, evaluator


def _timebins_from_window(window_bins: Iterable[int]) -> List[int]:
    """Converts a window [start, end_exclusive] to a list of integer time bins."""
    wb = list(int(b) for b in window_bins)
    if not wb:
        return []
    if len(wb) == 1:
        return [wb[0]]
    start = wb[0]
    end_exclusive = wb[1]
    if end_exclusive <= start:
        end_exclusive = start + 1
    return list(range(int(start), int(end_exclusive)))

# INSTRUCTIONS FOR CODING AGENTS AND DEVELOPERS: This function could be a source of inconsistency
# since it replicates regen_api_wrapper's _build_capacities_by_tv function
# If the objective function seem very high, one common culprit is the normalization of missing capacity values
def _build_capacities_by_tv(
    evaluator: NetworkEvaluator,
    indexer: TVTWIndexer,
) -> Dict[str, np.ndarray]:
    """
    Build per-bin capacities for each traffic volume and normalize them,
    mirroring the approach used by the server-side wrapper.

    This function is critical for providing the regen engine with the correct
    capacity information. It has two primary methods of building capacities:
    1.  **Preferred:** If a path to the capacity GeoJSON is available on the
        evaluator, it uses the `build_bin_capacities` and `normalize_capacities`
        functions from `parrhesia.optim.capacity`. This is more robust as it
        handles various capacity definitions (e.g., time-varying).
    2.  **Fallback:** If the path is not available, it constructs per-bin capacities
        from the `hourly_capacity_by_tv` map stored within the evaluator. This
        is less flexible and assumes capacities are constant within each hour.

    Normalization is a key step: it replaces zero or missing capacity values
    with a very large number (effectively infinite capacity) to signify that
    those bins are unconstrained. This prevents the optimizer from incorrectly
    penalizing traffic in volumes or times that simply lack a defined capacity.

    Args:
        evaluator: The NetworkEvaluator instance.
        indexer: The TVTWIndexer for time bin information.

    Returns:
        A dictionary mapping each traffic volume ID to a NumPy array of its
        per-bin normalized capacity values.
    """
    # Prefer the shared builder from parrhesia.optim.capacity when a source path is known
    caps_path = getattr(evaluator, "_capacities_path", None)
    if caps_path:
        try:
            # This is the preferred, more robust method using the original GeoJSON.
            raw_caps = build_bin_capacities(str(caps_path), indexer)
            capacities_by_tv = normalize_capacities(raw_caps)
            if capacities_by_tv:
                sample_items = list(capacities_by_tv.items())[:5]
                sample_stats = []
                for tv, arr in sample_items:
                    arr_np = np.asarray(arr, dtype=np.float64)
                    if arr_np.size == 0:
                        sample_stats.append(f"{tv}:empty")
                        continue
                    sample_stats.append(
                        f"{tv}:min={float(arr_np.min()):.1f},max={float(arr_np.max()):.1f}"
                    )
                print(
                    f"Regen: normalized capacities (from GeoJSON) ready for {len(capacities_by_tv)} TVs; samples: "
                    + "; ".join(sample_stats)
                )
            return capacities_by_tv
        except Exception as exc:
            print(f"Regen: capacity builder fallback due to error: {exc}")

    # Fallback method: construct capacities from the evaluator's hourly map.
    T = int(indexer.num_time_bins)
    bins_per_hour = int(indexer.rolling_window_size())

    # Construct raw per-bin capacities from evaluator's hourly map
    per_hour = getattr(evaluator, "hourly_capacity_by_tv", {}) or {}
    raw_capacities: Dict[str, np.ndarray] = {}
    for tv_id in indexer.tv_id_to_idx.keys():
        arr = np.zeros(T, dtype=np.float64)
        hours = per_hour.get(tv_id, {}) or {}
        for h, cap in hours.items():
            try:
                hour = int(h)
            except Exception:
                continue
            start = hour * bins_per_hour
            if start >= T:
                continue
            end = min(start + bins_per_hour, T)
            arr[start:end] = float(cap)
        raw_capacities[str(tv_id)] = arr

    if raw_capacities:
        non_positive = [tv for tv, arr in raw_capacities.items() if float(np.max(arr)) <= 0.0]
        if non_positive:
            print(
                f"Regen: capacity rows are non-positive for {len(non_positive)}/{len(raw_capacities)} TVs; sample="
                + ",".join([str(x) for x in non_positive[:5]])
            )

    # Normalize: treat missing/zero bins as unconstrained
    capacities_by_tv = normalize_capacities(raw_capacities)

    if capacities_by_tv:
        sample_items = list(capacities_by_tv.items())[:5]
        sample_stats = []
        for tv, arr in sample_items:
            arr_np = np.asarray(arr, dtype=np.float64)
            if arr_np.size == 0:
                sample_stats.append(f"{tv}:empty")
                continue
            sample_stats.append(
                f"{tv}:min={float(arr_np.min()):.1f},max={float(arr_np.max()):.1f}"
            )
        print(
            f"Regen: normalized capacities ready for {len(capacities_by_tv)} TVs; samples: "
            + "; ".join(sample_stats)
        )

    return capacities_by_tv


def _tv_centroids(
    gdf: gpd.GeoDataFrame,
    indexer: TVTWIndexer,
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate the geographic centroid for each traffic volume.

    Args:
        gdf: A GeoDataFrame containing the traffic volume geometries.
        indexer: The TVTWIndexer, used to identify the relevant TV IDs.

    Returns:
        A dictionary mapping each TV ID to its (latitude, longitude) centroid.
    """
    try:
        # Ensure the GeoDataFrame is in the standard WGS 84 coordinate system.
        geo = gdf.to_crs(epsg=4326) if gdf.crs and "4326" not in str(gdf.crs) else gdf
    except Exception:
        geo = gdf
    tv_ids = set(str(tv) for tv in indexer.tv_id_to_idx.keys())
    centroids: Dict[str, Tuple[float, float]] = {}
    for _, row in geo.iterrows():
        tv_id = row.get("traffic_volume_id")
        if tv_id is None:
            continue
        tv_key = str(tv_id)
        if tv_key not in tv_ids:
            continue
        geom = row.get("geometry")
        if geom is None or geom.is_empty:
            continue
        try:
            c = geom.centroid
            centroids[tv_key] = (float(c.y), float(c.x))
        except Exception:
            continue
    return centroids


def _travel_minutes_from_centroids(
    centroids: Mapping[str, Tuple[float, float]],
    *,
    speed_kts: float = 475.0,
) -> Dict[str, Dict[str, float]]:
    """
    Estimates the travel time in minutes between all pairs of TV centroids.

    This uses the haversine formula to calculate the great-circle distance
    between centroids and then converts this distance to an estimated travel
    time based on a constant assumed aircraft speed. This information is used
    by the regen engine to understand the spatial relationships between flows.

    Args:
        centroids: A mapping from TV ID to its (lat, lon) centroid.
        speed_kts: Assumed aircraft ground speed in knots.

    Returns:
        A nested dictionary representing a travel time matrix:
        `{source_tv_id: {dest_tv_id: travel_minutes}}`.
    """
    if not centroids:
        return {}
    ids = sorted(centroids.keys())
    lat_arr = np.asarray([centroids[i][0] for i in ids], dtype=np.float64)
    lon_arr = np.asarray([centroids[i][1] for i in ids], dtype=np.float64)
    # Compute all-pairs distance matrix efficiently with vectorized haversine.
    dist_nm = haversine_vectorized(lat_arr[:, None], lon_arr[:, None], lat_arr[None, :], lon_arr[None, :])
    minutes = (dist_nm / float(speed_kts)) * 60.0
    out: Dict[str, Dict[str, float]] = {}
    for i, src in enumerate(ids):
        out[src] = {ids[j]: float(minutes[i, j]) for j in range(len(ids))}
    return out


def _lookup_flow_payload(flows_payload: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    """Creates a fast lookup map from flow ID to its payload dictionary."""
    lookup: Dict[int, Mapping[str, Any]] = {}
    for flow in flows_payload.get("flows", []) or []:
        try:
            fid = int(flow.get("flow_id"))
        except Exception:
            continue
        lookup[fid] = flow
    return lookup


def _build_flow_summary(
    proposal,
    *,
    hotspot_payload: Mapping[str, Any],
    flows_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """
    Constructs a detailed summary for each flow within a regulation proposal.

    This function enriches the flow information from the proposal object with
    contextual data from the original hotspot and flow payloads, such as the
    number of flights and the demand (entrants) during the hotspot window. This
    is used for creating the final human-readable output table.

    Args:
        proposal: The proposal object from the regen engine.
        hotspot_payload: The input hotspot descriptor payload.
        flows_payload: The raw payload from the `compute_flows` API.

    Returns:
        A list of dictionaries, where each dictionary summarizes one flow
        affected by the proposed regulation.
    """
    window_bins = list(hotspot_payload.get("window_bins", []))
    timebins_h = _timebins_from_window(window_bins)
    start = timebins_h[0] if timebins_h else 0
    end_exclusive = timebins_h[-1] + 1 if timebins_h else 0
    meta = hotspot_payload.get("metadata", {}) or {}
    flow_to_flights = meta.get("flow_to_flights", {}) or {}
    proxies = meta.get("flow_proxies", {}) or {}
    payload_lookup = _lookup_flow_payload(flows_payload)
    summary: List[Dict[str, Any]] = []
    for flow in proposal.flows_info:
        fid = int(flow.get("flow_id"))
        fid_key = str(fid)
        payload_entry = payload_lookup.get(fid, {})
        demand = payload_entry.get("demand") or []
        entrants = 0.0
        # Calculate total entrants (demand) for the flow during the hotspot window.
        if demand and end_exclusive > start:
            entrants = float(sum(float(demand[t]) for t in range(start, min(end_exclusive, len(demand)))))
        elif proxies.get(fid_key):
            entrants = float(sum(float(x) for x in proxies[fid_key]))
        flights = flow_to_flights.get(fid_key) or []
        summary.append(
            {
                "flow_id": fid,
                "control_tv_id": flow.get("control_tv_id"),
                "baseline_rate_per_hour": float(flow.get("r0_i", 0.0)),
                "allowed_rate_per_hour": float(flow.get("R_i", 0.0)),
                "assigned_cut_per_hour": float(flow.get("lambda_cut_i", 0.0)),
                "entrants_in_window": entrants,
                "num_flights": int(flow.get("num_flights", len(flights))),
            }
        )
    summary.sort(key=lambda item: item["flow_id"])
    return summary


def _proposal_to_dict(
    proposal,
    *,
    hotspot_payload: Mapping[str, Any],
    flows_payload: Mapping[str, Any],
    exceedance_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Serializes a proposal object into a JSON-friendly dictionary.

    This function converts the internal proposal representation into a clean
    dictionary structure, suitable for printing, saving to a file, or sending
    over an API. It includes the flow summary, diagnostic information, and the
    predicted performance improvement.

    Args:
        proposal: The proposal object from the regen engine.
        hotspot_payload: The input hotspot descriptor payload.
        flows_payload: The raw payload from the `compute_flows` API.
        exceedance_stats: A dict with stats about the baseline hotspot exceedance.

    Returns:
        A dictionary representation of the proposal.
    """
    flows = _build_flow_summary(proposal, hotspot_payload=hotspot_payload, flows_payload=flows_payload)
    diag = dict(proposal.diagnostics)
    diag.setdefault("D_peak", float(exceedance_stats.get("D_peak", 0.0)))
    diag.setdefault("D_sum", float(exceedance_stats.get("D_total", 0.0)))
    diag.setdefault("window_bins_input", list(hotspot_payload.get("window_bins", [])))
    result = {
        "hotspot_id": proposal.hotspot_id,
        "control_volume_id": proposal.controlled_volume,
        "window_bins": [int(proposal.window.start_bin), int(proposal.window.end_bin)],
        "flows": flows,
        "predicted_improvement": {
            "delta_deficit_per_hour": float(proposal.predicted_improvement.delta_deficit_per_hour),
            "delta_objective_score": float(proposal.predicted_improvement.delta_objective_score),
        },
        "diagnostics": diag,
    }
    return result


def propose_regulations(
    *,
    hotspot_payload: Dict[str, Any],
    evaluator: NetworkEvaluator,
    indexer: TVTWIndexer,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Main entry point for generating regulation proposals for a given hotspot.

    This function orchestrates the entire regulation proposal process:
    1.  It prepares all necessary inputs: capacities, TV centroids, and travel
        times.
    2.  It calls the `compute_flows` API to identify the traffic flows passing
        through the hotspot TV during the specified time window.
    3.  It invokes the core `propose_regulations_for_hotspot` engine with all
        the prepared data to generate a list of candidate regulation proposals.
    4.  It computes baseline exceedance statistics for comparison.
    5.  Finally, it formats the raw proposals into dictionaries and returns the
        top `k` results.

    Args:
        hotspot_payload: A dictionary describing the hotspot, including the
            control TV ID and time window.
        evaluator: The configured NetworkEvaluator.
        indexer: The configured TVTWIndexer.
        top_k: If specified, returns only the top `k` proposals.

    Returns:
        A list of regulation proposals, each formatted as a dictionary.
    """
    control_tv = str(hotspot_payload.get("control_volume_id"))
    if not control_tv:
        raise ValueError("hotspot payload missing control volume id")
    window_bins = hotspot_payload.get("window_bins") or []
    timebins_h = _timebins_from_window(window_bins)
    if not timebins_h:
        raise ValueError("hotspot payload missing time window bins")

    flight_list = getattr(evaluator, "flight_list", None)
    if flight_list is None:
        raise ValueError("network evaluator missing flight_list reference")

    # Prepare shared inputs required by the regen engine.
    capacities_by_tv = _build_capacities_by_tv(evaluator, indexer)
    res_obj = getattr(evaluator, "_resources", None)
    if res_obj is not None:
        try:
            centroids = res_obj.tv_centroids
        except Exception:
            centroids = _tv_centroids(evaluator.traffic_volumes_gdf, indexer)
        try:
            travel_minutes_map = res_obj.travel_minutes()
        except Exception:
            travel_minutes_map = _travel_minutes_from_centroids(centroids)
    else:
        centroids = _tv_centroids(evaluator.traffic_volumes_gdf, indexer)
        travel_minutes_map = _travel_minutes_from_centroids(centroids)

    # Flow payload via API (reuse in-memory artifacts)
    # This step identifies the distinct flows of traffic that contribute to the
    # congestion in the hotspot control volume during the time window.
    set_global_resources(indexer, flight_list)
    direction_opts = {"mode": "coord_cosine", "tv_centroids": centroids} if centroids else {"mode": "coord_cosine"}
    flows_payload = compute_flows(tvs=[control_tv], timebins=timebins_h, direction_opts=direction_opts, threshold=leiden_params["threshold"], resolution=leiden_params["resolution"])

    # Flow-to-flights metadata from hotspot discovery (string keys)
    meta = hotspot_payload.get("metadata", {}) or {}
    flow_to_flights_in = meta.get("flow_to_flights", {}) or {}
    flow_to_flights: Dict[str, Sequence[str]] = {
        str(k): tuple(v or []) for k, v in flow_to_flights_in.items()
    }

    proposals: List[Dict[str, Any]] = []
    try:
        # Configure the regen engine. Using a fallback config here with relaxed
        # constraints to ensure it returns a result even for edge cases, which
        # is useful for debugging.
        fallback_cfg = RegenConfig(
            g_min=-float("inf"),
            rho_max=float("inf"),
            slack_min=-float("inf"),
            distinct_controls_required=False,
            raise_on_edge_cases=True,
            min_num_flights=4
        )


        
        # This is the core call to the regulation proposal engine.
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

    except ValueError as exc:
        print(f"[regen] Primary proposal attempt failed: {exc}")
    if not proposals:
        raise RuntimeError("regen engine returned no proposals even with relaxed config")

    # Compute the baseline exceedance (congestion) statistics for the hotspot.
    # This is used for context and to calculate the target for regulation.
    exceedance_stats = compute_hotspot_exceedance(
        indexer=indexer,
        flight_list=flight_list,
        capacities_by_tv=capacities_by_tv,
        hotspot_tv=control_tv,
        timebins_h=timebins_h,
    )
    proposals_to_emit: List[Dict[str, Any]] = []
    # Format each raw proposal into a serializable dictionary.
    for idx, proposal in enumerate(proposals):
        if top_k is not None and idx >= int(top_k):
            break
        proposals_to_emit.append(
            _proposal_to_dict(
                proposal,
                hotspot_payload=hotspot_payload,
                flows_payload=flows_payload,
                exceedance_stats=exceedance_stats,
            )
        )

    if not proposals_to_emit:
        raise RuntimeError("Requested top_k resulted in zero proposals being returned")

    return proposals_to_emit


def propose_top_regulations(
    *,
    hotspot_payload: Dict[str, Any],
    evaluator: NetworkEvaluator,
    indexer: TVTWIndexer,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    A convenience wrapper around `propose_regulations` to get the single best
    proposal.
    """
    return propose_regulations(
        hotspot_payload=hotspot_payload,
        evaluator=evaluator,
        indexer=indexer,
        top_k=top_k,
    )[0]


def main() -> None:
    """
    Main execution function for the test bench.
    """
    print("[regen] Resolving artifacts ...")
    occ_path, idx_path, caps_path = load_artifacts()
    print(f"[regen] occupancy: {occ_path}")
    print(f"[regen] indexer:   {idx_path}")
    print(f"[regen] capacities:{caps_path}")

    print("[regen] Loading data ...")
    flight_list, indexer, evaluator = build_data(occ_path, idx_path, caps_path)

    # Manually define the hotspot to be analyzed. In a real application, this
    # would come from an automated hotspot detection system.
    hotspot_payload = {
        "control_volume_id": "LFBZX15",
        "window_bins": [45, 49], # means [45, 46, 47, 48]
        "metadata": {}, # will be filled in below
        "mode": "manual",
    }

    # Build metadata in the same schema as HotspotInventory descriptors.
    # This metadata, particularly `flow_to_flights`, provides the ground truth
    # for which flights belong to which flow, which is essential for the regen
    # engine to correctly model the impact of regulations.
    # - flow_to_flights: { flow_id(str): [flight_id(str), ...] }
    # - flow_proxies: { flow_id(str): [entrants per bin within window] }
    control_tv = str(hotspot_payload["control_volume_id"])
    timebins_h = _timebins_from_window(hotspot_payload.get("window_bins", []))
    # First, compute the flows for the given hotspot TV and time window.
    set_global_resources(indexer, flight_list)
    res_obj = getattr(evaluator, "_resources", None)
    if res_obj is not None:
        try:
            centroids = res_obj.tv_centroids
        except Exception:
            centroids = _tv_centroids(evaluator.traffic_volumes_gdf, indexer)
    else:
        centroids = _tv_centroids(evaluator.traffic_volumes_gdf, indexer)
    direction_opts = {"mode": "coord_cosine", "tv_centroids": centroids} if centroids else {"mode": "coord_cosine"}
    flows_payload = compute_flows(
        tvs=[control_tv],
        timebins=timebins_h,
        direction_opts=direction_opts,
        threshold=leiden_params["threshold"],
        resolution=leiden_params["resolution"],
    )

    # Now, iterate through the computed flows to extract the required metadata.
    flow_to_flights: Dict[str, List[str]] = {}
    flow_proxies: Dict[str, List[float]] = {}
    for flow in flows_payload.get("flows", []) or []:
        # Normalize flow id to string
        try:
            fid_key = str(int(flow.get("flow_id")))
        except Exception:
            fid_key = str(flow.get("flow_id"))

        # Collect all flight IDs belonging to this flow.
        flights: List[str] = []
        for spec in flow.get("flights", []) or []:
            fid = spec.get("flight_id")
            if fid is not None:
                flights.append(str(fid))
        flow_to_flights[fid_key] = flights

        # Calculate demand proxies: entrants per bin within the window [t0, t1).
        # This gives a time-series view of how demand for the flow is distributed.
        demand = flow.get("demand") or []
        proxy: List[float] = []
        for b in timebins_h:
            try:
                proxy.append(float(demand[int(b)]))
            except Exception:
                proxy.append(0.0)
        flow_proxies[fid_key] = proxy

    hotspot_payload["metadata"] = {
        "flow_to_flights": flow_to_flights,
        "flow_proxies": flow_proxies,
    }
    print(
        "[regen] Using manual hotspot selection: "
        f"TV={hotspot_payload['control_volume_id']} window={hotspot_payload['window_bins']}"
    )

    print("[regen] Proposing regulations (per-flow) ...")
    # This is the main call to the proposal generation logic.
    proposals = propose_regulations(
        hotspot_payload=hotspot_payload,
        evaluator=evaluator,
        indexer=indexer,
    )
    print(f"[regen] Retrieved {len(proposals)} regulation candidate(s).")

    # Use the `rich` library to print a well-formatted summary of the proposals.
    # This makes the output much easier to review and understand.
    from rich.console import Console
    from rich.table import Table

    console = Console()

    for rank, proposal in enumerate(proposals, start=1):
        # Header info mirrors previous single-proposal view
        diag = proposal["diagnostics"]
        improvement = proposal.get("predicted_improvement", {})
        components_before = diag.get("score_components_before", {}) or {}
        components_after = diag.get("score_components_after", {}) or {}
        console.print(f"\n[bold green][regen] Proposal Summary #{rank}[/bold green]")
        console.print(f"Control Volume: [cyan]{proposal['control_volume_id']}[/cyan]")
        console.print(
            f"Window Bins: [cyan]{proposal['window_bins'][0]}-{proposal['window_bins'][1]}[/cyan]"
        )
        console.print(
            f"Target Exceedance to Remove: [yellow]{diag['E_target']:.1f}[/yellow] "
            f"(D_peak={diag['D_peak']:.1f}, D_sum={diag['D_sum']:.1f})"
        )
        delta_obj = float(improvement.get("delta_objective_score", 0.0))
        console.print(f"Predicted Objective Improvement: [yellow]{delta_obj:.3f}[/yellow]")

        # If available, show the breakdown of the objective function score
        # before and after the proposed regulation. This helps in understanding
        # what trade-offs the optimizer made.
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
                comp_table.add_row(
                    key,
                    f"{before_val:.3f}",
                    f"{after_val:.3f}",
                    f"{delta_val:+.3f}",
                )

            console.print(comp_table)

        # Display the specific regulations applied to each flow.
        table = Table(title="Flow Regulations")
        table.add_column("Flow ID", style="cyan", no_wrap=True)
        table.add_column("Control TV", style="magenta")
        table.add_column("Baseline Rate\n(per hour)", justify="right", style="green")
        table.add_column("Allowed Rate\n(per hour)", justify="right", style="yellow")
        table.add_column("Cut\n(per hour)", justify="right", style="red")
        table.add_column("Entrants in\nWindow", justify="right")
        table.add_column("Num\nFlights", justify="right")

        for flow in proposal["flows"]:
            table.add_row(
                str(flow["flow_id"]),
                str(flow["control_tv_id"]),
                f"{flow['baseline_rate_per_hour']:.1f}",
                f"{flow['allowed_rate_per_hour']:.1f}",
                f"{flow['assigned_cut_per_hour']:.0f}",
                f"{flow['entrants_in_window']:.0f}",
                str(flow["num_flights"]),
            )

        console.print(table)

    # Optional: write the full proposal data to a JSON file for later analysis.
    out_dir = REPO_ROOT / "agent_runs" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regen_proposal.json"
    output_payload = {
        "top_hotspot": hotspot_payload,
        "proposals": proposals,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2)
    print(f"[regen] Saved {len(proposals)} proposal(s): {out_path}")


if __name__ == "__main__":
    main()
