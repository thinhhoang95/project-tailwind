from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

from parrhesia.flows.flow_pipeline import build_global_flows, collect_hotspot_flights
from parrhesia.optim.sa_optimizer import prepare_flow_scheduling_inputs


@dataclass
class HotspotDiscoveryConfig:
    threshold: float = 0.0
    top_hotspots: int = 10
    top_flows: int = 4
    max_flights_per_flow: int = 20
    leiden_params: Optional[Dict[str, float | int]] = None
    direction_opts: Optional[Dict[str, Any]] = None


@dataclass
class HotspotDescriptor:
    control_volume_id: str
    window_bins: Tuple[int, int]
    candidate_flow_ids: Tuple[str, ...]
    hotspot_prior: float
    mode: str = "per_flow"
    metadata: Dict[str, Any] = field(default_factory=dict)


class HotspotInventory:
    """Builds and caches hotspot descriptors from evaluator + flights.

    The inventory exposes minimal payloads suitable for seeding PlanState.metadata
    under key "hotspot_candidates".
    """

    def __init__(
        self,
        evaluator: NetworkEvaluator,
        flight_list: FlightList,
        indexer: TVTWIndexer,
    ) -> None:
        self.evaluator = evaluator
        self.flight_list = flight_list
        self.indexer = indexer
        self._cache: Dict[Tuple[str, Tuple[int, int]], HotspotDescriptor] = {}

    # ------------------------------- Public API ---------------------------
    def build_from_segments(
        self,
        *,
        threshold: float = 0.0,
        top_hotspots: int = 10,
        top_flows: int = 4,
        max_flights_per_flow: int = 20,
        leiden_params: Optional[Mapping[str, float | int]] = None,
        direction_opts: Optional[Mapping[str, Any]] = None,
    ) -> List[HotspotDescriptor]:
        segs = self.evaluator.get_hotspot_segments(threshold=float(threshold))
        if not segs:
            return []

        # Filter to TVs known to indexer
        segs_filt: List[Dict[str, Any]] = []
        for seg in segs:
            tv = str(seg.get("traffic_volume_id"))
            if tv in self.indexer.tv_id_to_idx:
                segs_filt.append(seg)
        if not segs_filt:
            return []

        # Score/severity: prefer segments with higher overload metric if present
        def _seg_score(s: Mapping[str, Any]) -> float:
            v = s.get("severity")
            try:
                return float(v)
            except Exception:
                pass
            v = s.get("overload")
            try:
                return float(v)
            except Exception:
                pass
            return 0.0

        segs_ranked = sorted(segs_filt, key=_seg_score, reverse=True)
        if top_hotspots is not None and top_hotspots > 0:
            segs_ranked = segs_ranked[: int(top_hotspots)]

        out: List[HotspotDescriptor] = []
        for seg in segs_ranked:
            tv_id = str(seg.get("traffic_volume_id"))
            t0 = int(seg.get("start_bin", 0))
            t1_incl = int(seg.get("end_bin", t0))
            if t1_incl < t0:
                t1_incl = t0
            # Use half-open [t0, t1)
            window_bins = (t0, t1_incl + 1)

            desc = self._cache.get((tv_id, window_bins))
            if desc is None:
                desc = self._build_descriptor(
                    control_volume_id=tv_id,
                    window_bins=window_bins,
                    top_flows=top_flows,
                    max_flights_per_flow=max_flights_per_flow,
                    leiden_params=leiden_params,
                    direction_opts=direction_opts,
                )
                if desc is None:
                    continue
                self._cache[(tv_id, window_bins)] = desc
            out.append(desc)

        # Stable order by prior then tv_id,t0
        out.sort(key=lambda d: (-float(d.hotspot_prior), d.control_volume_id, int(d.window_bins[0])))
        return out

    def get(self, control_volume_id: str, window_bins: Tuple[int, int]) -> Optional[HotspotDescriptor]:
        return self._cache.get((str(control_volume_id), (int(window_bins[0]), int(window_bins[1]))))

    # ------------------------------ Internals -----------------------------
    def _build_descriptor(
        self,
        *,
        control_volume_id: str,
        window_bins: Tuple[int, int],
        top_flows: int,
        max_flights_per_flow: int,
        leiden_params: Optional[Mapping[str, float | int]],
        direction_opts: Optional[Mapping[str, Any]],
    ) -> Optional[HotspotDescriptor]:
        t0, t1 = int(window_bins[0]), int(window_bins[1])
        active_bins = list(range(t0, max(t0 + 1, t1)))
        if not active_bins:
            active_bins = [t0]

        # Flights touching (tv, window)
        union_ids, _meta = collect_hotspot_flights(
            self.flight_list, [control_volume_id], active_windows={control_volume_id: active_bins}
        )
        if not union_ids:
            return None

        # Cluster to global flows; then map to controlled volume
        flow_map = build_global_flows(
            self.flight_list,
            union_ids,
            hotspots=[control_volume_id],
            trim_policy="earliest_hotspot",
            leiden_params=dict(leiden_params or {}),
            direction_opts=dict(direction_opts or {}),
        )
        if not flow_map:
            return None

        flights_by_flow, ctrl_by_flow = prepare_flow_scheduling_inputs(
            flight_list=self.flight_list,
            flow_map=flow_map,
            hotspot_ids=[control_volume_id],
        )
        flows_for_ctrl: Dict[str, List[str]] = {}
        for flow_id, ctrl in ctrl_by_flow.items():
            if str(ctrl) != str(control_volume_id):
                continue
            specs = flights_by_flow.get(flow_id) or []
            if not specs:
                continue
            fids = [spec.get("flight_id") for spec in specs if spec and spec.get("flight_id")]
            if fids:
                flows_for_ctrl[str(flow_id)] = [str(x) for x in fids]
        if not flows_for_ctrl:
            return None

        # Entrants-based proxies (histograms) and severity
        win_len = max(1, int(window_bins[1]) - int(window_bins[0]))
        proxies: Dict[str, np.ndarray] = {}
        entrants_count: Dict[str, int] = {}
        total = 0
        # Derive entrants by scanning FlightList crossings over the window; cheap and deterministic
        reverse: Dict[str, str] = {}
        for fid, flights in flows_for_ctrl.items():
            for fl in flights:
                reverse[str(fl)] = str(fid)
        iter_fn = getattr(self.flight_list, "iter_hotspot_crossings", None)
        if callable(iter_fn):
            for fl_id, tv, _dt, t in iter_fn([control_volume_id], active_windows={control_volume_id: active_bins}):  # type: ignore[misc]
                flow = reverse.get(str(fl_id))
                if flow is None:
                    continue
                tt = int(t) - int(window_bins[0])
                if 0 <= tt < win_len:
                    proxies.setdefault(flow, np.zeros(win_len, dtype=float))[tt] += 1.0
                    entrants_count[flow] = entrants_count.get(flow, 0) + 1
                    total += 1

        # Trim flows by entrants count and cap per-flow flights
        flow_ids_sorted = sorted(flows_for_ctrl.keys(), key=lambda f: (-int(entrants_count.get(f, 0)), int(f)))
        trimmed: Dict[str, Tuple[str, ...]] = {}
        for f in flow_ids_sorted:
            flights = flows_for_ctrl[f][: int(max(1, max_flights_per_flow))]
            if flights:
                trimmed[str(f)] = tuple(flights)
            if len(trimmed) >= int(max(1, top_flows)):
                break
        if not trimmed:
            return None

        # Severity prior: total entrants in window (normalized later by MCTS priors)
        severity = float(sum(int(entrants_count.get(f, 0)) for f in trimmed))

        meta: Dict[str, Any] = {
            "flow_to_flights": {k: list(v) for k, v in trimmed.items()},
            "flow_proxies": {k: proxies.get(k, np.zeros(win_len, dtype=float)).tolist() for k in trimmed.keys()},
        }
        desc = HotspotDescriptor(
            control_volume_id=str(control_volume_id),
            window_bins=(int(window_bins[0]), int(window_bins[1])),
            candidate_flow_ids=tuple(sorted(trimmed.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))),
            hotspot_prior=severity,
            mode="per_flow",
            metadata=meta,
        )
        return desc

    # ------------------------------ Serialization -------------------------
    @staticmethod
    def to_candidate_payloads(descriptors: Iterable[HotspotDescriptor]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for d in descriptors:
            items.append(
                {
                    "control_volume_id": d.control_volume_id,
                    "window_bins": list(d.window_bins),
                    "candidate_flow_ids": list(d.candidate_flow_ids),
                    "mode": d.mode,
                    "metadata": dict(d.metadata),
                    "hotspot_prior": float(d.hotspot_prior),
                }
            )
        return items


__all__ = [
    "HotspotDiscoveryConfig",
    "HotspotDescriptor",
    "HotspotInventory",
]

