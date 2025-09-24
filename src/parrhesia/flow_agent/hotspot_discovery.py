from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

from parrhesia.flows.flow_pipeline import build_global_flows
from parrhesia.optim.sa_optimizer import prepare_flow_scheduling_inputs


@dataclass
class HotspotDiscoveryConfig:
    threshold: float = 0.0
    top_hotspots: int = 10
    top_flows: int = 4
    max_flights_per_flow: int = 20
    min_flights_per_flow: int = 5
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


@dataclass
class _CrossingCache:
    bins_by_flight: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    flights_by_bin: Dict[int, Tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_events(cls, events: Iterable[Tuple[str, int]]) -> "_CrossingCache":
        by_flight: Dict[str, set[int]] = {}
        by_bin: Dict[int, set[str]] = {}
        for fid, bin_idx in events:
            sfid = str(fid)
            b = int(bin_idx)
            by_flight.setdefault(sfid, set()).add(b)
            by_bin.setdefault(b, set()).add(sfid)
        return cls(
            bins_by_flight={k: tuple(sorted(v)) for k, v in by_flight.items()},
            flights_by_bin={k: tuple(sorted(v)) for k, v in by_bin.items()},
        )


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
        self._crossings_cache: Dict[str, _CrossingCache] = {}

    # ------------------------------- Public API ---------------------------
    def build_from_segments(
        self,
        *,
        threshold: float = 0.0,
        top_hotspots: int = 10,
        top_flows: int = 4,
        min_flights_per_flow: int = 5,
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

        # Pre-load crossing caches once for the TVs that survived filtering
        unique_tvs = {str(seg.get("traffic_volume_id")) for seg in segs_ranked}
        self._preload_crossings(unique_tvs)

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
                    min_flights_per_flow=min_flights_per_flow,
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
        min_flights_per_flow: int,
        max_flights_per_flow: int,
        leiden_params: Optional[Mapping[str, float | int]],
        direction_opts: Optional[Mapping[str, Any]],
    ) -> Optional[HotspotDescriptor]:
        t0, t1 = int(window_bins[0]), int(window_bins[1])
        active_bins = list(range(t0, max(t0 + 1, t1)))
        if not active_bins:
            active_bins = [t0]

        cache = self._crossings_cache.get(str(control_volume_id))
        if cache is None:
            return None

        active_set = set(int(b) for b in active_bins)
        union_ids = [fid for fid, bins in cache.bins_by_flight.items() if any(int(b) in active_set for b in bins)]
        union_ids.sort()
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
            flight_ids=union_ids,
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

        # Trim flows by entrants count and cap per-flow flights
        win_len = max(1, int(window_bins[1]) - int(window_bins[0]))
        reverse: Dict[str, str] = {}
        for fid, flights in flows_for_ctrl.items():
            for fl in flights:
                reverse[str(fl)] = str(fid)

        entrants_seed: Dict[str, int] = {}
        for b in active_bins:
            offset = int(b) - int(window_bins[0])
            if not (0 <= offset < win_len):
                continue
            for fl in cache.flights_by_bin.get(int(b), ()):  # type: ignore[arg-type]
                flow = reverse.get(str(fl))
                if flow is None:
                    continue
                entrants_seed[flow] = entrants_seed.get(flow, 0) + 1

        def _flow_sort_key(fid: str) -> Tuple[int, str]:
            entrants = int(entrants_seed.get(str(fid), 0))
            fid_str = str(fid)
            if fid_str.isdigit():
                order = f"{int(fid_str):020d}"
            else:
                order = fid_str
            return (-entrants, order)

        flow_ids_sorted = sorted(flows_for_ctrl.keys(), key=_flow_sort_key)

        min_required = max(0, int(min_flights_per_flow))
        cap_limit = max(1, int(max_flights_per_flow))
        top_limit = max(1, int(top_flows)) if top_flows is not None else None

        selected: Dict[str, Tuple[str, ...]] = {}
        for fid in flow_ids_sorted:
            all_flights = [str(x) for x in flows_for_ctrl[fid]]
            if not all_flights:
                continue
            if min_required > 0 and len(all_flights) < min_required:
                continue
            if top_limit is not None and len(selected) >= top_limit:
                continue
            selected[str(fid)] = tuple(all_flights[:cap_limit])

        all_flights_linear: List[str] = []
        for flights in flows_for_ctrl.values():
            all_flights_linear.extend(str(x) for x in flights)

        if not selected and not all_flights_linear:
            return None

        selected_flights = {fl for flights in selected.values() for fl in flights}
        remaining_flights: List[str] = []
        seen_remaining: set[str] = set()
        for fl in all_flights_linear:
            if fl in selected_flights:
                continue
            if fl in seen_remaining:
                continue
            seen_remaining.add(fl)
            remaining_flights.append(fl)

        if remaining_flights:
            # Keep a catch-all flow for the flights that failed the gating rules.
            remaining_flow_id = "remaining"
            suffix = 0
            while remaining_flow_id in selected:
                suffix += 1
                remaining_flow_id = f"remaining_{suffix}"
            selected[remaining_flow_id] = tuple(remaining_flights)

        if not selected:
            return None

        # Entrants-based proxies (histograms) and severity
        proxies: Dict[str, np.ndarray] = {flow_id: np.zeros(win_len, dtype=float) for flow_id in selected}
        entrants_count: Dict[str, int] = {flow_id: 0 for flow_id in selected}
        flight_to_flow: Dict[str, str] = {}
        for flow_id, flights in selected.items():
            for fl in flights:
                flight_to_flow[str(fl)] = flow_id

        for b in active_bins:
            offset = int(b) - int(window_bins[0])
            if not (0 <= offset < win_len):
                continue
            for fl in cache.flights_by_bin.get(int(b), ()):  # type: ignore[arg-type]
                flow = flight_to_flow.get(str(fl))
                if flow is None:
                    continue
                proxies[flow][offset] += 1.0
                entrants_count[flow] = entrants_count.get(flow, 0) + 1

        # Severity prior: total entrants in window (normalized later by MCTS priors)
        severity = float(sum(int(entrants_count.get(f, 0)) for f in selected))

        meta: Dict[str, Any] = {
            "flow_to_flights": {k: list(v) for k, v in selected.items()},
            "flow_proxies": {k: proxies.get(k, np.zeros(win_len, dtype=float)).tolist() for k in selected.keys()},
        }
        def _candidate_sort_key(flow_id: str) -> Tuple[int, int | str]:
            fid_str = str(flow_id)
            if fid_str.isdigit():
                return (0, int(fid_str))
            return (1, fid_str)

        desc = HotspotDescriptor(
            control_volume_id=str(control_volume_id),
            window_bins=(int(window_bins[0]), int(window_bins[1])),
            candidate_flow_ids=tuple(
                sorted(selected.keys(), key=_candidate_sort_key)
            ),
            hotspot_prior=severity,
            mode="per_flow",
            metadata=meta,
        )
        return desc

    def _preload_crossings(self, tv_ids: Iterable[str]) -> None:
        remaining = [str(tv) for tv in tv_ids if str(tv) not in self._crossings_cache]
        if not remaining:
            return

        events: Dict[str, List[Tuple[str, int]]] = {tv: [] for tv in remaining}
        iter_fn = getattr(self.flight_list, "iter_hotspot_crossings", None)
        if callable(iter_fn):
            for fid, tv, _dt, t in iter_fn(remaining, None):  # type: ignore[misc]
                key = str(tv)
                bucket = events.get(key)
                if bucket is not None:
                    bucket.append((str(fid), int(t)))
        else:
            # Fallback to scanning flight metadata once
            meta_map = getattr(self.flight_list, "flight_metadata", {}) or {}
            idx_obj = getattr(self.flight_list, "indexer", None)
            decode = getattr(idx_obj, "get_tvtw_from_index", None)
            if decode is None:
                bins_per_tv = int(getattr(self.flight_list, "num_time_bins_per_tv"))
                idx_to_tv_id = getattr(self.flight_list, "idx_to_tv_id")

                def _decode(val: int) -> Optional[Tuple[str, int]]:
                    tv_idx = int(val) // int(bins_per_tv)
                    tbin = int(val) % int(bins_per_tv)
                    tv_id = idx_to_tv_id.get(int(tv_idx))
                    if tv_id is None:
                        return None
                    return str(tv_id), int(tbin)

                decode_fn: Callable[[int], Optional[Tuple[str, int]]] = _decode
            else:
                decode_fn = lambda v: decode(int(v))  # type: ignore[misc]

            target = set(remaining)
            for fid, meta in meta_map.items():
                for iv in meta.get("occupancy_intervals", []) or []:
                    try:
                        tvtw_idx = int(iv.get("tvtw_index"))
                    except Exception:
                        continue
                    decoded = decode_fn(tvtw_idx)
                    if not decoded:
                        continue
                    tv_id, tbin = decoded
                    if str(tv_id) not in target:
                        continue
                    bucket = events.get(str(tv_id))
                    if bucket is not None:
                        bucket.append((str(fid), int(tbin)))

        for tv, ev in events.items():
            self._crossings_cache[tv] = _CrossingCache.from_events(ev)

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
