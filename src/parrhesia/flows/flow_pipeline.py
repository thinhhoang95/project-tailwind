from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
from datetime import datetime

from .flow_extractor import (
    compute_jaccard_similarity,
    run_leiden_from_similarity,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from project_tailwind.optimize.eval.flight_list import FlightList


def collect_hotspot_flights(
    flight_list: FlightList,
    hotspot_ids: Sequence[str],
    active_windows: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None,
) -> Tuple[List[str], Dict[str, Dict[str, object]]]:
    """
    Build the union of flights that cross any hotspot within active windows and
    capture earliest-crossing metadata per flight.

    Parameters
    ----------
    flight_list : FlightList
        Provider for flight occupancy and TVTW decoding helpers.
    hotspot_ids : Sequence[str]
        Traffic volume identifiers to consider as hotspots.
    active_windows : Optional[Sequence[int] | Dict[str, Sequence[int]]]
        Active time-bin windows to filter crossings. If a sequence is given, it is
        applied globally to all hotspots. If a dict is given, it maps hotspot_id to
        its allowed windows.

    Returns
    -------
    Tuple[List[str], Dict[str, Dict[str, object]]]
        - Sorted list of unique flight_ids that cross any (hotspot, window)
        - Metadata by flight_id with keys:
          - "first_crossings": Dict[hotspot_id, {"time_idx": int, "entry_dt": datetime}]
            Contains the earliest crossing for each hotspot that this flight touches.
          - "first_global": {"hotspot_id": str, "time_idx": int, "entry_dt": datetime}
            Contains the single earliest crossing across ALL hotspots for this flight.
            This identifies which hotspot the flight encounters first and is used for
            the earliest-median policy when selecting controlled volumes per flow.

    Notes
    -----
    - Crossings are sourced via `FlightList.iter_hotspot_crossings(...)`, which
      decodes TVTW indices and filters by `active_windows`.
    - If a flight crosses the same hotspot multiple times in the allowed windows,
      only the earliest is retained.
    Examples
    --------
    Minimal example with a concrete FlightList and TVTWIndexer:

    >>> from datetime import datetime
    >>> from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
    >>> from project_tailwind.optimize.eval.flight_list import FlightList
    >>> idx = TVTWIndexer(time_bin_minutes=30)
    >>> # Prepare two TVs and populate TVTW mappings
    >>> idx._tv_id_to_idx = {"TV1": 0, "TV2": 1}; idx._idx_to_tv_id = {0: "TV1", 1: "TV2"}
    >>> idx._populate_tvtw_mappings()
    >>> fl = FlightList(idx)
    >>> fl.flight_metadata = {
    ...   "F1": {
    ...     "takeoff_time": datetime(2025, 1, 1, 9, 0, 0),
    ...     "occupancy_intervals": [
    ...       {"tvtw_index": idx.get_tvtw_index("TV1", 18), "entry_time_s": 0, "exit_time_s": 60},
    ...       {"tvtw_index": idx.get_tvtw_index("TV2", 19), "entry_time_s": 120, "exit_time_s": 160},
    ...     ],
    ...   },
    ...   "F2": {
    ...     "takeoff_time": datetime(2025, 1, 1, 9, 15, 0),
    ...     "occupancy_intervals": [
    ...       {"tvtw_index": idx.get_tvtw_index("TV2", 19), "entry_time_s": 30, "exit_time_s": 90},
    ...     ],
    ...   },
    ... }
    >>> union_ids, meta = collect_hotspot_flights(fl, ["TV2"], active_windows=[19])
    >>> set(union_ids) == {"F1", "F2"}
    True
    >>> isinstance(meta["F1"]["first_crossings"]["TV2"]["time_idx"], int)
    True
    >>> meta["F1"]["first_global"]["hotspot_id"] in {"TV1", "TV2"}
    True
    """
    first_by_flight: Dict[str, Dict[str, Tuple[datetime, int]]] = {}
    union_ids: set[str] = set()

    # Prefer native helper when available; else derive from metadata
    iter_fn = getattr(flight_list, "iter_hotspot_crossings", None)
    if callable(iter_fn):
        for fid, h, entry_dt, time_idx in iter_fn(hotspot_ids, active_windows):  # type: ignore[misc]
            union_ids.add(str(fid))
            per_hotspot = first_by_flight.setdefault(str(fid), {})
            prev = per_hotspot.get(str(h))
            if prev is None or entry_dt < prev[0]:
                per_hotspot[str(h)] = (entry_dt, int(time_idx))
    else:
        # Decode tvtw_index -> (tv_id, time_idx)
        # Strategy: use indexer when present; else derive from tv_id_to_idx and bins_per_tv
        idx = getattr(flight_list, "indexer", None)
        decode = None
        if idx is not None and hasattr(idx, "get_tvtw_from_index"):
            decode = lambda j: idx.get_tvtw_from_index(int(j))
            bins_per_tv = None
            idx_to_tv_id = None
        else:
            bins_per_tv = int(getattr(flight_list, "num_time_bins_per_tv"))
            idx_to_tv_id = getattr(flight_list, "idx_to_tv_id")
            def decode(j: int):
                tv_idx = int(j) // int(bins_per_tv)
                tbin = int(j) % int(bins_per_tv)
                tv_id = str(idx_to_tv_id[int(tv_idx)])
                return tv_id, tbin

        # Normalize windows filter
        global_windows: Optional[set[int]] = None
        per_hot: Optional[Dict[str, set[int]]] = None
        if active_windows is None:
            pass
        elif isinstance(active_windows, dict):
            per_hot = {str(k): set(int(x) for x in v) for k, v in active_windows.items()}
        else:
            global_windows = set(int(x) for x in active_windows)

        hotspot_set = set(str(h) for h in hotspot_ids)
        meta_map = getattr(flight_list, "flight_metadata", {}) or {}
        for fid, meta in meta_map.items():
            takeoff = meta.get("takeoff_time")
            if not isinstance(takeoff, datetime):
                continue
            per_hotspot: Dict[str, Tuple[datetime, int]] = {}
            for iv in meta.get("occupancy_intervals", []) or []:
                try:
                    tvtw_idx = int(iv.get("tvtw_index"))
                except Exception:
                    continue
                decoded = decode(tvtw_idx)
                if not decoded:
                    continue
                tv_id, tbin = decoded
                tv_id = str(tv_id)
                if tv_id not in hotspot_set:
                    continue
                # Window filter
                allowed = True
                if per_hot is not None:
                    allowed_set = per_hot.get(tv_id)
                    allowed = allowed_set is not None and int(tbin) in allowed_set
                elif global_windows is not None:
                    allowed = int(tbin) in global_windows
                if not allowed:
                    continue
                # Earliest crossing by entry_time_s
                entry_s = iv.get("entry_time_s", 0)
                try:
                    entry_s = float(entry_s)
                except Exception:
                    entry_s = 0.0
                entry_dt = takeoff + (datetime.min - datetime.min)  # no-op for type hints
                entry_dt = takeoff + (entry_dt - entry_dt)  # preserve type
                entry_dt = takeoff + (datetime(1970,1,1) - datetime(1970,1,1))  # no-op
                # Proper timedelta construction
                from datetime import timedelta as _td
                entry_dt = takeoff + _td(seconds=float(entry_s))
                prev = per_hotspot.get(tv_id)
                if prev is None or entry_dt < prev[0]:
                    per_hotspot[tv_id] = (entry_dt, int(tbin))
            if per_hotspot:
                union_ids.add(str(fid))
                first_by_flight[str(fid)] = per_hotspot

    # Build metadata payload
    meta: Dict[str, Dict[str, object]] = {}
    for fid, per_hotspot in first_by_flight.items():
        # Compute global earliest across hotspots for this flight
        gh: Optional[str] = None
        gdt: Optional[datetime] = None
        gt: Optional[int] = None
        for h, (dt, t) in per_hotspot.items():
            if gdt is None or dt < gdt:
                gh, gdt, gt = h, dt, t
        # Per-hotspot structured dict
        per_hotspot_struct: Dict[str, Dict[str, object]] = {
            h: {"entry_dt": dt, "time_idx": int(t)} for h, (dt, t) in per_hotspot.items()
        }
        first_global_struct = (
            {"hotspot_id": gh, "entry_dt": gdt, "time_idx": int(gt)} if gh is not None else {}
        )
        meta[fid] = {
            "first_crossings": per_hotspot_struct,
            "first_global": first_global_struct,
        }

    return sorted(union_ids), meta


def build_global_flows(
    flight_list: FlightList,
    union_flight_ids: Sequence[str],
    hotspots: Optional[Sequence[Union[str, int]]] = None,
    trim_policy: str = "earliest_hotspot",
    leiden_params: Optional[Dict[str, Union[float, int]]] = None,
    direction_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """
    Partition union flights into flows using Jaccard + Leiden on TV footprints.

    Footprints are constructed via `FlightList.get_footprints_for_flights` with the
    provided `trim_policy`. For the default policy "earliest_hotspot", the given
    `hotspots` are forwarded to enable trimming up to and including the earliest
    crossing of any hotspot in the set.

    Parameters
    ----------
    flight_list : FlightList
        Provider for footprints and metadata.
    union_flight_ids : Sequence[str]
        De-duplicated set of flights to cluster.
    hotspots : Optional[Sequence[str | int]]
        IDs or indices of hotspots; used only when `trim_policy` requires it.
    trim_policy : str, default="earliest_hotspot"
        Policy for footprint trimming ("none" or "earliest_hotspot").
    leiden_params : Optional[Dict[str, Union[float, int]]]
        Parameters for the Leiden stage: keys "threshold" (float in [0,1]),
        "resolution" (float), and "seed" (int).

    Returns
    -------
    Dict[str, int]
        Mapping from flight_id to flow_id (0..k-1), with stable relabeling in
        order of first appearance.
    
    Examples
    --------
    Using a lightweight stub that supplies footprints directly:

    >>> import numpy as np
    >>> class _Stub:
    ...     def get_footprints_for_flights(self, ids, trim_policy="none", hotspots=None):
    ...         fps = {
    ...             "F1": np.array([1, 2, 3]),
    ...             "F2": np.array([2, 3, 4]),
    ...             "F3": np.array([10, 11]),
    ...             "F4": np.array([10, 12]),
    ...         }
    ...         return [fps[i] for i in ids]
    >>> fl_stub = _Stub()
    >>> ids = ["F1", "F2", "F3", "F4"]
    >>> flows = build_global_flows(fl_stub, ids, trim_policy="none", leiden_params={"threshold": 0.1, "resolution": 1.0, "seed": 0})
    >>> flows["F1"] == flows["F2"], flows["F3"] == flows["F4"], flows["F1"] != flows["F3"]
    (True, True, True)
    """

    print(f"build_global_flows: {flight_list}, {union_flight_ids}, {hotspots}, {trim_policy}, {leiden_params}, {direction_opts}")

    ids = list(union_flight_ids)
    if len(ids) == 0:
        return {}
    if len(ids) == 1:
        return {ids[0]: 0}

    # Default Leiden parameters
    threshold = float((leiden_params or {}).get("threshold", 0.1))
    resolution = float((leiden_params or {}).get("resolution", 1.0))
    seed_val = (leiden_params or {}).get("seed", None)
    seed: Optional[int]
    try:
        seed = int(seed_val) if seed_val is not None else None
    except Exception:
        seed = None

    # Helpers for direction-aware reweighting
    # Map hotspots to TV indices for trimming when computing direction vectors
    hotspot_tv_indices_for_dir: Optional[set[int]] = None
    if hotspots is not None:
        hotspot_tv_indices_for_dir = set()
        tv_map = getattr(flight_list, "tv_id_to_idx", {}) or {}
        for h in hotspots:
            if isinstance(h, str):
                idx_h = tv_map.get(h)
                if idx_h is not None:
                    hotspot_tv_indices_for_dir.add(int(idx_h))
            else:
                try:
                    hotspot_tv_indices_for_dir.add(int(h))
                except Exception:
                    pass
        if not hotspot_tv_indices_for_dir:
            hotspot_tv_indices_for_dir = None

    def _seq_indices_for_dir(fid: str) -> List[int]:
        """Chronological TV index sequence for a flight, deduping consecutive repeats."""
        if hasattr(flight_list, "get_flight_tv_sequence_indices"):
            try:
                arr = flight_list.get_flight_tv_sequence_indices(fid)  # type: ignore[misc]
                return list(map(int, getattr(arr, "tolist", lambda: list(arr))()))
            except Exception:
                pass
        meta = getattr(flight_list, "flight_metadata", {}).get(fid)
        if not meta:
            return []
        intervals = meta.get("occupancy_intervals", []) or []
        if not intervals:
            return []
        try:
            order = sorted(range(len(intervals)), key=lambda i: float(intervals[i].get("entry_time_s", 0.0)))
        except Exception:
            order = list(range(len(intervals)))
        bins_per_tv = int(getattr(flight_list, "num_time_bins_per_tv"))
        tv_indices = [(int(intervals[i].get("tvtw_index", 0)) // bins_per_tv) for i in order]
        out: List[int] = []
        for v in tv_indices:
            vv = int(v)
            if not out or out[-1] != vv:
                out.append(vv)
        return out

    # Build footprints according to trimming policy
    try:
        fps = flight_list.get_footprints_for_flights(
            ids, trim_policy=trim_policy, hotspots=hotspots
        )
    except Exception:
        # Generic fallback compatible with project_tailwind FlightList
        # Build footprints by deriving TV index sequences and optional trimming
        # Map hotspots to TV indices when provided
        hotspot_tv_indices: Optional[set[int]] = None
        if hotspots is not None:
            hotspot_tv_indices = set()
            tv_map = getattr(flight_list, "tv_id_to_idx", {}) or {}
            for h in hotspots:
                if isinstance(h, str):
                    idx_h = tv_map.get(h)
                    if idx_h is not None:
                        hotspot_tv_indices.add(int(idx_h))
                else:
                    try:
                        hotspot_tv_indices.add(int(h))
                    except Exception:
                        pass
            if not hotspot_tv_indices:
                hotspot_tv_indices = None

        def _seq_indices(fid: str) -> List[int]:
            # Prefer native helper if available
            if hasattr(flight_list, "get_flight_tv_sequence_indices"):
                arr = flight_list.get_flight_tv_sequence_indices(fid)  # type: ignore[misc]
                return list(map(int, arr.tolist()))
            # Derive from metadata
            meta = getattr(flight_list, "flight_metadata", {}).get(fid)
            if not meta:
                return []
            intervals = meta.get("occupancy_intervals", []) or []
            if not intervals:
                return []
            try:
                # Sort by entry_time_s
                order = sorted(range(len(intervals)), key=lambda i: float(intervals[i].get("entry_time_s", 0.0)))
            except Exception:
                order = list(range(len(intervals)))
            bins_per_tv = int(getattr(flight_list, "num_time_bins_per_tv"))
            tv_indices = [(int(intervals[i].get("tvtw_index", 0)) // bins_per_tv) for i in order]
            # compress consecutive duplicates
            out: List[int] = []
            for v in tv_indices:
                if not out or out[-1] != int(v):
                    out.append(int(v))
            return out

        fps = []
        for fid in ids:
            seq = _seq_indices(str(fid))
            if isinstance(trim_policy, str) and trim_policy == "earliest_hotspot" and hotspot_tv_indices:
                cut = None
                for i, v in enumerate(seq):
                    if int(v) in hotspot_tv_indices:
                        cut = i
                        break
                if cut is not None:
                    seq = seq[: cut + 1]
            # unique indices for Jaccard; order not required
            import numpy as _np
            fps.append(_np.unique(_np.asarray(seq, dtype=int)))

    S = compute_jaccard_similarity(fps)

    # Direction-aware reweighting (defaults to enabled via coord_cosine)
    if direction_opts is None:
        direction_opts = {"mode": "coord_cosine"}
    mode = str(direction_opts.get("mode", "coord_cosine")).strip().lower() if isinstance(direction_opts, dict) else "coord_cosine"
    if mode == "coord_cosine":
        # Expect tv_centroids mapping tv_id -> (lat, lon) in EPSG:4326; if not supplied, skip
        tv_centroids: Optional[Dict[str, Tuple[float, float]]] = None
        if isinstance(direction_opts, dict):
            tv_centroids = direction_opts.get("tv_centroids")  # type: ignore[assignment]
        if isinstance(tv_centroids, dict) and tv_centroids:
            import math
            idx_to_tv_id = getattr(flight_list, "idx_to_tv_id", {}) or {}

            def _unit_vec_from_seq(seq_tv_idx: List[int]) -> Optional[Tuple[float, float]]:
                if not seq_tv_idx:
                    return None
                tv_start = idx_to_tv_id.get(int(seq_tv_idx[0]))
                tv_end = idx_to_tv_id.get(int(seq_tv_idx[-1]))
                if tv_start is None or tv_end is None:
                    return None
                a = tv_centroids.get(str(tv_start))
                b = tv_centroids.get(str(tv_end))
                if not a or not b:
                    return None
                lat1, lon1 = float(a[0]), float(a[1])
                lat2, lon2 = float(b[0]), float(b[1])
                latm = math.radians(0.5 * (lat1 + lat2))
                dx = (lon2 - lon1) * math.cos(latm)
                dy = (lat2 - lat1)
                norm = math.hypot(dx, dy)
                if not (norm > 0.0):
                    return None
                return (dx / norm, dy / norm)

            use_trimmed = bool(direction_opts.get("use_trimmed", True)) if isinstance(direction_opts, dict) else True
            angle_gate_deg = float(direction_opts.get("angle_gate_deg", 90.0)) if isinstance(direction_opts, dict) else 90.0
            cosine_exponent = float(direction_opts.get("cosine_exponent", 1.0)) if isinstance(direction_opts, dict) else 1.0
            min_weight = float(direction_opts.get("min_weight", 0.0)) if isinstance(direction_opts, dict) else 0.0

            # Precompute per-flight unit vectors using first/last centroids; apply same trimming as footprints
            unit_vecs: List[Optional[Tuple[float, float]]] = []
            for fid in ids:
                seq = _seq_indices_for_dir(str(fid))
                if use_trimmed and hotspot_tv_indices_for_dir and isinstance(trim_policy, str) and trim_policy == "earliest_hotspot":
                    hit = None
                    for i, v in enumerate(seq):
                        if int(v) in hotspot_tv_indices_for_dir:
                            hit = i
                            break
                    if hit is not None:
                        seq = seq[: hit + 1]
                unit_vecs.append(_unit_vec_from_seq(seq))

            def _pair_weight(i: int, j: int) -> float:
                vi = unit_vecs[i]
                vj = unit_vecs[j]
                if vi is None or vj is None:
                    return 1.0
                cosv = vi[0] * vj[0] + vi[1] * vj[1]
                if cosv > 1.0:
                    cosv = 1.0
                elif cosv < -1.0:
                    cosv = -1.0
                try:
                    theta = math.degrees(math.acos(cosv))
                except Exception:
                    theta = 0.0 if cosv >= 1.0 else 180.0
                if theta > angle_gate_deg:
                    return float(min_weight)
                w = max(cosv, 0.0) ** float(cosine_exponent)
                return float(max(w, min_weight))

            # Apply to S (dense ndarray or sparse via COO)
            if hasattr(S, "tocoo"):
                coo = S.tocoo()
                vals = coo.data
                rows = coo.row
                cols = coo.col
                for k in range(vals.size):
                    i = int(rows[k]); j = int(cols[k])
                    if i >= j:
                        continue
                    vals[k] *= _pair_weight(i, j)
                S = coo
            else:
                n = S.shape[0]
                for i in range(n):
                    for j in range(i + 1, n):
                        wij = _pair_weight(i, j)
                        S[i, j] *= wij
                        S[j, i] = S[i, j]

    membership = run_leiden_from_similarity(S, threshold=threshold, resolution=resolution, seed=seed)

    # Remap to consecutive labels 0..k-1 in order of first appearance
    remap: Dict[int, int] = {}
    next_label = 0
    out: Dict[str, int] = {}
    for fid, m in zip(ids, membership):
        if m not in remap:
            remap[m] = next_label
            next_label += 1
        out[fid] = remap[m]
    return out


__all__ = [
    "collect_hotspot_flights",
    "build_global_flows",
]
