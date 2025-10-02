"""QueryAPIWrapper implements evaluation of flight query ASTs for the API."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from server_tailwind.core.resources import get_resources

Number = Union[int, float]


@dataclass(frozen=True)
class CrossSpec:
    """Normalized representation of a cross node for reuse across operators."""

    any_of: Tuple[int, ...]
    all_of: Tuple[int, ...]
    none_of: Tuple[int, ...]
    time_range: Tuple[int, int]
    mode: str

    def all_tv_indices(self) -> Tuple[int, ...]:
        if self.all_of:
            return self.all_of
        if self.any_of:
            return self.any_of
        return ()


class QueryAPIWrapper:
    """Evaluate normalized query ASTs against the global FlightList."""

    MAX_RESULTS = 50_000

    def __init__(self, resources=None):
        # Allow dependency injection for tests; default to shared resources
        """
        Initialize the QueryAPIWrapper and prepare internal resource-backed state and caches.
        
        Parameters:
            resources (optional): Dependency-injected resources object providing attributes used by the wrapper
                (e.g., `indexer`, `capacity_per_bin_matrix`, `hourly_capacity_by_tv`, `traffic_volumes_gdf`, and
                `flight_list`). If omitted, shared application resources are obtained automatically.
        
        Description:
            - Binds resource-backed attributes required for query evaluation and capacity lookups.
            - Creates empty caches and lookup stores that will be populated lazily or by calling refresh_flight_list.
            - Rebinds the current flight list via refresh_flight_list to build flight-derived state.
            - Initializes internal cache hit counter.
        """
        self._resources = resources or get_resources()
        self.indexer = getattr(self._resources, "indexer", None)
        self._capacity_per_bin_matrix: Optional[np.ndarray] = getattr(
            self._resources, "capacity_per_bin_matrix", None
        )
        self._hourly_capacity_by_tv: Dict[str, Dict[int, float]] = getattr(
            self._resources, "hourly_capacity_by_tv", {}
        )
        self._traffic_volumes_gdf = getattr(self._resources, "traffic_volumes_gdf", None)

        # Cached helpers and lookup stores (populated via refresh_flight_list)
        self._origin_to_rows: Dict[str, np.ndarray] = {}
        self._destination_to_rows: Dict[str, np.ndarray] = {}
        self._tv_bin_to_rows: Dict[int, Dict[int, np.ndarray]] = {}
        self._tv_time_mask_cache: Dict[Tuple[int, int, int, str], np.ndarray] = {}
        self._flight_entries_cache: Dict[str, np.ndarray] = {}
        self._arrival_time_cache: Dict[Tuple[str, str], datetime] = {}
        self._node_cache: Dict[str, np.ndarray] = {}
        self._debug_last_nodes: List[str] = []
        self._capacity_state_cache: Dict[Tuple[int, int, int, str, float], np.ndarray] = {}
        self._total_occupancy_vector: Optional[np.ndarray] = None

        # Region helpers
        self._region_cache: Dict[str, Tuple[str, ...]] = {}

        # Bind flight list-derived state
        self.refresh_flight_list(self._resources.flight_list)
        self._cache_hits = 0

    def refresh_flight_list(self, flight_list) -> None:
        """Rebind the flight list and rebuild cached sparse structures and lookups."""
        self.flight_list = flight_list

        self.time_bin_minutes = int(self.flight_list.time_bin_minutes)
        self.bins_per_tv = int(self.flight_list.num_time_bins_per_tv)
        self.tv_id_to_idx = {str(k): int(v) for k, v in self.flight_list.tv_id_to_idx.items()}
        self.idx_to_tv_id = {int(k): str(v) for k, v in self.flight_list.idx_to_tv_id.items()}

        self.num_flights = int(self.flight_list.num_flights)
        self.flight_ids = np.asarray(self.flight_list.flight_ids, dtype=object)

        self.occupancy_csr = self.flight_list.occupancy_matrix
        if not sparse.isspmatrix_csr(self.occupancy_csr):
            self.occupancy_csr = sparse.csr_matrix(self.occupancy_csr, dtype=np.float32)
        self.occupancy_csc = self.occupancy_csr.tocsc(copy=False)

        self.num_tvtws = int(self.flight_list.num_tvtws)

        self._flight_row_lookup = {
            fid: idx for idx, fid in enumerate(self.flight_list.flight_ids)
        }
        self._flight_meta = self.flight_list.flight_metadata
        self._takeoff_cache = {
            fid: meta.get("takeoff_time") for fid, meta in self._flight_meta.items()
        }

        # Reset cached structures so they repopulate lazily with fresh data
        self._origin_to_rows = {}
        self._destination_to_rows = {}
        self._tv_bin_to_rows = {}
        self._tv_time_mask_cache = {}
        self._flight_entries_cache = {}
        self._arrival_time_cache = {}
        self._node_cache = {}
        self._capacity_state_cache = {}
        self._total_occupancy_vector = None
        self._debug_last_nodes = []

        self._seconds_per_bin = self.time_bin_minutes * 60

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a query payload and return serialized results."""
        if not isinstance(payload, dict):
            raise ValueError("Request payload must be a JSON object")
        query = payload.get("query")
        if not isinstance(query, dict):
            raise ValueError("'query' field must be a JSON object")
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("'options' must be an object if provided")

        # Allow top-level query options to act as defaults
        merged_options = self._merge_options(query, options)
        select_mode = merged_options.get("select", "flight_ids")
        order_by = merged_options.get("order_by")
        limit = merged_options.get("limit")
        deduplicate = merged_options.get("deduplicate", True)
        debug = merged_options.get("debug", False)

        scope_mask: Optional[np.ndarray] = None
        scope_flights = merged_options.get("flight_ids")
        if scope_flights is not None:
            if isinstance(scope_flights, str):
                scope_flights = [scope_flights]
            if not isinstance(scope_flights, Iterable) or isinstance(scope_flights, dict):
                raise ValueError("'flight_ids' option must be a list of strings")
            indices: List[int] = []
            missing: List[str] = []
            for item in scope_flights:
                fid = str(item)
                row_idx = self._flight_row_lookup.get(fid)
                if row_idx is None:
                    missing.append(fid)
                else:
                    indices.append(int(row_idx))
            if missing:
                raise ValueError(f"Unknown flight id(s): {', '.join(sorted(set(missing)))}")
            scope_mask = np.zeros(self.num_flights, dtype=bool)
            if indices:
                scope_mask[np.asarray(indices, dtype=np.int32)] = True

        if limit is not None:
            try:
                limit = int(limit)
            except Exception as exc:  # noqa: BLE001
                raise ValueError("'limit' must be an integer") from exc
            if limit <= 0:
                raise ValueError("'limit' must be positive if provided")

        start_time = time.perf_counter()
        self._cache_hits = 0
        self._debug_last_nodes = []

        mask = self._evaluate_node(query)
        if scope_mask is not None:
            mask = mask & scope_mask
        matched_rows = np.where(mask)[0]

        if deduplicate:
            # mask already unique; no-op reserved for API contract compatibility
            pass

        total_matches = int(matched_rows.size)
        truncated = False
        effective_limit = limit
        if effective_limit is None and total_matches > self.MAX_RESULTS:
            effective_limit = self.MAX_RESULTS
            truncated = True
        if effective_limit is not None:
            matched_rows = matched_rows[: effective_limit]

        ordered_rows = self._apply_ordering(matched_rows, order_by)

        metadata: Dict[str, Any] = {
            "time_bin_minutes": self.time_bin_minutes,
            "bins_per_tv": self.bins_per_tv,
            "evaluation_ms": (time.perf_counter() - start_time) * 1000.0,
            "node_cache_hits": self._cache_hits,
            "result_size": int(matched_rows.size),
            "total_matches": total_matches,
        }
        if truncated:
            metadata["truncated"] = True
        if debug:
            metadata["explain"] = {"cache_keys": list(self._debug_last_nodes)}

        if scope_mask is not None:
            metadata["scope_flights"] = int(np.count_nonzero(scope_mask))

        if select_mode == "flight_ids":
            flight_ids = self.flight_ids[ordered_rows].tolist()
            return {"flight_ids": flight_ids, "metadata": metadata}
        if select_mode == "count":
            return {"count": total_matches, "metadata": metadata}
        if select_mode == "ids_and_times":
            ids_and_times = [
                self._build_flight_time_record(int(row_idx)) for row_idx in ordered_rows
            ]
            return {"ids_and_times": ids_and_times, "metadata": metadata}
        raise ValueError("Unsupported select mode. Expected 'flight_ids', 'count', or 'ids_and_times'.")

    # ------------------------------------------------------------------
    # Node evaluation dispatch
    # ------------------------------------------------------------------
    def _evaluate_node(self, node: Dict[str, Any]) -> np.ndarray:
        if not isinstance(node, dict):
            raise ValueError("Query nodes must be objects")
        node_type = node.get("type")
        if not isinstance(node_type, str):
            raise ValueError("Query node missing 'type'")

        cache_key = json.dumps(node, sort_keys=True)
        cached = self._node_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            self._debug_last_nodes.append(cache_key)
            return cached

        if node_type in ("and", "or"):
            mask = self._eval_boolean(node_type, node)
        elif node_type in ("not", "negation"):
            mask = self._eval_not(node)
        elif node_type == "cross":
            mask = self._eval_cross(node)
        elif node_type == "sequence":
            mask = self._eval_sequence(node)
        elif node_type == "origin":
            mask = self._eval_origin(node)
        elif node_type == "destination":
            mask = self._eval_destination(node)
        elif node_type == "arrival_window":
            mask = self._eval_arrival_window(node)
        elif node_type == "takeoff_window":
            mask = self._eval_takeoff_window(node)
        elif node_type == "geo_region_cross":
            mask = self._eval_geo_region_cross(node)
        elif node_type == "capacity_state":
            mask = self._eval_capacity_state(node)
        elif node_type == "duration_between":
            mask = self._eval_duration_between(node)
        elif node_type == "count_crossings":
            mask = self._eval_count_crossings(node)
        else:
            raise ValueError(f"Unsupported query node type '{node_type}'")

        self._node_cache[cache_key] = mask
        return mask

    # ------------------------------------------------------------------
    # Boolean combinators
    # ------------------------------------------------------------------
    def _eval_boolean(self, op: str, node: Dict[str, Any]) -> np.ndarray:
        children = node.get("children")
        if not isinstance(children, list) or not children:
            raise ValueError(f"'{op}' node requires a non-empty 'children' array")
        if op == "and":
            mask = np.ones(self.num_flights, dtype=bool)
            for child in children:
                mask &= self._evaluate_node(child)
            return mask
        if op == "or":
            mask = np.zeros(self.num_flights, dtype=bool)
            for child in children:
                mask |= self._evaluate_node(child)
            return mask
        raise ValueError(f"Unknown boolean operator '{op}'")

    def _eval_not(self, node: Dict[str, Any]) -> np.ndarray:
        child = node.get("child")
        if child is None:
            children = node.get("children")
            if isinstance(children, list) and len(children) == 1:
                child = children[0]
        if child is None:
            raise ValueError("'not' node requires a single child")
        return ~self._evaluate_node(child)

    # ------------------------------------------------------------------
    # Atomic predicates
    # ------------------------------------------------------------------
    def _eval_cross(self, node: Dict[str, Any]) -> np.ndarray:
        spec = self._build_cross_spec(node)
        return self._mask_for_cross_spec(spec)

    def _eval_sequence(self, node: Dict[str, Any]) -> np.ndarray:
        steps = node.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("'sequence' requires a non-empty 'steps' array")
        strict = bool(node.get("strict", False))
        within_spec = node.get("within")
        within_seconds: Optional[float] = None
        if within_spec is not None:
            within_seconds = self._duration_spec_to_seconds(within_spec)
            if within_seconds is None:
                raise ValueError("'within' must contain 'minutes' or 'bins'")

        cross_specs: List[CrossSpec] = []
        candidate_mask = np.ones(self.num_flights, dtype=bool)
        for step in steps:
            if not isinstance(step, dict) or step.get("type") != "cross":
                raise ValueError("Sequence steps must be 'cross' nodes")
            spec = self._build_cross_spec(step)
            cross_specs.append(spec)
            candidate_mask &= self._mask_for_cross_spec(spec)

        if not candidate_mask.any():
            return candidate_mask

        row_indices = np.where(candidate_mask)[0]
        matches = np.zeros(self.num_flights, dtype=bool)
        for row_idx in row_indices:
            flight_id = self.flight_ids[row_idx]
            entries = self._get_flight_entries(flight_id)
            if entries.size == 0:
                continue
            if self._flight_matches_sequence(entries, cross_specs, strict, within_seconds):
                matches[row_idx] = True
        return matches

    def _eval_origin(self, node: Dict[str, Any]) -> np.ndarray:
        return self._mask_for_airport(node, field="origin", cache=self._origin_to_rows)

    def _eval_destination(self, node: Dict[str, Any]) -> np.ndarray:
        return self._mask_for_airport(node, field="destination", cache=self._destination_to_rows)

    def _eval_arrival_window(self, node: Dict[str, Any]) -> np.ndarray:
        window = node.get("clock") or node.get("bins")
        if window is None:
            window = node.get("time")
        if window is None:
            raise ValueError("'arrival_window' requires a 'clock' or 'bins' specification")
        method = str(node.get("method", "last_crossing"))
        start_sec, end_sec = self._resolve_window_to_seconds(window)
        matches = np.zeros(self.num_flights, dtype=bool)
        for row_idx, flight_id in enumerate(self.flight_ids):
            arrival_dt = self._arrival_time_for_flight(flight_id, method)
            if arrival_dt is None:
                continue
            seconds = self._seconds_since_midnight(arrival_dt)
            if self._seconds_in_range(seconds, start_sec, end_sec):
                matches[row_idx] = True
        return matches

    def _eval_takeoff_window(self, node: Dict[str, Any]) -> np.ndarray:
        window = node.get("clock") or node.get("bins")
        if window is None:
            window = node.get("time")
        if window is None:
            raise ValueError("'takeoff_window' requires a 'clock' or 'bins' specification")
        start_sec, end_sec = self._resolve_window_to_seconds(window)
        matches = np.zeros(self.num_flights, dtype=bool)
        for row_idx, flight_id in enumerate(self.flight_ids):
            takeoff = self._takeoff_cache.get(flight_id)
            if not isinstance(takeoff, datetime):
                continue
            seconds = self._seconds_since_midnight(takeoff)
            if self._seconds_in_range(seconds, start_sec, end_sec):
                matches[row_idx] = True
        return matches

    def _eval_geo_region_cross(self, node: Dict[str, Any]) -> np.ndarray:
        region = node.get("region")
        if not isinstance(region, dict):
            raise ValueError("'geo_region_cross' requires a 'region' object")
        tv_ids = self._resolve_region_to_tv_ids(region)
        if not tv_ids:
            return np.zeros(self.num_flights, dtype=bool)
        cross_node = {"type": "cross", "tv": {"anyOf": list(tv_ids)}}
        time_spec = node.get("time")
        if time_spec is not None:
            cross_node["time"] = time_spec
        mode = node.get("mode")
        if mode is not None:
            cross_node["mode"] = mode
        return self._eval_cross(cross_node)

    def _eval_capacity_state(self, node: Dict[str, Any]) -> np.ndarray:
        if self._capacity_per_bin_matrix is None:
            return np.zeros(self.num_flights, dtype=bool)
        tv_field = node.get("tv")
        if tv_field is None:
            raise ValueError("'capacity_state' requires a 'tv' field")
        tv_ids = self._normalize_tv_selection(tv_field)
        if not tv_ids:
            raise ValueError("'capacity_state' tv selection resolved to empty set")
        time_spec = node.get("time")
        time_range = self._resolve_time_window(time_spec)
        condition = str(node.get("condition", "overloaded"))
        threshold_spec = node.get("threshold") or {}
        delta = float(threshold_spec.get("occupancy_minus_capacity", 0.0))
        if "z" in threshold_spec:
            raise ValueError("'threshold.z' is not supported; use 'occupancy_minus_capacity'")

        mask = np.zeros(self.num_flights, dtype=bool)
        for tv_id in tv_ids:
            tv_idx = self._tv_id_to_index(tv_id)
            base = tv_idx * self.bins_per_tv
            start_bin, end_bin = time_range
            rows = self._rows_for_capacity_state(tv_idx, start_bin, end_bin, condition, delta)
            if rows.size:
                mask[rows] = True
        return mask

    def _eval_duration_between(self, node: Dict[str, Any]) -> np.ndarray:
        from_node = node.get("from")
        to_node = node.get("to")
        if not isinstance(from_node, dict) or from_node.get("type") != "cross":
            raise ValueError("'duration_between.from' must be a cross node")
        if not isinstance(to_node, dict) or to_node.get("type") != "cross":
            raise ValueError("'duration_between.to' must be a cross node")
        op = node.get("op")
        if op not in {"<", "<=", ">", ">="}:
            raise ValueError("'duration_between.op' must be one of '<', '<=', '>', '>='")
        threshold_seconds = self._duration_spec_to_seconds(node.get("value"))
        if threshold_seconds is None:
            raise ValueError("'duration_between.value' must specify 'minutes' or 'bins'")

        from_spec = self._build_cross_spec(from_node)
        to_spec = self._build_cross_spec(to_node)
        from_mask = self._mask_for_cross_spec(from_spec)
        to_mask = self._mask_for_cross_spec(to_spec)
        candidate_rows = np.where(from_mask & to_mask)[0]
        matches = np.zeros(self.num_flights, dtype=bool)
        for row_idx in candidate_rows:
            flight_id = self.flight_ids[row_idx]
            entries = self._get_flight_entries(flight_id)
            if entries.size == 0:
                continue
            from_times = self._entry_times_matching(entries, from_spec)
            to_times = self._entry_times_matching(entries, to_spec)
            if not from_times.size or not to_times.size:
                continue
            duration = float(np.min(to_times) - np.max(from_times))
            if self._compare(duration, op, threshold_seconds):
                matches[row_idx] = True
        return matches

    def _eval_count_crossings(self, node: Dict[str, Any]) -> np.ndarray:
        tv_field = node.get("tv")
        if tv_field is None:
            raise ValueError("'count_crossings' requires a 'tv' field")
        tv_ids = self._normalize_tv_selection(tv_field)
        if not tv_ids:
            raise ValueError("'count_crossings' tv selection resolved to empty set")
        op = node.get("op")
        if op not in {">=", "<=", "==", ">", "<"}:
            raise ValueError("'count_crossings.op' must be a comparison operator")
        try:
            value = int(node.get("value"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError("'count_crossings.value' must be an integer") from exc
        time_range = self._resolve_time_window(node.get("time"))
        tv_indices = tuple(self._tv_id_to_index(tv_id) for tv_id in tv_ids)
        matches = np.zeros(self.num_flights, dtype=bool)
        for row_idx, flight_id in enumerate(self.flight_ids):
            entries = self._get_flight_entries(flight_id)
            if entries.size == 0:
                count = 0
            else:
                mask = np.isin(entries["tv_idx"], tv_indices)
                if mask.any():
                    bin_mask = (entries["time_idx"] >= time_range[0]) & (
                        entries["time_idx"] <= time_range[1]
                    )
                    mask &= bin_mask
                count = int(np.count_nonzero(mask))
            if self._compare(count, op, value):
                matches[row_idx] = True
        return matches

    # ------------------------------------------------------------------
    # Helper builders and resolvers
    # ------------------------------------------------------------------
    def _build_cross_spec(self, node: Dict[str, Any]) -> CrossSpec:
        tv_field = node.get("tv")
        any_of: Tuple[int, ...] = ()
        all_of: Tuple[int, ...] = ()
        none_of: Tuple[int, ...] = ()
        if isinstance(tv_field, str):
            any_of = (self._tv_id_to_index(tv_field),)
        elif isinstance(tv_field, list):
            ids = self._normalize_tv_selection(tv_field)
            any_of = tuple(self._tv_id_to_index(tv_id) for tv_id in ids)
        elif isinstance(tv_field, dict):
            if "anyOf" in tv_field:
                ids = self._normalize_tv_selection(tv_field["anyOf"])
                any_of = tuple(self._tv_id_to_index(tv_id) for tv_id in ids)
            if "allOf" in tv_field:
                ids = self._normalize_tv_selection(tv_field["allOf"])
                all_of = tuple(self._tv_id_to_index(tv_id) for tv_id in ids)
            if "noneOf" in tv_field:
                ids = self._normalize_tv_selection(tv_field["noneOf"])
                none_of = tuple(self._tv_id_to_index(tv_id) for tv_id in ids)
        else:
            raise ValueError("'cross.tv' must be a string, list, or object")
        if not (any_of or all_of):
            if not none_of:
                raise ValueError("'cross' tv selection resolved to empty set")
            # matching nothing but requiring exclusions -> start from all flights
            any_of = tuple(sorted(self.tv_id_to_idx.values()))
        mode = str(node.get("mode", "any"))
        time_range = self._resolve_time_window(node.get("time"))
        return CrossSpec(any_of=any_of, all_of=all_of, none_of=none_of, time_range=time_range, mode=mode)

    def _mask_for_cross_spec(self, spec: CrossSpec) -> np.ndarray:
        mask = np.ones(self.num_flights, dtype=bool)
        if spec.any_of:
            any_mask = np.zeros(self.num_flights, dtype=bool)
            for tv_idx in spec.any_of:
                any_mask |= self._mask_for_tv_and_range(tv_idx, spec.time_range, spec.mode)
            mask &= any_mask
        if spec.all_of:
            for tv_idx in spec.all_of:
                mask &= self._mask_for_tv_and_range(tv_idx, spec.time_range, spec.mode)
        if spec.none_of:
            for tv_idx in spec.none_of:
                mask &= ~self._mask_for_tv_and_range(tv_idx, spec.time_range, spec.mode)
        return mask

    def _mask_for_tv_and_range(
        self, tv_idx: int, time_range: Tuple[int, int], mode: str
    ) -> np.ndarray:
        start_bin, end_bin = time_range
        key = (int(tv_idx), int(start_bin), int(end_bin), str(mode))
        cached = self._tv_time_mask_cache.get(key)
        if cached is not None:
            return cached

        rows_list: List[np.ndarray] = []
        for bin_idx in range(start_bin, end_bin + 1):
            rows = self._rows_for_tv_bin(tv_idx, bin_idx, mode)
            if rows.size:
                rows_list.append(rows)
        if rows_list:
            union_rows = np.unique(np.concatenate(rows_list))
        else:
            union_rows = np.empty(0, dtype=np.int32)
        mask = np.zeros(self.num_flights, dtype=bool)
        if union_rows.size:
            mask[union_rows] = True
        self._tv_time_mask_cache[key] = mask
        return mask

    def _rows_for_tv_bin(self, tv_idx: int, bin_idx: int, mode: str) -> np.ndarray:
        tv_cache = self._tv_bin_to_rows.setdefault(int(tv_idx), {})
        cached = tv_cache.get(int(bin_idx))
        if cached is not None:
            return cached
        column_idx = int(tv_idx) * self.bins_per_tv + int(bin_idx)
        if column_idx < 0 or column_idx >= self.num_tvtws:
            rows = np.empty(0, dtype=np.int32)
        else:
            column = self.occupancy_csc.getcol(column_idx)
            rows = column.indices.astype(np.int32, copy=False)
        tv_cache[int(bin_idx)] = rows
        return rows

    def _rows_for_capacity_state(
        self, tv_idx: int, start_bin: int, end_bin: int, condition: str, delta: float
    ) -> np.ndarray:
        key = (int(tv_idx), int(start_bin), int(end_bin), condition, float(delta))
        cached = self._capacity_state_cache.get(key)
        if cached is not None:
            return cached
        rows_list: List[np.ndarray] = []
        base = int(tv_idx) * self.bins_per_tv
        total = self._get_total_occupancy_vector()
        for bin_offset in range(start_bin, end_bin + 1):
            tvtw_idx = base + bin_offset
            if tvtw_idx < 0 or tvtw_idx >= self.num_tvtws:
                continue
            occupancy = float(total[tvtw_idx])
            cap = float(self._capacity_per_bin_matrix[int(tv_idx), bin_offset])
            if cap < 0:
                continue
            if condition == "overloaded":
                ok = occupancy - cap > delta
            elif condition == "under_capacity":
                ok = cap - occupancy > delta
            elif condition == "near_capacity":
                ok = abs(occupancy - cap) <= max(delta, 0.0)
            else:
                raise ValueError("'capacity_state.condition' must be 'overloaded', 'near_capacity', or 'under_capacity'")
            if ok:
                rows = self._rows_for_tv_bin(tv_idx, bin_offset, mode="any")
                if rows.size:
                    rows_list.append(rows)
        if rows_list:
            combined = np.unique(np.concatenate(rows_list))
        else:
            combined = np.empty(0, dtype=np.int32)
        self._capacity_state_cache[key] = combined
        return combined

    def _get_total_occupancy_vector(self) -> np.ndarray:
        if self._total_occupancy_vector is None:
            total = self.occupancy_csr.sum(axis=0)
            self._total_occupancy_vector = np.asarray(total).ravel().astype(np.float32, copy=False)
        return self._total_occupancy_vector

    def _merge_options(self, query: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(options)
        for key in ("select", "order_by", "limit", "deduplicate", "debug", "flight_ids"):
            if key in query and key not in merged:
                merged[key] = query[key]
        return merged

    def _apply_ordering(self, rows: np.ndarray, order_by: Optional[str]) -> np.ndarray:
        if rows.size == 0:
            return rows
        if not order_by:
            return rows
        order_by = str(order_by)
        if order_by == "takeoff_time":
            takeoff_times = [self._takeoff_cache.get(self.flight_ids[idx]) for idx in rows]
            order = np.argsort([
                dt.timestamp() if isinstance(dt, datetime) else math.inf for dt in takeoff_times
            ])
            return rows[order]
        if order_by in {"first_crossing_time", "last_crossing_time"}:
            is_last = order_by == "last_crossing_time"
            times = []
            for idx in rows:
                entries = self._get_flight_entries(self.flight_ids[idx])
                if entries.size == 0:
                    times.append(math.inf)
                    continue
                sec = entries["entry_time_s"].max() if is_last else entries["entry_time_s"].min()
                times.append(float(sec))
            order = np.argsort(times)
            return rows[order]
        if order_by == "dest":
            dests = [self._flight_meta[self.flight_ids[idx]].get("destination") for idx in rows]
            order = np.argsort([dest or "" for dest in dests])
            return rows[order]
        return rows

    def _build_flight_time_record(self, row_idx: int) -> Dict[str, Any]:
        flight_id = self.flight_ids[row_idx]
        record: Dict[str, Any] = {"flight_id": flight_id}
        entries = self._get_flight_entries(flight_id)
        if entries.size:
            takeoff = self._takeoff_cache.get(flight_id)
            if isinstance(takeoff, datetime):
                first_dt = takeoff + timedelta(seconds=float(entries["entry_time_s"].min()))
                last_dt = takeoff + timedelta(seconds=float(entries["entry_time_s"].max()))
                record["first_crossing_time"] = first_dt.time().strftime("%H:%M:%S")
                record["last_crossing_time"] = last_dt.time().strftime("%H:%M:%S")
        return record

    def _flight_matches_sequence(
        self,
        entries: np.ndarray,
        specs: Sequence[CrossSpec],
        strict: bool,
        within_seconds: Optional[float],
    ) -> bool:
        positions: List[int] = []
        cursor = -1
        for spec in specs:
            found_idx = self._find_next_entry(entries, spec, after=cursor)
            if found_idx < 0:
                return False
            if strict and cursor >= 0 and found_idx != cursor + 1:
                return False
            positions.append(found_idx)
            cursor = found_idx
        if within_seconds is not None and positions:
            first_time = float(entries["entry_time_s"][positions[0]])
            last_time = float(entries["entry_time_s"][positions[-1]])
            if last_time - first_time > within_seconds:
                return False
        return True

    def _find_next_entry(self, entries: np.ndarray, spec: CrossSpec, after: int) -> int:
        time_start, time_end = spec.time_range
        for idx in range(after + 1, entries.size):
            tv_idx = int(entries["tv_idx"][idx])
            time_idx = int(entries["time_idx"][idx])
            if time_idx < time_start or time_idx > time_end:
                continue
            if spec.none_of and tv_idx in spec.none_of:
                continue
            tv_match = False
            if spec.all_of:
                tv_match = tv_idx in spec.all_of
            elif spec.any_of:
                tv_match = tv_idx in spec.any_of
            else:
                tv_match = True
            if tv_match:
                return idx
        return -1

    def _entry_times_matching(self, entries: np.ndarray, spec: CrossSpec) -> np.ndarray:
        time_start, time_end = spec.time_range
        mask = (entries["time_idx"] >= time_start) & (entries["time_idx"] <= time_end)
        if spec.none_of:
            mask &= ~np.isin(entries["tv_idx"], spec.none_of)
        if spec.all_of:
            mask &= np.isin(entries["tv_idx"], spec.all_of)
        elif spec.any_of:
            mask &= np.isin(entries["tv_idx"], spec.any_of)
        return entries["entry_time_s"][mask]

    def _get_flight_entries(self, flight_id: str) -> np.ndarray:
        cached = self._flight_entries_cache.get(flight_id)
        if cached is not None:
            return cached
        meta = self._flight_meta.get(flight_id)
        if not meta:
            arr = np.empty(0, dtype=self._entry_dtype)
            self._flight_entries_cache[flight_id] = arr
            return arr
        intervals = meta.get("occupancy_intervals", []) or []
        if not intervals:
            arr = np.empty(0, dtype=self._entry_dtype)
            self._flight_entries_cache[flight_id] = arr
            return arr
        data = []
        for interval in intervals:
            tvtw_index = int(interval["tvtw_index"])
            tv_idx = tvtw_index // self.bins_per_tv
            time_idx = tvtw_index % self.bins_per_tv
            entry_time = float(interval.get("entry_time_s", 0.0))
            data.append((tv_idx, time_idx, entry_time))
        arr = np.array(data, dtype=self._entry_dtype)
        order = np.argsort(arr["entry_time_s"], kind="mergesort")
        arr = arr[order]
        self._flight_entries_cache[flight_id] = arr
        return arr

    @property
    def _entry_dtype(self) -> np.dtype:
        return np.dtype([("tv_idx", np.int32), ("time_idx", np.int32), ("entry_time_s", np.float32)])

    def _arrival_time_for_flight(self, flight_id: str, method: str) -> Optional[datetime]:
        key = (flight_id, method)
        cached = self._arrival_time_cache.get(key)
        if cached is not None:
            return cached
        takeoff = self._takeoff_cache.get(flight_id)
        if not isinstance(takeoff, datetime):
            return None
        entries = self._get_flight_entries(flight_id)
        if entries.size == 0:
            return None
        if method not in {"last_crossing", "approach_tv", "provided"}:
            raise ValueError("arrival_window.method must be 'last_crossing', 'approach_tv', or 'provided'")
        if method != "last_crossing":
            # TODO: implement alternative strategies when enriched data is available
            method = "last_crossing"
        last_time = float(entries["entry_time_s"].max())
        arrival = takeoff + timedelta(seconds=last_time)
        self._arrival_time_cache[key] = arrival
        return arrival

    def _resolve_region_to_tv_ids(self, region: Dict[str, Any]) -> Tuple[str, ...]:
        if "tv_ids" in region:
            ids = self._normalize_tv_selection(region["tv_ids"])
            return tuple(ids)
        key = json.dumps(region, sort_keys=True)
        cached = self._region_cache.get(key)
        if cached is not None:
            return cached
        if self._traffic_volumes_gdf is None:
            return ()
        try:
            import shapely.geometry as geom
        except Exception:
            return ()
        if "bbox" in region:
            try:
                bounds = [float(x) for x in region["bbox"]]
                if len(bounds) != 4:
                    raise ValueError
                polygon = geom.box(bounds[0], bounds[1], bounds[2], bounds[3])
            except Exception as exc:  # noqa: BLE001
                raise ValueError("region.bbox must be [minLon, minLat, maxLon, maxLat]") from exc
        elif "polygon" in region:
            polygon = geom.shape(region["polygon"])
        else:
            raise ValueError("region must specify 'tv_ids', 'bbox', or 'polygon'")
        gdf = self._traffic_volumes_gdf
        matches: List[str] = []
        for _, row in gdf.iterrows():
            tv_id = row.get("traffic_volume_id")
            if tv_id is None:
                continue
            if tv_id not in self.tv_id_to_idx:
                continue
            geom_obj = row.get("geometry")
            try:
                if geom_obj is not None and polygon.intersects(geom_obj):
                    matches.append(str(tv_id))
            except Exception:
                continue
        matches = sorted(set(matches))
        self._region_cache[key] = tuple(matches)
        return self._region_cache[key]

    def _normalize_tv_selection(self, value: Any) -> Tuple[str, ...]:
        if isinstance(value, str):
            return (self._validate_tv_id(value),)
        if isinstance(value, Sequence):
            ids = [self._validate_tv_id(str(tv)) for tv in value]
            if not ids:
                raise ValueError("TV selection cannot be empty")
            return tuple(ids)
        raise ValueError("TV selection must be a string or list of strings")

    def _validate_tv_id(self, tv_id: str) -> str:
        tv_id = str(tv_id)
        if tv_id not in self.tv_id_to_idx:
            raise ValueError(f"Unknown traffic volume id '{tv_id}'")
        return tv_id

    def _tv_id_to_index(self, tv_id: str) -> int:
        return int(self.tv_id_to_idx[self._validate_tv_id(tv_id)])

    def _resolve_time_window(self, time_spec: Optional[Dict[str, Any]]) -> Tuple[int, int]:
        if time_spec is None:
            return (0, self.bins_per_tv - 1)
        inclusive = bool(time_spec.get("inclusive", True))
        bins_spec = time_spec.get("bins")
        if bins_spec is not None:
            start = int(bins_spec.get("from", 0))
            end = int(bins_spec.get("to", self.bins_per_tv - 1))
        else:
            clock_spec = time_spec.get("clock") or time_spec
            if not isinstance(clock_spec, dict):
                raise ValueError("Time window requires 'clock' or 'bins' object")
            clock_from = self._parse_clock_to_seconds(clock_spec.get("from"))
            clock_to = self._parse_clock_to_seconds(clock_spec.get("to"))
            start = int(clock_from // self._seconds_per_bin)
            end = int(clock_to // self._seconds_per_bin)
        start = max(0, min(int(start), self.bins_per_tv - 1))
        end = max(0, min(int(end), self.bins_per_tv - 1))
        if end < start:
            raise ValueError("Time window end must be >= start")
        if not inclusive:
            end = max(start, end - 1)
        return (start, end)

    def _resolve_window_to_seconds(self, spec: Dict[str, Any]) -> Tuple[int, int]:
        if "from" in spec or "to" in spec:
            # direct spec
            start = self._parse_clock_to_seconds(spec.get("from"))
            end = self._parse_clock_to_seconds(spec.get("to"))
        elif "bins" in spec:
            bins_spec = spec["bins"]
            start = int(bins_spec.get("from", 0)) * self._seconds_per_bin
            end = int(bins_spec.get("to", self.bins_per_tv - 1)) * self._seconds_per_bin
        elif "clock" in spec:
            return self._resolve_window_to_seconds(spec["clock"])
        else:
            raise ValueError("Time window spec must contain 'clock' or 'bins'")
        start = int(max(0, min(start, 24 * 3600 - 1)))
        end = int(max(0, min(end, 24 * 3600 - 1)))
        if end < start:
            raise ValueError("Time window seconds end must be >= start")
        return (start, end)

    def _parse_clock_to_seconds(self, value: Union[str, Number, None]) -> int:
        if value is None:
            return 0
        s = str(value).strip()
        if not s:
            return 0
        if s.count(":") in {1, 2}:
            parts = [int(p) for p in s.split(":")]
            if len(parts) == 2:
                hour, minute = parts
                second = 0
            else:
                hour, minute, second = parts
        else:
            if not s.isdigit():
                raise ValueError("Clock strings must be HH:MM[:SS] or numeric HHMM[SS]")
            if len(s) not in {4, 6}:
                raise ValueError("Numeric clock strings must be HHMM or HHMMSS")
            hour = int(s[0:2])
            minute = int(s[2:4])
            second = int(s[4:6]) if len(s) == 6 else 0
        if hour == 24 and minute == 0 and second == 0:
            return 24 * 3600 - 1
        if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
            raise ValueError("Clock time components out of range")
        return hour * 3600 + minute * 60 + second

    def _duration_spec_to_seconds(self, spec: Optional[Dict[str, Any]]) -> Optional[float]:
        if spec is None:
            return None
        if "minutes" in spec:
            return float(spec["minutes"]) * 60.0
        if "bins" in spec:
            return float(spec["bins"]) * float(self._seconds_per_bin)
        return None

    def _seconds_since_midnight(self, dt: datetime) -> int:
        return dt.hour * 3600 + dt.minute * 60 + dt.second

    def _seconds_in_range(self, value: int, start: int, end: int) -> bool:
        if start <= end:
            return start <= value <= end
        # wrap-around window
        return value >= start or value <= end

    def _mask_for_airport(
        self, node: Dict[str, Any], *, field: str, cache: Dict[str, np.ndarray]
    ) -> np.ndarray:
        airport_value = node.get("airport")
        if airport_value is None:
            raise ValueError(f"'{node.get('type')}' requires an 'airport' field")
        airports = (
            [airport_value]
            if isinstance(airport_value, str)
            else list(airport_value)
        )
        rows = np.zeros(self.num_flights, dtype=bool)
        for airport in airports:
            airport_str = str(airport)
            rows_for_airport = cache.get(airport_str)
            if rows_for_airport is None:
                indices: List[int] = []
                for fid, idx in self._flight_row_lookup.items():
                    meta = self._flight_meta.get(fid)
                    if meta and meta.get(field) == airport_str:
                        indices.append(idx)
                rows_for_airport = np.array(indices, dtype=np.int32)
                cache[airport_str] = rows_for_airport
            if rows_for_airport.size:
                rows[rows_for_airport] = True
        return rows

    def _compare(self, lhs: float, op: str, rhs: float) -> bool:
        if op == "<":
            return lhs < rhs
        if op == "<=":
            return lhs <= rhs
        if op == ">":
            return lhs > rhs
        if op == ">=":
            return lhs >= rhs
        if op == "==":
            return lhs == rhs
        raise ValueError(f"Unsupported comparison operator '{op}'")
