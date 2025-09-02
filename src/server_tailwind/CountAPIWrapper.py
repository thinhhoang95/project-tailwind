"""
CountAPIWrapper module

Provides an API wrapper for computing original occupancy counts over
traffic volumes and time windows, with optional per-category breakdowns.

It reuses a single FlightList instance loaded at process start to
avoid repeatedly loading large JSON files into memory.
"""

from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path
import numpy as np

# Ensure imports from project src are available
import sys
project_root = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(project_root))

from project_tailwind.optimize.eval.flight_list import FlightList


class CountAPIWrapper:
    """
    Wrapper around the FlightList to compute occupancy counts efficiently,
    supporting optional category/group breakdowns.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2)
        # Load once and reuse
        self._flight_list = FlightList(
            occupancy_file_path="output/so6_occupancy_matrix_with_times.json",
            tvtw_indexer_path="output/tvtw_indexer.json",
        )

        # Cached overall totals (across all flights)
        self._total_occupancy_vector: Optional[np.ndarray] = None

    # ---------- Helpers ----------
    @property
    def time_bin_minutes(self) -> int:
        return int(self._flight_list.time_bin_minutes)

    @property
    def bins_per_tv(self) -> int:
        return int(self._flight_list.num_time_bins_per_tv)

    def _parse_time_to_seconds(self, value: str) -> int:
        s = str(value).strip()
        if ":" in s:
            parts = s.split(":")
            if len(parts) not in (2, 3):
                raise ValueError("Time must be HH:MM or HH:MM:SS")
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) == 3 else 0
        else:
            if not s.isdigit() or len(s) not in (4, 6):
                raise ValueError("Time must be numeric 'HHMM' or 'HHMMSS'")
            hour = int(s[0:2])
            minute = int(s[2:4])
            second = int(s[4:6]) if len(s) == 6 else 0
        if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
            raise ValueError("Time contains invalid components")
        return hour * 3600 + minute * 60 + second

    def _format_time_window(self, bin_offset: int) -> str:
        start_total_min = int(bin_offset * self.time_bin_minutes)
        start_hour = start_total_min // 60
        start_min = start_total_min % 60

        end_total_min = start_total_min + self.time_bin_minutes
        if end_total_min == 24 * 60:
            end_str = "24:00"
        else:
            end_hour = (end_total_min // 60) % 24
            end_min = end_total_min % 60
            end_str = f"{end_hour:02d}:{end_min:02d}"

        start_str = f"{start_hour:02d}:{start_min:02d}"
        return f"{start_str}-{end_str}"

    def _get_or_build_total_vector(self) -> np.ndarray:
        if self._total_occupancy_vector is not None:
            return self._total_occupancy_vector
        vec = self._flight_list.get_total_occupancy_by_tvtw().astype(np.float32, copy=False)
        self._total_occupancy_vector = vec
        return vec

    # ---------- Core computation ----------
    def _aggregate_vector_for_rows(self, rows_arr: Optional[np.ndarray]) -> np.ndarray:
        """
        Build a flat occupancy vector across TVs and all bins, either for all flights
        (when rows_arr is None) or for a filtered set of flight rows.
        """
        if rows_arr is None:
            return self._get_or_build_total_vector()
        if rows_arr.size == 0:
            num_tvs = len(self._flight_list.tv_id_to_idx)
            return np.zeros(num_tvs * self.bins_per_tv, dtype=np.float32)
        sub_sum = self._flight_list.occupancy_matrix[rows_arr, :].sum(axis=0)
        return np.asarray(sub_sum).ravel().astype(np.float32, copy=False)

    def _apply_rolling_hour(self, matrix_2d: np.ndarray, window_bins: int) -> np.ndarray:
        """
        Apply a forward-looking rolling sum of size window_bins along axis=1.
        For each bin j, result[:, j] = sum(matrix_2d[:, j : j+window_bins]) truncated at end.
        """
        if window_bins <= 1:
            return matrix_2d
        num_tvs, num_bins = matrix_2d.shape
        # cumulative sum with zero at start for easy windowed differences
        cs = np.cumsum(np.concatenate([np.zeros((num_tvs, 1), dtype=matrix_2d.dtype), matrix_2d], axis=1), axis=1)
        out = np.empty_like(matrix_2d, dtype=matrix_2d.dtype)
        for j in range(num_bins):
            j2 = min(num_bins, j + window_bins)
            out[:, j] = cs[:, j2] - cs[:, j]
        return out
    def _compute_counts(
        self,
        *,
        traffic_volume_ids: Optional[List[str]],
        from_time_str: Optional[str],
        to_time_str: Optional[str],
        categories: Optional[Dict[str, List[str]]],
        flight_ids: Optional[List[str]],
        include_overall: bool,
        rank_by: str,
        rolling_hour: bool,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        # Validate TVs if provided
        tv_map = self._flight_list.tv_id_to_idx
        if traffic_volume_ids is not None:
            unknown_tvs = [tv for tv in traffic_volume_ids if tv not in tv_map]
            if unknown_tvs:
                raise ValueError(f"Unknown traffic_volume_ids: {unknown_tvs}")

        # Determine time bin range
        if (from_time_str and not to_time_str) or (to_time_str and not from_time_str):
            raise ValueError("Both 'from_time_str' and 'to_time_str' must be provided together")

        if from_time_str and to_time_str:
            start_seconds = self._parse_time_to_seconds(from_time_str)
            end_seconds = self._parse_time_to_seconds(to_time_str)
            if end_seconds < start_seconds:
                raise ValueError("'to_time_str' must be greater than or equal to 'from_time_str'")
            seconds_per_bin = self.time_bin_minutes * 60
            start_bin = start_seconds // seconds_per_bin
            end_bin = end_seconds // seconds_per_bin
            # Clamp to valid range
            start_bin = max(0, min(int(start_bin), self.bins_per_tv - 1))
            end_bin = max(0, min(int(end_bin), self.bins_per_tv - 1))
            if start_bin > end_bin:
                raise ValueError("Computed start bin is after end bin; check time inputs")
        else:
            start_bin = 0
            end_bin = self.bins_per_tv - 1

        # Prepare labels
        labels = [self._format_time_window(i) for i in range(int(start_bin), int(end_bin) + 1)]

        # Collections
        counts: Dict[str, List[int]] = {}
        mentioned_counts: Dict[str, List[int]] = {}
        by_category: Dict[str, Dict[str, List[int]]] = {}
        by_category_mentioned: Dict[str, Dict[str, List[int]]] = {}
        missing_flight_ids: List[str] = []

        # Categories take precedence over flight filter for choosing which flights to aggregate
        use_flight_filter = False
        valid_rows_for_filter: List[int] = []
        if (not categories) and flight_ids:
            for fid in flight_ids:
                idx = self._flight_list.flight_id_to_row.get(str(fid))
                if idx is None:
                    missing_flight_ids.append(str(fid))
                else:
                    valid_rows_for_filter.append(int(idx))
            use_flight_filter = True

        rows_arr_for_overall: Optional[np.ndarray]
        if categories:
            # union of all category flights for overall/top-k ranking
            union_rows: List[int] = []
            seen_rows = set()
            for flist in categories.values():
                if not isinstance(flist, list):
                    continue
                for fid in flist:
                    idx = self._flight_list.flight_id_to_row.get(fid)
                    if idx is None:
                        missing_flight_ids.append(str(fid))
                    else:
                        ii = int(idx)
                        if ii not in seen_rows:
                            seen_rows.add(ii)
                            union_rows.append(ii)
            rows_arr_for_overall = np.asarray(union_rows, dtype=np.int32) if union_rows else np.asarray([], dtype=np.int32)
        elif use_flight_filter:
            rows_arr_for_overall = np.asarray(valid_rows_for_filter, dtype=np.int32) if valid_rows_for_filter else np.asarray([], dtype=np.int32)
        else:
            rows_arr_for_overall = None  # all flights

        # Build aggregated vector for the chosen flight set and reshape to (num_tvs, bins)
        num_total_tvs = len(tv_map)
        aggregated_vec = self._aggregate_vector_for_rows(rows_arr_for_overall)
        per_tv_matrix = aggregated_vec.reshape((num_total_tvs, self.bins_per_tv))

        # Rolling-hour window setup
        if rolling_hour:
            window_bins = int(np.ceil(60.0 / float(self.time_bin_minutes)))
            window_bins = max(1, window_bins)
            rolled_matrix = self._apply_rolling_hour(per_tv_matrix, window_bins)
        else:
            rolled_matrix = per_tv_matrix

        # Rank TVs by selected criterion over the requested time range
        slice_start = int(start_bin)
        slice_end_inclusive = int(end_bin)
        sliced = rolled_matrix[:, slice_start : slice_end_inclusive + 1]
        if rank_by != "total_count":
            # For now only total_count supported; fall back to total_count
            rank_by = "total_count"
        scores = sliced.sum(axis=1)
        # Top-K indices
        k = min(int(top_k), num_total_tvs)
        top_indices = np.argsort(-scores, kind="stable")[:k]

        # Build reverse map idx -> tv_id
        idx_to_tv = [None] * num_total_tvs
        for tv_id, idx in tv_map.items():
            idx_to_tv[int(idx)] = tv_id

        # Populate counts for top-k TVs
        for idx in top_indices:
            tv_id = idx_to_tv[int(idx)]
            vec = sliced[int(idx), :]
            counts[tv_id] = [int(x) for x in np.asarray(vec).ravel().tolist()]

        # If specific TVs were mentioned, also include them under mentioned_counts
        if traffic_volume_ids:
            for tv in traffic_volume_ids:
                row = int(tv_map[tv])
                vec = sliced[row, :]
                mentioned_counts[tv] = [int(x) for x in np.asarray(vec).ravel().tolist()]

        # Category-specific counts for the same set of TVs as in counts (top-k)
        flights_considered: int
        if categories and isinstance(categories, dict):
            considered_set = set()
            selected_tv_indices = [int(i) for i in top_indices]
            selected_tv_ids = [idx_to_tv[i] for i in selected_tv_indices]
            for cat_name, flist in categories.items():
                if not isinstance(flist, list):
                    continue
                rows: List[int] = []
                for fid in flist:
                    idx = self._flight_list.flight_id_to_row.get(fid)
                    if idx is None:
                        missing_flight_ids.append(str(fid))
                    else:
                        rows.append(int(idx))
                        considered_set.add(fid)

                cat_tv_counts: Dict[str, List[int]] = {}
                cat_mentioned_tv_counts: Dict[str, List[int]] = {}
                if rows:
                    rows_arr = np.asarray(rows, dtype=np.int32)
                    # aggregate for category across all TVs, then roll and slice once
                    cat_vec = self._aggregate_vector_for_rows(rows_arr)
                    cat_matrix = cat_vec.reshape((num_total_tvs, self.bins_per_tv))
                    if rolling_hour:
                        cat_matrix = self._apply_rolling_hour(cat_matrix, window_bins)
                    cat_sliced = cat_matrix[:, slice_start : slice_end_inclusive + 1]
                    for tv_id, tv_idx in zip(selected_tv_ids, selected_tv_indices):
                        vec = cat_sliced[int(tv_idx), :]
                        cat_tv_counts[tv_id] = [int(x) for x in np.asarray(vec).ravel().tolist()]
                    if traffic_volume_ids:
                        for tv in traffic_volume_ids:
                            row = int(tv_map[tv])
                            vec = cat_sliced[row, :]
                            cat_mentioned_tv_counts[tv] = [int(x) for x in np.asarray(vec).ravel().tolist()]
                else:
                    zeros = [0] * (int(end_bin) - int(start_bin) + 1)
                    for tv_id in selected_tv_ids:
                        cat_tv_counts[tv_id] = list(zeros)
                    if traffic_volume_ids:
                        for tv in traffic_volume_ids:
                            cat_mentioned_tv_counts[tv] = list(zeros)

                by_category[str(cat_name)] = cat_tv_counts
                if traffic_volume_ids:
                    by_category_mentioned[str(cat_name)] = cat_mentioned_tv_counts

            flights_considered = len(considered_set)
        else:
            if use_flight_filter:
                flights_considered = len(valid_rows_for_filter)
            else:
                flights_considered = int(self._flight_list.num_flights)

        # Build response
        # Build response
        ranked_tv_ids = [idx_to_tv[int(i)] for i in top_indices]
        resp: Dict[str, Any] = {
            "time_bin_minutes": int(self.time_bin_minutes),
            "timebins": {
                "start_bin": int(start_bin),
                "end_bin": int(end_bin),
                "labels": labels,
            },
            "counts": counts,
            "metadata": {
                "num_tvs": int(len(ranked_tv_ids)),
                "num_bins": int(end_bin) - int(start_bin) + 1,
                "total_flights_considered": int(flights_considered),
                "rank_by": str(rank_by),
                "top_k": int(top_k),
                "rolling_hour": bool(rolling_hour),
                "rolling_window_minutes": 60,
                "ranked_tv_ids": ranked_tv_ids,
            },
        }

        if traffic_volume_ids:
            resp["mentioned_counts"] = mentioned_counts
            resp["metadata"]["num_mentioned"] = int(len(traffic_volume_ids))

        # Attach by_category and/or missing flights info when present
        if by_category:
            # Deduplicate missing flight IDs, preserve order
            if missing_flight_ids:
                seen = set()
                unique_missing: List[str] = []
                for fid in missing_flight_ids:
                    if fid not in seen:
                        seen.add(fid)
                        unique_missing.append(fid)
                resp["metadata"]["missing_flight_ids"] = unique_missing
            resp["by_category"] = by_category
            if by_category_mentioned:
                resp["by_category_mentioned"] = by_category_mentioned

        # If flight filter used without categories, also include missing ids
        if (not by_category) and missing_flight_ids:
            seen = set()
            unique_missing: List[str] = []
            for fid in missing_flight_ids:
                if fid not in seen:
                    seen.add(fid)
                    unique_missing.append(fid)
            resp["metadata"]["missing_flight_ids"] = unique_missing

        return resp

    # ---------- Async facade ----------
    async def get_original_counts(
        self,
        *,
        traffic_volume_ids: Optional[List[str]] = None,
        from_time_str: Optional[str] = None,
        to_time_str: Optional[str] = None,
        categories: Optional[Dict[str, List[str]]] = None,
        flight_ids: Optional[List[str]] = None,
        include_overall: bool = True,
        rank_by: str = "total_count",
        rolling_hour: bool = True,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._compute_counts(
                traffic_volume_ids=traffic_volume_ids,
                from_time_str=from_time_str,
                to_time_str=to_time_str,
                categories=categories,
                flight_ids=flight_ids,
                include_overall=include_overall,
                rank_by=rank_by,
                rolling_hour=rolling_hour,
                top_k=top_k,
            ),
        )

    def __del__(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
