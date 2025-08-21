"""
Heuristic feature computation to guide candidate selection for Tabu search.

Features per flight:
- multiplicity score: number of overloaded (traffic_volume_id, hour) pairs intersected
- footprint set: traffic volumes visited by the flight
- slack valley score: measures how deep capacity slack valleys are around the
  flight's scheduled times in the TVs it traverses (higher is worse slack)

Also provides ranking helper to combine features with configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Set, Tuple, Iterable, Optional

import numpy as np

from ..eval.flight_list import FlightList
from ..eval.network_evaluator import NetworkEvaluator


@dataclass(frozen=True)
class FlightFeatureValues:
    """Container for precomputed per-flight features.

    Attributes
    ----------
    footprint_tv_ids: set of traffic volume ids traversed by this flight
    multiplicity_by_hour: count of overloaded (tv, hour) intersections
    slack_valley_score: positive number, larger means deeper shortage of slack around schedule
    slack_min_p5: the most severe (lowest) 5th percentile slack across TVs visited
    slack_mean_over_tvs: mean of per-TV mean slacks around schedule
    """

    footprint_tv_ids: Set[str]
    multiplicity_by_hour: int
    slack_valley_score: float
    slack_min_p5: float
    slack_mean_over_tvs: float


class FlightFeatures:
    """Compute and store heuristic features per flight for candidate selection.

    Notes
    -----
    - Uses hourly-based multiplicity (overloaded (tv, hour) pairs)
    - Uses evaluator's occupancy snapshot for slack computation; make sure to
      call with a consistent `NetworkEvaluator` state.
    """

    def __init__(
        self,
        flight_list: FlightList,
        evaluator: NetworkEvaluator,
        overload_threshold: float = 0.0,
        *,
        limit_to_flight_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.flight_list = flight_list
        self.evaluator = evaluator
        self.overload_threshold = float(overload_threshold)
        # If provided, compute features only for these flight ids
        self._flight_id_pool: Set[str] = (
            set(limit_to_flight_ids) if limit_to_flight_ids is not None else set(self.flight_list.flight_ids)
        )

        time_start = time.time()
        # Ensure evaluator caches are populated (hourly occupancy matrix)
        try:
            _ = self.evaluator.compute_excess_traffic_vector()
        except Exception:
            # If anything goes wrong, proceed; slack metrics will gracefully degrade
            pass
        time_end = time.time()
        print(f"Excess traffic vector computation took {time_end - time_start} seconds")

        # Precompute helpers
        self._bins_per_hour: int = 60 // int(self.flight_list.time_bin_minutes)
        self._num_tvtws: int = self.flight_list.num_tvtws
        self._num_tvs: int = len(self.flight_list.tv_id_to_idx)
        self._num_time_bins_per_tv: int = self._num_tvtws // max(1, self._num_tvs)

        # Reverse map for tv_idx -> tv_id
        self._idx_to_tv_id: Dict[int, str] = {
            idx: tv_id for tv_id, idx in self.flight_list.tv_id_to_idx.items()
        }

        # Per-flight schedule map: flight -> { tv_id -> set(hours) }
        self._hours_by_tv: Dict[str, Dict[str, Set[int]]] = self._build_hours_by_tv()

        # Overloaded set by (tv_id, hour)
        self._overloaded_by_hour: Set[Tuple[str, int]] = self._compute_overloaded_by_hour()

        # Precompute features per flight
        time_start = time.time()
        self._features: Dict[str, FlightFeatureValues] = self._compute_features_all()
        time_end = time.time()
        print(f"Features computation took {time_end - time_start} seconds")

    # -------------------------------------------------------
    # Core public API
    # -------------------------------------------------------
    def get(self, flight_id: str) -> FlightFeatureValues:
        return self._features[flight_id]

    def get_footprint(self, flight_id: str) -> Set[str]:
        return set(self._features[flight_id].footprint_tv_ids)

    def multiplicity(self, flight_id: str) -> int:
        return int(self._features[flight_id].multiplicity_by_hour)

    def slack_valley(self, flight_id: str) -> float:
        return float(self._features[flight_id].slack_valley_score)

    def compute_seed_footprint(self, seed_flight_ids: Iterable[str]) -> Set[str]:
        seed_set: Set[str] = set()
        for fid in seed_flight_ids:
            vals = self._features.get(fid)
            if vals is not None:
                seed_set.update(vals.footprint_tv_ids)
        return seed_set

    def rank_candidates(
        self,
        seed_footprint_tv_ids: Set[str],
        candidate_flight_ids: Optional[Iterable[str]] = None,
        *,
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """Rank candidate flights combining multiplicity/similarity/slack.

        Parameters
        ----------
        seed_footprint_tv_ids: set of TV ids that represent seed footprint
        candidate_flight_ids: iterable of flight ids to rank; if None, rank all
        weights: mapping with keys 'multiplicity', 'similarity', 'slack'
        normalize: if True, normalize raw components to [0, 1] across pool
        top_k: optionally return only the top K results

        Returns
        -------
        list of dicts with keys: flight_id, score, components
        """
        pool = (
            list(candidate_flight_ids)
            if candidate_flight_ids is not None
            else list(self._features.keys())
        )

        # Defaults favor multiplicity, then slack, then similarity
        w = {"multiplicity": 0.5, "similarity": 0.2, "slack": 0.3}
        if weights:
            w.update(weights)

        # Gather raw components
        raw_mult: List[float] = []
        raw_sim: List[float] = []
        raw_slack: List[float] = []
        for fid in pool:
            feats = self._features[fid]
            raw_mult.append(float(feats.multiplicity_by_hour))
            raw_slack.append(float(feats.slack_valley_score))
            raw_sim.append(
                _jaccard_similarity(feats.footprint_tv_ids, seed_footprint_tv_ids)
            )

        mult_arr = np.asarray(raw_mult, dtype=float)
        sim_arr = np.asarray(raw_sim, dtype=float)
        slack_arr = np.asarray(raw_slack, dtype=float)

        if normalize:
            mult_arr = _safe_minmax_norm(mult_arr)
            # similarity already ∈ [0,1]
            slack_arr = _safe_minmax_norm(slack_arr)

        scores = w["multiplicity"] * mult_arr + w["similarity"] * sim_arr + w["slack"] * slack_arr

        results: List[Dict[str, object]] = []
        for i, fid in enumerate(pool):
            results.append(
                {
                    "flight_id": fid,
                    "score": float(scores[i]),
                    "components": {
                        "multiplicity": float(mult_arr[i]),
                        "similarity": float(sim_arr[i]),
                        "slack": float(slack_arr[i]),
                    },
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        if top_k is not None and top_k > 0:
            return results[:top_k]
        return results

    # -------------------------------------------------------
    # Internal computations
    # -------------------------------------------------------
    def _build_hours_by_tv(self) -> Dict[str, Dict[str, Set[int]]]:
        """Map each flight to the set of (tv_id -> hours) it traverses.

        Hours are coarse-grained from the occupancy intervals' TVTW indices.
        """
        hours_by_tv: Dict[str, Dict[str, Set[int]]] = {}
        for flight_id in self._flight_id_pool:
            by_tv: Dict[str, Set[int]] = {}
            intervals = self.flight_list.flight_metadata[flight_id][
                "occupancy_intervals"
            ]
            for interval in intervals:
                tvtw_idx = int(interval["tvtw_index"])
                if tvtw_idx < 0 or tvtw_idx >= self._num_tvtws:
                    continue
                tv_row = tvtw_idx // max(1, self._num_time_bins_per_tv)
                time_idx = tvtw_idx % max(1, self._num_time_bins_per_tv)
                tv_id = self._idx_to_tv_id.get(tv_row)
                if tv_id is None:
                    continue
                hour = int(time_idx // max(1, self._bins_per_hour))
                by_tv.setdefault(tv_id, set()).add(hour)
            hours_by_tv[flight_id] = by_tv
        return hours_by_tv

    def _compute_overloaded_by_hour(self) -> Set[Tuple[str, int]]:
        per_hour = self.evaluator.get_hotspot_flights(
            threshold=self.overload_threshold, mode="hour"
        )
        overloaded: Set[Tuple[str, int]] = set()
        for item in per_hour:
            try:
                if bool(item.get("is_overloaded", False)):
                    tv_id = str(item.get("traffic_volume_id"))
                    hour = int(item.get("hour"))
                    overloaded.add((tv_id, hour))
            except Exception:
                continue
        return overloaded

    def _compute_features_all(self) -> Dict[str, FlightFeatureValues]:
        features: Dict[str, FlightFeatureValues] = {}
        # Pre-fetch evaluator caches
        hourly_occupancy = getattr(self.evaluator, "last_hourly_occupancy_matrix", None)
        capacity_by_tv = getattr(self.evaluator, "hourly_capacity_by_tv", {})
        tv_id_to_row = getattr(self.evaluator, "tv_id_to_row_idx", self.flight_list.tv_id_to_idx)

        # Build dense capacity matrix (rows: tv rows per evaluator mapping, cols: hours)
        cap_mat: Optional[np.ndarray] = None
        occ_mat: Optional[np.ndarray] = None
        num_rows: int = 0
        num_hours: int = 0
        if isinstance(hourly_occupancy, np.ndarray) and hourly_occupancy.size > 0:
            occ_mat = hourly_occupancy.astype(np.float32, copy=False)
            num_rows, num_hours = int(occ_mat.shape[0]), int(occ_mat.shape[1])
            cap_mat = np.zeros_like(occ_mat, dtype=np.float32)
            # Fill capacities from mapping {tv_id -> {hour -> cap}}
            for tv_id, per_hour in capacity_by_tv.items():
                row_idx = tv_id_to_row.get(tv_id)
                if row_idx is None or row_idx < 0 or row_idx >= num_rows:
                    continue
                if not isinstance(per_hour, dict):
                    continue
                for h, cap in per_hour.items():
                    try:
                        hh = int(h)
                        if 0 <= hh < num_hours:
                            cap_mat[row_idx, hh] = float(cap)
                    except Exception:
                        continue

        print(f'There are {len(self._hours_by_tv)} flights to compute features for')
        for fid, by_tv in self._hours_by_tv.items():
            footprint = set(by_tv.keys())

            # Multiplicity: count of overloaded (tv, hour) intersected
            mult = 0
            if self._overloaded_by_hour:
                for tv_id, hours in by_tv.items():
                    for h in hours:
                        if (tv_id, h) in self._overloaded_by_hour:
                            mult += 1

            # Slack valley metrics (vectorized grouped-indices approach)
            slack_min_p5 = float("inf")
            slack_mean_over_tvs = 0.0
            if occ_mat is not None and cap_mat is not None:
                row_idx_list: List[np.ndarray] = []
                col_idx_list: List[np.ndarray] = []
                grp_rows: List[np.ndarray] = []
                for tv_id, hours in by_tv.items():
                    row_idx = tv_id_to_row.get(tv_id)
                    if row_idx is None or row_idx < 0 or row_idx >= num_rows or not hours:
                        continue
                    try:
                        hs = np.fromiter((int(h) for h in hours), dtype=np.int32)
                    except Exception:
                        continue
                    if hs.size == 0:
                        continue
                    # Clip to valid hour range
                    hs = hs[(hs >= 0) & (hs < num_hours)]
                    if hs.size == 0:
                        continue
                    row_idx_list.append(np.full(hs.shape, int(row_idx), dtype=np.int32))
                    col_idx_list.append(hs)
                    grp_rows.append(np.full(hs.shape, int(row_idx), dtype=np.int32))

                if row_idx_list:
                    rows = np.concatenate(row_idx_list)
                    cols = np.concatenate(col_idx_list)
                    slacks = (cap_mat[rows, cols] - occ_mat[rows, cols]).astype(np.float32, copy=False)

                    tv_rows = np.concatenate(grp_rows)
                    order = np.argsort(tv_rows)
                    tv_rows = tv_rows[order]
                    slacks = slacks[order]

                    uniq, idx, counts = np.unique(tv_rows, return_index=True, return_counts=True)
                    sums = np.add.reduceat(slacks, idx)
                    per_tv_means_arr = sums / counts

                    # Compute min 5th percentile across TVs using NumPy percentile per group (exact behavior)
                    p5_min = np.inf
                    start = 0
                    for c in counts:
                        # segment for a single tv row
                        seg = slacks[start:start + c]
                        if seg.size == 0:
                            start += c
                            continue
                        # Match np.percentile default behavior for 1D arrays
                        try:
                            val = float(np.percentile(seg.astype(np.float64, copy=False), 5))
                        except Exception:
                            val = float(np.nan)
                        if val < p5_min:
                            p5_min = val
                        start += c

                    slack_min_p5 = 0.0 if not np.isfinite(p5_min) else float(p5_min)
                    slack_mean_over_tvs = float(per_tv_means_arr.mean()) if per_tv_means_arr.size else 0.0

            if slack_min_p5 == float("inf"):
                # No data available; default to neutral values
                slack_min_p5 = 0.0

            # Positive valley score: deeper negative slack -> higher score
            slack_valley_score = float(max(0.0, -slack_min_p5))

            features[fid] = FlightFeatureValues(
                footprint_tv_ids=footprint,
                multiplicity_by_hour=int(mult),
                slack_valley_score=slack_valley_score,
                slack_min_p5=float(slack_min_p5),
                slack_mean_over_tvs=float(slack_mean_over_tvs),
            )
        return features


def _safe_minmax_norm(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values)) if values.size > 0 else 0.0
    vmax = float(np.max(values)) if values.size > 0 else 1.0
    if vmax - vmin <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def _jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(inter) / float(union) if union > 0 else 0.0


