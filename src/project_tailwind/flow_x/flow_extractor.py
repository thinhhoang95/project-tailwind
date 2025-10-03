from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import math
import numpy as np
import csv
import json
import heapq
from pathlib import Path
from rich.progress import Progress, TaskID


if TYPE_CHECKING:
    from project_tailwind.optimize.eval.flight_list import FlightList


class FlowXExtractor:
    """
    Find a spatially coherent group of flights feeding a hotspot using the
    spectral relaxation approach described in `prompts/flow_x/prompt_for_flow_x.md`.

    This implementation is designed to work directly with the structures returned by
    `NetworkEvaluator.get_hotspot_flights` (either mode="hour" or mode="bin").
    It uses TV (traffic volume) rows as the sector universe S, and per-time-bin
    sets of TV rows A[f][t] derived from the occupancy intervals in the FlightList
    metadata.

    High-level steps per the prompt:
    - Build candidate upstream references R from TVs encountered upstream of the hotspot
      among candidate flights.
    - For each r in R, align per-flight upstream sequences from r to H and build a
      weighted-Jaccard similarity matrix.
    - Apply a spectral relaxation (top eigenvector) and perform a threshold sweep on
      the induced order to select the best-scoring subset.
    - Return the best reference r* and its group.
    """

    def __init__(self, flight_list: "FlightList", debug_verbose_path: Optional[str] = "output/flow_extractor"):
        if TYPE_CHECKING:
            # Only for type checkers; at runtime we avoid heavy imports here
            from project_tailwind.optimize.eval.flight_list import FlightList as _FL  # noqa: F401

        # Duck-typing-friendly defensive check
        if not hasattr(flight_list, "time_bin_minutes") or not hasattr(flight_list, "tv_id_to_idx"):
            raise TypeError("flight_list must be a FlightList instance")

        self.flight_list = flight_list
        # Debug output directory (can be overridden per-call)
        self.debug_verbose_path: Optional[str] = debug_verbose_path

        # Grid constants
        self.time_bin_minutes: int = self.flight_list.time_bin_minutes
        self.bins_per_hour: int = 60 // self.time_bin_minutes
        self.num_tvtws: int = self.flight_list.num_tvtws
        self.num_tvs: int = len(self.flight_list.tv_id_to_idx)
        self.num_time_bins_per_tv: int = self.num_tvtws // self.num_tvs

        # tv_id <-> row mappings
        self.tv_id_to_row: Dict[str, int] = self.flight_list.tv_id_to_idx
        # Build inverse; initialize with None for safety
        self.row_to_tv_id: List[Optional[str]] = [None] * self.num_tvs
        for tv_id, row in self.tv_id_to_row.items():
            if 0 <= row < self.num_tvs:
                self.row_to_tv_id[row] = tv_id

    # ------------------------------- Utilities -------------------------------
    def _tvtw_to_tv_row_and_hour(self, tvtw_index: int) -> Tuple[int, int]:
        tv_row = int(tvtw_index) // self.num_time_bins_per_tv
        bin_offset = int(tvtw_index) % self.num_time_bins_per_tv
        hour = bin_offset // self.bins_per_hour
        return tv_row, hour

    def _tv_row_to_id(self, tv_row: int) -> str:
        tv_id = self.row_to_tv_id[int(tv_row)]
        if tv_id is None:
            raise ValueError(f"Unknown tv_row {tv_row}")
        return tv_id

    @staticmethod
    def _sanitize_for_filename(value: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(value))

    def _build_flight_timebin_tvsets(
        self, flight_ids: Sequence[str]
    ) -> Dict[str, List[set]]:
        """
        For each flight_id, build a list of length T=num_time_bins_per_tv where entry t is a set
        of TV rows occupied by the flight at time-of-day bin t.
        """
        T = self.num_time_bins_per_tv
        result: Dict[str, List[set]] = {}
        fm = self.flight_list.flight_metadata
        for fid in flight_ids:
            # Initialize per-time-bin sets lazily to avoid large allocations for empty flights
            tvsets: List[set] = [set() for _ in range(T)]
            for interval in fm[fid]["occupancy_intervals"]:
                tvtw_index = int(interval["tvtw_index"])
                tv_row = tvtw_index // T
                t = tvtw_index % T
                tvsets[t].add(tv_row)
            result[fid] = tvsets
        return result

    def _earliest_entry_in_hour(
        self, tvsets: List[set], hotspot_tv_row: int, hour: int
    ) -> Optional[int]:
        start = int(hour) * self.bins_per_hour
        end = start + self.bins_per_hour
        for t in range(start, min(end, self.num_time_bins_per_tv)):
            if hotspot_tv_row in tvsets[t]:
                return t
        return None

    # ------------------------- Weighted Jaccard & W --------------------------
    @staticmethod
    def _jaccard(X: Iterable[int], Y: Iterable[int], w: Dict[int, float]) -> float:
        if len(w) > 0:
            raise ValueError("Weighted Jaccard is not supported. It has been deprecated.")
        set_x = set(X)
        set_y = set(Y)
        if not set_x and not set_y:
            return 1.0
        inter = set_x & set_y
        union = set_x | set_y
        num = float(len(inter))
        den = float(len(union))
        return 0.0 if den <= 0.0 else num / den

    @staticmethod
    def _compute_sector_weights_from_frequency(
        sequences: Dict[str, List[set]]
    ) -> Dict[int, float]:
        """
        Option B from the prompt: down-weight large/ubiquitous sectors using
        w[s] = 1 / log(2 + freq[s]), where freq[s] counts appearances across all sequences.
        """
        freq: Dict[int, int] = {}
        for seq in sequences.values():
            for sset in seq:
                for s in sset:
                    freq[s] = freq.get(s, 0) + 1
        weights: Dict[int, float] = {}
        for s, c in freq.items():
            weights[s] = 1.0 / math.log(2.0 + float(c))
        return weights

    def _build_similarity_matrix(
        self,
        flights: List[str],
        A_r: Dict[str, List[set]],
        tau: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build the flight-by-flight similarity matrix using a time-agnostic
        weighted Jaccard similarity over the union of TVs encountered between
        the reference r and the hotspot H for each flight.

        If tau is provided, zero out entries below tau. If alpha is provided,
        apply adaptive sparsification: keep entries >= (mean - alpha * std).
        """
        idx = list(flights)
        m = len(idx)
        W = np.zeros((m, m), dtype=np.float32) # similarity matrix

        if m < 2:
            return W, idx

        # Unweighted Jaccard: no sector weights needed
        w = {}

        # Fill upper triangle
        for a in range(m):
            fa = idx[a]
            seq_a = A_r[fa]
            for b in range(a + 1, m):
                # if a == 1 and b == 2:
                #     print(f"🔍 Debugging flight {idx[a]} and {idx[b]}")
                fb = idx[b]
                seq_b = A_r[fb]
                # Time-agnostic comparison: union sets across the entire window
                Sa = set().union(*seq_a)
                Sb = set().union(*seq_b)
                s = self._jaccard(Sa, Sb, w)
                W[a, b] = s
                W[b, a] = s

        # Optional sparsification
        if tau is not None:
            W[W < float(tau)] = 0.0
        elif alpha is not None and m >= 2:
            # Compute mean/std over off-diagonals only
            triu_vals = W[np.triu_indices(m, k=1)]
            if triu_vals.size > 0:
                mu = float(np.mean(triu_vals))
                sd = float(np.std(triu_vals))
                thresh = mu - float(alpha) * sd
                W[W < thresh] = 0.0

        # Zero diagonal by definition
        np.fill_diagonal(W, 0.0)
        return W, idx

    # ---------------------------- Spectral + Sweep ----------------------------
    @staticmethod
    def _spectral_group(
        W: np.ndarray,
        lam: float = None,  # group size control parameter: higher lam means smaller groups
        normalize_by_degree: bool = False,
        average_objective: bool = True,
        k_max: Optional[int] = None,
        tv_id: Optional[str] = None,
        time_bin_start: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Spectral relaxation: take the top eigenvector of W (or D^{-1/2} W D^{-1/2}),
        order vertices by it, then perform a one-pass threshold sweep to pick a subset.
        Returns (indices_selected, best_score).
        """


        if average_objective:
            raise Exception("Average objective is not yet supported. It has been deprecated.")
        
        m = int(W.shape[0])
        if m == 0:
            return np.zeros((0,), dtype=np.int64), float("-inf")
        if m == 1:
            return np.array([0], dtype=np.int64), 0.0

        Wn = W
        if normalize_by_degree:
            d = np.maximum(W.sum(axis=1), 1e-8)
            Dm12 = 1.0 / np.sqrt(d)
            Wn = (Dm12[:, None] * W) * Dm12[None, :]

        # Top eigenvector via eigh (dense, symmetric)
        try:
            vals, vecs = np.linalg.eigh(Wn)
            v = vecs[:, -1]
        except np.linalg.LinAlgError:
            raise Exception("Singular matrix")
            # Fallback: use power iteration
            v = np.random.default_rng(0).normal(size=m).astype(np.float64)
            v /= np.linalg.norm(v) + 1e-12
            for _ in range(50):
                v = Wn @ v
                nrm = np.linalg.norm(v)
                if nrm <= 1e-12:
                    break
                v /= nrm

        order = np.argsort(-v)

        in_set = np.zeros(m, dtype=bool)
        pair_sum = 0.0
        best_score = -1e18
        best_k = 0
        kcap = m if k_max is None else min(int(k_max), m)

        # SWEEPING THE NUMBER OF MEMBERS IN THE COHERENT GROUP
        # The idea is that the eigenvector is a continuous value vector, and we need a binary vector to know which flights are in the group.
        # It iterates through group sizes, k, from 1 up to kcap.
        # For each k, it considers a candidate group made up of the top k flights from the ordered list (by the magnitude from the eigenvector)
        # It then calculates the objective score for that entire candidate group.
        # It compares this score to the best_score seen so far. If the current group's score is better, it updates best_score and remembers the current size k as the new best_k.

        debug_info = []
        for k in range(1, kcap + 1):  # k: group size
            i = int(order[k - 1])
            # sum weights to current set
            if in_set.any():
                pair_sum += float(W[i, in_set].sum())
            in_set[i] = True

            score = float("nan")
            if k >= 2:
                if average_objective:
                    denom = (k * (k - 1)) / 2.0
                    score = pair_sum / denom if denom > 0 else 0.0
                else:
                    score = pair_sum - float(
                        lam
                    ) * k  # k controls the size of the group: higher lam means smaller groups
                if score > best_score:
                    best_score = score
                    best_k = k
            
            if log_file and k <= 10:
                debug_info.append({"k": k, "pair_sum": pair_sum, "score": score})


        if log_file and tv_id and time_bin_start:
            with open(log_file, "a") as f:
                f.write(
                    f"--- Processing traffic_volume_id: {tv_id}, time_bin: {time_bin_start}, kcap: {kcap}, m = {m}, matrix_size: {W.shape} ---\n"
                )
                if m > 1:
                    upper_tri_indices = np.triu_indices(m, k=1)
                    top_10_sim = sorted(W[upper_tri_indices], reverse=True)[:10]
                    f.write(f"Top 10 similarity values in W: {top_10_sim}\n")

                for info in debug_info:
                    f.write(
                        f"k={info['k']}, pair_sum={info['pair_sum']:.4f}, score={info['score']:.4f}\n"
                    )
                
                f.write(f"Best k: {best_k}, Best score: {best_score}\n")
                f.write(
                    f"--- End of processing for {tv_id}, {time_bin_start} ---\n\n"
                )

        if best_k < 2:
            best_k = min(2, m)
        selected = order[:best_k]
        return selected, float(best_score)

    # ---------------------------- Alignment & R set ---------------------------
    def _collect_candidate_references(
        self,
        flights: Sequence[str],
        A: Dict[str, List[set]],
        H_tv_row: int,
        H_entry_by_flight: Dict[str, int],
        min_flights_per_ref: int = 3,
        max_references: Optional[int] = None,
        include_hotspot_bin_in_refs: bool = True
    ) -> List[int]:
        """
        Collect TVs that appear upstream of H for the candidate flights, excluding H itself.
        Filter to those observed for at least `min_flights_per_ref` distinct flights, and
        optionally keep at most `max_references` by descending frequency.
        """
        counts: Dict[int, int] = {}
        for fid in flights:
            tH = H_entry_by_flight.get(fid)
            if tH is None:
                continue
            tvs_seen: set = set()
            seq = A[fid]
            # Include tH to capture upstream TVs that may quantize into the hotspot bin
            for t in range(0, max(0, tH + 1 if include_hotspot_bin_in_refs else tH)):
                tvs_seen.update(seq[t])
            tvs_seen.discard(H_tv_row)
            for s in tvs_seen:
                counts[s] = counts.get(s, 0) + 1

        # Filter and sort
        refs = [s for s, c in counts.items() if c >= int(min_flights_per_ref)]
        refs.sort(key=lambda s: counts[s], reverse=True)
        if max_references is not None and len(refs) > int(max_references):
            refs = refs[: int(max_references)]
        return refs

    def _align_from_reference_to_H(
        self,
        r_tv_row: int,
        H_tv_row: int,
        flights: Sequence[str],
        A: Dict[str, List[set]],
        H_entry_by_flight: Dict[str, int],
        min_aligned_bins: int = 2
    ) -> Tuple[List[str], Dict[str, List[set]]]:
        """
        For each flight, if it passes r before H, build the aligned sequence from r to H-1.
        Returns (eligible_flights, aligned_sequences).
        """
        eligible: List[str] = []
        A_r: Dict[str, List[set]] = {}
        for fid in flights:
            # print(f"  ✈️  Flight {fid}")
            # Debugging info:
            # fid: flight identifier
            tH = H_entry_by_flight.get(fid)
            if tH is None or tH <= 0:
                continue
            seq = A[fid]

            # flattened_seq = [tv for tv_set in seq for tv in tv_set]
            
            # tv_to_lookfor = 'MASB5WL'
            # hotspot_tv = 'MASB5KL'
            # hotspot_tv_row = self.tv_id_to_row[hotspot_tv]
            # tv_to_lookfor_row = self.tv_id_to_row[tv_to_lookfor]

            # first_tv_to_lookfor_idx = None
            # hotspot_tv_idx = None
            # for t, tv_set in enumerate(seq):
            #     if tv_to_lookfor_row in tv_set and first_tv_to_lookfor_idx is None:
            #         first_tv_to_lookfor_idx = t
            #     if hotspot_tv_row in tv_set:
            #         hotspot_tv_idx = t
            #         break

            # assert first_tv_to_lookfor_idx <= hotspot_tv_idx, f"First {tv_to_lookfor} at index {first_tv_to_lookfor_idx} must be before Hotspot {hotspot_tv} at index {hotspot_tv_idx}"


            # if tv_to_lookfor_row in flattened_seq: # for debugging, we only print out the sequence if it contains MASB5WL

            #     for tv_set in seq:
            #         tv_ids = ''
            #         if tv_set:
            #             tv_ids = ' '.join(self._tv_row_to_id(tv) for tv in tv_set)
            #         if tv_ids != '':
            #             print(f"📍{tv_ids}")
            
            # Find earliest entry at/after r prior to H
            t_r: Optional[int] = None
            # Allow r to occur at tH due to bin quantization
            for t in range(0, tH + 1):
            # for t in range(tH, -1, -1):
                if r_tv_row in seq[t]:
                    t_r = t
                    break
            if t_r is None:
                continue
            # Only proceed if tH - t_r >= min_aligned_bins (i.e., the aligned sequence has at least min_aligned_bins bins)
            if tH - t_r < min_aligned_bins:
                # print(f"❌ Skipping due to not enough bins: tH - t_r = {tH - t_r} (required: {min_aligned_bins})")
                continue

            # Build aligned sequence; allow r at tH by taking the tH bin minus H
            if t_r < tH:
                # Slice [t_r, tH)
                aligned = seq[t_r:tH]
            else:
                # t_r == tH: use a single-bin sequence at tH without the hotspot TV
                last = set(seq[tH])
                last.discard(H_tv_row)
                if not last:
                    continue
                aligned = [last]
            if len(aligned) < min_aligned_bins:
                continue
            eligible.append(fid)
            A_r[fid] = aligned

            # print(f"✅ Admitted as eligible flight")


        return eligible, A_r

    # ---------------------------- File Writing Utilities ---------------------
    def _write_footprint_csv(
        self,
        debug_dir: Path,
        ref_tv_id: str,
        hotspot_tv_id: str,
        hotspot_hour: int,
        candidate_flight_ids: Sequence[str],
        A: Dict[str, List[set]]
    ) -> None:
        safe_ref = self._sanitize_for_filename(ref_tv_id)
        safe_hotspot_tv = self._sanitize_for_filename(hotspot_tv_id)
        footprint_path = debug_dir / f"footprint_r_{safe_ref}_tv_{safe_hotspot_tv}_hour_{int(hotspot_hour)}.csv"
        
        with footprint_path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "hotspot_tv_id",
                "hotspot_hour",
                "reference_tv_id",
                "flight_id",
                "time_bin",
                "tv_ids",
            ])
            for fid in candidate_flight_ids:
                seq = A.get(fid, [])
                for t, sset in enumerate(seq):
                    if not sset:
                        continue
                    tv_ids = sorted(self._tv_row_to_id(s) for s in sset)
                    writer.writerow([
                        hotspot_tv_id,
                        int(hotspot_hour),
                        ref_tv_id,
                        fid,
                        t,
                        ";".join(tv_ids),
                    ])

    def _write_similarity_csv(
        self,
        debug_dir: Path,
        ref_tv_id: str,
        hotspot_tv_id: str,
        hotspot_hour: int,
        W: np.ndarray,
        idx: List[str]
    ) -> None:
        if W.size > 0 and len(idx) > 0:
            safe_ref = self._sanitize_for_filename(ref_tv_id)
            safe_hotspot_tv = self._sanitize_for_filename(hotspot_tv_id)
            sim_path = debug_dir / f"similarity_r_{safe_ref}_tv_{safe_hotspot_tv}_hour_{int(hotspot_hour)}.csv"
            
            with sim_path.open("w", newline="") as sp:
                writer = csv.writer(sp)
                header = ["hotspot_tv_id", "hotspot_hour", "reference_tv_id", "flight_id"] + list(idx)
                writer.writerow(header)
                for a, fa in enumerate(idx):
                    row_vals = [hotspot_tv_id, int(hotspot_hour), ref_tv_id, fa]
                    row_vals.extend([float(W[a, b]) for b in range(len(idx))])
                    writer.writerow(row_vals)

    def _write_txt_log(
        self,
        txt_log_path: Path,
        r_tv_row: int,
        group_flights: List[str],
        score_adjusted: float,
        avg_sim: float,
        mean_path_len: float,
        hotspot_tv_id: str,
        hotspot_hour: int
    ) -> None:
        row_data = {
            "reference_sector": self._tv_row_to_id(r_tv_row),
            "group_flights": group_flights,
            "score": float(score_adjusted),
            "avg_pairwise_similarity": float(avg_sim),
            "group_size": int(len(group_flights)),
            "mean_path_length": float(mean_path_len),
            "hotspot": {"traffic_volume_id": hotspot_tv_id, "hour": int(hotspot_hour)},
        }
        with txt_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row_data, ensure_ascii=False) + "\n")

    # ---------------------------- Public API methods -------------------------
    def find_group_from_hotspot_hour(
        self,
        hotspot_tv_id: str,
        hotspot_hour: int,
        candidate_flight_ids: Sequence[str],
        *,
        auto_collapse_group_output: bool = True,
        min_flights_per_ref: int = 3,
        max_references: Optional[int] = 20,
        tau: Optional[float] = None,
        alpha_sparsification: Optional[float] = 0.0, # adaptive sparsification parameter
        group_size_lam: float = None,  # spectral relaxation parameter, required if average_objective is False
        normalize_by_degree: bool = False,
        average_objective: bool = True, 
        k_max_trajectories_per_group: Optional[int] = None,
        max_groups: int = None,
        path_length_gamma: float = None, # higher gamma prefers longer paths
        debug_verbose_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Core driver: align upstream sequences from each candidate reference r to hotspot H,
        build similarity matrices, run spectral relaxation + sweep, and pick the best group.
        Returns a dict with keys: reference_sector, group_flights, score, avg_pairwise_similarity,
        group_size, and hotspot info.
        """
        groups = self.find_groups_from_hotspot_hour(
            hotspot_tv_id,
            int(hotspot_hour),
            candidate_flight_ids,
            auto_collapse_group_output=auto_collapse_group_output,
            min_flights_per_ref=min_flights_per_ref,
            max_references=max_references,
            tau=tau,
            sparsification_alpha=alpha_sparsification,
            group_size_lam=group_size_lam,
            normalize_by_degree=normalize_by_degree,
            average_objective=average_objective,
            k_max_trajectories_per_group=k_max_trajectories_per_group,
            max_groups=max_groups,
            path_length_gamma=path_length_gamma,
            debug_verbose_path=debug_verbose_path,
        )
        if groups:
            return groups
        return {
            "reference_sector": None,
            "group_flights": [],
            "score": float("-inf"),
            "avg_pairwise_similarity": 0.0,
            "group_size": 0,
            "hotspot": {"traffic_volume_id": hotspot_tv_id, "hour": int(hotspot_hour)},
        }

    def find_groups_from_hotspot_hour(
        self,
        hotspot_tv_id: str,
        hotspot_hour: int,
        candidate_flight_ids: Sequence[str],
        *,
        auto_collapse_group_output: bool = True,
        min_flights_per_ref: int = 3,
        max_references: Optional[int] = 500,
        tau: Optional[float] = None,
        sparsification_alpha: Optional[float] = 0.0, # adaptive sparsification parameter
        group_size_lam: float = None, # group size control parameter: higher lam means smaller groups
        normalize_by_degree: bool = False,
        average_objective: bool = True,
        k_max_trajectories_per_group: Optional[int] = None,
        max_groups: int = 3,
        min_group_size: int = 2,
        # New knobs: (A) path-length reward
        path_length_gamma: float = None, # higher gamma prefers longer paths
        debug_verbose_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ranking procedure: evaluate grouping solutions for all candidate reference sectors
        using the full, consistent flight list; keep only the top `max_groups` by score.
        Returns a list of up to `max_groups` group dicts with `group_rank` added.
        """

        if not candidate_flight_ids:
            return []

        if hotspot_tv_id not in self.tv_id_to_row:
            raise ValueError(f"Unknown traffic_volume_id: {hotspot_tv_id}")

        H_tv_row = self.tv_id_to_row[hotspot_tv_id]

        # --------- DEBUG CSV OUTPUTS ---------
        debug_dir = None
        txt_log_dir = None
        txt_log_path = None

        # Resolve/debug directory
        # debug_dir = Path(debug_verbose_path or self.debug_verbose_path or "output/flow_extractor")
        # try:
        #     debug_dir.mkdir(parents=True, exist_ok=True)
        # except Exception:
        #     debug_dir = None  # type: ignore[assignment]

        # # Prepare exhaustive TXT logging directory and file path
        # safe_hotspot_tv_for_log = self._sanitize_for_filename(hotspot_tv_id)
        # txt_log_dir = Path("output/flow_extraction")
        # try:
        #     txt_log_dir.mkdir(parents=True, exist_ok=True)
        # except Exception:
        #     txt_log_dir = None  # type: ignore[assignment]
        # txt_log_path = (
        #     (txt_log_dir / f"flow_groups_{safe_hotspot_tv_for_log}_hour_{int(hotspot_hour)}.txt")
        #     if txt_log_dir is not None
        #     else None
        # )

        if len(candidate_flight_ids) < max(3, min_group_size):
            return []

        # Build per-flight TV sets across the full candidate list (fixed universe)
        A = self._build_flight_timebin_tvsets(candidate_flight_ids)

        # Compute hotspot entry times for the full list
        H_entry: Dict[str, int] = {}
        for fid in candidate_flight_ids:
            t = self._earliest_entry_in_hour(A[fid], H_tv_row, int(hotspot_hour))
            if t is not None:
                H_entry[fid] = int(t)
        if len(H_entry) < max(3, min_group_size):
            return []

        # Collect candidate reference upstream traffic volumes once for the full list
        R = self._collect_candidate_references(
            candidate_flight_ids,
            A,
            H_tv_row,
            H_entry,
            min_flights_per_ref=min_flights_per_ref,
            max_references=max_references,
        )
        if not R:
            return []

        # Maintain only the top-N groups via a min-heap of (score, counter, group_dict)
        top_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        seq_counter = 0
        num_references_scanned = 0

        with Progress() as progress:
            task = progress.add_task(
                f"[green]Scanning reference sectors for hotspot {hotspot_tv_id}...", 
                total=len(R)
            )
            
            for r_tv_row in R:
                num_references_scanned += 1
                progress.update(task, advance=1, description=f"[green]Processing reference {self._tv_row_to_id(r_tv_row)}...")
                
                # print("--------------------------------")
                # print(f"🪂 Debugging reference TV: {self._tv_row_to_id(r_tv_row)} as row {r_tv_row}")

                # Align sequences from r to H over the full, fixed set
                # Iterates through flights: The method loops through each flight in candidate_flight_ids.
                # Checks for hotspot entry: It first checks if a flight enters the hotspot H_tv_row by looking up its entry time in the H_entry dictionary.
                # Finds reference point: If the flight enters the hotspot, the code then looks for the time it passed through the reference traffic volume r_tv_row, ensuring this happens on or before it reaches the hotspot.
                # Filters by duration: The method makes sure there's a minimum number of time steps between passing the reference point and reaching the hotspot.
                # Extracts trajectory: For flights that meet all these criteria, it extracts the portion of their trajectory between the reference point and the hotspot.
                # Returns results: The method returns two things:
                # flights_r: A list of the flight IDs that met the criteria.
                # A_r: A dictionary where keys are the eligible flight IDs and values are their extracted trajectory segments.
                flights_r, A_r = self._align_from_reference_to_H(
                    r_tv_row, H_tv_row, candidate_flight_ids, A, H_entry
                )
                if len(flights_r) < max(3, min_group_size):
                    # print(f"❌ Skipping due to not enough flights: {len(flights_r)} (required: {max(3, min_group_size)})")
                    continue
                else:
                    # print(f"✅ Found {len(flights_r)} flights that pass reference sector {self._tv_row_to_id(r_tv_row)} before hotspot. Will build similarity matrix for these flights.")
                    pass

                W, idx = self._build_similarity_matrix(
                    flights_r, A_r, tau=tau, alpha=sparsification_alpha if sparsification_alpha and sparsification_alpha > 0 else None
                )

                # print(f"✅ Similarity matrix built. Size: {W.shape}")

                # --------- DEBUG CSV OUTPUTS ---------
                try:
                    if debug_dir is not None:
                        ref_tv_id = self._tv_row_to_id(r_tv_row)
                        # 1) Traffic volume footprint from A (TV IDs, not indices)
                        self._write_footprint_csv(debug_dir, ref_tv_id, hotspot_tv_id, hotspot_hour, candidate_flight_ids, A)
                        # 2) Similarity matrix W for current reference
                        self._write_similarity_csv(debug_dir, ref_tv_id, hotspot_tv_id, hotspot_hour, W, idx)
                except Exception:
                    pass

                if W.size == 0:
                    continue

                selected_idx, score = self._spectral_group(
                    W,
                    lam=float(group_size_lam),
                    normalize_by_degree=bool(normalize_by_degree),
                    average_objective=bool(average_objective),
                    k_max=k_max_trajectories_per_group,
                    tv_id=hotspot_tv_id,
                    time_bin_start=str(hotspot_hour),
                    log_file="spectral_group_debug.txt",
                )
                if len(selected_idx) < W.shape[0]:
                    # print(f"✅ Looks like what it should be: selected {len(selected_idx)} < {W.shape[0]}")
                    pass
                else:
                    # print(f"❌ Looks like what it should not be: selected {len(selected_idx)} = {W.shape[0]}")
                    pass

                # Compute metrics and decide whether to keep
                group_flights = [idx[i] for i in selected_idx.tolist()]
                k = len(selected_idx)
                if k >= 2:
                    tri = W[np.ix_(selected_idx, selected_idx)]
                    pair_sum = float(np.triu(tri, k=1).sum())
                    avg_sim = pair_sum / (k * (k - 1) / 2.0)
                else:
                    avg_sim = 0.0

                mean_path_len = float(np.mean([len(A_r[fid]) for fid in group_flights])) if group_flights else 0.0
                score_adjusted = float(score) + float(path_length_gamma) * float(mean_path_len)

                # Log every evaluated state to TXT (JSON lines)
                try:
                    if txt_log_path is not None:
                        self._write_txt_log(txt_log_path, r_tv_row, group_flights, score_adjusted, avg_sim, mean_path_len, hotspot_tv_id, hotspot_hour)
                except Exception:
                    pass

                # Enforce minimum group size for ranking
                if selected_idx.size < min_group_size:
                    continue

                # Insert into top-N heap
                group_item: Dict[str, Any] = {
                    "reference_sector": self._tv_row_to_id(r_tv_row),
                    "group_flights": group_flights,
                    "score": float(score_adjusted),
                    "avg_pairwise_similarity": float(avg_sim),
                    "group_size": int(len(group_flights)),
                    "mean_path_length": float(mean_path_len),
                    "hotspot": {"traffic_volume_id": hotspot_tv_id, "hour": int(hotspot_hour)},
                }

                if len(top_heap) < int(max_groups):
                    heapq.heappush(top_heap, (float(score_adjusted), seq_counter, group_item))
                else:
                    if top_heap and float(score_adjusted) > float(top_heap[0][0]):
                        heapq.heapreplace(top_heap, (float(score_adjusted), seq_counter, group_item))
                seq_counter += 1

        print(f"✔️ Scanned {num_references_scanned} reference sectors per {len(R)} reference sectors")

        if not top_heap:
            return []

        # Extract and sort by descending score
        ordered = sorted(top_heap, key=lambda x: (-x[0], x[1]))

        # Optionally collapse duplicate groups by their flight membership, keeping highest score
        if bool(auto_collapse_group_output):
            seen_keys: set = set()
            collapsed: List[Dict[str, Any]] = []
            for _, __, grp in ordered:
                flights_key = frozenset(grp.get("group_flights", []))
                if flights_key in seen_keys:
                    continue
                seen_keys.add(flights_key)
                collapsed.append(grp)
            # Re-assign ranks after collapsing
            for rank, grp in enumerate(collapsed, start=1):
                grp["group_rank"] = rank
            return collapsed

        # Default: no collapsing, just assign ranks
        results: List[Dict[str, Any]] = []
        for rank, (_, __, grp) in enumerate(ordered, start=1):
            grp["group_rank"] = rank
            results.append(grp)
        return results

    def find_group_from_hotspot_bin(
        self,
        hotspot_tvtw_index: int,
        candidate_flight_ids: Sequence[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper for TVTW-index hotspot; hour is derived from the index.
        """
        tv_row, hour = self._tvtw_to_tv_row_and_hour(int(hotspot_tvtw_index))
        tv_id = self._tv_row_to_id(tv_row)
        return self.find_group_from_hotspot_hour(tv_id, int(hour), candidate_flight_ids, **kwargs)

    def find_groups_from_hotspot_bin(
        self,
        hotspot_tvtw_index: int,
        candidate_flight_ids: Sequence[str],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        tv_row, hour = self._tvtw_to_tv_row_and_hour(int(hotspot_tvtw_index))
        tv_id = self._tv_row_to_id(tv_row)
        return self.find_groups_from_hotspot_hour(tv_id, int(hour), candidate_flight_ids, **kwargs)

    def find_group_from_evaluator_item(
        self, hotspot_item: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Accepts one item returned from NetworkEvaluator.get_hotspot_flights(...):
        - If it contains {"traffic_volume_id", "hour", "flight_ids"}, uses hour-mode.
        - If it contains {"tvtw_index", "flight_ids"}, uses bin-mode.
        Returns a dict with the best reference and group.
        """
        if not hotspot_item or "flight_ids" not in hotspot_item:
            return {
                "reference_sector": None,
                "group_flights": [],
                "score": float("-inf"),
                "avg_pairwise_similarity": 0.0,
                "group_size": 0,
                "hotspot": None,
            }

        flight_ids = hotspot_item.get("flight_ids", [])
        if "traffic_volume_id" in hotspot_item and "hour" in hotspot_item:
            return self.find_group_from_hotspot_hour(
                hotspot_item["traffic_volume_id"],
                int(hotspot_item["hour"]),
                flight_ids,
                **kwargs,
            )

        if "tvtw_index" in hotspot_item:
            return self.find_group_from_hotspot_bin(
                int(hotspot_item["tvtw_index"]),
                flight_ids,
                **kwargs,
            )

        raise ValueError("Unsupported hotspot item format")

    def find_groups_from_evaluator_item(
        self, hotspot_item: Dict[str, Any], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Multi-group variant of find_group_from_evaluator_item. Returns a list of groups
        with at most `max_groups` items when provided in kwargs.
        """
        if not hotspot_item or "flight_ids" not in hotspot_item:
            return []
        flight_ids = hotspot_item.get("flight_ids", [])
        if "traffic_volume_id" in hotspot_item and "hour" in hotspot_item:
            return self.find_groups_from_hotspot_hour(
                hotspot_item["traffic_volume_id"],
                int(hotspot_item["hour"]),
                flight_ids,
                **kwargs,
            )
        if "tvtw_index" in hotspot_item:
            return self.find_groups_from_hotspot_bin(
                int(hotspot_item["tvtw_index"]),
                flight_ids,
                **kwargs,
            )
        raise ValueError("Unsupported hotspot item format")


