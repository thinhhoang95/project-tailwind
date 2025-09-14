from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Any
import numpy as np


def minutes_to_bin_offsets(
    travel_minutes: Mapping[str, Mapping[str, float]],
    time_bin_minutes: int,
) -> Dict[str, Dict[str, int]]:
    """
    Convert nested mapping of minutes[src][dst] -> float to integer bin offsets.
    Rounds to nearest integer number of bins using time_bin_minutes.
    """
    out: Dict[str, Dict[str, int]] = {}
    denom = max(1, int(time_bin_minutes))
    for src, rows in (travel_minutes or {}).items():
        d: Dict[str, int] = {}
        for dst, m in (rows or {}).items():
            try:
                bins = int(round(float(m) / float(denom)))
            except Exception:
                bins = 0
            d[str(dst)] = int(bins)
        out[str(src)] = d
    return out


def flow_offsets_from_ctrl(
    control_tv_id: Optional[str],
    tv_id_to_idx: Mapping[str, int],
    bin_offsets: Mapping[str, Mapping[str, int]],
    *,
    flow_flight_ids: Optional[Sequence[str]] = None,
    flight_list: Optional[Any] = None,
    hotspots: Optional[Sequence[str]] = None,
    trim_policy: Optional[str] = None,
    direction_sign_mode: Optional[str] = None,
    tv_centroids: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Optional[Dict[int, int]]:
    """
    Build τ_{G,s} for a flow controlled at `control_tv_id`.

    Returns mapping tv_row_index -> offset_bins from control to that TV. If
    control_tv_id is None or not present in bin_offsets, returns None.
    """
    if control_tv_id is None:
        return None
    row_map = {str(k): int(v) for k, v in tv_id_to_idx.items()}
    ctrl = str(control_tv_id)
    row_offsets = bin_offsets.get(ctrl)
    if row_offsets is None:
        return None
    # Build absolute magnitudes |τ| based on available offsets, with symmetry fallback
    abs_tau_by_row: Dict[int, int] = {}
    for tv, row in row_map.items():
        if tv == ctrl:
            abs_tau_by_row[int(row)] = 0
            continue
        off = row_offsets.get(tv)
        if off is None:
            # Symmetric fallback: use reverse entry if present
            rev = bin_offsets.get(str(tv), {}).get(ctrl)
            if rev is not None:
                off = int(rev)
        if off is not None:
            abs_tau_by_row[int(row)] = int(abs(int(off)))

    # Back-compat: if no flow context or signs disabled, return non-negative magnitudes
    if (
        not direction_sign_mode
        or direction_sign_mode in {"disabled", "none", "off"}
        or flow_flight_ids is None
        or flight_list is None
    ):
        if flow_flight_ids is None or flight_list is None:
            print(
                "[metaopt.flow_offsets_from_ctrl] No flow context provided; "
                "returning non-negative τ magnitudes (back-compat)."
            )
        return dict(abs_tau_by_row)

    # Infer per-TV signs relative to control using the requested mode, with fallback
    ctrl_row = row_map.get(ctrl)
    if ctrl_row is None:
        print(
            f"[metaopt.flow_offsets_from_ctrl] Control TV '{ctrl}' missing from row map; "
            "returning non-negative τ magnitudes."
        )
        return dict(abs_tau_by_row)

    # Normalize hotspot ids to row indices for optional trimming
    hotspot_rows: Optional[set[int]] = None
    if hotspots:
        hotspot_rows = set()
        for h in hotspots:
            r = row_map.get(str(h))
            if r is not None:
                hotspot_rows.add(int(r))
        if len(hotspot_rows) == 0:
            hotspot_rows = None

    def _infer_signs_order_vs_ctrl() -> Dict[int, int]:
        votes: Dict[int, int] = {}
        for fid in flow_flight_ids:
            try:
                seq = flight_list.get_flight_tv_sequence_indices(str(fid))  # np.ndarray of row indices
                if not isinstance(seq, np.ndarray):
                    seq = np.asarray(seq, dtype=np.int64)
            except Exception:
                continue
            if seq.size == 0:
                continue
            # Optional trim to earliest hotspot
            if trim_policy and isinstance(trim_policy, str) and trim_policy.lower().startswith("earliest_hotspot") and hotspot_rows:
                cut: Optional[int] = None
                for i, v in enumerate(seq.tolist()):
                    if int(v) in hotspot_rows:
                        cut = i
                        break
                if cut is not None:
                    seq = seq[: cut + 1]
                # If trim eliminates control later in sequence, that's acceptable: control must be within prefix to vote
            # First-visit positions
            pos: Dict[int, int] = {}
            for i, v in enumerate(seq.tolist()):
                vv = int(v)
                if vv not in pos:
                    pos[vv] = i
            if int(ctrl_row) not in pos:
                # If the control TV is not visited in this sequence, skip this flight for voting
                continue
            p_ctrl = int(pos[int(ctrl_row)])
            for tv_row, p_tv in pos.items():
                if int(tv_row) == int(ctrl_row):
                    continue
                votes[int(tv_row)] = votes.get(int(tv_row), 0) + (1 if int(p_tv) > p_ctrl else -1 if int(p_tv) < p_ctrl else 0)
        # Collapse to signs
        signs: Dict[int, int] = {}
        for tv_row, v in votes.items():
            if v > 0:
                signs[int(tv_row)] = 1
            elif v < 0:
                signs[int(tv_row)] = -1
            else:
                signs[int(tv_row)] = 0
        return signs

    def _infer_signs_vector_centroid() -> Dict[int, int]:
        signs: Dict[int, int] = {}
        if not tv_centroids:
            print(
                "[metaopt.flow_offsets_from_ctrl] vector_centroid mode requested but "
                "no tv_centroids provided; using 0 signs."
            )
            return signs
        # Average flight direction vector (first -> last) across flights
        acc = np.zeros(2, dtype=np.float64)
        cnt = 0
        for fid in flow_flight_ids:
            try:
                seq = flight_list.get_flight_tv_sequence_indices(str(fid))
                if not isinstance(seq, np.ndarray):
                    seq = np.asarray(seq, dtype=np.int64)
            except Exception:
                continue
            if seq.size < 2:
                continue
            first = int(seq[0])
            last = int(seq[-1])
            tv_first = flight_list.idx_to_tv_id.get(int(first)) if hasattr(flight_list, "idx_to_tv_id") else None
            tv_last = flight_list.idx_to_tv_id.get(int(last)) if hasattr(flight_list, "idx_to_tv_id") else None
            if tv_first is None or tv_last is None:
                continue
            c1 = tv_centroids.get(str(tv_first))
            c2 = tv_centroids.get(str(tv_last))
            if not c1 or not c2:
                continue
            v = np.asarray([float(c2[1]) - float(c1[1]), float(c2[0]) - float(c1[0])], dtype=np.float64)  # (dx, dy)
            n = np.linalg.norm(v)
            if n <= 0:
                continue
            acc += v / n
            cnt += 1
        if cnt == 0:
            print(
                "[metaopt.flow_offsets_from_ctrl] vector_centroid could not compute a mean flight direction; using 0 signs."
            )
            return signs
        mean_v = acc / float(max(1, cnt))
        # Sign per TV via dot product with (control -> TV)
        ctrl_id = str(control_tv_id)
        c_ctrl = tv_centroids.get(ctrl_id)
        if not c_ctrl:
            print(
                "[metaopt.flow_offsets_from_ctrl] vector_centroid missing centroid for control TV; using 0 signs."
            )
            return signs
        for tv, row in row_map.items():
            if str(tv) == ctrl_id:
                continue
            c_tv = tv_centroids.get(str(tv))
            if not c_tv:
                continue
            u = np.asarray([float(c_tv[1]) - float(c_ctrl[1]), float(c_tv[0]) - float(c_ctrl[0])], dtype=np.float64)
            n = np.linalg.norm(u)
            if n <= 0:
                continue
            u /= n
            dot = float(mean_v.dot(u))
            signs[int(row)] = 1 if dot >= 0.0 else -1
        return signs

    mode = str(direction_sign_mode or "order_vs_ctrl").lower()
    if mode not in {"order_vs_ctrl", "vector_centroid"}:
        # Interpret any other truthy value as enabling order_vs_ctrl
        mode = "order_vs_ctrl"

    signs_primary = _infer_signs_order_vs_ctrl() if mode == "order_vs_ctrl" else _infer_signs_vector_centroid()
    # Fallback: for TVs with zero/unknown sign, try the other method
    signs_secondary: Dict[int, int] = {}
    if mode == "order_vs_ctrl":
        # If few/no flights visit a TV or tie, geometric fallback
        if tv_centroids:
            signs_secondary = _infer_signs_vector_centroid()
            if signs_secondary:
                print(
                    "[metaopt.flow_offsets_from_ctrl] Using vector_centroid fallback for TVs with insufficient order evidence."
                )
    else:
        # If geometric unavailable or tie, try order-based
        signs_secondary = _infer_signs_order_vs_ctrl()

    out: Dict[int, int] = {}
    for row, mag in abs_tau_by_row.items():
        if int(row) == int(ctrl_row):
            out[int(row)] = 0
            continue
        s = signs_primary.get(int(row))
        if s is None or int(s) == 0:
            s = signs_secondary.get(int(row), 0)
        out[int(row)] = int(int(s) * int(mag))

    return out
