"""Rate allocation utilities."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .types import Bundle, RateCut


def compute_e_target(D_vec: Sequence[float], *, mode: str = "q95", fallback_to_peak: bool = True) -> float:
    if not D_vec:
        return 0.0
    arr = np.asarray(D_vec, dtype=np.float64)
    if mode == "q95":
        e_target = float(np.quantile(arr, 0.95))
    elif mode == "peak":
        e_target = float(arr.max()) if arr.size else 0.0
    else:
        raise ValueError(f"Unsupported target mode {mode}")
    if fallback_to_peak and e_target <= 0.0:
        e_target = float(arr.max()) if arr.size else 0.0
    return max(0.0, e_target)


def occupancy_to_entrance(occupancy_target: float, *, dwell_minutes: float | None, alpha: float) -> float:
    if dwell_minutes is not None and dwell_minutes > 0:
        return float(occupancy_target) * (60.0 / float(dwell_minutes))
    return float(occupancy_target) * float(alpha)


def rate_cuts_for_bundle(
    bundle: Bundle,
    *,
    E_target: float,
    bins_per_hour: int,
) -> List[RateCut]:
    if not bundle.flows:
        return []
    weights = dict(bundle.weights_by_flow)
    total_weight = sum(float(weights.get(fs.flow_id, 0.0)) for fs in bundle.flows)
    if total_weight <= 0.0:
        w_uniform = 1.0 / float(len(bundle.flows))
        weights = {fs.flow_id: w_uniform for fs in bundle.flows}
    cuts: List[RateCut] = []
    any_positive = False
    for fs in bundle.flows:
        diag = fs.diagnostics
        baseline = max(0, int(round(diag.r0_i)))
        weight = float(weights.get(fs.flow_id, 0.0))
        cut = int(round(weight * float(E_target)))
        cut = min(cut, baseline)
        if cut > 0 and baseline > 0:
            cut = max(1, cut)
        if cut > 0:
            any_positive = True
        allowed = max(0, baseline - cut)
        cuts.append(
            RateCut(
                flow_id=fs.flow_id,
                baseline_rate_r0=float(diag.r0_i),
                cut_per_hour_lambda=int(cut),
                allowed_rate_R=int(allowed),
            )
        )
    if not any_positive:
        return []
    return cuts


def distribute_hourly_rate_to_bins(rate_per_hour: int, *, bins_per_hour: int, start_bin: int, end_bin: int) -> np.ndarray:
    duration = max(0, int(end_bin) - int(start_bin) + 1)
    if duration <= 0:
        return np.zeros(0, dtype=np.int64)
    rate = max(0, int(rate_per_hour))
    q, r = divmod(rate, int(bins_per_hour))
    arr = np.full(duration, q, dtype=np.int64)
    for idx in range(duration):
        global_bin = int(start_bin) + idx
        pos_in_hour = global_bin % int(bins_per_hour)
        if pos_in_hour < r:
            arr[idx] += 1
    return arr
