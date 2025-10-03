"""Local search over rate cuts."""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

from .types import RateCut


def _clone_with_cut(rate: RateCut, new_cut: int) -> RateCut:
    baseline_int = max(0, int(round(rate.baseline_rate_r0)))
    cut = max(0, min(int(new_cut), baseline_int))
    allowed = max(0, baseline_int - cut)
    return RateCut(
        flow_id=rate.flow_id,
        baseline_rate_r0=rate.baseline_rate_r0,
        cut_per_hour_lambda=int(cut),
        allowed_rate_R=int(allowed),
    )


def local_search_variants(
    bundle,
    base_cuts: Sequence[RateCut],
    *,
    steps: Sequence[int] = (-2, -1, 1, 2),
    max_variants: int = 8,
    use_percent: bool = False,
    percent_lower: float = 0.05,
    percent_upper: float = 0.50,
    percent_step: float = 0.05,
) -> List[Sequence[RateCut]]:
    """Generate nearby rate variants around the base allocation.

    When ``use_percent`` is False (default), explore small integer deltas on
    ``cut_per_hour_lambda`` like before.

    When ``use_percent`` is True, explore adjustments to the allowed rate ``R``
    around the target allowed rate from ``base_cuts`` using +/- percentage of
    the original (baseline) rate ``r0``. Percent inputs may be provided either
    as fractions (e.g., 0.05) or whole percentages (e.g., 5 for 5%).
    """

    variants: List[Sequence[RateCut]] = []
    seen: set[Tuple[int, ...]] = set()

    def add_variant(cuts: Sequence[RateCut]) -> None:
        key = tuple(rc.cut_per_hour_lambda for rc in cuts)
        if key in seen:
            return
        seen.add(key)
        variants.append(cuts)

    add_variant(tuple(base_cuts))
    if len(variants) >= max_variants:
        return variants

    def _build_percent_list(lower: float, upper: float, step: float) -> List[float]:
        # Normalize percentage inputs: accept 5 or 0.05 as "5%"
        def _norm(x: float) -> float:
            if x is None:
                return 0.0
            val = float(x)
            if val < 0:
                val = -val
            if val > 1.0:
                # Treat values in [1, 100] as percent
                val = val / 100.0
            return val

        lo = _norm(lower)
        hi = _norm(upper)
        if hi < lo:
            lo, hi = hi, lo
        st = _norm(step)
        if st <= 0.0:
            st = hi  # single step at upper bound
        # Build inclusive [lo, hi] with stable rounding
        count = max(1, int((hi - lo) / st + 1e-9) + 1)
        vals = [min(hi, lo + i * st) for i in range(count)]
        # Ensure uniqueness and monotonicity
        dedup: List[float] = []
        for v in vals:
            if not dedup or abs(v - dedup[-1]) > 1e-9:
                dedup.append(v)
        return dedup

    # Independent tweaks per flow
    if not use_percent:
        for idx, rate in enumerate(base_cuts):
            baseline_int = max(0, int(round(rate.baseline_rate_r0)))
            for step in steps:
                if step == 0:
                    continue
                new_cut = max(0, min(rate.cut_per_hour_lambda + int(step), baseline_int))
                cuts = list(base_cuts)
                cuts[idx] = _clone_with_cut(rate, new_cut)
                add_variant(tuple(cuts))
                if len(variants) >= max_variants:
                    return variants
    else:
        percents = _build_percent_list(percent_lower, percent_upper, percent_step)
        for idx, rate in enumerate(base_cuts):
            baseline_int = max(0, int(round(rate.baseline_rate_r0)))
            if baseline_int <= 0:
                continue
            base_allowed = max(0, min(int(rate.allowed_rate_R), baseline_int))
            for p in percents:
                # symmetric +/- adjustments around target allowed rate
                delta = int(round(p * baseline_int))
                if delta <= 0 and p > 0.0:
                    delta = 1  # ensure at least a 1-flight/h change when baseline exists
                for sign in (-1, 1):
                    new_allowed = max(0, min(base_allowed + sign * delta, baseline_int))
                    new_cut = max(0, min(baseline_int - new_allowed, baseline_int))
                    cuts = list(base_cuts)
                    cuts[idx] = _clone_with_cut(rate, new_cut)
                    add_variant(tuple(cuts))
                    if len(variants) >= max_variants:
                        return variants

    # Coupled tweaks preserving approximate total
    if len(base_cuts) >= 2:
        if not use_percent:
            for (i, rate_i), (j, rate_j) in combinations(enumerate(base_cuts), 2):
                baseline_i = max(0, int(round(rate_i.baseline_rate_r0)))
                baseline_j = max(0, int(round(rate_j.baseline_rate_r0)))
                for step in steps:
                    if step == 0:
                        continue
                    new_i = max(0, min(rate_i.cut_per_hour_lambda + int(step), baseline_i))
                    new_j = max(0, min(rate_j.cut_per_hour_lambda - int(step), baseline_j))
                    cuts = list(base_cuts)
                    cuts[i] = _clone_with_cut(rate_i, new_i)
                    cuts[j] = _clone_with_cut(rate_j, new_j)
                    add_variant(tuple(cuts))
                    if len(variants) >= max_variants:
                        return variants
        else:
            percents = _build_percent_list(percent_lower, percent_upper, percent_step)
            for (i, rate_i), (j, rate_j) in combinations(enumerate(base_cuts), 2):
                baseline_i = max(0, int(round(rate_i.baseline_rate_r0)))
                baseline_j = max(0, int(round(rate_j.baseline_rate_r0)))
                if baseline_i <= 0 and baseline_j <= 0:
                    continue
                base_allowed_i = max(0, min(int(rate_i.allowed_rate_R), baseline_i))
                base_allowed_j = max(0, min(int(rate_j.allowed_rate_R), baseline_j))
                for p in percents:
                    delta_i = int(round(p * baseline_i)) if baseline_i > 0 else 0
                    delta_j = int(round(p * baseline_j)) if baseline_j > 0 else 0
                    if delta_i <= 0 and p > 0.0 and baseline_i > 0:
                        delta_i = 1
                    if delta_j <= 0 and p > 0.0 and baseline_j > 0:
                        delta_j = 1
                    # Variation A: increase R_i, decrease R_j
                    new_allowed_i = max(0, min(base_allowed_i + delta_i, baseline_i))
                    new_allowed_j = max(0, min(base_allowed_j - delta_j, baseline_j))
                    new_cut_i = max(0, min(baseline_i - new_allowed_i, baseline_i))
                    new_cut_j = max(0, min(baseline_j - new_allowed_j, baseline_j))
                    cuts = list(base_cuts)
                    cuts[i] = _clone_with_cut(rate_i, new_cut_i)
                    cuts[j] = _clone_with_cut(rate_j, new_cut_j)
                    add_variant(tuple(cuts))
                    if len(variants) >= max_variants:
                        return variants
                    # Variation B: decrease R_i, increase R_j
                    new_allowed_i = max(0, min(base_allowed_i - delta_i, baseline_i))
                    new_allowed_j = max(0, min(base_allowed_j + delta_j, baseline_j))
                    new_cut_i = max(0, min(baseline_i - new_allowed_i, baseline_i))
                    new_cut_j = max(0, min(baseline_j - new_allowed_j, baseline_j))
                    cuts = list(base_cuts)
                    cuts[i] = _clone_with_cut(rate_i, new_cut_i)
                    cuts[j] = _clone_with_cut(rate_j, new_cut_j)
                    add_variant(tuple(cuts))
                    if len(variants) >= max_variants:
                        return variants
    return variants
