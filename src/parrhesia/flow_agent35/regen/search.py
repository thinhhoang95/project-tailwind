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
) -> List[Sequence[RateCut]]:
    """Generate nearby rate-cut variants around the base allocation."""

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

    # Independent tweaks per flow
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

    # Coupled tweaks preserving approximate total
    if len(base_cuts) >= 2:
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
    return variants
