"""Bundle construction helpers."""
from __future__ import annotations

from typing import Dict, List, Sequence

from .types import Bundle, FlowScore


def _weights_from_scores(flows: Sequence[FlowScore]) -> Dict[int, float]:
    scores = [max(0.0, float(f.score)) for f in flows]
    total = float(sum(scores))
    if total <= 0.0:
        w = 1.0 / float(len(flows)) if flows else 0.0
        return {fs.flow_id: w for fs in flows}
    return {fs.flow_id: float(score) / total for fs, score in zip(flows, scores)}


def build_candidate_bundles(
    scored: Sequence[FlowScore],
    *,
    max_bundle_size: int = 5,
    distinct_controls_required: bool = True,
) -> List[Bundle]:
    """Build candidate bundles of top-scoring flows."""

    bundles: List[Bundle] = []
    sorted_flows = [fs for fs in scored if fs.score > 0.0]
    if not sorted_flows:
        return bundles
    limit = min(len(sorted_flows), max_bundle_size, 3)
    for size in range(1, limit + 1):
        subset = sorted_flows[:size]
        if distinct_controls_required and size > 1:
            ctrls = [fs.control_tv_id for fs in subset]
            if len(set(ctrls)) != len(ctrls):
                continue
        weights = _weights_from_scores(subset)
        bundles.append(Bundle(flows=list(subset), weights_by_flow=weights))
    return bundles
