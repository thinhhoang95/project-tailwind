from __future__ import annotations

from typing import Dict, Hashable, Iterable, Mapping, Sequence, Tuple


def decide_union_or_separate(
    features_ij: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> str:
    """
    Heuristic decision based on pairwise features and thresholds.

    Returns 'union' or 'separate'.
    thresholds keys expected: 'tau_ov', 'tau_sl', 'tau_pr', 'tau_orth'
    """
    ov = float(features_ij.get("overlap", 0.0))
    sl = float(features_ij.get("slack_corr", 0.0))
    pr = float(features_ij.get("price_gap", 0.0))
    orth = float(features_ij.get("orth", 0.0))
    tau_ov = float(thresholds.get("tau_ov", float("inf")))
    tau_sl = float(thresholds.get("tau_sl", 0.0))
    tau_pr = float(thresholds.get("tau_pr", float("inf")))
    tau_orth = float(thresholds.get("tau_orth", 0.8))

    # Union if: low temporal overlap, high slack correlation, small price gap
    if ov <= tau_ov and sl >= tau_sl and pr <= tau_pr:
        return "union"
    # Separate if: highly orthogonal or large price gap
    if orth >= tau_orth or pr >= tau_pr:
        return "separate"
    # Default to separate for safety
    return "separate"


def cluster_flows(
    flow_ids: Sequence[Hashable],
    pairwise_feature_map: Mapping[Tuple[Hashable, Hashable], Mapping[str, float]],
    thresholds: Mapping[str, float],
) -> Dict[Hashable, int]:
    """
    Simple clustering: build an undirected graph linking pairs that should 'union'.
    Return mapping flow_id -> group_id (0..k-1) by connected components.
    """
    # Build adjacency
    adj: Dict[Hashable, set] = {f: set() for f in flow_ids}
    for (i, j), feats in pairwise_feature_map.items():
        if decide_union_or_separate(feats, thresholds) == "union":
            adj[i].add(j)
            adj[j].add(i)

    # Connected components via DFS
    group_id = 0
    labels: Dict[Hashable, int] = {}
    visited: set = set()
    for f in flow_ids:
        if f in visited:
            continue
        stack = [f]
        visited.add(f)
        labels[f] = group_id
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):  # type: ignore[arg-type]
                if v not in visited:
                    visited.add(v)
                    labels[v] = group_id
                    stack.append(v)
        group_id += 1

    return labels

