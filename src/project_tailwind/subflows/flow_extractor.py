from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse


try:
    import igraph as ig  # type: ignore
    import leidenalg as la  # type: ignore
except Exception:  # pragma: no cover - lazy import checked at runtime in run_leiden_from_similarity
    ig = None  # type: ignore
    la = None  # type: ignore


# Local import for type hints and API access
from ..optimize.eval.flight_list import FlightList


def _build_binary_csr_from_footprints(footprints: Sequence[np.ndarray]) -> Tuple[sparse.csr_matrix, int]:
    n_flights = len(footprints)
    if n_flights == 0:
        return sparse.csr_matrix((0, 0), dtype=np.int32), 0

    # Compress global TV indices to a local contiguous space to keep matrix narrow
    all_vals: List[int] = []
    for fp in footprints:
        if fp.size > 0:
            all_vals.extend(fp.tolist())
    if not all_vals:
        return sparse.csr_matrix((n_flights, 0), dtype=np.int32), 0

    unique_vals = np.asarray(sorted(set(all_vals)), dtype=np.int64)
    local_index: Dict[int, int] = {int(v): i for i, v in enumerate(unique_vals.tolist())}

    indptr = np.zeros(n_flights + 1, dtype=np.int64)
    indices_list: List[int] = []
    for i, fp in enumerate(footprints):
        if fp.size == 0:
            indptr[i + 1] = indptr[i]
            continue
        # Ensure uniqueness per row and map to local indices
        row_vals = np.unique(fp).tolist()
        row_idx = [local_index[int(v)] for v in row_vals]
        row_idx.sort()
        indices_list.extend(row_idx)
        indptr[i + 1] = indptr[i] + len(row_idx)

    indices = np.asarray(indices_list, dtype=np.int64)
    data = np.ones(indices.shape[0], dtype=np.int32)
    X = sparse.csr_matrix((data, indices, indptr), shape=(n_flights, unique_vals.size), dtype=np.int32)
    return X, n_flights


def compute_jaccard_similarity(
    footprints: Sequence[np.ndarray],
    dense_limit: int = 512,
) -> Union[np.ndarray, sparse.coo_matrix]:
    """
    Compute Jaccard similarity between flight TV footprints using sparse operations.

    Returns a dense matrix if the number of flights is small (<= dense_limit),
    otherwise returns a sparse COO matrix containing non-zero similarities.
    Diagonal entries are set to 1.0.
    """
    X, n = _build_binary_csr_from_footprints(footprints)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if X.shape[1] == 0:
        # No features at all: return identity similarities
        if n <= dense_limit:
            return np.eye(n, dtype=np.float32)
        rows = np.arange(n, dtype=np.int64)
        return sparse.coo_matrix((np.ones(n, dtype=np.float32), (rows, rows)), shape=(n, n))

    # Intersection counts via sparse matmul
    M = X @ X.T  # csr @ csr -> csr
    if not sparse.isspmatrix_csr(M):
        M = M.tocsr()

    # Set sizes per row
    sizes = np.asarray(X.getnnz(axis=1)).ravel().astype(np.int64)

    # Work with COO to vectorize over non-zero pairs
    M_coo: sparse.coo_matrix = M.tocoo(copy=False)
    inter = M_coo.data.astype(np.float32)
    rows = M_coo.row.astype(np.int64)
    cols = M_coo.col.astype(np.int64)

    # Compute unions per nonzero pair
    union = (sizes[rows] + sizes[cols]).astype(np.float32) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)

    # Ensure diagonal is exactly 1.0
    diag_mask = rows == cols
    if np.any(diag_mask):
        sims[diag_mask] = 1.0

    if n <= dense_limit:
        S = np.zeros((n, n), dtype=np.float32)
        S[rows, cols] = sims
        # Ensure symmetric (matmul should already ensure symmetry, but guard anyway)
        S[cols, rows] = sims
        np.fill_diagonal(S, 1.0)
        return S

    # Return COO sparse similarities
    return sparse.coo_matrix((sims, (rows, cols)), shape=(n, n))


def run_leiden_from_similarity(
    S_or_sparse: Union[np.ndarray, sparse.spmatrix],
    threshold: float = 0.1,
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Build an undirected graph from similarities, threshold edges, and run Leiden.
    If there are no edges, return singletons. For n<=1, return trivial membership.
    """
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be within [0, 1]")

    # Build thresholded, undirected edge list first
    n: int
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []

    if isinstance(S_or_sparse, np.ndarray):
        n = int(S_or_sparse.shape[0])
        if n <= 1:
            return [0] * n
        S = S_or_sparse
        for i in range(n):
            for j in range(i + 1, n):
                w = float(S[i, j])
                if w >= threshold and w > 0.0:
                    edges.append((i, j))
                    weights.append(w)
    else:
        coo = S_or_sparse.tocoo()
        n = int(coo.shape[0])
        if n <= 1:
            return [0] * n
        for i, j, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
            if i >= j:
                continue
            if w >= threshold and w > 0.0:
                edges.append((int(i), int(j)))
                weights.append(float(w))

    if n <= 1:
        return [0] * n
    if len(edges) == 0:
        return list(range(n))

    # Try Leiden if available; otherwise fallback to connected components on the thresholded graph
    try:
        if ig is None or la is None:  # pragma: no cover
            import igraph as ig  # type: ignore  # noqa: F401
            import leidenalg as la  # type: ignore  # noqa: F401

        g = ig.Graph(n=n, edges=edges, directed=False)  # type: ignore[name-defined]
        if len(weights) > 0:
            g.es["weight"] = weights  # type: ignore[attr-defined]
        partition = la.find_partition(  # type: ignore[name-defined]
            g,
            la.RBConfigurationVertexPartition,  # type: ignore[attr-defined]
            weights="weight" if len(weights) > 0 else None,
            resolution_parameter=float(resolution),
            seed=seed,
        )
        membership: List[int] = list(map(int, partition.membership))  # type: ignore[attr-defined]
        return membership
    except Exception:
        # Fallback: compute connected components on the thresholded graph
        # This yields deterministic grouping when edges are well-separated by threshold
        row_idx = np.fromiter((i for i, _ in edges), dtype=np.int64)
        col_idx = np.fromiter((j for _, j in edges), dtype=np.int64)
        data = np.ones(len(edges), dtype=np.int32)
        adj = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
        # make symmetric
        adj = adj + adj.T
        from scipy.sparse.csgraph import connected_components

        _, labels = connected_components(adj, directed=False)
        return labels.tolist()


def assign_communities_for_hotspot(
    flight_list: FlightList,
    flight_ids: Sequence[str],
    traffic_volume_id: str,
    threshold: float = 0.1,
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    High-level wrapper: compute TV footprints for hotspot flights, build Jaccard
    similarities, run Leiden, and map back to flight identifiers.
    """
    flight_ids = list(flight_ids)
    if len(flight_ids) == 0:
        return {}
    if len(flight_ids) == 1:
        return {flight_ids[0]: 0}

    # Validate existence of flights
    missing = [fid for fid in flight_ids if fid not in flight_list.flight_metadata]
    if missing:
        raise ValueError(f"Unknown flight ids: {missing[:3]}{'...' if len(missing)>3 else ''}")

    # Determine hotspot TV index for prefix trimming using traffic_volume_id
    if traffic_volume_id not in flight_list.tv_id_to_idx:
        raise ValueError(f"Unknown traffic_volume_id: {traffic_volume_id}")
    hotspot_tv_index: Optional[int] = int(flight_list.tv_id_to_idx[traffic_volume_id])

    # Build footprints (unique TV indices), trimmed up to first occurrence of hotspot TV
    footprints = flight_list.get_footprints_for_flights(flight_ids, hotspot_tv_index)

    # Compute similarities and run Leiden
    S_or_sparse = compute_jaccard_similarity(footprints)
    membership = run_leiden_from_similarity(S_or_sparse, threshold=threshold, resolution=resolution, seed=seed)

    # Remap membership to consecutive 0..k-1 in order of first appearance for stability
    remap: Dict[int, int] = {}
    next_label = 0
    out: Dict[str, int] = {}
    for fid, m in zip(flight_ids, membership):
        if m not in remap:
            remap[m] = next_label
            next_label += 1
        out[fid] = remap[m]
    return out


