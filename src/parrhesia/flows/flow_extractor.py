from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
try:  # Optional SciPy import
    from scipy import sparse  # type: ignore
except Exception:  # pragma: no cover - allow running without SciPy
    sparse = None  # type: ignore


try:
    import igraph as ig  # type: ignore
    import leidenalg as la  # type: ignore
except Exception:  # pragma: no cover - lazy import checked at runtime in run_leiden_from_similarity
    ig = None  # type: ignore
    la = None  # type: ignore


# Local import for type hints and API access
if TYPE_CHECKING:  # Only for type checking; avoid hard dependency at runtime
    from ..optim.flight_list import FlightList


def _build_binary_csr_from_footprints(footprints: Sequence[np.ndarray]):
    n_flights = len(footprints)
    if n_flights == 0:
        if sparse is not None:
            return sparse.csr_matrix((0, 0), dtype=np.int32), 0
        else:
            return np.zeros((0, 0), dtype=np.int32), 0

    # Compress global TV indices to a local contiguous space to keep matrix narrow
    all_vals: List[int] = []
    for fp in footprints:
        if fp.size > 0:
            all_vals.extend(fp.tolist())
    if not all_vals:
        if sparse is not None:
            return sparse.csr_matrix((n_flights, 0), dtype=np.int32), 0
        else:
            return np.zeros((n_flights, 0), dtype=np.int32), 0

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
    if sparse is not None:
        X = sparse.csr_matrix((data, indices, indptr), shape=(n_flights, unique_vals.size), dtype=np.int32)
        return X, n_flights
    else:
        # Dense fallback when SciPy is unavailable
        X = np.zeros((n_flights, unique_vals.size), dtype=np.int32)
        start = 0
        for i in range(n_flights):
            end = indptr[i + 1]
            cols = indices[start:end]
            X[i, cols] = 1
            start = end
        return X, n_flights


def compute_jaccard_similarity(
    footprints: Sequence[np.ndarray],
    dense_limit: int = 512,
) -> Union[np.ndarray, Any]:
    """
    Compute pairwise Jaccard similarity between traffic‑volume footprints.

    Each footprint is a 1D array of integer TV indices visited by a flight. Rows are
    binarized and compressed to a local column space for efficiency. Diagonal entries
    are set to 1.0.

    Parameters
    ----------
    footprints : Sequence[np.ndarray]
        Each element is a 1D array of TV indices for a flight. Duplicates within a
        single footprint are ignored.
    dense_limit : int, default=512
        If the number of flights (n) is <= dense_limit, return a dense float32 array
        of shape (n, n). Otherwise, return a sparse COO matrix (if SciPy is
        available); if SciPy is not available, always return a dense array.

    Returns
    -------
    Union[np.ndarray, sparse.coo_matrix]
        Pairwise Jaccard similarity matrix with values in [0, 1]. Diagonal is 1.0.

    Notes
    -----
    - If there are no flights, returns a (0, 0) matrix.
    - If there are flights but no features at all, returns an identity matrix.
    - When SciPy is not installed, this function always returns a dense ndarray.

    Examples
    --------
    Dense output for small n:

    >>> import numpy as np
    >>> fps = [np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([10])]
    >>> S = compute_jaccard_similarity(fps, dense_limit=10)
    >>> S.shape
    (3, 3)
    >>> round(float(S[0, 1]), 2)  # intersection {2,3} over union {1,2,3,4} -> 2/4
    0.5

    Sparse output for larger n (if SciPy is available):

    >>> S2 = compute_jaccard_similarity(fps, dense_limit=1)
    >>> getattr(S2, 'getformat', lambda: 'dense')()
    'coo'
    """
    X, n = _build_binary_csr_from_footprints(footprints)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if X.shape[1] == 0:
        # No features at all: return identity similarities
        if n <= dense_limit:
            return np.eye(n, dtype=np.float32)
        if sparse is not None:
            rows = np.arange(n, dtype=np.int64)
            return sparse.coo_matrix((np.ones(n, dtype=np.float32), (rows, rows)), shape=(n, n))
        else:
            return np.eye(n, dtype=np.float32)

    # Intersection counts via matmul
    if sparse is not None and not isinstance(X, np.ndarray):
        M = X @ X.T  # csr @ csr -> csr
        if not sparse.isspmatrix_csr(M):
            M = M.tocsr()
        sizes = np.asarray(X.getnnz(axis=1)).ravel().astype(np.int64)
        M_coo = M.tocoo(copy=False)
        inter = M_coo.data.astype(np.float32)
        rows = M_coo.row.astype(np.int64)
        cols = M_coo.col.astype(np.int64)
        union = (sizes[rows] + sizes[cols]).astype(np.float32) - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)
        diag_mask = rows == cols
        if np.any(diag_mask):
            sims[diag_mask] = 1.0
        if n <= dense_limit:
            S = np.zeros((n, n), dtype=np.float32)
            S[rows, cols] = sims
            S[cols, rows] = sims
            np.fill_diagonal(S, 1.0)
            return S
        return sparse.coo_matrix((sims, (rows, cols)), shape=(n, n))

    # Dense fallback (SciPy not available)
    A = X.astype(np.float32)
    inter = A @ A.T  # (n x d) @ (d x n) => (n x n)
    sizes = A.sum(axis=1).ravel()
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            u = sizes[i] + sizes[j] - inter[i, j]
            Sij = float(inter[i, j] / u) if u > 0 else (1.0 if i == j else 0.0)
            S[i, j] = Sij
            S[j, i] = Sij
    np.fill_diagonal(S, 1.0)
    return S


def run_leiden_from_similarity(
    S_or_sparse: Union[np.ndarray, Any],
    threshold: float = 0.1,
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Build an undirected graph from a similarity matrix, threshold edges, and run Leiden.

    Edges with similarity >= `threshold` are added to the graph (undirected). If Leiden
    is unavailable at runtime, a deterministic connected-components fallback is used.
    If there are no edges after thresholding, each node is its own community. For
    n <= 1, returns a trivial membership.

    Parameters
    ----------
    S_or_sparse : np.ndarray | sparse.spmatrix
        Pairwise similarity matrix with values in [0, 1]. Dense or sparse.
    threshold : float, default=0.1
        Minimum similarity required to create an edge.
    resolution : float, default=1.0
        Resolution parameter for the Leiden RBConfigurationVertexPartition.
    seed : Optional[int]
        Random seed passed to Leiden to improve reproducibility.

    Returns
    -------
    List[int]
        Community label (0..k-1) for each node in order.

    Raises
    ------
    ValueError
        If `threshold` is not within [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> S = np.array([[1.0, 0.6, 0.0],
    ...               [0.6, 1.0, 0.0],
    ...               [0.0, 0.0, 1.0]], dtype=np.float32)
    >>> run_leiden_from_similarity(S, threshold=0.5, resolution=1.0, seed=42)
    [0, 0, 1]
    """
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be within [0, 1]")

    # Build thresholded, undirected edge list first
    n: int
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []

    if isinstance(S_or_sparse, np.ndarray) or sparse is None:
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
        adj_list = {i: [] for i in range(n)}
        for (i, j) in edges:
            adj_list[i].append(j)
            adj_list[j].append(i)

        labels = [-1] * n
        comp = 0
        for v in range(n):
            if labels[v] != -1:
                continue
            # BFS/DFS
            stack = [v]
            labels[v] = comp
            while stack:
                u = stack.pop()
                for w in adj_list[u]:
                    if labels[w] == -1:
                        labels[w] = comp
                        stack.append(w)
            comp += 1
        return labels


def assign_communities_for_hotspot(
    flight_list: "FlightList",
    flight_ids: Sequence[str],
    traffic_volume_id: str,
    threshold: float = 0.1,
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Cluster a set of flights that hit a given hotspot traffic‑volume.

    For each flight, a TV-footprint is built and trimmed up to the earliest
    occurrence of the hotspot. Jaccard similarities are computed over those
    footprints, Leiden clustering is applied, and membership labels are remapped
    to consecutive integers (0..k-1) in order of first appearance for stability.

    Parameters
    ----------
    flight_list : FlightList
        Provider for footprints and metadata; must expose `get_footprints_for_flights`
        and `tv_id_to_idx` mappings.
    flight_ids : Sequence[str]
        Flight identifiers to cluster.
    traffic_volume_id : str
        The hotspot TV id used to trim footprints to the earliest occurrence.
    threshold : float, default=0.1
        Edge threshold for the similarity graph.
    resolution : float, default=1.0
        Resolution parameter for Leiden.
    seed : Optional[int]
        Random seed for Leiden.

    Returns
    -------
    Dict[str, int]
        Mapping from `flight_id` to community label (0..k-1).

    Raises
    ------
    ValueError
        If any `flight_ids` are unknown or if `traffic_volume_id` is not present in
        the indexer.

    Examples
    --------
    >>> # assuming `flight_list` is a populated FlightList
    >>> community_map = assign_communities_for_hotspot(
    ...     flight_list,
    ...     ["F1", "F2", "F3"],
    ...     traffic_volume_id="TV123",
    ...     threshold=0.2,
    ...     resolution=1.0,
    ...     seed=0,
    ... )
    >>> isinstance(community_map, dict)
    True
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

    # Validate hotspot exists in indexer
    if traffic_volume_id not in flight_list.tv_id_to_idx:
        raise ValueError(f"Unknown traffic_volume_id: {traffic_volume_id}")

    # Build footprints (unique TV indices), trimmed up to first occurrence of the hotspot
    footprints = flight_list.get_footprints_for_flights(
        flight_ids,
        trim_policy="earliest_hotspot",
        hotspots=[traffic_volume_id],
    )

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


def assign_communities_global(
    flight_list: "FlightList",
    flight_ids: Sequence[str],
    *,
    trim_policy: str = "none",
    threshold: float = 0.1,
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Cluster a provided set of flights globally using Jaccard similarity of footprints.

    This general-purpose variant delegates footprint trimming to
    `flight_list.get_footprints_for_flights` via `trim_policy`.

    Parameters
    ----------
    flight_list : FlightList
        Provider for footprints and metadata.
    flight_ids : Sequence[str]
        Flight identifiers to cluster.
    trim_policy : str, default="none"
        Policy understood by `FlightList.get_footprints_for_flights` (e.g., "none").
    threshold : float, default=0.1
        Edge threshold for the similarity graph.
    resolution : float, default=1.0
        Resolution parameter for Leiden.
    seed : Optional[int]
        Random seed for Leiden.

    Returns
    -------
    Dict[str, int]
        Mapping `flight_id` -> community label (0..k-1). Labels are remapped to
        consecutive integers in order of first appearance for stability.

    Raises
    ------
    ValueError
        If none of the provided `flight_ids` are known to `flight_list`.

    Examples
    --------
    >>> # assuming `flight_list` is a populated FlightList
    >>> community_map = assign_communities_global(
    ...     flight_list,
    ...     ["F1", "F2", "F3", "F4"],
    ...     trim_policy="none",
    ...     threshold=0.15,
    ...     resolution=1.0,
    ...     seed=42,
    ... )
    >>> sorted(community_map.keys())
    ['F1', 'F2', 'F3', 'F4']
    """
    flight_ids = list(flight_ids)
    if len(flight_ids) == 0:
        return {}
    if len(flight_ids) == 1:
        return {flight_ids[0]: 0}

    # Validate existence of flights if possible
    missing = [fid for fid in flight_ids if getattr(flight_list, "flight_metadata", {}).get(fid) is None]
    # Skip strict error to allow lightweight stubs; only filter known
    if missing and len(missing) == len(flight_ids):
        raise ValueError("No known flights in flight_ids")

    # Build footprints as unique TV indices with desired trimming
    try:
        footprints = flight_list.get_footprints_for_flights(flight_ids, trim_policy=trim_policy)  # type: ignore[attr-defined]
    except TypeError:
        # Backward-compat: older signature without keyword
        footprints = flight_list.get_footprints_for_flights(flight_ids)  # type: ignore[attr-defined]

    S_or_sparse = compute_jaccard_similarity(footprints)
    membership = run_leiden_from_similarity(S_or_sparse, threshold=threshold, resolution=resolution, seed=seed)

    remap: Dict[int, int] = {}
    next_label = 0
    out: Dict[str, int] = {}
    for fid, m in zip(flight_ids, membership):
        if m not in remap:
            remap[m] = next_label
            next_label += 1
        out[fid] = remap[m]
    return out
