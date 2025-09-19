Can you help me implement the following flow extraction algorithm according to the following description?

# Requirements
- The implementation must work with the returned flight list from `network_evaluation`'s `get_hotspot_flights` method.
- Please implement the algorithm in as many files as you want in `src/project_tailwind/flow_x` directory.
- Afterwards, modify the `test_network_evaluation.py` file to run the code. Feel free to remove existing components in there, the old algorithm is now deprecated because it fell short.
- Beware that the drop-in example below may contain errors. It is only there for your reference.

# Algorithm Description

# 0) Inputs & notation

* Flights: `F = {f1, …, fN}`
* Sectors: `S` (may overlap; may have containment hierarchy)
* Time bins: discrete `t = 1..T`
* Hotspot sector: `H`
* Candidate upstream references: `R ⊆ S`
* For each flight `f` and time bin `t`, you have the *sector set* it occupies:

  * `A[f][t] ⊆ S` (multi-label per bin because of overlap)

### Optional structures (helpful!)

* Sector DAG `G_S` (parent ⊃ child relationships).
* Sector areas (or “children counts”) if you’ll weight large umbrella sectors down.

---

# 1) Preprocess overlap (pick 1 of 2)

### Option A — Atomization (exact, more work)

1. Build polygon arrangement of all sectors and compute **atoms** (non-overlapping cells).
2. Map each sector to the subset of atoms it contains.
3. Replace each `A[f][t]` with the **set of atoms** occupied.
   → Similarity computed on atoms avoids double counting.

### Option B — Hierarchy/size weights (fast, practical)

1. Compute per-sector weight `w[s]` (examples):

   * `w[s] = 1 / log(2 + children_count[s])`
   * or `w[s] = 1 / area[s]` (rescaled to be O(1))
2. You’ll use **weighted Jaccard** on sector sets:

   $$
   J_w(X,Y) = \frac{\sum_{s\in X\cap Y} w_s}{\sum_{s\in X\cup Y} w_s}
   $$

---

# 2) Align footprints from each reference `r` to hotspot `H`

For each `r ∈ R`:

1. **Eligibility**: keep flights that pass `r` before `H`.
2. **Alignment**: for each eligible flight `f`, build a *reference-aligned* sequence

   ```
   A_r[f][1], A_r[f][2], …, A_r[f][T_r(f)]
   ```

   where bin `1` is the first bin at/after `r`, and the last bin is the one at/just before `H`.
   (pad/truncate sequences to a common `T_r` if you want fixed-length comparison, e.g., by clipping long ones and dropping very short ones.)

Implementation hint: Precompute for each (flight, sector) the entry/exit bins; then slice the original series for `[entry_r(f) .. entry_H(f)]`.

---

# 3) Build the similarity matrix for each reference `r`

For each pair of eligible flights `f,g`:

1. Compute per-bin similarity

   $$
   \text{sim}_t(f,g) = J_w\!\big(A_r[f][t], A_r[g][t]\big)
   $$
2. Aggregate across bins, e.g. simple mean (robust default):

   $$
   W^{(r)}_{fg} = \frac{1}{T_r}\sum_{t=1}^{T_r} \text{sim}_t(f,g)
   $$

   (You can use a trimmed mean or Huber mean if you expect outlier bins.)

**Sparsify** (recommended for speed): set `W^{(r)}_{fg} = 0` if above mean minus `α·std` or if `< τ`. Keep matrix symmetric, zero diagonal.

Complexities (for reference): `O(|F_r|^2 T_r · k̄)` where `k̄` ≈ average sectors per bin after preprocessing.

---

# 4) Spectral relaxation per reference `r`

We want a high-similarity subset. Relax the discrete problem to the Rayleigh quotient:

* Compute the **top eigenvector** `v` of `W^{(r)}` (largest eigenvalue).
  Use `scipy.sparse.linalg.eigsh(W, k=1, which="LA")` or power iteration for large sparse.

* `v[i]` is a soft score for flight `i`.

### Threshold sweep to get a discrete group

1. Sort flights by `v` descending → order `π(1), π(2), …, π(m)` with `m=|F_r|`.

2. Iterate `k = 2..Kmax` (e.g., `Kmax = m` or a cap like 200):

   * Add flight `i = π(k)` to the current set `G_k`.
   * Maintain `pair_sum_k = pair_sum_{k-1} + ∑_{j∈G_{k-1}} W_{ij}` (O(deg) if sparse).
   * Score options:

     * **Unnormalized**: `score_k = pair_sum_k - λ·k`
     * **Avg pairwise**: `score_k = pair_sum_k / comb(k,2)` (optionally minus a size penalty)
   * Track `k*` that maximizes your chosen score.

3. Final group for reference `r`: `G_r = {π(1..k*)}`; record `score_r`.

**Which score?**

* If you want explicit size control: use `pair_sum - λ·k`, tune `λ` by stability or validation.
* If you want “pure density”: use average pairwise similarity `pair_sum / comb(k,2)` and, if needed, post-filter a minimum size `k_min`.

---

# 5) Pick the best reference sector

Run step 4 for each candidate `r ∈ R` (in parallel if possible).
Choose `r* = argmax_r score_r` and output `(r*, G_{r*})`.

---

# 6) Outputs & diagnostics

Return:

* `reference_sector = r*`
* `group_flights = G_{r*}` (IDs)
* `score`, `|G|`, `avg_pairwise_similarity`
* Heatmaps: per-bin average Jaccard across the group (quick quality check)
* Top contributing sectors: rank sectors by their average presence weight within the group vs out-group (helps interpret the pattern)

---

# 7) Robustness knobs

* **Bin weighting**: give bins near the hotspot more weight (e.g., geometric weights) if that matters operationally.
* **Length standardization**: resample aligned sequences to a fixed `T_r` with temporal interpolation of occupancy (binary → probabilities) and use weighted Jaccard on probabilities (becomes Tanimoto on reals).
* **Row/col normalization**: normalize `W` by degree to reduce hub effects: `W_norm = D^{-1/2} W D^{-1/2}`; use its top eigenvector instead.
* **Multiple groups**: after extracting `(r*, G*)`, zero out rows/cols of `G*` and rerun to peel the next coherent group.

---

# 8) Edge cases to handle

* **Very few eligible flights for a reference**: skip if `|F_r| < 3`.
* **All-zeros after sparsification**: back off threshold `τ` or use the dense `W`.
* **Dominant umbrella sectors**: if you didn’t atomize, make sure weights `w[s]` down-weight them sufficiently; sanity-check by removing them and re-running (sensitivity test).

---

# 9) Python skeleton (drop-in)

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from math import comb

# ---------- Weighted Jaccard ----------
def weighted_jaccard(X, Y, w):
    # X, Y: iterables of sector IDs; w: dict sector->weight
    if not X and not Y:
        return 1.0
    Xs, Ys = set(X), set(Y)
    inter = Xs & Ys
    union = Xs | Ys
    num = sum(w[s] for s in inter)
    den = sum(w[s] for s in union)
    return 0.0 if den == 0 else num / den

# ---------- Build W for a given reference r ----------
def build_similarity_matrix(flights, A_r, w, tau=None):
    """
    flights: list of flight IDs eligible for r
    A_r: dict flight_id -> list[ set(sector) ] aligned bins from r to H
    w: dict sector -> weight
    tau: optional threshold to sparsify
    returns: (W, idx) where W is numpy array or csr_matrix, idx maps position->flight_id
    """
    idx = list(flights)
    m = len(idx)
    W = np.zeros((m, m), dtype=np.float32)
    # assume equal length sequences; if not, use min length or resample
    T = min(len(A_r[f]) for f in idx)
    for a in range(m):
        f = idx[a]
        for b in range(a+1, m):
            g = idx[b]
            sims = [weighted_jaccard(A_r[f][t], A_r[g][t], w) for t in range(T)]
            s = float(np.mean(sims))
            W[a,b] = W[b,a] = s
    if tau is not None:
        # sparsify
        M = (W >= tau).astype(np.float32) * W
        W = csr_matrix(M)
    return W, idx

# ---------- Spectral + sweep ----------
def spectral_group(W, lam=0.0, k_max=None, normalize_by_degree=False, average_objective=True):
    """
    W: numpy array or csr_matrix, symmetric, zero diag
    lam: size penalty (only used if average_objective=False or you subtract afterward)
    k_max: cap on group size
    normalize_by_degree: if True, use D^{-1/2} W D^{-1/2}
    average_objective: if True, maximize avg pairwise similarity; else maximize sum - lam*k
    """
    m = W.shape[0]
    if normalize_by_degree:
        if hasattr(W, "tocsr"):
            d = np.array(W.sum(axis=1)).ravel()
        else:
            d = W.sum(axis=1)
        d = np.maximum(d, 1e-8)
        Dm12 = 1.0 / np.sqrt(d)
        if hasattr(W, "multiply"):  # sparse
            D = csr_matrix(np.diag(Dm12))
            Wn = D @ W @ D
        else:
            Wn = (Dm12[:,None] * W) * Dm12[None,:]
    else:
        Wn = W

    # Top eigenvector
    if hasattr(Wn, "tocsr"):
        vals, vecs = eigsh(Wn, k=1, which="LA")
        v = vecs[:,0]
    else:
        vals, vecs = np.linalg.eigh(Wn)
        v = vecs[:, -1]
    order = np.argsort(-v)

    # Prepare incremental sums for sweep
    in_set = np.zeros(m, dtype=bool)
    pair_sum = 0.0
    best_score, best_k = -1e18, 0
    kmax = m if k_max is None else min(k_max, m)

    for k in range(1, kmax+1):
        i = order[k-1]
        if hasattr(W, "tocsr"):
            # sum to existing set
            if in_set.any():
                # gather weights W[i, j] for j in set
                js = np.where(in_set)[0]
                pair_sum += W[i, js].sum()
        else:
            pair_sum += W[i, in_set].sum()
        in_set[i] = True

        if k >= 2:
            if average_objective:
                denom = comb(k, 2)
                score = pair_sum / denom
            else:
                score = pair_sum - lam * k
            if score > best_score:
                best_score, best_k = score, k

    selected = order[:max(best_k, 2)]  # ensure at least 2
    return selected, best_score

# ---------- Outer loop across references ----------
def find_spatially_coherent_group(R, H, flights_all, A, sector_weights, lam=0.0, tau=None,
                                  normalize_by_degree=False, average_objective=True, k_max=None):
    """
    R: iterable of candidate reference sectors
    H: hotspot sector ID
    flights_all: list of all flights
    A: dict f -> list over global time bins of set(sector)
    sector_weights: dict sector->weight
    Returns: dict with best reference, group flight IDs, and metrics
    """
    best = {"ref": None, "group": None, "score": -1e18, "avg_similarity": None}
    for r in R:
        # 1) eligible flights & aligned sequences
        flights_r, A_r = align_from_reference_to_H(r, H, flights_all, A)  # you implement
        if len(flights_r) < 3:
            continue
        # 2) similarity matrix
        W, idx = build_similarity_matrix(flights_r, A_r, sector_weights, tau=tau)
        # 3) spectral + sweep
        selected_idx, score = spectral_group(W, lam=lam, k_max=k_max,
                                             normalize_by_degree=normalize_by_degree,
                                             average_objective=average_objective)
        group_flights = [idx[i] for i in selected_idx]
        # compute avg pairwise similarity for reporting
        if hasattr(W, "tocsr"):
            sub = W[selected_idx][:, selected_idx].toarray()
        else:
            sub = W[np.ix_(selected_idx, selected_idx)]
        k = len(selected_idx)
        pair_sum = sub[np.triu_indices(k, 1)].sum()
        avg_sim = pair_sum / comb(k, 2) if k >= 2 else 0.0

        if score > best["score"]:
            best.update({"ref": r, "group": group_flights, "score": float(score), "avg_similarity": float(avg_sim)})

    return best

# --- You must provide this based on your data layout ---
def align_from_reference_to_H(r, H, flights_all, A):
    """
    Return (flights_r, A_r) where:
      - flights_r: list of flight IDs that pass r before H
      - A_r: dict flight_id -> list[ set(sector) ] aligned bins from r to H (inclusive/exclusive as you define)
    """
    raise NotImplementedError
```

---

# 10) Validation & tuning

* **Stability**: bootstrap flights (resample with replacement), re-run, and report how often each flight appears in `G` and how often each `r` wins.
* **Sensitivity to weights**: halve/double `w[s]` for umbrella sectors and check if the group changes materially.
* **λ / τ selection**: choose by maximizing out-of-bag average similarity (split flights into halves; compute `G` on half A; evaluate average similarity on B restricted to its top-scoring members).

