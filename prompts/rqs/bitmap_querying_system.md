Can you help me implement the following route information querying system as a Python class from the following description.

> **Important:** the json data also contains the distance property, and it is expected that the methodology described below should be able to return the distance as well in all circumstances.

# Inputs
- The route file is located at `output/route_distances.json`.

# Instructions
- No need to implement the merisa-trie.
- Always sanity check your code as you write.
- Implement your code in `src/project_tailwind/rqs`.
- Write small examples to show how to use the code as well.

---


## Data model (built once, queried many times)

Use **ID-based posting lists + roaring bitmaps**:

* Assign a dense `route_id ∈ [0, N)` to each route string.
* Store the occupancy vectors in a single contiguous `uint32` array with an **offsets table**:

  * `vec_data: uint32[]` – concatenation of all vectors.
  * `vec_off[i], vec_off[i+1]` gives the slice for `route_id = i`.
  * This avoids Python lists and lets you return a **zero-copy memoryview** or NumPy view.
* Build **inverted indexes** with **Roaring bitmaps** (via `pyroaring`):

  * `OD_index[(origin, dest)] -> BitMap` of `route_id`s.
  * `TVTW_index[tvtw] -> BitMap` of `route_id`s whose vector contains that TVTW.
* Keep two minimal maps:

  * `route_str_to_id: dict[str, int]` – (if memory is tight, replace with a **marisa-trie**).
  * `route_id_to_str: array of offsets into one big UTF-8 blob` (so you can list route strings fast).

**Why roaring bitmaps?** The key ops you need (union, intersection, difference) are SIMD-optimized and run in microseconds even on hundreds of thousands of IDs, with compact memory.

---

## Build step (Python; runs once, saves compact artifacts)

```python
# pip install orjson pyroaring numpy marisa-trie (optional)
import orjson, numpy as np
from pyroaring import BitMap
from collections import defaultdict

# 1) Parse JSON (use orjson for speed)
with open("impact_vectors.json", "rb") as f:
    obj = orjson.loads(f.read())

routes = list(obj.keys())
N = len(routes)

# 2) Assign route IDs and gather TVTWs
route_str_to_id = {r:i for i, r in enumerate(routes)}
origin = np.empty(N, dtype=object)
dest   = np.empty(N, dtype=object)

# Occupancy vectors to contiguous buffer
vec_off = np.zeros(N+1, dtype=np.int64)
vec_chunks = []
for i, r in enumerate(routes):
    toks = r.split()
    origin[i], dest[i] = toks[0], toks[-1]
    v = obj[r]  # e.g., [1706, 1754, 1802, 2522]  # :contentReference[oaicite:5]{index=5}
    vec_off[i+1] = vec_off[i] + len(v)
    vec_chunks.append(np.asarray(v, dtype=np.uint32))
vec_data = np.concatenate(vec_chunks) if vec_chunks else np.array([], dtype=np.uint32)

# 3) Build OD index
OD_index = defaultdict(BitMap)
for i, r in enumerate(routes):
    OD_index[(origin[i], dest[i])].add(i)

# 4) Build TVTW index (posting lists)
TVTW_index = defaultdict(BitMap)
for i in range(N):
    start, end = vec_off[i], vec_off[i+1]
    for tv in vec_data[start:end]:
        TVTW_index[int(tv)].add(i)

# 5) (Optional) compress route strings to a single blob + offsets to reduce Python overhead
```

> For the full \~300k set, you’ll want to **serialize** these artifacts to disk after the build (e.g., using `np.save` for arrays and `BitMap.serialize()` for roaring bitmaps) and **memory-map** them on load so the process starts fast and stays read-only.

---

## Query API (all O(log chunks) or better; practical \~μs)

```python
import numpy as np
from pyroaring import BitMap

def get_vector(route: str) -> memoryview | np.ndarray:
    """(1) Given a route string, return its occupancy vector without copies."""
    i = route_str_to_id.get(route)
    if i is None:
        return None
    start, end = int(vec_off[i]), int(vec_off[i+1])
    # return a zero-copy view; convert to list only if you must
    return memoryview(vec_data[start:end])

def get_routes_by_OD(origin: str, dest: str) -> list[str]:
    """(3) All routes for an (O,D)."""
    bm = OD_index.get((origin, dest))
    if not bm:
        return []
    # Map IDs back to strings; prefer prebuilt id->string via offsets
    return [routes[i] for i in bm]  # iteration over BitMap is fast

def get_routes_avoiding_OD(origin: str, dest: str, banned_tvtws: list[int]) -> list[str]:
    """(2) All (O,D) routes that avoid all TVTW indices in banned_tvtws."""
    cand = OD_index.get((origin, dest))
    if not cand:
        return []
    # Union the 'bad' postings and subtract
    bad = BitMap()
    for tv in banned_tvtws:
        b = TVTW_index.get(int(tv))
        if b:
            bad |= b
    ok = cand - bad
    return [routes[i] for i in ok]

def get_routes_matching_OD_with_or_without(origin: str, dest: str,
                                           tvtws: list[int], require_all=False) -> list[str]:
    """
    Variant: return routes that (a) contain ANY tvtw (require_all=False), or
    (b) contain ALL tvtws (require_all=True), restricted to (O,D).
    """
    cand = OD_index.get((origin, dest))
    if not cand:
        return []
    if not tvtws:
        return [routes[i] for i in cand]
    if require_all:
        bm = cand.copy()
        for tv in tvtws:
            bm &= TVTW_index.get(int(tv), BitMap())  # intersection
        return [routes[i] for i in bm]
    else:
        bm = BitMap()
        for tv in tvtws:
            bm |= TVTW_index.get(int(tv), BitMap())  # union
        bm &= cand
        return [routes[i] for i in bm]
```

**Latency notes (on commodity CPUs):**

* `BitMap` set ops (`|`, `&`, `-`) over 300k IDs are typically **tens of microseconds**.
* Mapping IDs→strings is a minor cost if you keep them in a big UTF-8 blob with offsets (avoid Python `str` allocations in hot paths unless the caller actually needs them).
* `get_vector()` is O(1) and returns a **zero-copy** slice over `vec_data`.

---

## Memory + throughput tips

* **Roaring bitmaps** compress automatic­ally; the memory is roughly proportional to the posting-list density. They can also be serialized and memory-mapped.
* Keep the **route strings off the hot path**:

  * Represent route strings as a single `bytes` blob and an `int32` offsets array. Only decode to Python `str` when returning to the user/UI.
  * If you need fast route-string lookups with lower memory, replace the Python dict with a **marisa-trie** (very compact, supports exact lookup).
* Store occupancy vectors **sorted & deduplicated** (if multiple routes share the same vector) and map `route_id -> vector_id`. This can materially shrink memory and improve cache locality.
* If your workload is highly repetitive, consider a small **LRU cache** keyed by `(O,D, frozenset(banned))` of the last results; however, raw bitmap ops are fast enough that this is optional.

---

## Sanity check with your sample

* Route `"BKPR ALELU EDDM"` → vector `[1706, 1754, 1802, 2522]` via `get_vector`.&#x20;
* `(origin='BKPR', dest='EDJA')` would include, e.g., `"BKPR ERKIR EDNL EDJA"`; avoiding `{2474}` excludes it because its vector contains `2474`.&#x20;

---