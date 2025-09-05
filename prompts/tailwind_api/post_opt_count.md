### Revised plan (no SA rerun; aggregate from existing result)

- We will add a lightweight endpoint that takes the prior `/automatic_rate_adjustment` response and returns aggregated pre/post occupancy counts and per-bin capacity for exactly the TVs in that result (all targets + all ripples), across all T bins.

### Endpoint
- Method: POST
- Path: `/autorate_occupancy`
- Purpose: Aggregate pre/post occupancy counts per TV across flows from an existing autorate result; add capacity arrays aligned to bins.
- No SA optimization is executed here.

### Request JSON
- Required:
  - `autorate_result` (object): The exact JSON returned by `/automatic_rate_adjustment`.
- Optional:
  - `include_capacity` (boolean, default true)

Notes:
- We strictly adhere to the TVs in `autorate_result`:
  - Targets: `autorate_result.tvs` (order preserved)
  - Ripples: unique TV order from `autorate_result.ripple_cells` (append after targets; dedupe)

### Response JSON
- `time_bin_minutes` (int)
- `num_bins` (int)
- `tv_ids_order` (string[]) — targets in given order, then ripples (dedup)
- `timebins`:
  - `labels` (string[]) — `HH:MM-HH:MM` for all bins 0..T-1
- `pre_counts` (object): `{ tv_id: int[T] }` aggregated baseline earliest-crossing counts (sum over flows)
- `post_counts` (object): `{ tv_id: int[T] }` aggregated realized occupancy under optimized delays (sum over flows)
- `capacity` (object, when `include_capacity`): `{ tv_id: float[T] }` hourly capacity value repeated per bin; `-1` if unknown

### Aggregation logic (from autorate result only; no recompute)
- TV set/order:
  - `target_order = autorate_result.tvs`
  - `ripple_order = unique order of tvs from autorate_result.ripple_cells`
  - `tv_ids_order = target_order + (ripple_order \ target_order)`
  - For TVs that appear in both, treat them as targets to avoid double-counting.

- Pre counts per TV (length T):
  - For each flow in `autorate_result.flows`:
    - If `tv` in `flow.pre_target_demands`, add that array;
    - else if in `flow.pre_ripple_demands`, add that array;
    - else add zeros.
  - Sum across flows.

- Post counts per TV (length T):
  - For each flow in `autorate_result.flows`:
    - If `tv` in `flow.target_occupancy_opt`, add that array;
    - else if in `flow.ripple_occupancy_opt`, add that array;
    - else add zeros.
  - Sum across flows.

- Binning and labels:
  - `T = autorate_result.num_time_bins` (fallback to app indexer if missing).
  - `time_bin_minutes = app_resources.indexer.time_bin_minutes`.
  - Build `labels` for all bins 0..T-1.

- Capacity per bin:
  - Use app resources `capacity_per_bin_matrix` (row = TV, columns = T), where each hour’s capacity value is repeated across its bins; unknown = `-1`.
  - Return only for TVs in `tv_ids_order`.

### Minimal server implementation sketch
```python
def compute_autorate_occupancy(payload: dict) -> dict:
  res = payload.get("autorate_result") or {}
  flows = res.get("flows") or []
  T = int(res.get("num_time_bins") or get_resources().indexer.num_time_bins)
  tbm = int(get_resources().indexer.time_bin_minutes)

  # TV order: targets then ripples (dedup)
  targets = [str(t) for t in (res.get("tvs") or [])]
  ripple_cells = res.get("ripple_cells") or []
  ripples_order = []
  seen = set()
  for tv, _b in ripple_cells:
    s = str(tv)
    if s not in seen:
      ripples_order.append(s); seen.add(s)
  tv_ids_order = targets + [tv for tv in ripples_order if tv not in set(targets)]

  def zeros(): return [0]*T
  pre = {tv: [0]*T for tv in tv_ids_order}
  post = {tv: [0]*T for tv in tv_ids_order}

  # Aggregate pre/post across flows with target precedence to avoid double-counting
  for f in flows:
    pre_t = f.get("pre_target_demands") or {}
    pre_r = f.get("pre_ripple_demands") or {}
    occ_t = f.get("target_occupancy_opt") or {}
    occ_r = f.get("ripple_occupancy_opt") or {}

    for tv in tv_ids_order:
      src = pre_t.get(tv) or pre_r.get(tv) or zeros()
      dst = pre[tv]
      for i in range(T): dst[i] += int(src[i] if i < len(src) else 0)

      src2 = occ_t.get(tv) or occ_r.get(tv) or zeros()
      dst2 = post[tv]
      for i in range(T): dst2[i] += int(src2[i] if i < len(src2) else 0)

  # Capacity per bin
  capacity = {}
  if payload.get("include_capacity", True):
    mat = get_resources().capacity_per_bin_matrix
    fl = get_resources().flight_list
    for tv in tv_ids_order:
      row = fl.tv_id_to_idx.get(tv)
      if row is None: capacity[tv] = [-1]*T
      else: capacity[tv] = [float(x) for x in (mat[int(row), :].tolist())]

  # Labels
  def label(k):
    start = k*tbm; end = start+tbm
    sh, sm = divmod(start, 60); eh, em = divmod(end, 60); eh %= 24
    return f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"
  labels = [label(i) for i in range(T)]

  return {
    "time_bin_minutes": tbm,
    "num_bins": T,
    "tv_ids_order": tv_ids_order,
    "timebins": {"labels": labels},
    "pre_counts": pre,
    "post_counts": post,
    "capacity": capacity
  }
```
