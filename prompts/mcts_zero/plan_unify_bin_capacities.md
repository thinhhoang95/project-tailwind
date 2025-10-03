### Code used in each place (for reference)

Testbench builder:
```160:216:/mnt/d/project-tailwind/examples/regen/regen_test_bench_custom_tvtw.py
def _build_capacities_by_tv(
    evaluator: NetworkEvaluator,
    indexer: TVTWIndexer,
) -> Dict[str, np.ndarray]:
    """
    Build per-bin capacities for each traffic volume and normalize them,
    mirroring the approach used by the server-side wrapper.
    """
    T = int(indexer.num_time_bins)
    bins_per_hour = int(indexer.rolling_window_size())

    # Construct raw per-bin capacities from evaluator's hourly map
    per_hour = getattr(evaluator, "hourly_capacity_by_tv", {}) or {}
    raw_capacities: Dict[str, np.ndarray] = {}
    for tv_id in indexer.tv_id_to_idx.keys():
        arr = np.zeros(T, dtype=np.float64)
        hours = per_hour.get(tv_id, {}) or {}
        for h, cap in hours.items():
            try:
                hour = int(h)
            except Exception:
                continue
            start = hour * bins_per_hour
            if start >= T:
                continue
            end = min(start + bins_per_hour, T)
            arr[start:end] = float(cap)
        raw_capacities[str(tv_id)] = arr

    if raw_capacities:
        non_positive = [tv for tv, arr in raw_capacities.items() if float(np.max(arr)) <= 0.0]
        if non_positive:
            print(
                f"Regen: capacity rows are non-positive for {len(non_positive)}/{len(raw_capacities)} TVs; sample="
                + ",".join([str(x) for x in non_positive[:5]])
            )

    # Normalize: treat missing/zero bins as unconstrained
    capacities_by_tv = normalize_capacities(raw_capacities)

    if capacities_by_tv:
        sample_items = list(capacities_by_tv.items())[:5]
        sample_stats = []
        for tv, arr in sample_items:
            arr_np = np.asarray(arr, dtype=np.float64)
            if arr_np.size == 0:
                sample_stats.append(f"{tv}:empty")
                continue
            sample_stats.append(
                f"{tv}:min={float(arr_np.min()):.1f},max={float(arr_np.max()):.1f}"
            )
        print(
            f"Regen: normalized capacities ready for {len(capacities_by_tv)} TVs; samples: "
            + "; ".join(sample_stats)
        )

    return capacities_by_tv
```

Library builder:
```52:107:/mnt/d/project-tailwind/src/parrhesia/optim/capacity.py
def build_bin_capacities(
    geojson_path: str,
    indexer: TVTWIndexer,
) -> Dict[str, np.ndarray]:
    """
    Build per-TV per-bin capacity arrays C_v(t) from a GeoJSON file.

    The GeoJSON is expected to be a FeatureCollection where each feature's
    properties include:
      - "traffic_volume_id": TV identifier (string)
      - "capacity": mapping of hour ranges (e.g., "6:00-7:00") to integers

    Policy for outside provided hours: bins with no provided hour range are
    filled with 0.

    Returns a mapping: tv_id -> numpy array of shape (T,), where T is the
    number of time bins in the indexer, containing integer capacities.
    """
    # Initialize all TVs to zeros by default
    T = indexer.num_time_bins
    capacities: Dict[str, np.ndarray] = {
        tv_id: np.zeros(T, dtype=np.int64) for tv_id in indexer.tv_id_to_idx.keys()
    }

    # Load GeoJSON (avoid geopandas dependency; parse JSON directly)
    with open(geojson_path, "r") as f:
        data = json.load(f)

    features: List[Dict[str, Any]] = data.get("features", []) or []
    for feat in features:
        props = feat.get("properties", {}) or {}
        tv_id = props.get("traffic_volume_id")
        if not tv_id or tv_id not in capacities:
            # Skip TVs not present in the indexer mapping
            continue
        cap_map = props.get("capacity") or {}
        if not isinstance(cap_map, dict):
            continue
        arr = capacities[tv_id]
        for hour_key, val in cap_map.items():
            try:
                v = int(val)
            except Exception:
                # Skip non-integer capacity values
                continue
            try:
                start_bin, end_bin = _parse_hour_range_to_bins(str(hour_key), indexer)
            except Exception:
                # Skip malformed hour strings but continue others
                continue
            if end_bin <= start_bin:
                continue
            arr[start_bin:end_bin] = v
        capacities[tv_id] = arr

    return capacities
```

### Key differences
- Input source
  - Testbench: uses `evaluator.hourly_capacity_by_tv` (built from a `GeoDataFrame`).
  - Library: reads GeoJSON from a path and parses properties directly (no geopandas).
- Time parsing/granularity
  - Testbench: collapses to whole hours via `int(h)` and repeats per hour across `bins_per_hour`; ignores minutes.
  - Library: parses full "HH:MM-HH:MM" ranges via regex; supports partial hours and minute precision; maps to `[start_bin, end_bin)`.
- Defaults and normalization
  - Both initialize unspecified bins to 0; testbench then calls `normalize_capacities` (same utility) to map non-positive to 9999. Library leaves zeros; callers typically normalize afterwards (e.g., in `base_evaluation.py`).
- Dtypes
  - Testbench builds `float64` arrays; library builds `int64` arrays, which become `float64` after normalization.
- Coverage and ordering
  - Both align to `indexer.tv_id_to_idx`. Library skips features not in the indexer; testbench loops all indexer TVs and only fills where the evaluator map has entries.
- Robustness
  - Testbench accepts only integer hour keys; library robustly parses full hour-range strings and skips malformed ones.
- Behavior change to expect if you switch
  - Minute-level ranges in the GeoJSON will start to matter (more precise). If your data use exact "HH:00-HH:00" keys, the results should be equivalent.

### Can we use capacity.py here?
Yes. Recommended for consistency with the rest of the codebase and minute-accurate parsing. Minimal integration change:

- Import the builder and keep normalization:
```python
from parrhesia.optim.capacity import build_bin_capacities, normalize_capacities
```

- Preserve the original capacities path so we can call the builder:
Add after creating the `NetworkEvaluator` in `build_data`:
```python
evaluator._capacities_path = str(caps_path)
```

- Update `_build_capacities_by_tv` to use the library when the path is available, with a fallback to the current logic:
```python
def _build_capacities_by_tv(evaluator: NetworkEvaluator, indexer: TVTWIndexer) -> Dict[str, np.ndarray]:
    caps_path = getattr(evaluator, "_capacities_path", None)
    if caps_path:
        raw = build_bin_capacities(str(caps_path), indexer)
        return normalize_capacities(raw)
    # Fallback to existing hourly approach (unchanged) ...
```

Notes:
- This will re-read the GeoJSON once per run; fine for the testbench.
- After the swap, partial-hour capacity ranges will be respected; objective/exceedance values may shift slightly compared to the legacy hourly approximation.

- Keep the normalization call; it matches how capacities are used elsewhere (treating missing/invalid bins as unconstrained).

- Switching to `capacity.py` is feasible and will likely improve consistency with other modules (e.g., `base_evaluation.py`).

- Proposed changes:
  - Save `caps_path` on `evaluator` in `build_data`.
  - Replace `_build_capacities_by_tv` body to call `build_bin_capacities` + `normalize_capacities` when the path is available; keep current implementation as a fallback.