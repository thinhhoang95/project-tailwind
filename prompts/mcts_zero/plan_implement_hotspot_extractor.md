The goal is to implement a hotspot segment extractor from resources, implemented as a separate module that will be called by `regen`.

- You’ll reuse the same rolling-hour segment logic as `NetworkEvaluator.get_hotspot_segments`, but wire it to `AppResources` so it uses the exact `flight_list`, `time_bin_minutes`, and `capacity_per_bin_matrix`. Check the example in `regen_second_order.py`.

Reference for the algorithm we’ll mirror:
```1146:1155:/mnt/d/project-tailwind/src/server_tailwind/airspace/network_evaluator_for_api.py
def get_hotspot_segments(self, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Detect hotspots using a sliding rolling-hour count at each bin (stride = time_bin_minutes).
        A bin is overloaded when (rolling_count - capacity_per_bin) > threshold and capacity_per_bin >= 0.
        Consecutive overloaded bins for the same TV are merged into contiguous segments.
```
```1186:1194:/mnt/d/project-tailwind/src/server_tailwind/airspace/network_evaluator_for_api.py
        window_bins = int(np.ceil(60.0 / float(int(self.time_bin_minutes))))
        window_bins = max(1, window_bins)
        rolling = self._apply_rolling_hour_forward(per_tv, window_bins)

        # Capacity per bin matrix (hourly value repeated across bins in that hour)
        cap = self._build_capacity_per_bin_matrix()

        segments: List[Dict[str, Any]] = []
        thr = float(threshold)
```

What to add in `src/parrhesia/flow_agent35/regen/hotspot_segment_extractor.py`:
- A small extractor that:
  - Pulls occupancy from `res.flight_list`.
  - Computes a forward-looking 60-minute rolling sum per TV.
  - Subtracts `res.capacity_per_bin_matrix` (hourly cap repeated per bin).
  - Merges contiguous overloaded bins into segments.
- A tiny helper to convert a segment into a `hotspot_payload` you can pass to `propose_regulations`.

Reference implementation:

```python
from typing import Any, Dict, List, Mapping, Optional
import numpy as np

def _rolling_hour_forward(matrix_2d: np.ndarray, window_bins: int) -> np.ndarray:
    if window_bins <= 1:
        return matrix_2d.astype(np.float32, copy=False)
    num_tvs, num_bins = matrix_2d.shape
    cs = np.cumsum(
        np.concatenate([np.zeros((num_tvs, 1), dtype=np.float32), matrix_2d.astype(np.float32, copy=False)], axis=1),
        axis=1,
        dtype=np.float64,
    )
    out = np.empty_like(matrix_2d, dtype=np.float32)
    for j in range(num_bins):
        j2 = min(num_bins, j + window_bins)
        out[:, j] = cs[:, j2] - cs[:, j]
    return out

def extract_hotspot_segments_from_resources(
    *,
    threshold: float = 0.0,
    resources: Optional[AppResources] = None,
) -> List[Dict[str, Any]]:
    res = (resources or get_resources()).preload_all()
    fl = res.flight_list

    num_tvs = len(fl.tv_id_to_idx)
    if num_tvs == 0:
        return []

    total_occ = fl.get_total_occupancy_by_tvtw().astype(np.float32, copy=False)
    num_tvtws = int(fl.num_tvtws)
    bins_per_tv = int(num_tvtws // num_tvs)
    if bins_per_tv <= 0:
        return []

    per_tv = np.zeros((num_tvs, bins_per_tv), dtype=np.float32)
    tv_items = sorted(fl.tv_id_to_idx.items(), key=lambda kv: kv[1])
    for tv_id, row_idx in tv_items:
        start = int(row_idx) * bins_per_tv
        end = start + bins_per_tv
        per_tv[int(row_idx), :] = total_occ[start:end]

    window_bins = max(1, int(np.ceil(60.0 / float(int(fl.time_bin_minutes)))))
    rolling = _rolling_hour_forward(per_tv, window_bins)

    cap = res.capacity_per_bin_matrix
    if cap is None or cap.shape != (num_tvs, bins_per_tv):
        raise RuntimeError("capacity_per_bin_matrix is missing or has unexpected shape")

    def _label(bin_offset: int) -> str:
        start_total_min = int(bin_offset * int(fl.time_bin_minutes))
        return f"{(start_total_min // 60) % 24:02d}:{start_total_min % 60:02d}"

    segments: List[Dict[str, Any]] = []
    thr = float(threshold)

    for tv_id, row_idx in tv_items:
        r = int(row_idx)
        cap_row = cap[r, :]
        roll_row = rolling[r, :]

        valid = cap_row >= 0.0
        diff = np.where(valid, roll_row - cap_row, -np.inf)
        overloaded = diff > thr

        i = 0
        while i < bins_per_tv:
            if not overloaded[i]:
                i += 1
                continue
            start_bin = i
            j = i + 1
            while j < bins_per_tv and overloaded[j]:
                j += 1
            end_bin = j - 1

            seg_slice = slice(start_bin, end_bin + 1)
            seg_diff = diff[seg_slice]
            seg_roll = roll_row[seg_slice]
            seg_cap = cap_row[seg_slice]

            max_excess = float(np.max(seg_diff)) if seg_diff.size > 0 else 0.0
            sum_excess = float(np.sum(seg_diff[seg_diff > -np.inf])) if seg_diff.size > 0 else 0.0
            peak_rolling = float(np.max(seg_roll)) if seg_roll.size > 0 else 0.0
            cap_min = float(np.min(seg_cap)) if seg_cap.size > 0 else -1.0
            cap_max = float(np.max(seg_cap)) if seg_cap.size > 0 else -1.0

            segments.append(
                {
                    "traffic_volume_id": str(tv_id),
                    "start_bin": int(start_bin),          # bin offset within day (shared across TVs)
                    "end_bin": int(end_bin),              # inclusive
                    "start_label": _label(start_bin),
                    "end_label": _label(end_bin),
                    "time_bin_minutes": int(fl.time_bin_minutes),
                    "window_minutes": 60,
                    "max_excess": max_excess,
                    "sum_excess": sum_excess,
                    "peak_rolling_count": peak_rolling,
                    "capacity_stats": {"min": cap_min, "max": cap_max},
                }
            )
            i = j

    segments.sort(
        key=lambda s: (-float(s.get("max_excess", 0.0)),
                       str(s.get("traffic_volume_id", "")),
                       int(s.get("start_bin", 0))))
    return segments

def segment_to_hotspot_payload(seg: Mapping[str, Any]) -> Dict[str, Any]:
    # regen uses [start, end_exclusive]; segments are inclusive
    return {
        "control_volume_id": str(seg["traffic_volume_id"]),
        "window_bins": [int(seg["start_bin"]), int(seg["end_bin"]) + 1],
        "metadata": {},
        "mode": "inventory",
    }
```

How to use it:
```python
res = get_resources().preload_all()
segments = extract_hotspot_segments_from_resources(threshold=0.0, resources=res)
if not segments:
    raise SystemExit("No hotspots detected.")
top_seg = segments[0]
hotspot_payload = segment_to_hotspot_payload(top_seg)
```

Notes:
- This matches `NetworkEvaluator.get_hotspot_segments` semantics, but plugs into the exact `AppResources` instance already driving flows/regen. No duplicate path loading, and capacity comes from `res.capacity_per_bin_matrix`.
- `start_bin`/`end_bin` are day-relative bin offsets, so they can be used directly with `compute_flows` when restricted to a specific TV.

- Built the rolling-hour and segment merge logic inline to avoid importing the evaluator class into examples; you get the same behavior with minimal coupling.

- If your capacity file has TVs not in the indexer, they’re naturally ignored via the `tv_id_to_idx` order and the `cap_row >= 0` mask.

- This extractor runs entirely off the in-memory `res` artifacts, so it’s consistent with any prior delta edits to the `flight_list`.

- You can also pass a positive `threshold` if you want to ignore minor overloads.
