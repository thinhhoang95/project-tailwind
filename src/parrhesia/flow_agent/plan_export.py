from __future__ import annotations

import json
from pathlib import Path

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .state import PlanState
from .agent import RunInfo


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _bin_label(indexer: Any, bin_idx: int) -> str:
    try:
        tw_map = getattr(indexer, "time_window_map", {}) or {}
        return str(tw_map.get(int(bin_idx), f"bin{int(bin_idx)}"))
    except Exception:
        return f"bin{int(bin_idx)}"


def save_plan_to_file(
    state: PlanState,
    info: RunInfo,
    indexer: Any,
    *,
    out_dir: Union[str, Path, None] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Persist the final regulation plan and metadata to a JSON file.

    - Output directory defaults to the same directory as the run logs (info.log_path or info.debug_log_path).
    - File name defaults to plan_<timestamp>.json derived from the log file name; falls back to plan_final.json.
    - Includes per-flow committed rates when available, or a blanket_rate otherwise.
    """
    # Resolve output directory
    if out_dir is None:
        base_path = info.log_path or info.debug_log_path
        out_dir_path = Path(base_path).parent if base_path else Path("agent_runs")
    else:
        out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Resolve filename
    if filename is None:
        stamp: Optional[str] = None
        try:
            base_path = info.log_path or info.debug_log_path
            if base_path:
                stem = Path(base_path).stem  # e.g., run_20250101_120000
                parts = stem.split("_", 1)
                if len(parts) == 2 and parts[1]:
                    stamp = parts[1]
        except Exception:
            stamp = None
        filename = f"plan_{stamp}.json" if stamp else "plan_final.json"

    out_path = out_dir_path / filename

    # Build plan payload
    plan_items: List[Dict[str, Any]] = []
    seen_specs: Set[Tuple[str, Tuple[int, int], Tuple[str, ...], str, Tuple[Tuple[str, int], ...] | int]] = set()
    for reg in getattr(state, "plan", []) or []:
        try:
            t0 = _safe_int(reg.window_bins[0])
            t1 = _safe_int(reg.window_bins[1])
        except Exception:
            t0, t1 = 0, 1

        raw_flow_ids = getattr(reg, "flow_ids", ()) or ()
        flow_ids = tuple(str(fid) for fid in raw_flow_ids)
        mode = str(getattr(reg, "mode", "per_flow"))
        rates = getattr(reg, "committed_rates", None)

        # Skip regulations with no flows or no effective rates
        valid = False
        rates_per_flow_out: Optional[Dict[str, int]] = None
        blanket_rate_out: Optional[int] = None
        if isinstance(rates, dict):
            cleaned = {str(k): _safe_int(v) for k, v in (rates or {}).items() if _safe_int(v) > 0}
            if cleaned and flow_ids:
                valid = True
                rates_per_flow_out = cleaned
        else:
            br = _safe_int(rates) if rates is not None else 0
            if br > 0 and flow_ids:
                valid = True
                blanket_rate_out = br

        if not valid:
            continue

        item: Dict[str, Any] = {
            "control_volume_id": str(getattr(reg, "control_volume_id", "")),
            "window_bins": [t0, t1],
            # Labels are informational; end label shown for the last included bin (t1-1)
            "window_labels": {
                "start": _bin_label(indexer, t0),
                "end": _bin_label(indexer, max(0, t1 - 1)),
            },
            "mode": mode,
            "flow_ids": list(flow_ids),
        }

        if rates_per_flow_out is not None:
            item["rates_per_flow"] = rates_per_flow_out
            item["blanket_rate"] = None
            canonical_rates: Tuple[Tuple[str, int], ...] | int = tuple(sorted(rates_per_flow_out.items()))
        else:
            item["rates_per_flow"] = None
            item["blanket_rate"] = blanket_rate_out
            canonical_rates = int(blanket_rate_out or 0)

        spec_key = (
            item["control_volume_id"],
            (t0, t1),
            tuple(item["flow_ids"]),
            item["mode"],
            canonical_rates,
        )
        if spec_key in seen_specs:
            continue
        seen_specs.add(spec_key)
        plan_items.append(item)

    payload: Dict[str, Any] = {
        "objective": float((getattr(info, "summary", {}) or {}).get("objective", 0.0)),
        "commits": _safe_int(getattr(info, "commits", 0)),
        "stop_reason": getattr(info, "stop_reason", None),
        "stop_info": getattr(info, "stop_info", None) or {},
        "action_counts": getattr(info, "action_counts", {}) or {},
        "time_bin_minutes": _safe_int(getattr(indexer, "time_bin_minutes", 60)),
        "plan": plan_items,
    }

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    return out_path


__all__ = ["save_plan_to_file"]


