from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class SearchLogger:
    """Lightweight JSONL logger for search traces and summaries.

    Usage:
      logger = SearchLogger.to_timestamped("output/flow_agent_runs")
      logger.event("run_start", {"cfg": {...}})
      ...
      logger.event("run_end", {"plan_size": 2, "best": {...}})
      logger.close()
    """

    path: str
    _fh: Optional[Any] = None

    # ------------------------------ Construction ---------------------------
    @staticmethod
    def to_timestamped(base_dir: str, prefix: str = "run") -> "SearchLogger":
        os.makedirs(base_dir, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(base_dir, f"{prefix}_{stamp}.jsonl")
        return SearchLogger.open(path)

    @staticmethod
    def open(path: str) -> "SearchLogger":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        log = SearchLogger(path=path)
        log._fh = open(path, "w", encoding="utf-8")
        return log

    # --------------------------------- API --------------------------------
    def event(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self._fh is None:
            return
        row = {
            "ts": _now_iso(),
            "type": str(kind),
            **(payload or {}),
        }
        self._fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None


__all__ = ["SearchLogger"]


# ----------------------------- JSON helpers ------------------------------
def _json_default(obj: Any):
    try:
        import numpy as _np  # local import to avoid hard dependency for simple runs
    except Exception:
        _np = None  # type: ignore
    try:
        from datetime import datetime as _dt, date as _date, time as _time
    except Exception:
        _dt = _date = _time = None  # type: ignore

    if _np is not None:
        if isinstance(obj, (_np.generic,)):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    if _dt is not None and isinstance(obj, _dt):
        return obj.isoformat()
    if _date is not None and isinstance(obj, _date):
        return obj.isoformat()
    if _time is not None and isinstance(obj, _time):
        return obj.isoformat()
    # Fallback: stringify
    return str(obj)
