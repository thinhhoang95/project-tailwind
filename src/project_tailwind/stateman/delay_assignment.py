"""Delay assignment table utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

from .types import DelayMapping, DelayMergePolicy


class DelayAssignmentTable:
    """Container storing per-flight delay assignments in minutes."""

    def __init__(self, delays: Optional[Mapping[str, int]] = None):
        self._delays: Dict[str, int] = {}
        if delays:
            for flight_id, delay_minutes in delays.items():
                self[flight_id] = delay_minutes

    # --- basic mapping protocol -------------------------------------------------
    def __getitem__(self, flight_id: str) -> int:
        return self._delays[flight_id]

    def __setitem__(self, flight_id: str, delay_minutes: int) -> None:
        self._delays[str(flight_id)] = self._normalize_delay(delay_minutes)

    def __contains__(self, flight_id: object) -> bool:
        return flight_id in self._delays

    def __iter__(self) -> Iterator[str]:
        return iter(self._delays)

    def __len__(self) -> int:
        return len(self._delays)

    def items(self) -> Iterable[tuple[str, int]]:
        return self._delays.items()

    def get(self, flight_id: str, default: int = 0) -> int:
        return self._delays.get(flight_id, default)

    def copy(self) -> "DelayAssignmentTable":
        return DelayAssignmentTable(self._delays)

    # --- factories ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, int]) -> "DelayAssignmentTable":
        return cls(data)

    def to_dict(self) -> DelayMapping:
        return dict(self._delays)

    @classmethod
    def load_json(cls, path: str | Path) -> "DelayAssignmentTable":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("DelayAssignmentTable JSON payload must be a mapping")
        return cls({str(k): int(v) for k, v in payload.items()})

    def save_json(self, path: str | Path) -> None:
        path_obj = Path(path)
        data = {flight_id: max(1, int(delay)) for flight_id, delay in self.nonzero_items()}
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)

    @classmethod
    def load_csv(cls, path: str | Path) -> "DelayAssignmentTable":
        delays: Dict[str, int] = {}
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("DelayAssignmentTable CSV missing header")
            expected_fields = {"flight_id", "delay_minutes"}
            if set(reader.fieldnames) >= expected_fields:
                flight_field = "flight_id"
                delay_field = "delay_minutes"
            else:
                raise ValueError("DelayAssignmentTable CSV must contain flight_id and delay_minutes columns")
            for row in reader:
                flight_id = str(row.get(flight_field, "")).strip()
                if not flight_id:
                    continue
                delay_val = row.get(delay_field)
                if delay_val is None:
                    continue
                delays[flight_id] = cls._normalize_static(delay_val)
        return cls(delays)

    def save_csv(self, path: str | Path) -> None:
        path_obj = Path(path)
        with path_obj.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["flight_id", "delay_minutes"])
            for flight_id, delay_minutes in self.nonzero_items():
                writer.writerow([flight_id, max(1, int(delay_minutes))])

    # --- operations ----------------------------------------------------------------
    def merge(self, other: Mapping[str, int] | "DelayAssignmentTable", *, policy: DelayMergePolicy = "overwrite") -> "DelayAssignmentTable":
        if policy not in {"overwrite", "max", "sum"}:
            raise ValueError(f"Unsupported merge policy: {policy}")
        merged = self.copy()
        source = other._delays if isinstance(other, DelayAssignmentTable) else other
        for flight_id, incoming_delay in source.items():
            incoming = self._normalize_delay(incoming_delay)
            current = merged._delays.get(flight_id, 0)
            if policy == "overwrite":
                merged._delays[flight_id] = incoming
            elif policy == "max":
                merged._delays[flight_id] = max(current, incoming)
            elif policy == "sum":
                merged._delays[flight_id] = current + incoming
        return merged

    def nonzero_items(self) -> Iterable[tuple[str, int]]:
        return ((flight_id, delay) for flight_id, delay in self._delays.items() if delay > 0)

    # --- helpers -------------------------------------------------------------------
    @staticmethod
    def _normalize_static(value: object) -> int:
        try:
            delay_int = int(value)
        except Exception as exc:
            raise ValueError(f"Invalid delay value: {value!r}") from exc
        if delay_int < 0:
            raise ValueError("Delay minutes cannot be negative")
        return delay_int

    def _normalize_delay(self, value: object) -> int:
        return self._normalize_static(value)


__all__ = ["DelayAssignmentTable"]
