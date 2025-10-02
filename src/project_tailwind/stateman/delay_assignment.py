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
        """
        Initialize the DelayAssignmentTable with an optional mapping of flight delays.
        
        If `delays` is provided, each (flight_id, delay_minutes) pair is normalized and stored in the table.
        
        Parameters:
            delays (Optional[Mapping[str, int]]): Mapping from flight identifier to delay in minutes. Keys are flight IDs and values are delay amounts; entries are validated and converted to integers before storing.
        
        Raises:
            ValueError: If any provided delay value is invalid or negative.
        """
        self._delays: Dict[str, int] = {}
        if delays:
            for flight_id, delay_minutes in delays.items():
                self[flight_id] = delay_minutes

    # --- basic mapping protocol -------------------------------------------------
    def __getitem__(self, flight_id: str) -> int:
        """
        Retrieve the delay in minutes for the specified flight ID.
        
        Returns:
            int: Delay in minutes for the given flight ID.
        """
        return self._delays[flight_id]

    def __setitem__(self, flight_id: str, delay_minutes: int) -> None:
        """
        Store a normalized delay for the given flight identifier.
        
        Parameters:
            flight_id (str): Flight identifier under which the delay will be stored; will be converted to a string.
            delay_minutes (int | object): Delay value to store; will be normalized to an integer representing minutes.
        
        Raises:
            ValueError: If `delay_minutes` cannot be converted to an integer or is negative.
        """
        self._delays[str(flight_id)] = self._normalize_delay(delay_minutes)

    def __contains__(self, flight_id: object) -> bool:
        """
        Check whether a flight identifier exists in the table.
        
        Parameters:
            flight_id (object): The flight identifier to look up; any object may be provided and membership is determined by key equality.
        
        Returns:
            bool: `True` if `flight_id` is present in the table, `False` otherwise.
        """
        return flight_id in self._delays

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the stored flight IDs in the table.
        
        Returns:
            Iterator[str]: An iterator yielding each flight ID present in the table.
        """
        return iter(self._delays)

    def __len__(self) -> int:
        """
        Return the number of stored flight delay entries.
        
        Returns:
            int: The count of flight-delay mappings in the table.
        """
        return len(self._delays)

    def items(self) -> Iterable[tuple[str, int]]:
        """
        Return an iterable of stored flight ID and delay pairs.
        
        Each item yielded is a (flight_id, delay_minutes) tuple reflecting the current contents of the table.
        
        Returns:
            Iterable[tuple[str, int]]: An iterable of (flight_id, delay_minutes) pairs.
        """
        return self._delays.items()

    def get(self, flight_id: str, default: int = 0) -> int:
        """
        Return the stored delay for a flight or the provided default if the flight is not present.
        
        Parameters:
            flight_id (str): Flight identifier to look up.
            default (int): Value to return when `flight_id` is not in the table.
        
        Returns:
            int: Delay in minutes for `flight_id`, or `default` if missing.
        """
        return self._delays.get(flight_id, default)

    def copy(self) -> "DelayAssignmentTable":
        """
        Create a new DelayAssignmentTable containing the same delay assignments.
        
        Returns:
            DelayAssignmentTable: A new instance with a shallow copy of the current mapping of flight IDs to delay minutes.
        """
        return DelayAssignmentTable(self._delays)

    # --- factories ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, int]) -> "DelayAssignmentTable":
        """
        Create a DelayAssignmentTable populated from a mapping of flight IDs to delay minutes.
        
        Parameters:
            data (Mapping[str, int]): Mapping of flight ID to delay minutes used to populate the table.
        
        Returns:
            DelayAssignmentTable: A new table containing the entries from `data`, with delays normalized.
        """
        return cls(data)

    def to_dict(self) -> DelayMapping:
        """
        Return a plain dictionary mapping flight IDs to their assigned delay minutes.
        
        Returns:
            DelayMapping: A shallow copy of the internal mapping where keys are flight ID strings and values are delay minutes as integers.
        """
        return dict(self._delays)

    @classmethod
    def load_json(cls, path: str | Path) -> "DelayAssignmentTable":
        """
        Create a DelayAssignmentTable from a JSON file.
        
        Parameters:
            path (str | Path): Filesystem path to a JSON file containing a mapping of flight IDs to delay minutes.
        
        Returns:
            DelayAssignmentTable: Table populated from the JSON mapping; keys are converted to strings and values to integers.
        
        Raises:
            ValueError: If the JSON top-level value is not a mapping.
        """
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("DelayAssignmentTable JSON payload must be a mapping")
        return cls({str(k): int(v) for k, v in payload.items()})

    def save_json(self, path: str | Path) -> None:
        """
        Write the table's nonzero delay assignments to a JSON file.
        
        Only flights with a delay greater than zero are written. Each delay is written as an integer of at least 1; JSON object keys are flight IDs and values are the normalized delays. The output is pretty-printed with keys sorted.
        
        Parameters:
            path (str | Path): Filesystem path to write the JSON output to.
        """
        path_obj = Path(path)
        data = {flight_id: max(1, int(delay)) for flight_id, delay in self.nonzero_items()}
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)

    @classmethod
    def load_csv(cls, path: str | Path) -> "DelayAssignmentTable":
        """
        Create a DelayAssignmentTable from a CSV file containing flight delay rows.
        
        Reads a CSV at `path` expecting a header with at least the columns `flight_id` and
        `delay_minutes`. Each row with a non-empty `flight_id` and a `delay_minutes`
        value is normalized and stored; rows missing `flight_id` or `delay_minutes`
        are skipped.
        
        Parameters:
            path (str | Path): Path to the CSV file to read.
        
        Returns:
            DelayAssignmentTable: A new instance populated with parsed delays.
        
        Raises:
            ValueError: If the CSV is missing a header, does not contain the required
                `flight_id` and `delay_minutes` columns, or if any delay value is
                invalid according to the normalization rules.
        """
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
        """
        Write nonzero delay assignments to a CSV file with header "flight_id,delay_minutes".
        
        Writes a CSV at the given path containing one row per flight whose stored delay is greater than zero. Each row contains the flight ID and the delay minutes coerced to an integer and written as at least 1 (i.e., max(1, int(delay))). The file is written using UTF-8 encoding and includes the header row.
        """
        path_obj = Path(path)
        with path_obj.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["flight_id", "delay_minutes"])
            for flight_id, delay_minutes in self.nonzero_items():
                writer.writerow([flight_id, max(1, int(delay_minutes))])

    # --- operations ----------------------------------------------------------------
    def merge(self, other: Mapping[str, int] | "DelayAssignmentTable", *, policy: DelayMergePolicy = "overwrite") -> "DelayAssignmentTable":
        """
        Create a new DelayAssignmentTable by merging delays from another mapping or table according to the given policy.
        
        Parameters:
            other (Mapping[str, int] | DelayAssignmentTable): Source of delays to merge; keys are flight IDs and values are delay minutes.
            policy (str): Merge strategy to apply for each flight. Supported values are:
                - "overwrite": replace existing delay with incoming delay
                - "max": keep the larger of existing and incoming delays
                - "sum": add incoming delay to existing delay
        
        Returns:
            DelayAssignmentTable: A new table containing the merged delays; the original table is not modified.
        
        Raises:
            ValueError: If `policy` is not one of "overwrite", "max", or "sum".
        """
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
        """
        Iterate over stored (flight_id, delay) pairs with delays greater than zero.
        
        Returns:
            Iterable[tuple[str, int]]: An iterable of (flight_id, delay_minutes) tuples for each stored flight whose delay is > 0.
        """
        return ((flight_id, delay) for flight_id, delay in self._delays.items() if delay > 0)

    # --- helpers -------------------------------------------------------------------
    @staticmethod
    def _normalize_static(value: object) -> int:
        """
        Convert a value to a non-negative integer representing delay minutes.
        
        Parameters:
        	value (object): A value convertible to int (e.g., int, str, float). 
        
        Returns:
        	int: The converted delay in minutes (>= 0).
        
        Raises:
        	ValueError: If the value cannot be converted to an integer or if the resulting integer is negative.
        """
        try:
            delay_int = int(value)
        except Exception as exc:
            raise ValueError(f"Invalid delay value: {value!r}") from exc
        if delay_int < 0:
            raise ValueError("Delay minutes cannot be negative")
        return delay_int

    def _normalize_delay(self, value: object) -> int:
        """
        Normalize an input value into a validated delay measured in minutes.
        
        Parameters:
            value (object): A value representing a delay; may be an int-like or string.
        
        Returns:
            int: The normalized delay as an integer number of minutes.
        
        Raises:
            ValueError: If the value cannot be converted to an integer or if the resulting delay is negative.
        """
        return self._normalize_static(value)


__all__ = ["DelayAssignmentTable"]
