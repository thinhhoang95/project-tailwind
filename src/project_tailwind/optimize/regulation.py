from typing import List, Optional


class Regulation:
    def __init__(self, regulation_str: str):
        self.raw_str = regulation_str
        parts = self.raw_str.split()
        if len(parts) != 4:
            raise ValueError("Regulation string must have 4 parts")

        self.ref_loc, self.filter_cond, self.rate, self.time_windows_str = parts

        self.type, self.location = self.ref_loc.split('_', 1)
        if self.type != 'TV':
            raise NotImplementedError("Only 'TV' type is supported for reference location.")

        self.filter_type, self.filter_value = self.filter_cond.split('_', 1)

        self.rate = int(self.rate)
        self.time_windows = self._parse_time_windows(self.time_windows_str)

        # Optional: explicit flight targeting (overrides parser if provided)
        self.target_flight_ids: Optional[List[str]] = None

    @classmethod
    def from_components(
        cls,
        *,
        location: str,
        rate: int,
        time_windows: List[int],
        filter_type: str = 'IC',
        filter_value: str = '__',
        target_flight_ids: Optional[List[str]] = None,
    ) -> "Regulation":
        """
        Build a Regulation directly from components. Optionally attach an explicit
        list of targeted flight identifiers which will be used instead of parsing.
        """
        # Construct a human-readable raw string for logging/debugging consistency
        def _windows_to_str(wins: List[int]) -> str:
            if not wins:
                return ""
            # Compact consecutive ranges for readability
            wins_sorted = sorted(int(w) for w in wins)
            ranges: List[str] = []
            start = prev = wins_sorted[0]
            for w in wins_sorted[1:]:
                if w == prev + 1:
                    prev = w
                    continue
                ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
                start = prev = w
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            return ",".join(ranges)

        raw = f"TV_{location} {filter_type}_{filter_value} {int(rate)} {_windows_to_str(time_windows)}"
        obj = cls(raw)
        # Overwrite with authoritative values (parsing above mirrors these anyway)
        obj.location = location
        obj.rate = int(rate)
        obj.time_windows = list(time_windows)
        obj.filter_type = filter_type
        obj.filter_value = filter_value
        obj.target_flight_ids = list(target_flight_ids) if target_flight_ids else None
        return obj

    def _parse_time_windows(self, time_windows_str: str) -> List[int]:
        """
        Parse time windows string supporting both comma-separated values and ranges.
        Examples:
        - "36,37,38" -> [36, 37, 38]
        - "36-38" -> [36, 37, 38]
        - "36,39-41,45" -> [36, 39, 40, 41, 45]
        """
        time_windows = []
        parts = time_windows_str.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                # Handle range like "47-49"
                start_str, end_str = part.split('-', 1)
                start = int(start_str.strip())
                end = int(end_str.strip())
                time_windows.extend(range(start, end + 1))
            else:
                # Handle single value
                time_windows.append(int(part))
        
        return time_windows