from typing import List


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