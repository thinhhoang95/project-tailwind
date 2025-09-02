from typing import Optional, Any, Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import compute_flows directly from the parrhesia package
from parrhesia.api.flows import compute_flows
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer


class FlowsWrapper:
    """
    Wrapper around parrhesia flows API to decouple HTTP handling from business logic.

    Exposes an async method to compute flows given query-like parameters, while
    running the heavy work in a background thread to avoid blocking the event loop.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2)
        # Load TVTW indexer once for time bin conversion
        self._tvtw_indexer = TVTWIndexer.load("output/tvtw_indexer.json")

    async def get_flows(
        self,
        *,
        tvs: str,
        from_time_str: Optional[str] = None,
        to_time_str: Optional[str] = None,
        threshold: Optional[float] = None,
        resolution: Optional[float] = None,
    ) -> Any:
        """
        Compute flows for the given traffic volumes and optional time bins.

        Args:
            tvs: Comma-separated traffic volume IDs.
            timebins: Optional comma-separated list of time bin indices (0 = midnight).
            threshold: Optional Jaccard cutoff in [0,1] for clustering.
            resolution: Optional Leiden resolution (>0), higher yields more clusters.
        """

        if not isinstance(tvs, str) or not tvs.strip():
            raise ValueError("Parameter 'tvs' is required and cannot be empty")

        # Parse TVs list
        tv_list: List[str] = [s.strip() for s in tvs.split(",") if s.strip()]

        # Convert from/to time strings to bin indices if provided
        bins_list: Optional[List[int]] = None
        if (from_time_str is not None and str(from_time_str).strip()) or (
            to_time_str is not None and str(to_time_str).strip()
        ):
            if not (from_time_str and to_time_str):
                raise ValueError("Both 'from_time_str' and 'to_time_str' must be provided together")

            def _parse_time_to_seconds(value: str) -> int:
                s = str(value).strip()
                if ":" in s:
                    parts = s.split(":")
                    if len(parts) not in (2, 3):
                        raise ValueError("Time must be HH:MM or HH:MM:SS")
                    hour = int(parts[0])
                    minute = int(parts[1])
                    second = int(parts[2]) if len(parts) == 3 else 0
                else:
                    if not s.isdigit() or len(s) not in (4, 6):
                        raise ValueError("Time must be numeric 'HHMM' or 'HHMMSS'")
                    hour = int(s[0:2])
                    minute = int(s[2:4])
                    second = int(s[4:6]) if len(s) == 6 else 0
                if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
                    raise ValueError("Time contains invalid components")
                return hour * 3600 + minute * 60 + second

            start_seconds = _parse_time_to_seconds(from_time_str)
            end_seconds = _parse_time_to_seconds(to_time_str)

            if end_seconds < start_seconds:
                raise ValueError("'to_time_str' must be greater than or equal to 'from_time_str'")

            seconds_per_bin = int(self._tvtw_indexer.time_bin_minutes) * 60
            num_bins = int(1440 // int(self._tvtw_indexer.time_bin_minutes))
            start_bin = start_seconds // seconds_per_bin
            end_bin = end_seconds // seconds_per_bin

            # Clamp to valid range just in case
            start_bin = max(0, min(int(start_bin), num_bins - 1))
            end_bin = max(0, min(int(end_bin), num_bins - 1))

            if start_bin > end_bin:
                raise ValueError("Computed start bin is after end bin; check time inputs")

            bins_list = list(range(int(start_bin), int(end_bin) + 1))

        # Validate optional clustering params if provided
        if threshold is not None:
            try:
                tval = float(threshold)
            except Exception:
                raise ValueError("Parameter 'threshold' must be a float")
            if not (0.0 <= tval <= 1.0):
                raise ValueError("Parameter 'threshold' must be in [0, 1]")
            threshold = tval

        if resolution is not None:
            try:
                rval = float(resolution)
            except Exception:
                raise ValueError("Parameter 'resolution' must be a float")
            if not (rval > 0.0):
                raise ValueError("Parameter 'resolution' must be > 0")
            resolution = rval

        loop = asyncio.get_event_loop()

        def _compute() -> Any:
            return compute_flows(
                tvs=tv_list,
                timebins=bins_list,
                threshold=threshold,
                resolution=resolution,
            )

        try:
            return await loop.run_in_executor(self._executor, _compute)
        except Exception:
            # Let callers decide on response mapping (e.g., HTTP 500)
            raise

    def __del__(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)


