from typing import Optional, Any, Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import compute_flows directly from the parrhesia package
from parrhesia.api.flows import compute_flows


class FlowsWrapper:
    """
    Wrapper around parrhesia flows API to decouple HTTP handling from business logic.

    Exposes an async method to compute flows given query-like parameters, while
    running the heavy work in a background thread to avoid blocking the event loop.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def get_flows(
        self,
        *,
        tvs: str,
        timebins: Optional[str] = None,
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

        # Parse time bins if provided
        bins_list: Optional[List[int]] = None
        if timebins is not None and str(timebins).strip():
            try:
                bins_list = [int(x.strip()) for x in str(timebins).split(",") if x.strip()]
            except Exception:
                raise ValueError("Parameter 'timebins' must be a comma-separated list of integers")

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


