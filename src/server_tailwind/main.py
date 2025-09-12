"""
FastAPI server for airspace traffic analysis.
Provides endpoints for traffic volume occupancy analysis.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from typing import Dict, Any, Optional
from .airspace.airspace_api_wrapper import AirspaceAPIWrapper
from .deepflow.flows_api_wrapper import FlowsWrapper
from .CountAPIWrapper import CountAPIWrapper
from .core.resources import get_resources
from parrhesia.api.base_evaluation import compute_base_evaluation
from parrhesia.api.automatic_rate_adjustment import compute_automatic_rate_adjustment
try:
    import parrhesia.api.resources as parr_res
except Exception:
    parr_res = None  # type: ignore[assignment]


app = FastAPI(title="Airspace Traffic Analysis API", version="1.0.0", debug=True)

# Preload global resources and initialize wrappers
_res = get_resources().preload_all()
# Register shared resources with parrhesia so its modules can reuse them
if parr_res is not None:
    try:
        parr_res.set_global_resources(_res.indexer, _res.flight_list)
    except Exception as _e:
        # Non-fatal: parrhesia will fall back to disk loading
        print(f"Warning: failed to register parrhesia resources: {_e}")
airspace_wrapper = AirspaceAPIWrapper()
flows_wrapper = FlowsWrapper()
count_wrapper = CountAPIWrapper()

# Auth utilities (kept separate so endpoints remain pure)
from .auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

@app.get("/")
async def root(current_user: dict = Depends(get_current_user)):
    """Root endpoint providing API information."""
    return {"message": "Airspace Traffic Analysis API", "version": "1.0.0"}


@app.post("/token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token = create_access_token({"sub": user["username"]}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {
        "access_token": token,
        "token_type": "bearer",
        "display_name": user.get("display_name"),
        "organization": user.get("organization"),
    }


@app.get("/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    return {"hello": current_user["username"]}

@app.get("/tv_count")
async def get_tv_count(traffic_volume_id: str, current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get occupancy count for all time windows of a specific traffic volume.
    
    Args:
        traffic_volume_id: The traffic volume ID to analyze
        
    Returns:
        Dictionary with time windows as keys and occupancy counts as values
        Format: {"06:00-06:15": 42, "06:15-06:30": 35, ...}
    """
    try:
        result = await airspace_wrapper.get_traffic_volume_occupancy(traffic_volume_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/tv_count_with_capacity")
async def get_tv_count_with_capacity(traffic_volume_id: str, current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get occupancy counts for all time windows of a specific traffic volume,
    along with the hourly capacity from the GeoJSON.
    
    Returns a dictionary with keys:
    - traffic_volume_id
    - occupancy_counts: {"HH:MM-HH:MM": int}
    - hourly_capacity: {"HH:00-HH+1:00": float}
    - metadata
    """
    try:
        result = await airspace_wrapper.get_traffic_volume_occupancy_with_capacity(traffic_volume_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/slack_distribution")
async def get_slack_distribution(
    traffic_volume_id: str, ref_time_str: str, sign: str, delta_min: float = 0.0,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get slack distribution across all TVs by shifting the reference bin by nominal travel time.

    Query params:
    - traffic_volume_id: source TV id
    - ref_time_str: HHMM, HHMMSS, HH:MM or HH:MM:SS
    - sign: "plus" or "minus"
    - delta_min: additional time shift in minutes (can be negative); applied after travel-time shift
    """
    try:
        result = await airspace_wrapper.get_slack_distribution(
            traffic_volume_id=traffic_volume_id,
            ref_time_str=ref_time_str,
            sign=sign,
            delta_min=delta_min,
        )
        return result
    except ValueError as e:
        msg = str(e)
        if "sign must be one of" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=404, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/traffic_volumes")
async def get_traffic_volumes(current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get list of available traffic volume IDs.
    
    Returns:
        Dictionary with available traffic volume IDs and metadata
    """
    try:
        result = await airspace_wrapper.get_available_traffic_volumes()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/tv_flights")
async def get_tv_flights(traffic_volume_id: str, current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get flight identifiers grouped by time window for a specific traffic volume.

    Returns a mapping like {"06:00-06:15": ["flight1", "flight2", ...], ...}.
    """
    try:
        result = await airspace_wrapper.get_traffic_volume_flight_ids(traffic_volume_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/tv_flights_ordered")
async def get_tv_flights_ordered(traffic_volume_id: str, ref_time_str: str, current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get all flights for a traffic volume ordered by proximity to ref_time_str.

    ref_time_str should be numeric in HHMMSS (or HHMM) format, e.g., "084510" for 08:45:10.
    """
    try:
        result = await airspace_wrapper.get_traffic_volume_flights_ordered_by_ref_time(
            traffic_volume_id, ref_time_str
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/flow_extraction_legacy")
async def get_flow_extraction_legacy(
    traffic_volume_id: str,
    ref_time_str: str,
    threshold: float = 0.8,
    resolution: float = 1.0,
    flight_ids: Optional[str] = None,
    seed: Optional[int] = None,
    limit: Optional[int] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Legacy flow extraction using the original implementation under project_tailwind.

    Parameters:
    - traffic_volume_id: TV identifier
    - ref_time_str: reference time in HHMMSS (or HHMM) format; HH:MM and HH:MM:SS also accepted
    - threshold: similarity threshold for graph edges (default 0.8)
    - resolution: Leiden resolution parameter (default 1.0)
    - flight_ids: optional comma-separated flight IDs to cluster; if provided, community detection runs only on these flights
    - seed: optional random seed for Leiden
    - limit: optional cap on number of closest flights to include
    """
    try:
        result = await airspace_wrapper.get_flow_extraction(
            traffic_volume_id=traffic_volume_id,
            ref_time_str=ref_time_str,
            threshold=threshold,
            resolution=resolution,
            flight_ids=flight_ids,
            seed=seed,
            limit=limit,
        )
        return result
    except ValueError as e:
        # invalid TV id or invalid ref time, etc.
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/flow_extraction")
async def get_flow_extraction(
    traffic_volume_id: str,
    ref_time_str: str,
    threshold: float = 0.8,
    resolution: float = 1.0,
    flight_ids: Optional[str] = None,
    seed: Optional[int] = None,
    limit: Optional[int] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Flow extraction that uses parrhesia's implementation to assign community labels
    to flights near a reference time.

    Parameters:
    - traffic_volume_id: TV identifier
    - ref_time_str: reference time in HHMMSS (or HHMM) format; HH:MM and HH:MM:SS also accepted
    - threshold: similarity threshold for graph edges (default 0.8)
    - resolution: Leiden resolution parameter (default 1.0)
    - flight_ids: optional comma-separated flight IDs to cluster; if provided, community detection runs only on these flights
    - seed: optional random seed for Leiden
    - limit: optional cap on number of closest flights to include
    """
    try:
        result = await airspace_wrapper.get_flow_extraction_parrhesia(
            traffic_volume_id=traffic_volume_id,
            ref_time_str=ref_time_str,
            threshold=threshold,
            resolution=resolution,
            flight_ids=flight_ids,
            seed=seed,
            limit=limit,
        )
        return result
    except ValueError as e:
        # invalid TV id or invalid ref time, etc.
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/flows")
async def get_flows(
    tvs: str = Query(..., description="Comma-separated traffic volume IDs"),
    from_time_str: Optional[str] = Query(None, description="Start time (HHMM or HHMMSS or HH:MM[:SS])"),
    to_time_str: Optional[str] = Query(None, description="End time (HHMM or HHMMSS or HH:MM[:SS])"),
    threshold: Optional[float] = Query(None, description="Jaccard cutoff in [0,1] for clustering (default 0.1)"),
    resolution: Optional[float] = Query(None, description="Leiden resolution (>0), higher yields more clusters (default 1.0)"),
    current_user: dict = Depends(get_current_user),
) -> Any:
    try:
        return await flows_wrapper.get_flows(
            tvs=tvs,
            from_time_str=from_time_str,
            to_time_str=to_time_str,
            threshold=threshold,
            resolution=resolution,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute flows: {str(e)}")

@app.post("/original_counts")
async def original_counts(request: Dict[str, Any], current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Compute occupancy counts over traffic volumes with optional time range, ranking and rolling-hour mode.

    Request JSON:
    - traffic_volume_ids: list[str] (optional). If provided, a dedicated 'mentioned_counts' field is returned for these TVs.
    - from_time_str: str (optional; HHMM, HHMMSS, HH:MM, or HH:MM:SS)
    - to_time_str: str (optional; same formats; required if from_time_str provided)
    - categories: dict[str -> list[str]] (optional category -> flight_ids)
    - flight_ids: list[str] (optional; ignored if categories present). When present and categories are absent, only these flights are counted.
    - rank_by: str (optional; default "total_count"). Supported: "total_count", "total_excess".
    - rolling_hour: bool (optional; default true). When true, each bin's count is the sum over the next hour.
    - include_overall: bool (accepted for backward-compatibility; counts are always included for the top 50 TVs)

    Always returns counts for the top_k=50 TVs ranked by 'rank_by' over the selected time range.
    Also returns capacity arrays per bin aligned with the returned counts: 
    - 'capacity' for the top-k TVs, and 'mentioned_capacity' when 'traffic_volume_ids' are provided.
    """
    try:
        tvs = request.get("traffic_volume_ids")
        if tvs is not None and not isinstance(tvs, list):
            raise HTTPException(status_code=400, detail="'traffic_volume_ids' must be a list when provided")

        from_time_str = request.get("from_time_str")
        to_time_str = request.get("to_time_str")
        categories = request.get("categories")
        flight_ids = request.get("flight_ids")
        include_overall = bool(request.get("include_overall", True))
        rank_by = str(request.get("rank_by", "total_excess"))
        rolling_hour = bool(request.get("rolling_hour", True))

        result = await count_wrapper.get_original_counts(
            traffic_volume_ids=[str(x) for x in tvs] if isinstance(tvs, list) else None,
            from_time_str=str(from_time_str) if from_time_str is not None else None,
            to_time_str=str(to_time_str) if to_time_str is not None else None,
            categories=categories if isinstance(categories, dict) else None,
            flight_ids=[str(x) for x in flight_ids] if isinstance(flight_ids, list) else None,
            include_overall=include_overall,
            rank_by=rank_by,
            rolling_hour=rolling_hour,
        )
        return result
    except HTTPException:
        raise
    except ValueError as e:
        msg = str(e)
        if msg.startswith("Unknown traffic_volume_ids"):
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/hotspots")
async def get_hotspots(threshold: float = 0.0, current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get list of hotspots (traffic volume and time bin where capacity exceeds demands).
    
    Returns hotspots with traffic_volume_id, time bin, z_max, z_sum, and other statistics.
    
    Args:
        threshold: Minimum excess traffic to consider as overloaded (default: 0.0)
    """
    try:
        result = await airspace_wrapper.get_hotspots(threshold)
        return result
    except Exception as e:
        print(f"Error in get_hotspots: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/regulation_ranking_tv_flights_ordered")
async def get_regulation_ranking_tv_flights_ordered(
    traffic_volume_id: str,
    ref_time_str: str,
    seed_flight_ids: str,
    duration_min: Optional[int] = None,
    top_k: Optional[int] = None,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Return flights passing a traffic volume near a reference time, ordered by proximity.

    Parameters:
    - traffic_volume_id: TV identifier
    - ref_time_str: reference time in HHMMSS (or HHMM) format
    - seed_flight_ids: comma-separated seed flight IDs (accepted but not used)
    - duration_min: optional positive integer; keep only flights whose entry time into the TV is in [ref_time_str, ref_time_str + duration_min]
    - top_k: optional limit on number of results

    Returns flights with arrival information. Scores and components are omitted.
    """
    try:
        result = await airspace_wrapper.get_regulation_ranking_tv_flights_ordered(
            traffic_volume_id=traffic_volume_id,
            ref_time_str=ref_time_str,
            seed_flight_ids=seed_flight_ids,
            duration_min=duration_min,
            top_k=top_k,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/regulation_plan_simulation")
async def regulation_plan_simulation(request: Dict[str, Any], current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Simulate a regulation plan and return:
    - per-flight delays and delay stats
    - objective and component breakdown
    - rolling-hour occupancy arrays (pre and post) for all bins of the top-K busiest TVs
      (busiest defined as TVs with highest max(pre_rolling_count - hourly_capacity)
      over the active regulation time windows)

    Request JSON keys:
    - regulations: List[str|object]
    - weights: Optional[dict]
    - top_k: Optional[int] (number of TVs to include; default 25)
    - include_excess_vector: Optional[bool]
    """
    try:
        regs = request.get("regulations", [])
        weights = request.get("weights")
        top_k = int(request.get("top_k", 25))
        include_excess_vector = bool(request.get("include_excess_vector", False))

        result = await airspace_wrapper.run_regulation_plan_simulation(
            regs,
            weights=weights,
            top_k=top_k,
            include_excess_vector=include_excess_vector,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/base_evaluation")
def post_base_evaluation(payload: dict, current_user: dict = Depends(get_current_user)):
    # Minimal validation: require targets and flows to be present
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    if not payload.get("targets"):
        raise HTTPException(status_code=400, detail="'targets' is required")
    if not payload.get("flows"):
        raise HTTPException(status_code=400, detail="'flows' is required")
    try:
        result = compute_base_evaluation(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"Exception in /base_evaluation: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to compute base evaluation: {e}")
    return result


@app.post("/automatic_rate_adjustment")
def post_automatic_rate_adjustment(payload: dict, current_user: dict = Depends(get_current_user)):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body required")
    if not payload.get("targets"):
        raise HTTPException(status_code=400, detail="'targets' is required")
    if not payload.get("flows"):
        raise HTTPException(status_code=400, detail="'flows' is required")
    try:
        result = compute_automatic_rate_adjustment(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"Exception in /automatic_rate_adjustment: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize: {e}")
    return result


@app.post("/autorate_occupancy")
def post_autorate_occupancy(payload: dict, current_user: dict = Depends(get_current_user)):
    """
    Aggregate pre/post occupancy across flows for TVs present in a prior
    /automatic_rate_adjustment response. No optimization is run.

    Request JSON:
      - autorate_result: object (required)
      - include_capacity: bool (optional; default true)
      - rolling_hour: bool (optional; default true)
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body required")
    if not payload.get("autorate_result"):
        raise HTTPException(status_code=400, detail="'autorate_result' is required")
    try:
        include_capacity = bool(payload.get("include_capacity", True))
        rolling_hour = bool(payload.get("rolling_hour", True))
        return count_wrapper.compute_autorate_occupancy(
            payload.get("autorate_result") or {}, include_capacity=include_capacity, rolling_hour=rolling_hour
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"Exception in /autorate_occupancy: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to compute autorate occupancy: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
