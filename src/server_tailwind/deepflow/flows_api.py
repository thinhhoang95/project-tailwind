from typing import Optional
from fastapi import FastAPI, HTTPException, Query

# Import compute_flows directly from the parrhesia package
from parrhesia.api.flows import compute_flows
from parrhesia.api.base_evaluation import compute_base_evaluation


@app.get("/flows")
def get_flows(
    tvs: str = Query(..., description="Comma-separated traffic volume IDs"),
    timebins: Optional[str] = Query(None, description="Comma-separated time bin indices (0=midnight)"),
    threshold: Optional[float] = Query(None, description="Jaccard cutoff in [0,1] for clustering (default 0.1)"),
    resolution: Optional[float] = Query(None, description="Leiden resolution (>0), higher yields more clusters (default 1.0)"),
):
    if not tvs.strip():
        raise HTTPException(status_code=400, detail="Parameter 'tvs' is required and cannot be empty")
    tv_list = [s.strip() for s in tvs.split(",") if s.strip()]
    bins_list = None
    if timebins is not None and timebins.strip():
        try:
            bins_list = [int(x.strip()) for x in timebins.split(",") if x.strip()]
        except Exception:
            raise HTTPException(status_code=400, detail="Parameter 'timebins' must be a comma-separated list of integers")

    # Validate optional clustering params if provided
    if threshold is not None:
        try:
            tval = float(threshold)
        except Exception:
            raise HTTPException(status_code=400, detail="Parameter 'threshold' must be a float")
        if not (0.0 <= tval <= 1.0):
            raise HTTPException(status_code=400, detail="Parameter 'threshold' must be in [0, 1]")
        threshold = tval
    if resolution is not None:
        try:
            rval = float(resolution)
        except Exception:
            raise HTTPException(status_code=400, detail="Parameter 'resolution' must be a float")
        if not (rval > 0.0):
            raise HTTPException(status_code=400, detail="Parameter 'resolution' must be > 0")
        resolution = rval

    try:
        result = compute_flows(
            tvs=tv_list,
            timebins=bins_list,
            threshold=threshold,
            resolution=resolution,
        )
    except Exception as e:
        import traceback
        print(f"Exception in get_flows: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to compute flows: {e}")
    return result





@app.post("/base_evaluation")
def post_base_evaluation(payload: dict):
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