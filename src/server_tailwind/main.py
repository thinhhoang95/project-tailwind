"""
FastAPI server for airspace traffic analysis.
Provides endpoints for traffic volume occupancy analysis.
"""

from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional
from .airspace.airspace_api_wrapper import AirspaceAPIWrapper

app = FastAPI(title="Airspace Traffic Analysis API", version="1.0.0")

# Initialize the airspace API wrapper
airspace_wrapper = AirspaceAPIWrapper()

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {"message": "Airspace Traffic Analysis API", "version": "1.0.0"}

@app.get("/tv_count")
async def get_tv_count(traffic_volume_id: str) -> Dict[str, Any]:
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
async def get_tv_count_with_capacity(traffic_volume_id: str) -> Dict[str, Any]:
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

@app.get("/traffic_volumes")
async def get_traffic_volumes() -> Dict[str, Any]:
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
async def get_tv_flights(traffic_volume_id: str) -> Dict[str, Any]:
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
async def get_tv_flights_ordered(traffic_volume_id: str, ref_time_str: str) -> Dict[str, Any]:
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

@app.get("/hotspots")
async def get_hotspots(threshold: float = 0.0) -> Dict[str, Any]:
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
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/regulation_ranking_tv_flights_ordered")
async def get_regulation_ranking_tv_flights_ordered(
    traffic_volume_id: str,
    ref_time_str: str,
    seed_flight_ids: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Rank flights passing a traffic volume near a reference time using heuristic features.

    Parameters:
    - traffic_volume_id: TV identifier
    - ref_time_str: reference time in HHMMSS (or HHMM) format
    - seed_flight_ids: comma-separated seed flight IDs
    - top_k: optional limit on number of results

    Returns ranked flights with arrival time, score and component breakdown.
    """
    try:
        result = await airspace_wrapper.get_regulation_ranking_tv_flights_ordered(
            traffic_volume_id=traffic_volume_id,
            ref_time_str=ref_time_str,
            seed_flight_ids=seed_flight_ids,
            top_k=top_k,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)