"""
Test script for the Airspace Traffic Analysis API.

This script demonstrates how to use the API endpoints to retrieve
traffic volume occupancy data.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

from server_tailwind.airspace.airspace_api_wrapper import AirspaceAPIWrapper


async def test_api_functionality():
    """Test the API wrapper functionality directly."""
    
    print("=== Testing Airspace API Wrapper ===")
    
    try:
        # Initialize the API wrapper
        print("1. Initializing AirspaceAPIWrapper...")
        wrapper = AirspaceAPIWrapper()
        print("   ✅ API wrapper initialized")
        
        # Test getting available traffic volumes
        print("2. Getting available traffic volumes...")
        available_tvs = await wrapper.get_available_traffic_volumes()
        print(f"   ✅ Found {available_tvs['count']} traffic volumes")
        print(f"   First 5 traffic volumes: {available_tvs['available_traffic_volumes'][:5]}")
        
        # Test getting occupancy counts for a specific traffic volume
        if available_tvs['available_traffic_volumes']:
            test_tv_id = available_tvs['available_traffic_volumes'][0]
            print(f"3. Getting occupancy counts for traffic volume: {test_tv_id}")
            
            occupancy_data = await wrapper.get_traffic_volume_occupancy(test_tv_id)
            
            print(f"   ✅ Retrieved occupancy data for {test_tv_id}")
            print(f"   Total time windows: {occupancy_data['metadata']['total_time_windows']}")
            print(f"   Total flights in TV: {occupancy_data['metadata']['total_flights_in_tv']}")
            
            # Show first few time windows
            occupancy_counts = occupancy_data['occupancy_counts']
            first_windows = list(occupancy_counts.items())[:5]
            print("   First 5 time windows:")
            for time_window, count in first_windows:
                print(f"     {time_window}: {count} flights")
            
            # Test with a non-existent traffic volume
            print("4. Testing with non-existent traffic volume...")
            try:
                await wrapper.get_traffic_volume_occupancy("NONEXISTENT_TV")
            except ValueError as e:
                print(f"   ✅ Correctly handled non-existent TV: {e}")
        
        print("\n=== API Test Completed Successfully ===")
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_with_http_client():
    """Test using HTTP client (requires server to be running)."""
    
    print("\n=== Testing with HTTP Client ===")
    print("Note: This requires the server to be running on http://localhost:8000")
    
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            # Test root endpoint
            response = await client.get("http://localhost:8000/")
            print(f"Root endpoint: {response.status_code} - {response.json()}")
            
            # Test traffic volumes endpoint
            response = await client.get("http://localhost:8000/traffic_volumes")
            if response.status_code == 200:
                data = response.json()
                print(f"Traffic volumes: Found {data['count']} volumes")
                
                if data['available_traffic_volumes']:
                    test_tv = data['available_traffic_volumes'][0]
                    
                    # Test tv_count endpoint
                    response = await client.get(f"http://localhost:8000/tv_count?traffic_volume_id={test_tv}")
                    if response.status_code == 200:
                        tv_data = response.json()
                        print(f"TV Count for {test_tv}: {tv_data['metadata']['total_flights_in_tv']} total flights")
                    else:
                        print(f"TV Count request failed: {response.status_code}")
            else:
                print(f"Traffic volumes request failed: {response.status_code}")
                
    except ImportError:
        print("   httpx not available, install with: pip install httpx")
    except Exception as e:
        print(f"   HTTP test failed (server may not be running): {e}")


if __name__ == "__main__":
    # Test the API wrapper directly
    asyncio.run(test_api_functionality())
    
    # Test with HTTP client if httpx is available
    asyncio.run(test_with_http_client())