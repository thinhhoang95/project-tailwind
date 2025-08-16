#!/usr/bin/env python3
"""
Extract hotspot data containing tvtw_index and export to output directory.
"""

import sys
import json
from pathlib import Path

# Add project source to path
project_root = Path(__file__).parent / "src"
sys.path.insert(0, str(project_root))

import geopandas as gpd
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator


def load_traffic_volumes_gdf():
    """Load the traffic volumes GeoDataFrame."""
    return gpd.read_file(
        "D:/project-cirrus/cases/traffic_volumes_simplified.geojson"
    )


def extract_hotspots_to_file():
    """Extract all hotspots with tvtw_index and save to output file."""
    
    print("=== Extracting Hotspot Data with tvtw_index ===")
    
    # Check if required data files exist
    required_files = [
        "output/tvtw_indexer.json",
        "output/so6_occupancy_matrix_with_times.json",
    ]
    missing_files = [p for p in required_files if not Path(p).exists()]
    if missing_files:
        print(f"X Missing required files: {missing_files}")
        print("Please ensure these files exist in the project root.")
        return
    
    try:
        # Load traffic volumes and flights
        print("1. Loading traffic volumes GeoDataFrame...")
        traffic_volumes_gdf = load_traffic_volumes_gdf()
        print(f"   OK Loaded {len(traffic_volumes_gdf)} traffic volumes")

        print("2. Loading Flight List...")
        flight_list = FlightList(
            occupancy_file_path="output/so6_occupancy_matrix_with_times.json",
            tvtw_indexer_path="output/tvtw_indexer.json",
        )
        print(
            f"   OK Loaded {flight_list.num_flights} flights with {flight_list.num_tvtws} TVTWs"
        )

        # Initialize evaluator
        print("3. Initializing NetworkEvaluator...")
        evaluator = NetworkEvaluator(traffic_volumes_gdf, flight_list)
        print("   OK NetworkEvaluator initialized")

        # Retrieve hotspots (per bin) with tvtw_index
        print("4. Retrieving all hotspot flights per TVTW bin...")
        per_bin = evaluator.get_hotspot_flights(threshold=0.0, mode="bin")
        print(f"   Found {len(per_bin)} hotspot bins")
        
        # Output to file
        output_path = "output/hotspots_with_tvtw_index.json"
        print(f"5. Saving hotspot data to {output_path}...")
        
        # Prepare data for export
        export_data = {
            "metadata": {
                "description": "Hotspot data with tvtw_index for visualization",
                "total_bins": len(per_bin),
                "time_bin_minutes": flight_list.time_bin_minutes,
                "extraction_timestamp": str(Path(__file__).stat().st_mtime)
            },
            "hotspots": per_bin
        }
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        
        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"   Successfully saved {len(per_bin)} hotspot bins to {output_path}")
        
        # Show sample of data
        if per_bin:
            sample = per_bin[0]
            print(f"   Sample hotspot: tvtw_index={sample['tvtw_index']}, flights={len(sample['flight_ids'])}")
            
            # Show a few more samples - loading indexer from the saved file
            from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
            indexer = TVTWIndexer.load("output/tvtw_indexer.json")
            
            for i, hotspot in enumerate(per_bin[:5], 1):
                tvtw_info = indexer.get_human_readable_tvtw(hotspot['tvtw_index'])
                if tvtw_info:
                    tv_name, time_window = tvtw_info
                    print(f"     [{i}] TVTW {hotspot['tvtw_index']}: {tv_name} at {time_window} - {len(hotspot['flight_ids'])} flights")
        
        print(f"\n=== Hotspot Extraction Completed Successfully ===")
        print(f"Data saved to: {output_path}")
        
    except Exception as e:
        print(f"X Hotspot extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    extract_hotspots_to_file()