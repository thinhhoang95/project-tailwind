import sys
from pathlib import Path

project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

import geopandas as gpd
import numpy as np
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator

# Rich formatting
try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def load_traffic_volumes_gdf():
    """Load the traffic volumes GeoDataFrame."""
    return gpd.read_file(
        "D:/project-cirrus/cases/scenarios/wxm_sm_ih_maxpool.geojson"
        # "D:/project-cirrus/output/scenarios/summer_good_wx_well_staffed_low.geojson"
    )


def find_busiest_traffic_volumes_by_hour():
    """Find the top 10 busiest traffic_volume_ids (by occupancy_count) for each hour."""
    
    print("=== Finding Top 10 Busiest Traffic Volumes by Hour ===")
    
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
        print(f"   OK Loaded {flight_list.num_flights} flights with {flight_list.num_tvtws} TVTWs")
        
        # Initialize evaluator
        print("3. Initializing NetworkEvaluator...")
        evaluator = NetworkEvaluator(traffic_volumes_gdf, flight_list)
        print("   OK NetworkEvaluator initialized")
        
        # Get hourly occupancy data
        print("4. Computing hourly occupancy data...")
        evaluator.compute_excess_traffic_vector()  # This populates last_hourly_occupancy_matrix
        hourly_occupancy_matrix = evaluator.last_hourly_occupancy_matrix
        
        if hourly_occupancy_matrix is None:
            print("X Failed to compute hourly occupancy matrix")
            return
        
        # Create reverse mapping from row index to traffic volume ID
        row_idx_to_tv_id = {}
        for tv_id, tv_idx in evaluator.tv_id_to_idx.items():
            row_idx = evaluator.tv_id_to_row_idx[tv_id]
            row_idx_to_tv_id[row_idx] = tv_id
        
        print("5. Finding top 10 busiest traffic volumes for each hour...")
        
        # For each hour (0-23), find top 10 traffic volumes with highest occupancy
        for hour in range(24):
            if hour >= hourly_occupancy_matrix.shape[1]:
                continue
                
            # Get occupancy counts for all traffic volumes at this hour
            hour_occupancy = hourly_occupancy_matrix[:, hour]
            
            # Find indices with non-zero occupancy
            non_zero_indices = np.where(hour_occupancy > 0)[0]
            if len(non_zero_indices) == 0:
                continue
            
            # Get top 10 (or fewer if less than 10 exist)
            top_n = min(10, len(non_zero_indices))
            top_indices = np.argsort(hour_occupancy)[-top_n:][::-1]  # Sort descending
            
            # Create table data
            table_data = []
            for rank, idx in enumerate(top_indices, 1):
                occupancy = hour_occupancy[idx]
                if occupancy <= 0:
                    continue
                tv_id = row_idx_to_tv_id.get(idx, "Unknown")
                table_data.append((rank, tv_id, int(occupancy)))
            
            if not table_data:
                continue
            
            # Display with rich table or fallback
            if RICH_AVAILABLE:
                table = Table(title=f"Top 10 Busiest Traffic Volumes - Hour {hour:02d}:00")
                table.add_column("Rank", justify="right", style="cyan")
                table.add_column("Traffic Volume ID", style="magenta")
                table.add_column("Occupancy Count", justify="right", style="green")
                
                for rank, tv_id, occupancy in table_data:
                    table.add_row(str(rank), tv_id, str(occupancy))
                
                console.print(table)
                console.print()
            else:
                print(f"\nTop 10 Busiest Traffic Volumes - Hour {hour:02d}:00")
                print("Rank | Traffic Volume ID | Occupancy Count")
                print("-" * 50)
                for rank, tv_id, occupancy in table_data:
                    print(f"{rank:4d} | {tv_id:<17} | {occupancy:13d}")
                print()
        
        print("=== Top 10 Busiest Traffic Volume Analysis Completed ===")
        
    except Exception as e:
        print(f"X Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    find_busiest_traffic_volumes_by_hour()