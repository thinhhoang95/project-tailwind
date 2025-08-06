"""
Test script for FlightList and NetworkEvaluator implementation.
"""

import sys
import os
import geopandas as gpd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.project_tailwind.optimize.eval.flight_list import FlightList
from src.project_tailwind.optimize.eval.network_evaluator import NetworkEvaluator


def test_flight_list():
    """Test FlightList functionality."""
    print("=" * 60)
    print("TESTING FLIGHTLIST")
    print("=" * 60)
    
    # File paths
    occupancy_file = r"D:\project-tailwind\output\so6_occupancy_matrix_with_times.json"
    tvtw_indexer_file = r"D:\project-tailwind\output\tvtw_indexer.json"
    
    print(f"Loading flight data from: {occupancy_file}")
    print(f"Loading TVTW indexer from: {tvtw_indexer_file}")
    
    # Initialize FlightList
    flight_list = FlightList(occupancy_file, tvtw_indexer_file)
    
    # Display summary statistics
    stats = flight_list.get_summary_stats()
    print(f"\nFlight Data Summary:")
    print(f"  Number of flights: {stats['num_flights']:,}")
    print(f"  Number of TVTWs: {stats['num_tvtws']:,}")
    print(f"  Number of traffic volumes: {stats['num_traffic_volumes']:,}")
    print(f"  Time bin size: {stats['time_bin_minutes']} minutes")
    print(f"  Matrix sparsity: {stats['matrix_sparsity']:.4f}")
    print(f"  Total TVTW occupancies: {stats['total_tvtw_occupancies']:,}")
    print(f"  Max occupancy per TVTW: {stats['max_occupancy_per_tvtw']:.1f}")
    print(f"  Average occupancy per TVTW: {stats['avg_occupancy_per_tvtw']:.2f}")
    
    # Test getting occupancy for a specific flight
    first_flight_id = flight_list.flight_ids[0]
    print(f"\nTesting occupancy vector for flight {first_flight_id}:")
    
    occupancy_vector = flight_list.get_occupancy_vector(first_flight_id)
    non_zero_indices = occupancy_vector.nonzero()[0]
    print(f"  Occupancy vector length: {len(occupancy_vector)}")
    print(f"  Non-zero entries: {len(non_zero_indices)}")
    print(f"  First few non-zero TVTW indices: {non_zero_indices[:5].tolist()}")
    
    # Test flight metadata
    metadata = flight_list.get_flight_metadata(first_flight_id)
    print(f"\nFlight metadata for {first_flight_id}:")
    print(f"  Origin: {metadata['origin']}")
    print(f"  Destination: {metadata['destination']}")
    print(f"  Takeoff time: {metadata['takeoff_time']}")
    print(f"  Distance: {metadata['distance']:.2f} km")
    print(f"  Number of occupancy intervals: {len(metadata['occupancy_intervals'])}")
    
    return flight_list


def test_network_evaluator(flight_list):
    """Test NetworkEvaluator functionality."""
    print("\n" + "=" * 60)
    print("TESTING NETWORKEVALUATOR")
    print("=" * 60)
    
    # Load traffic volumes
    traffic_volumes_file = r"D:\project-cirrus\cases\traffic_volumes_simplified.geojson"
    print(f"Loading traffic volumes from: {traffic_volumes_file}")
    
    traffic_volumes_gdf = gpd.read_file(traffic_volumes_file)
    print(f"Loaded {len(traffic_volumes_gdf)} traffic volumes")
    
    # Initialize NetworkEvaluator
    evaluator = NetworkEvaluator(traffic_volumes_gdf, flight_list)
    
    # Compute excess traffic vector
    print("\nComputing excess traffic vector...")
    excess_vector = evaluator.compute_excess_traffic_vector()
    
    overload_count = (excess_vector > 0).sum()
    total_excess = excess_vector.sum()
    max_excess = excess_vector.max()
    
    print(f"  Total TVTWs: {len(excess_vector):,}")
    print(f"  Overloaded TVTWs: {overload_count:,}")
    print(f"  Total excess traffic: {total_excess:.1f}")
    print(f"  Maximum excess traffic: {max_excess:.1f}")
    
    # Get overloaded TVTWs details
    print("\nTop 10 most overloaded TVTWs:")
    overloaded_tvtws = evaluator.get_overloaded_tvtws()
    
    for i, tvtw in enumerate(overloaded_tvtws[:10], 1):
        print(f"  {i:2d}. TVTW {tvtw['tvtw_index']:5d} (TV: {tvtw['traffic_volume_id']}):")
        print(f"      Hourly Context: Occupancy={tvtw['hourly_occupancy']:.1f}, Capacity={tvtw['hourly_capacity']:.1f}")
        print(f"      TVTW Bin:       Occupancy={tvtw['occupancy']:6.1f}, Capacity={tvtw['capacity_per_bin']:6.1f}")
        print(f"      Excess (TVTW scale): {tvtw['excess']:6.1f}, Utilization: {tvtw['utilization_ratio']:.2f}")

    # Compute horizon metrics
    print("\nHorizon metrics (first 1000 time windows):")
    horizon_metrics = evaluator.compute_horizon_metrics(1000)
    print(f"  z_max (max excess): {horizon_metrics['z_max']:.1f}")
    print(f"  z_sum (total excess): {horizon_metrics['z_sum']:.1f}")
    print(f"  Horizon windows: {horizon_metrics['horizon_windows']:,}")
    
    # Capacity utilization statistics
    print("\nCapacity utilization statistics:")
    util_stats = evaluator.get_capacity_utilization_stats()
    print(f"  Mean utilization: {util_stats['mean_utilization']:.3f}")
    print(f"  Max utilization: {util_stats['max_utilization']:.3f}")
    print(f"  System utilization: {util_stats['system_utilization']:.3f}")
    print(f"  Total capacity: {util_stats['total_capacity']:,.0f}")
    print(f"  Total demand: {util_stats['total_demand']:,.0f}")
    print(f"  Overloaded TVTWs: {util_stats['overloaded_tvtws']:,}")
    print(f"  Overload percentage: {util_stats['overload_percentage']:.2f}%")
    
    # Traffic volume summary
    print("\nTop 10 traffic volumes by excess traffic:")
    tv_summary = evaluator.get_traffic_volume_summary()
    top_10 = tv_summary.head(10)
    
    for idx, row in top_10.iterrows():
        print(f"  {row['traffic_volume_id']:12s}: "
              f"Excess={row['total_excess']:6.1f}, "
              f"Util={row['utilization_ratio']:.3f}, "
              f"Overloaded bins={row['overloaded_bins']}")
    
    return evaluator


def main():
    """Main test function."""
    print("Testing FlightList and NetworkEvaluator Implementation")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test FlightList
        flight_list = test_flight_list()
        
        # Test NetworkEvaluator
        evaluator = test_network_evaluator(flight_list)
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Export results for inspection
        output_file = r"D:\project-tailwind\output\overload_analysis_test.json"
        print(f"\nExporting test results to: {output_file}")
        evaluator.export_results(output_file, horizon_time_windows=1000)
        print("Results exported successfully!")
        
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
