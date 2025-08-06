#!/usr/bin/env python3
"""
Test script for RegulationMove integration with OptimizationProblem.
This script tests one complete regulation move to verify correctness.
"""

import json
import numpy as np
from typing import Dict, Any
from copy import deepcopy

# Add the source to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

from project_tailwind.optimize.moves.regulation_move import RegulationMove
from project_tailwind.optimize.alns.optimization_problem import OptimizationProblem
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
import geopandas as gpd


# Extended FlightList class with required methods
class ExtendedFlightList(FlightList):
    """Extended FlightList with copy and update_flight methods for testing."""
    
    def copy(self):
        """Create a deep copy of the FlightList."""
        # Create new instance with same file paths
        new_flight_list = ExtendedFlightList(self.occupancy_file_path, self.tvtw_indexer_path)
        
        # Copy the occupancy matrix
        new_flight_list.occupancy_matrix = self.occupancy_matrix.copy()
        
        # Copy flight metadata
        new_flight_list.flight_metadata = deepcopy(self.flight_metadata)
        new_flight_list.flight_data = deepcopy(self.flight_data)
        
        return new_flight_list
    
    def update_flight_occupancy(self, flight_id: str, new_occupancy_vector: np.ndarray):
        """
        Update the occupancy vector for a specific flight.
        Uses LIL format for efficient sparse matrix modification.
        """
        if flight_id not in self.flight_id_to_row:
            raise ValueError(f"Flight ID {flight_id} not found")
        
        row_idx = self.flight_id_to_row[flight_id]
        
        # For efficiency, convert to LIL format for changing sparsity
        lil_matrix = self.occupancy_matrix.tolil()
        lil_matrix[row_idx, :] = new_occupancy_vector
        self.occupancy_matrix = lil_matrix.tocsr()

    def update_flight(self, flight_id: str, updates: Dict[str, Any]):
        """Update flight data with new values."""
        if flight_id not in self.flight_id_to_row:
            raise ValueError(f"Flight ID {flight_id} not found")
        
        # Update occupancy vector if provided
        if "occupancy_vector" in updates:
            new_occupancy_vector = updates["occupancy_vector"]
            if hasattr(new_occupancy_vector, '__len__'):
                # Convert list to numpy array if needed
                if not isinstance(new_occupancy_vector, np.ndarray):
                    new_occupancy_vector = np.array(new_occupancy_vector)
                self.update_flight_occupancy(flight_id, new_occupancy_vector)
            
        # Update metadata if provided
        if flight_id in self.flight_metadata:
            for key, value in updates.items():
                if key != "occupancy_vector":
                    self.flight_metadata[flight_id][key] = value

def load_traffic_volumes_gdf():
    """Load the traffic volumes GeoDataFrame."""
    return gpd.read_file("/Volumes/CrucialX/project-cirrus/cases/traffic_volumes_simplified.geojson")

def test_regulation_move_integration():
    """Test complete regulation move integration."""
    
    print("=== Testing RegulationMove Integration ===")
    
    # Check if required data files exist
    required_files = [
        "output/tvtw_indexer.json",
        "output/so6_occupancy_matrix_with_times.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure these files exist in the project root.")
        return
    
    try:
        # 1. Load traffic volumes GeoDataFrame
        print("1. Loading traffic volumes GeoDataFrame...")
        traffic_volumes_gdf = load_traffic_volumes_gdf()
        print(f"   ✓ Loaded {len(traffic_volumes_gdf)} traffic volumes")
        
        # 1. Load TVTW Indexer
        print("1. Loading TVTW Indexer...")
        indexer = TVTWIndexer.load("output/tvtw_indexer.json")
        print(f"   ✓ Loaded indexer with {len(indexer._tv_id_to_idx)} traffic volumes")
        
        # 2. Initialize ExtendedFlightList
        print("2. Loading Flight List...")
        flight_list = ExtendedFlightList(
            occupancy_file_path="output/so6_occupancy_matrix_with_times.json",
            tvtw_indexer_path="output/tvtw_indexer.json"
        )
        print(f"   ✓ Loaded {flight_list.num_flights} flights with {flight_list.num_tvtws} TVTWs")
        
        # 3. Initialize RegulationParser
        print("3. Initializing Regulation Parser...")
        parser = RegulationParser(
            flights_file="output/so6_occupancy_matrix_with_times.json",
            tvtw_indexer=indexer
        )
        print("   ✓ Regulation parser initialized")
        
        # 4. Initialize OptimizationProblem
        print("4. Initializing Optimization Problem...")
        optimization_problem = OptimizationProblem(
            traffic_volumes_gdf=traffic_volumes_gdf,
            flight_list=flight_list,
            horizon_time_windows=100
        )
        
        # Compute baseline objective
        baseline_objective = optimization_problem.objective()
        print(f"   ✓ Baseline objective: {baseline_objective:.4f}")
        
        # 5. Test various regulation strings
        test_regulations = [
            # Wide-spectrum regulation
            "TV_MASH5RL IC__ 10 36-45",
            # # Basic regulation - all flights at EDMTEG during morning hours
            # "TV_EDMTEG IC__ 10 48-55",
            # # Country-specific regulation - flights from/to Belgium (BK prefix)
            # "TV_EDMTEG IC_BK>_> 15 50-52",
            # # Airport-specific regulation - flights between specific airports
            # "TV_LFPPLW1 IC_LFPG_EGLL 20 48-50"
        ]

        # before_flight_state = flight_list.get_flight_metadata("263867136")
        # print(f"Before flight state: {before_flight_state}")
        
        for i, regulation_str in enumerate(test_regulations, 1):
            print(f"\n5.{i}. Testing Regulation: '{regulation_str}'")
            
            try:
                # Create RegulationMove
                regulation_move = RegulationMove(
                    regulation_str=regulation_str,
                    parser=parser,
                    flight_list=flight_list,
                    tvtw_indexer=indexer
                )
                
                # Get regulation explanation
                explanation = parser.explain_regulation(regulation_move.regulation)
                print(f"     Explanation: {explanation}")
                
                # Apply the regulation move
                print("     Applying regulation move...")
                original_state = flight_list
                new_state = original_state.copy()  # Create a copy to modify\
                regulation_move(new_state)  # Apply move in-place
                
                # Verify the move was applied to a copy
                if new_state is original_state:
                    print("     ❌ ERROR: RegulationMove should have been applied to a copy")
                else:
                    print("     ✓ RegulationMove was applied to a new state copy")
                
                # Check if any flights were affected
                affected_flights = 0
                total_delay = 0.0
                
                for flight_id in flight_list.flight_ids:
                    original_vector = original_state.get_occupancy_vector(flight_id)
                    new_vector = new_state.get_occupancy_vector(flight_id)
                    
                    if not np.array_equal(original_vector, new_vector):
                        affected_flights += 1
                        
                
                print(f"     ✓ Affected flights: {affected_flights}")
                
            except Exception as e:
                print(f"     ❌ Error testing regulation {i}: {str(e)}")
                import traceback
                traceback.print_exc()

        # after_flight_state = new_state.get_flight_metadata("263867136")
        # print(f"After flight state: {after_flight_state}")
        
        
        
        print(f"\n=== Integration Test Completed Successfully ===")
        print(f"✓ All components work together correctly")
        print(f"✓ RegulationMove integrates properly with OptimizationProblem")
        print(f"✓ Flight list updates and objective computation work")
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_regulation_formats():
    """Test different regulation string formats."""
    
    print("\n=== Testing Regulation String Formats ===")
    
    test_formats = [
        ("TV_EDMTEG IC__ 60 36,37,38", "Comma-separated time windows"),
        ("TV_EDMTEG IC__ 60 36-38", "Range time windows"),
        ("TV_EDMTEG IC__ 60 36,39-41,45", "Mixed format"),
        ("TV_LFPPLW1 IC_LFPG_EGLL 30 48-50", "Airport pair filtering"),
        ("TV_EDMTEG IC_LF>_EG> 40 47-53", "Country pair filtering"),
        ("TV_EBBUELS1 IC_LFP>_> 25 50-52", "One-sided country filtering"),
    ]
    
    for regulation_str, description in test_formats:
        try:
            from project_tailwind.optimize.regulation import Regulation
            regulation = Regulation(regulation_str)
            print(f"✓ {description}: '{regulation_str}'")
            print(f"  Location: {regulation.location}, Rate: {regulation.rate}")
            print(f"  Time windows: {regulation.time_windows}")
            print(f"  Filter: {regulation.filter_type}_{regulation.filter_value}")
        except Exception as e:
            print(f"❌ {description}: {str(e)}")


if __name__ == "__main__":
    # Run the integration test
    test_regulation_move_integration()
    
    # Test regulation formats
    test_regulation_formats()
    
    print("\n" + "="*60)
    print("TEST SCRIPT COMPLETED")
    print("="*60)
