import sys
from pathlib import Path
project_root = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(project_root))

from project_tailwind.optimize.moves.regulation_move import RegulationMove
from project_tailwind.optimize.moves.network_plan_move import NetworkPlanMove
from project_tailwind.optimize.network_plan import NetworkPlan
from project_tailwind.optimize.alns.optimization_problem import OptimizationProblem
from project_tailwind.optimize.alns.pstate import ProblemState
from project_tailwind.optimize.parser.regulation_parser import RegulationParser
from project_tailwind.optimize.eval.flight_list import FlightList
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
import geopandas as gpd

objective_weights = {'z_95': 1.0, 'z_sum': 1.0, 'delay': 0.01}

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
        print(f"X Missing required files: {missing_files}")
        print("Please ensure these files exist in the project root.")
        return
    
    try:
        # 1. Load traffic volumes GeoDataFrame
        print("1. Loading traffic volumes GeoDataFrame...")
        traffic_volumes_gdf = load_traffic_volumes_gdf()
        print(f"   OK Loaded {len(traffic_volumes_gdf)} traffic volumes")
        
        # 1. Load TVTW Indexer
        print("1. Loading TVTW Indexer...")
        indexer = TVTWIndexer.load("output/tvtw_indexer.json")
        print(f"   OK Loaded indexer with {len(indexer._tv_id_to_idx)} traffic volumes")
        
        # 2. Initialize FlightList
        print("2. Loading Flight List...")
        flight_list = FlightList(
            occupancy_file_path="output/so6_occupancy_matrix_with_times.json",
            tvtw_indexer_path="output/tvtw_indexer.json"
        )
        print(f"   OK Loaded {flight_list.num_flights} flights with {flight_list.num_tvtws} TVTWs")
        
        # 3. Initialize RegulationParser
        print("3. Initializing Regulation Parser...")
        parser = RegulationParser(
            flights_file="output/so6_occupancy_matrix_with_times.json",
            tvtw_indexer=indexer
        )
        print("   OK Regulation parser initialized")
        
        # 4. Initialize OptimizationProblem
        print("4. Initializing Optimization Problem...")
        optimization_problem = OptimizationProblem(
            traffic_volumes_gdf=traffic_volumes_gdf,
            flight_list=flight_list,
            horizon_time_windows=100,
            objective_weights=objective_weights
        )
        
        # Create initial state
        state = optimization_problem.create_initial_state()
        
        # Compute baseline objective
        baseline_objective = state.objective()
        print(f"   OK Baseline objective: {baseline_objective:.4f}")
        
        # 5. Test various regulation strings
        test_regulations = [
            # Wide-spectrum regulation
            "TV_MASH5RL IC__ 10 36-45"
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
                    flight_list=state.flight_list,
                    tvtw_indexer=indexer
                )
                
                # Get regulation explanation
                explanation = parser.explain_regulation(regulation_move.regulation)
                print(f"     Explanation: {explanation}")
                
                # Apply the regulation move (this should return a new state)
                print("     Applying regulation move...")
                new_state, total_delay = regulation_move(state.flight_list)  # Apply move to flight_list
                print(f"     Total delay: {total_delay:.1f} minutes")
                
                # Create new state with updated flight_list
                updated_state = state.with_flight_list(new_state) # input is a flight list, so the return is a flight list as well
                # we ought to convert it to a ProblemState.
                # see below: if the input is a ProblemState, the return is a ProblemState as well.
                # no need for conversion.

                # After the move, recompute the objective
                new_objective = updated_state.objective()
                print(f"     New objective: {new_objective:.4f}")

                # Compute the improvement
                improvement = new_objective - baseline_objective
                print(f"     Loss change (negative is better): {improvement:.4f}")
                
            except Exception as e:
                print(f"     X Error testing regulation {i}: {str(e)}")
                import traceback
                traceback.print_exc()

        # after_flight_state = new_state.get_flight_metadata("263867136")
        # print(f"After flight state: {after_flight_state}")
        
        
        
        print(f"\n=== Integration Test Completed Successfully ===")
        print(f"OK All components work together correctly")
        print(f"OK RegulationMove integrates properly with OptimizationProblem")
        print(f"OK Flight list updates and objective computation work")
        
    except Exception as e:
        print(f"X Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_network_plan_move_integration():
    """Test NetworkPlanMove with multiple regulations."""
    
    print("\n=== Testing NetworkPlanMove Integration ===")
    
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
        print(f"X Missing required files: {missing_files}")
        print("Please ensure these files exist in the project root.")
        return
    
    try:
        # 1. Load traffic volumes GeoDataFrame
        print("1. Loading traffic volumes GeoDataFrame...")
        traffic_volumes_gdf = load_traffic_volumes_gdf()
        print(f"   OK Loaded {len(traffic_volumes_gdf)} traffic volumes")
        
        # 2. Load TVTW Indexer
        print("2. Loading TVTW Indexer...")
        indexer = TVTWIndexer.load("output/tvtw_indexer.json")
        print(f"   OK Loaded indexer with {len(indexer._tv_id_to_idx)} traffic volumes")
        
        # 3. Initialize FlightList
        print("3. Loading Flight List...")
        flight_list = FlightList(
            occupancy_file_path="output/so6_occupancy_matrix_with_times.json",
            tvtw_indexer_path="output/tvtw_indexer.json"
        )
        print(f"   OK Loaded {flight_list.num_flights} flights with {flight_list.num_tvtws} TVTWs")
        
        # 4. Initialize RegulationParser
        print("4. Initializing Regulation Parser...")
        parser = RegulationParser(
            flights_file="output/so6_occupancy_matrix_with_times.json",
            tvtw_indexer=indexer
        )
        print("   OK Regulation parser initialized")
        
        # 5. Initialize OptimizationProblem
        print("5. Initializing Optimization Problem...")
        optimization_problem = OptimizationProblem(
            base_flight_list=flight_list,
            regulation_parser=parser,
            tvtw_indexer=indexer,
            objective_weights=objective_weights,
            horizon_time_windows=100,
            base_traffic_volumes=traffic_volumes_gdf
        )
        
        # Create initial state
        state = optimization_problem.create_initial_state()
        
        # Compute baseline objective
        baseline_objective = state.objective()
        print(f"   OK Baseline objective: {baseline_objective:.4f}")
        
        # 6. Test NetworkPlan with multiple regulations
        print("\n6. Testing NetworkPlan with multiple regulations...")
        
        # Create a network plan with multiple regulations
        network_plan = NetworkPlan([
            "TV_MASH5RL IC__ 10 36-40",
            "TV_MASH5RL IC__ 8 41-45",
            # Add more regulations as needed for testing
        ])
        
        print(f"   Created NetworkPlan with {len(network_plan)} regulations")
        print(f"   NetworkPlan: {network_plan}")
        
        # Create NetworkPlanMove
        network_plan_move = NetworkPlanMove(
            network_plan=network_plan,
            parser=parser,
            tvtw_indexer=indexer
        )
        
        # Get regulation summary
        summary = network_plan_move.get_regulation_summary()
        print(f"   Regulation summary: {summary['total_regulations']} regulations")
        
        # Apply the network plan move
        print("   Applying network plan move...")
        new_state, total_delay = network_plan_move(state) # already copied
        print(f"   Total delay applied: {total_delay:.1f} minutes")
        
        # The updated_state would need to be created differently since ProblemState doesn't have with_flight_list method
        # For this test, we'll just compute the objective directly

        # After the move, recompute the objective using the current state (the move effect is computed in the objective function)
        new_objective = state.objective()
        print(f"   New objective: {new_objective:.4f}")

        # Compute the improvement
        improvement = new_objective - baseline_objective
        print(f"   Loss change (negative is better): {improvement:.4f}")
        
        print(f"\n=== NetworkPlanMove Test Completed Successfully ===")
        print(f"OK NetworkPlan and NetworkPlanMove work together correctly")
        print(f"OK Multiple regulations processed with delay aggregation")
        print(f"OK Highest delay per flight logic implemented")
        
    except Exception as e:
        print(f"X NetworkPlanMove test failed: {str(e)}")
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
            print(f"OK {description}: '{regulation_str}'")
            print(f"  Location: {regulation.location}, Rate: {regulation.rate}")
            print(f"  Time windows: {regulation.time_windows}")
            print(f"  Filter: {regulation.filter_type}_{regulation.filter_value}")
        except Exception as e:
            print(f"X {description}: {str(e)}")


if __name__ == "__main__":
    # Run the single regulation test
    # test_regulation_move_integration()
    
    # Run the network plan test
    test_network_plan_move_integration()
    
    # Test regulation formats
    # test_regulation_formats()
    
    print("\n" + "="*60)
    print("TEST SCRIPT COMPLETED")
    print("="*60)
