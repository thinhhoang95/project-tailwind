"""
Test script for delay and reroute operators.
Tests both operators using flights from the SO6 occupancy matrix.
"""

import json
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np

from project_tailwind.impact_eval.operators.delay import delay_operator
from project_tailwind.impact_eval.operators.reroute import find_alternative_route
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.rqs.bitmap_querying_system import RouteQuerySystem


def display_occupancy_vector_human_readable(
    occupancy_vector: List[int], 
    indexer: TVTWIndexer, 
    flight_id: str = None
) -> None:
    """
    Display traffic volumes and time windows in human readable format.
    Sorts the trajectory by time bins for chronological display.
    
    Args:
        occupancy_vector: List of TVTW indices
        indexer: TVTWIndexer instance
        flight_id: Optional flight identifier for display
    """
    print(f"\n{'='*60}")
    if flight_id:
        print(f"Flight {flight_id} - Trajectory Analysis")
    else:
        print("Trajectory Analysis")
    print(f"{'='*60}")
    
    if not occupancy_vector:
        print("Empty occupancy vector")
        return
    
    print(f"Total sectors/time windows: {len(occupancy_vector)}")
    print(f"Time bin size: {indexer.time_bin_minutes} minutes")
    
    # Create list of (original_index, tvtw_index, readable_info, time_bin_idx) for sorting
    trajectory_data = []
    for i, tvtw_index in enumerate(occupancy_vector):
        readable = indexer.get_human_readable_tvtw(tvtw_index)
        tvtw_tuple = indexer.get_tvtw_from_index(tvtw_index)
        time_bin_idx = tvtw_tuple[1] if tvtw_tuple else 9999  # Put unknowns at end
        
        trajectory_data.append({
            'original_index': i,
            'tvtw_index': tvtw_index,
            'readable': readable,
            'time_bin_idx': time_bin_idx
        })
    
    # Sort by time bin index (chronological order)
    trajectory_data.sort(key=lambda x: x['time_bin_idx'])
    
    print("\nTrajectory details (sorted chronologically):")
    print(f"{'Orig#':<6} {'Sector':<15} {'Time Window':<12} {'TVTW Index':<10} {'Sorted#':<8}")
    print("-" * 70)
    
    for sorted_idx, data in enumerate(trajectory_data):
        if data['readable']:
            sector, time_window = data['readable']
            print(f"{data['original_index']+1:<6} {sector:<15} {time_window:<12} {data['tvtw_index']:<10} {sorted_idx+1:<8}")
        else:
            print(f"{data['original_index']+1:<6} {'UNKNOWN':<15} {'UNKNOWN':<12} {data['tvtw_index']:<10} {sorted_idx+1:<8}")
    
    # Show time span using sorted data
    if len(trajectory_data) > 1:
        first_data = trajectory_data[0]
        last_data = trajectory_data[-1]
        
        if first_data['readable'] and last_data['readable']:
            first_time = first_data['readable'][1]
            last_time = last_data['readable'][1]
            print(f"\nFlight spans from {first_time} to {last_time} (chronologically)")
            
        # Show if trajectory was reordered
        original_order = [d['original_index'] for d in trajectory_data]
        if original_order != sorted(original_order):
            print(f"Note: Original trajectory was reordered for chronological display")
            print(f"Original order: {[i+1 for i in range(len(occupancy_vector))]}")
            print(f"Time-sorted order: {[d['original_index']+1 for d in trajectory_data]}")


def load_flight_data(so6_file_path: str) -> Dict[str, List[int]]:
    """Load flight occupancy vectors from SO6 file."""
    with open(so6_file_path, 'r') as f:
        return json.load(f)


def load_cirrus_flight_data(cirrus_csv_path: str) -> pd.DataFrame:
    """Load flight data from Cirrus CSV file."""
    return pd.read_csv(cirrus_csv_path)


def get_flight_info_from_cirrus(flight_id: str, cirrus_df: pd.DataFrame) -> Optional[Dict]:
    """
    Extract flight information from Cirrus data for a given flight ID.
    
    Returns:
        Dict with origin, destination, takeoff_time, or None if not found
    """
    # Find the first segment for this flight (lowest sequence number)
    flight_segments = cirrus_df[cirrus_df['flight_identifier'] == int(flight_id)]
    
    if flight_segments.empty:
        return None
    
    # Get the first segment (takeoff)
    first_segment = flight_segments.loc[flight_segments['sequence'].idxmin()]
    
    origin = first_segment['origin_aerodrome']
    destination = first_segment['destination_aerodrome']
    
    # Parse date and time
    date_str = str(first_segment['date_begin_segment'])  # e.g., "230801"
    time_int = int(first_segment['time_begin_segment'])  # e.g., 50207 (seconds since midnight)
    
    # Convert date: 230801 -> 2023-08-01
    if len(date_str) == 6:
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
    else:
        print(f"Warning: Unexpected date format: {date_str}")
        year, month, day = 2023, 8, 1  # Default fallback
    
    # Convert time: HHMMSS integer (e.g., 91000 means 09:10:00) to hours, minutes, seconds
    hours = time_int // 10000
    minutes = (time_int % 10000) // 100
    seconds = time_int % 100
    
    takeoff_time = datetime(year, month, day, hours, minutes, seconds)
    
    return {
        'origin': origin,
        'destination': destination,
        'takeoff_time': takeoff_time,
        'flight_segments': len(flight_segments)
    }


def test_delay_operator(
    flight_id: str, 
    occupancy_vector: List[int], 
    delay_minutes: int,
    indexer: TVTWIndexer
) -> List[int]:
    """
    Test the delay operator on a flight.
    
    Args:
        flight_id: Flight identifier
        occupancy_vector: Original trajectory
        delay_minutes: Delay to apply in minutes
        indexer: TVTWIndexer instance
        
    Returns:
        Delayed occupancy vector
    """
    print(f"\n{'*'*80}")
    print(f"TESTING DELAY OPERATOR - {delay_minutes} minute delay")
    print(f"{'*'*80}")
    
    print("\nORIGINAL TRAJECTORY:")
    display_occupancy_vector_human_readable(occupancy_vector, indexer, flight_id)
    
    try:
        delayed_vector = delay_operator(occupancy_vector, delay_minutes, indexer)
        
        print(f"\nDELAYED TRAJECTORY (+{delay_minutes} minutes):")
        display_occupancy_vector_human_readable(delayed_vector, indexer, flight_id)
        
        print(f"\nDelay operation summary:")
        print(f"- Original sectors: {len(occupancy_vector)}")
        print(f"- Delayed sectors: {len(delayed_vector)}")
        print(f"- Time shift: {delay_minutes // indexer.time_bin_minutes} time bins")
        
        return delayed_vector
        
    except Exception as e:
        print(f"ERROR in delay operator: {e}")
        return occupancy_vector


def test_reroute_operator(
    flight_id: str,
    occupancy_vector: List[int],
    indexer: TVTWIndexer,
    rqs: RouteQuerySystem,
    cirrus_df: pd.DataFrame,
    sector_to_avoid_idx: int = None
) -> Optional[Tuple[str, np.ndarray, float]]:
    """
    Test the reroute operator on a flight.
    
    Args:
        flight_id: Flight identifier
        occupancy_vector: Original trajectory
        indexer: TVTWIndexer instance
        rqs: RouteQuerySystem instance
        cirrus_df: Cirrus flight data DataFrame
        sector_to_avoid_idx: Index of sector to avoid (if None, picks middle sector)
        
    Returns:
        Alternative route tuple or None
    """
    print(f"\n{'*'*80}")
    print(f"TESTING REROUTE OPERATOR")
    print(f"{'*'*80}")
    
    print("\nORIGINAL TRAJECTORY:")
    display_occupancy_vector_human_readable(occupancy_vector, indexer, flight_id)
    
    if not occupancy_vector:
        print("Cannot reroute empty trajectory")
        return None
    
    # Get real flight info from Cirrus data
    flight_info = get_flight_info_from_cirrus(flight_id, cirrus_df)
    if not flight_info:
        print(f"Warning: Flight {flight_id} not found in Cirrus data, using dummy values")
        origin = "KJFK"
        destination = "KLAX"
        takeoff_time = datetime(2023, 8, 1, 8, 0)
    else:
        origin = flight_info['origin']
        destination = flight_info['destination']
        takeoff_time = flight_info['takeoff_time']
    
    print(f"\nFLIGHT INFORMATION (from Cirrus data):")
    print(f"Flight ID: {flight_id}")
    print(f"Origin: {origin}")
    print(f"Destination: {destination}")
    print(f"Takeoff time: {takeoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if flight_info:
        print(f"Total segments: {flight_info['flight_segments']}")
    
    # Select sector to avoid (middle of trajectory if not specified)
    if sector_to_avoid_idx is None:
        sector_to_avoid_idx = len(occupancy_vector) // 2
    
    if sector_to_avoid_idx >= len(occupancy_vector):
        sector_to_avoid_idx = len(occupancy_vector) - 1
    
    tvtw_to_avoid = occupancy_vector[sector_to_avoid_idx]
    readable_avoid = indexer.get_human_readable_tvtw(tvtw_to_avoid)
    
    print(f"\nAVOIDING SECTOR:")
    if readable_avoid:
        sector_name, time_window = readable_avoid
        print(f"Sector: {sector_name}")
        print(f"Time: {time_window}")
        print(f"TVTW Index: {tvtw_to_avoid}")
    else:
        print(f"TVTW Index: {tvtw_to_avoid} (details unknown)")
    
    print(f"\nReroute parameters:")
    print(f"Origin: {origin}")
    print(f"Destination: {destination}")
    print(f"Takeoff time: {takeoff_time.strftime('%H:%M:%S')}")
    
    try:
        result = find_alternative_route(
            origin=origin,
            destination=destination,
            takeoff_time=takeoff_time,
            overloaded_tvtws=[tvtw_to_avoid],
            rqs=rqs,
            tvtw_indexer=indexer
        )
        
        if result:
            route_str, new_impact_vector, distance = result
            
            print(f"\nALTERNATIVE ROUTE FOUND:")
            print(f"Route: {route_str}")
            print(f"Distance: {distance:.2f} km")
            
            print(f"\nNEW TRAJECTORY:")
            display_occupancy_vector_human_readable(
                new_impact_vector.tolist(), indexer, f"{flight_id}_rerouted"
            )
            
            print(f"\nReroute operation summary:")
            print(f"- Original sectors: {len(occupancy_vector)}")
            print(f"- New route sectors: {len(new_impact_vector)}")
            print(f"- Avoided TVTW: {tvtw_to_avoid}")
            
            return result
        else:
            print("\nNo alternative route found!")
            return None
            
    except Exception as e:
        print(f"ERROR in reroute operator: {e}")
        return None


def interactive_flight_selection(flight_data: Dict[str, List[int]], cirrus_df: pd.DataFrame) -> str:
    """Allow user to select a flight ID interactively."""
    available_flights = list(flight_data.keys())
    
    print(f"\nAvailable flights in SO6 data: {len(available_flights)}")
    print("Sample flight IDs:", available_flights[:10], "..." if len(available_flights) > 10 else "")
    
    while True:
        flight_id = input("\nEnter flight ID to test (or 'random' for random selection): ").strip()
        
        if flight_id.lower() == 'random':
            flight_id = random.choice(available_flights)
            print(f"Randomly selected: {flight_id}")
            return flight_id
        
        if flight_id in flight_data:
            # Check if flight exists in Cirrus data
            cirrus_info = get_flight_info_from_cirrus(flight_id, cirrus_df)
            if cirrus_info:
                print(f"✓ Flight {flight_id} found in both SO6 and Cirrus data")
                print(f"  Route: {cirrus_info['origin']} → {cirrus_info['destination']}")
                print(f"  Takeoff: {cirrus_info['takeoff_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"⚠ Flight {flight_id} found in SO6 but not in Cirrus data")
            return flight_id
        else:
            print(f"Flight ID '{flight_id}' not found in SO6 data. Please try again.")


def main():
    """Main test function."""
    print("Loading test data and initializing systems...")
    
    # Load flight data
    so6_file = "output/so6_occupancy_matrix.json"
    try:
        flight_data = load_flight_data(so6_file)
        print(f"Loaded {len(flight_data)} flights from {so6_file}")
    except Exception as e:
        print(f"Error loading flight data: {e}")
        return
    
    # Load Cirrus flight data
    cirrus_file = "/Volumes/CrucialX/project-cirrus/cases/flights_20230801.csv"
    try:
        cirrus_df = load_cirrus_flight_data(cirrus_file)
        print(f"Loaded {len(cirrus_df)} flight segments from {cirrus_file}")
        unique_flights = cirrus_df['flight_identifier'].nunique()
        print(f"Total unique flights in Cirrus data: {unique_flights}")
    except Exception as e:
        print(f"Error loading Cirrus flight data: {e}")
        return
    
    # Initialize indexer (you may need to adjust these paths)
    try:
        # Try to load existing indexer or create one
        indexer_file = "output/tvtw_indexer.json"
        try:
            indexer = TVTWIndexer.load(indexer_file)
            print(f"Loaded TVTWIndexer from {indexer_file}")
        except:
            # Create a basic indexer if loading fails
            indexer = TVTWIndexer(time_bin_minutes=30)
            print("Created new TVTWIndexer with 30-minute bins")
            
    except Exception as e:
        print(f"Error initializing TVTWIndexer: {e}")
        return
    
    # Initialize RouteQuerySystem (you may need to adjust the path)
    try:
        routes_file = "output/route_distances.json"  # Adjust path as needed
        rqs = RouteQuerySystem(routes_file)
        print(f"Loaded RouteQuerySystem from {routes_file}")
    except Exception as e:
        print(f"Warning: Could not load RouteQuerySystem: {e}")
        print("Reroute tests will be skipped")
        rqs = None
    
    # Interactive flight selection
    # test_flight_id = interactive_flight_selection(flight_data, cirrus_df)
    test_flight_id = "263871294"
    test_occupancy_vector = flight_data[test_flight_id]
    
    print(f"\nSelected flight {test_flight_id} for testing")
    print(f"Original trajectory has {len(test_occupancy_vector)} sectors")
    
    # Test delay operator with different delays
    # delay_minutes = [15, 30, 60, 120]
    delay_minutes = [] 
    
    for delay in delay_minutes:
        delayed_vector = test_delay_operator(
            test_flight_id, 
            test_occupancy_vector, 
            delay, 
            indexer
        )
        
        # Brief pause between tests
        input(f"\nPress Enter to continue to next delay test...")
    
    # Test reroute operator if RQS is available
    if rqs:
        print(f"\n{'='*80}")
        print("Starting reroute operator tests...")
        print(f"{'='*80}")
        
        test_reroute_operator(
            test_flight_id,
            test_occupancy_vector,
            indexer,
            rqs,
            cirrus_df
        )
    
    print(f"\n{'='*80}")
    print("All tests completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()