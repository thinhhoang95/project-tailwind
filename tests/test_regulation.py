import json
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.parser.regulation_parser import Regulation, RegulationParser
from project_tailwind.casa.casa import run_readapted_casa

if __name__ == '__main__':
    # Example usage:
    # 1. Initialize the TVTWIndexer
    # This assumes the indexer has been built and saved previously.
    indexer = TVTWIndexer.load("output/tvtw_indexer.json")

    # 2. Initialize the RegulationParser
    parser = RegulationParser(
        flights_file="output/so6_occupancy_matrix.json",
        tvtw_indexer=indexer
    )
    
    regulation_str = "TV_EDMTEG IC__ 10 48-55"
    
    regulation = Regulation(regulation_str)
    
    # 4. Parse the regulation
    matched_flights = parser.parse(regulation)
    
    print(f"Regulation: '{regulation.raw_str}'")
    print(f"Explanation: {parser.explain_regulation(regulation)}")
    print(f"Matched {len(matched_flights)} flights:")
    print(json.dumps(matched_flights, indent=2))
    
    # 5. Calculate delays using CASA algorithm
    if matched_flights:
        flight_ids = matched_flights
        delays = run_readapted_casa(
            so6_occupancy_path="output/so6_occupancy_matrix_with_times.json",
            identifier_list=flight_ids,
            reference_location=regulation.location,
            tvtw_indexer=indexer,
            hourly_rate=regulation.rate,
            active_time_windows=regulation.time_windows
        )
        
        print(f"\nDelay calculations:")
        for flight_id, delay_minutes in delays.items():
            print(f"Flight {flight_id}: {delay_minutes:.2f} minutes delay")
    else:
        print("No flights matched - no delays to calculate")
