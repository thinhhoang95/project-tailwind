import json
from typing import List, Dict, Any, Tuple
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.regulation import Regulation 

class RegulationParser:
    def __init__(self, flights_file: str, tvtw_indexer: TVTWIndexer):
        with open(flights_file, 'r') as f:
            self.flights_data: Dict[str, Dict[str, Any]] = json.load(f)
        self.tvtw_indexer = tvtw_indexer

    def parse(self, regulation: Regulation) -> List[str]:
        """
        Parses a single regulation and returns a list of flight identifiers that match.
        """
        matching_flights = []

        # Step 1: Filter by reference location and time windows
        flights_at_ref_loc = self._filter_by_ref_loc_and_time(regulation)

        # Step 2: Apply filtering condition
        for flight_id in flights_at_ref_loc:
            flight_info = self.flights_data.get(flight_id)
            if not flight_info:
                continue

            if self._matches_filter_condition(flight_info, regulation):
                matching_flights.append(flight_id)

        return matching_flights

    def _filter_by_ref_loc_and_time(self, regulation: Regulation) -> List[str]:
        """
        Filter flights that pass through the reference location within the given time windows.
        """
        candidate_flights = []
        for flight_id, flight_data in self.flights_data.items():
            occupancy_vector = flight_data.get("occupancy_vector", [])
            for tvtw_idx in occupancy_vector:
                tv_name, time_idx = self.tvtw_indexer.get_tvtw_from_index(tvtw_idx)
                if tv_name == regulation.location and time_idx in regulation.time_windows:
                    candidate_flights.append(flight_id)
                    break # Move to the next flight
        return candidate_flights

    def _matches_filter_condition(self, flight_info: Dict[str, Any], regulation: Regulation) -> bool:
        """
        Check if a flight matches the filtering condition of the regulation.
        """
        origin = flight_info.get("origin", "")
        destination = flight_info.get("destination", "")

        if regulation.filter_type == 'IC':
            return self._matches_ic_filter(origin, destination, regulation.filter_value)
        elif regulation.filter_type == 'TV':
            return self._matches_tv_filter(flight_info.get("occupancy_vector", []), regulation.filter_value)
        else:
            return False

    def _matches_ic_filter(self, origin: str, destination: str, filter_value: str) -> bool:
        """
        Check ICAO code based filtering conditions.
        The filter supports the following syntaxes (each side separated by an underscore "_"):
        1. Airport  pair : ``LIMC_EGLL``  - exact origin and destination ICAO codes.
        2. Country  pair : ``LI>_ED>``    - prefixes ending with ``>``. Either side can be omitted (e.g. ``LI>_>`` or ``>_ED>``).
        3. City     pair : ``LIM>_EGL_``  - prefixes ending with ``>`` / ``_``. Either side can be omitted as well.
        4. Wildcard pair : ``_``           - single underscore disables filtering completely.

        Historically only the *both-sides* variants of 2 & 3 were supported. This implementation also
        accepts one-sided wild-cards, i.e. an empty prefix on either the origin or destination side.
        
        Order matters: LI>_EG> means Italy to England (not England to Italy).
        """
        # Fully wildcard – match every flight
        if filter_value == '_':
            return True

        # Ensure we have the origin / destination delimiter
        if '_' not in filter_value:
            return False

        orig_pattern, dest_pattern = filter_value.split('_', 1)

        def _match(code: str, pattern: str) -> bool:
            """Return True if *code* matches *pattern* according to the rules above."""
            # Empty segment or single underscore – wildcard (matches everything)
            if pattern in ('', '_'):
                return True

            # Country prefix – e.g. "ED>" or just ">" (wildcard)
            if pattern.endswith('>'):
                prefix = pattern[:-1]  # strip the trailing marker
                return (not prefix) or code.startswith(prefix)

            # City prefix – e.g. "EGL_" or just "_" (wildcard)
            if pattern.endswith('_') and pattern != '_':
                prefix = pattern[:-1]
                return (not prefix) or code.startswith(prefix)

            # Exact airport code
            return code == pattern

        # Order is important: origin must match orig_pattern and destination must match dest_pattern
        return _match(origin, orig_pattern) and _match(destination, dest_pattern)
            
    def _matches_tv_filter(self, occupancy_vector: List[int], filter_value: str) -> bool:
        """
        Check traffic volume based filtering conditions.
        Supports multiple traffic volumes on each side: TV1,TV2_TV3,TV4
        Order matters: the flight must pass through the "from" traffic volumes before the "to" traffic volumes.
        """
        from_tvs_str, to_tvs_str = filter_value.split('_', 1)
        
        # Parse comma-separated traffic volumes
        from_tvs = [tv.strip() for tv in from_tvs_str.split(',') if tv.strip()]
        to_tvs = [tv.strip() for tv in to_tvs_str.split(',') if tv.strip()]
        
        # Get ordered list of traffic volumes that the flight passes through
        flight_tvs_ordered = []
        for idx in occupancy_vector:
            tv_name, _ = self.tvtw_indexer.get_tvtw_from_index(idx)
            if tv_name not in flight_tvs_ordered:  # Avoid duplicates while preserving order
                flight_tvs_ordered.append(tv_name)
        
        # Check if all "from" traffic volumes are present
        from_matches = all(tv in flight_tvs_ordered for tv in from_tvs)
        # Check if all "to" traffic volumes are present
        to_matches = all(tv in flight_tvs_ordered for tv in to_tvs)
        
        if not (from_matches and to_matches):
            return False
        
        # Ensure ordering: the last "from" TV must appear before the first "to" TV
        if from_tvs and to_tvs:
            # Find the latest position of any "from" TV
            last_from_pos = max(flight_tvs_ordered.index(tv) for tv in from_tvs if tv in flight_tvs_ordered)
            # Find the earliest position of any "to" TV
            first_to_pos = min(flight_tvs_ordered.index(tv) for tv in to_tvs if tv in flight_tvs_ordered)
            
            # The last "from" TV must come before the first "to" TV
            return last_from_pos < first_to_pos
        
        return True

    def explain_regulation(self, regulation: Regulation) -> str:
        """
        Convert a regulation into human-readable English explanation.
        """
        explanation_parts = []
        
        # Location part
        explanation_parts.append(f"Apply traffic regulation at traffic volume '{regulation.location}'")
        
        # Time windows part
        time_windows_desc = self._format_time_windows(regulation.time_windows)
        explanation_parts.append(f"during time windows {time_windows_desc}")
        
        # Rate part
        explanation_parts.append(f"with a maximum rate of {regulation.rate} flights per time window")
        
        # Filter condition part
        filter_desc = self._explain_filter_condition(regulation.filter_type, regulation.filter_value)
        explanation_parts.append(f"for flights {filter_desc}")
        
        return ", ".join(explanation_parts) + "."
    
    def _format_time_windows(self, time_windows: List[int]) -> str:
        """Format time windows list into readable string with actual times."""
        if not time_windows:
            return "none"
        
        # Group consecutive numbers into ranges and convert to time format
        ranges = []
        start = time_windows[0]
        end = start
        
        for i in range(1, len(time_windows)):
            if time_windows[i] == end + 1:
                end = time_windows[i]
            else:
                ranges.append(self._format_time_range(start, end))
                start = time_windows[i]
                end = start
        
        # Add the last range
        ranges.append(self._format_time_range(start, end))
        
        return ", ".join(ranges)
    
    def _format_time_range(self, start_idx: int, end_idx: int) -> str:
        """Format a time window range into readable time format."""
        start_time = self.tvtw_indexer.time_window_map.get(start_idx, f"TW{start_idx}")
        
        if start_idx == end_idx:
            return f"{start_time} (TW{start_idx})"
        else:
            end_time = self.tvtw_indexer.time_window_map.get(end_idx, f"TW{end_idx}")
            # Extract just the start time of the first window and end time of the last window
            start_time_only = start_time.split('-')[0]
            end_time_only = end_time.split('-')[1]
            return f"{start_time_only}-{end_time_only} (TW{start_idx}-{end_idx})"
    
    def _explain_filter_condition(self, filter_type: str, filter_value: str) -> str:
        """Explain the filter condition in human-readable terms."""
        if filter_type == 'IC':
            return self._explain_ic_filter(filter_value)
        elif filter_type == 'TV':
            return self._explain_tv_filter(filter_value)
        else:
            return f"with unknown filter type '{filter_type}'"
    
    def _explain_ic_filter(self, filter_value: str) -> str:
        """Explain ICAO code filter in human-readable terms."""
        if filter_value == '_':
            return "from any origin to any destination"
        
        if '_' not in filter_value:
            return f"with invalid filter format '{filter_value}'"
        
        orig_pattern, dest_pattern = filter_value.split('_', 1)
        
        orig_desc = self._explain_icao_pattern(orig_pattern, "origin")
        dest_desc = self._explain_icao_pattern(dest_pattern, "destination")
        
        return f"from {orig_desc} to {dest_desc}"
    
    def _explain_icao_pattern(self, pattern: str, location_type: str) -> str:
        """Explain a single ICAO pattern."""
        if pattern in ('', '_'):
            return f"any {location_type}"
        
        if pattern.endswith('>'):
            prefix = pattern[:-1]
            if not prefix:
                return f"any {location_type}"
            return f"{location_type}s in country/region '{prefix}*'"
        
        if pattern.endswith('_') and pattern != '_':
            prefix = pattern[:-1]
            if not prefix:
                return f"any {location_type}"
            return f"{location_type}s in city area '{prefix}*'"
        
        return f"{location_type} '{pattern}'"
    
    def _explain_tv_filter(self, filter_value: str) -> str:
        """Explain traffic volume filter in human-readable terms."""
        if '_' not in filter_value:
            return f"with invalid TV filter format '{filter_value}'"
        
        from_tvs_str, to_tvs_str = filter_value.split('_', 1)
        
        from_tvs = [tv.strip() for tv in from_tvs_str.split(',') if tv.strip()]
        to_tvs = [tv.strip() for tv in to_tvs_str.split(',') if tv.strip()]
        
        if not from_tvs and not to_tvs:
            return "passing through any traffic volumes"
        
        parts = []
        if from_tvs:
            if len(from_tvs) == 1:
                parts.append(f"passing through traffic volume '{from_tvs[0]}'")
            else:
                tv_list = "', '".join(from_tvs)
                parts.append(f"passing through traffic volumes '{tv_list}'")
        
        if to_tvs:
            if len(to_tvs) == 1:
                parts.append(f"then through traffic volume '{to_tvs[0]}'")
            else:
                tv_list = "', '".join(to_tvs)
                parts.append(f"then through traffic volumes '{tv_list}'")
        
        return " ".join(parts)

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

    # 3. Create a Regulation object
    # Example: TV_EBBUELS1 IC_LIMC_EGLL 60 36,37
    # Note: The prompt has a slightly different format. Re-checking...
    # The prompt format is: <REFERENCE LOCATION OR TRAFFIC VOLUME> <FILTERING CONDITION> <RATE> <TIME WINDOWS>
    # Example from prompt: TV_LFPPLW1 LFP*_LI* 60 36,37 -- This is not matching the IC_ or TV_ prefix for filter.
    # Let's adjust the parser based on the prompt's examples.
    # The prompt example: <LFPPLW1> <LFP* > LI*> <60> <36, 37>
    # This example seems to have a different structure.
    # Let's stick to the documented format: TV_EBBUELS1 IC_LIMC_EGLL 60 "36,37"
    
    # regulation_str = "TV_EBBUELS1 IC_LIMC_EGLL 60 36-38"
    
    # regulation = Regulation(regulation_str)
    
    # # 4. Parse the regulation
    # matched_flights = parser.parse(regulation)
    
    # print(f"Regulation: '{regulation.raw_str}'")
    # print(f"Matched {len(matched_flights)} flights:")
    # print(json.dumps(matched_flights, indent=2))


    # Example for country-pair with range notation
    regulation_str_country = "TV_EDMTEG IC_BK>_> 60 47-53" # IC__ means no filtering
    
    regulation_country = Regulation(regulation_str_country)
    matched_flights_country = parser.parse(regulation_country)
    print(f"Regulation: '{regulation_country.raw_str}'")
    print(f"Matched {len(matched_flights_country)} flights:")
    print(json.dumps(matched_flights_country, indent=2))


