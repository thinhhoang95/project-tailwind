from typing import List, Dict, Any, Optional
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer
from project_tailwind.optimize.regulation import Regulation
from server_tailwind.core.resources import AppResources, get_resources

class RegulationParser:
    def __init__(self, resources: Optional[AppResources] = None):
        """
        Regulation parser backed by shared AppResources.

        Uses the process-wide FlightList and TVTWIndexer from resources instead of
        loading any files directly.
        """
        self.resources: AppResources = resources or get_resources()
        self.flight_list = self.resources.flight_list
        self.tvtw_indexer: TVTWIndexer = self.resources.indexer

    def parse(self, regulation: Regulation) -> List[str]:
        """
        Parses a single regulation and returns a list of flight identifiers that match.
        """
        matching_flights = []

        # Step 1: Filter by reference location and time windows (via resources)
        flights_at_ref_loc = self._filter_by_ref_loc_and_time(regulation)

        # Step 2: Apply filtering condition
        for flight_id in flights_at_ref_loc:
            if self._matches_filter_condition(flight_id, regulation):
                matching_flights.append(flight_id)

        return matching_flights

    def _get_occupancy_vector(self, flight_meta: Dict[str, Any]) -> List[int]:
        """
        Extract the list of TVTW indices from flight metadata.
        Assumes the presence of `occupancy_intervals` entries with `tvtw_index`.
        """
        intervals = flight_meta.get("occupancy_intervals") or []
        return [int(item["tvtw_index"]) for item in intervals if "tvtw_index" in item]

    def _filter_by_ref_loc_and_time(self, regulation: Regulation) -> List[str]:
        """
        Filter flights that pass through the reference location within the given time windows.
        """
        # Use the shared FlightList helper to find crossings
        hits = set()
        for fid, _tv_id, _entry_dt, _tw in self.flight_list.iter_hotspot_crossings(
            [regulation.location], active_windows=regulation.time_windows
        ):
            hits.add(fid)
        return list(hits)

    def _matches_filter_condition(self, flight_id: str, regulation: Regulation) -> bool:
        """
        Check if a flight matches the filtering condition of the regulation.
        """
        meta = self.flight_list.flight_metadata.get(flight_id)
        if not meta:
            return False
        origin = meta.get("origin", "")
        destination = meta.get("destination", "")

        if regulation.filter_type == 'IC':
            return self._matches_ic_filter(origin, destination, regulation.filter_value)
        elif regulation.filter_type == 'TV':
            occupancy_vector = self._get_occupancy_vector(meta)
            return self._matches_tv_filter(occupancy_vector, regulation.filter_value)
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
    # Example usage with shared resources
    resources = get_resources().preload_all()
    parser = RegulationParser(resources)

    regulation_str_country = "TV_EDMTEG IC_BK>_> 60 47-53"
    regulation_country = Regulation(regulation_str_country)
    matched_flights_country = parser.parse(regulation_country)
    print(f"Regulation: '{regulation_country.raw_str}'")
    print(f"Matched {len(matched_flights_country)} flights:")
    print(matched_flights_country)

