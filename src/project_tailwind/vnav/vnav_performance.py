from typing import List, Tuple


class Performance:
    """
    Models aircraft performance characteristics based on speed and vertical speed profiles.
    It handles climb, cruise, and descent phases of flight.
    """

    def __init__(
        self,
        climb_speed_profile: List[Tuple[float, float]],
        descent_speed_profile: List[Tuple[float, float]],
        climb_vertical_speed_profile: List[Tuple[float, float]],
        descent_vertical_speed_profile: List[Tuple[float, float]],
        cruise_altitude_ft: float,
        cruise_speed_kts: float,
    ):
        """
        Initializes the Performance model.

        Args:
            climb_speed_profile: TAS profile for climb [(alt_thresh_ft, speed_kts), ...].
            descent_speed_profile: TAS profile for descent [(alt_thresh_ft, speed_kts), ...].
            climb_vertical_speed_profile: VS profile for climb [(alt_thresh_ft, vs_fpm), ...].
                                          VS values should be positive.
            descent_vertical_speed_profile: VS profile for descent [(alt_thresh_ft, vs_fpm), ...].
                                            VS values should be negative for descent.
            cruise_altitude_ft: The designated cruise altitude in feet.
            cruise_speed_kts: The designated cruise true airspeed in knots.

        Raises:
            ValueError: If profiles are empty or not correctly terminated with float('inf').
        """
        if not all(
            [
                climb_speed_profile,
                descent_speed_profile,
                climb_vertical_speed_profile,
                descent_vertical_speed_profile,
            ]
        ):
            raise ValueError(
                "All profiles (climb/descent speed and vertical speed) must be provided and non-empty."
            )

        if not (
            climb_speed_profile[-1][0] == float("inf")
            and descent_speed_profile[-1][0] == float("inf")
            and climb_vertical_speed_profile[-1][0] == float("inf")
            and descent_vertical_speed_profile[-1][0] == float("inf")
        ):
            raise ValueError(
                "All profiles must end with a float('inf') altitude threshold."
            )

        self.cruise_altitude_ft = cruise_altitude_ft
        self.cruise_speed_kts = cruise_speed_kts

        self._combined_climb_profile = self._create_combined_profile(
            climb_speed_profile, climb_vertical_speed_profile
        )
        self._combined_descent_profile = self._create_combined_profile(
            descent_speed_profile, descent_vertical_speed_profile
        )

    def _get_value_at_alt(
        self, altitude_ft: float, profile: List[Tuple[float, float]]
    ) -> float:
        """
        Helper to get a value (speed or VS) from a simple profile at a given altitude.
        Assumes profile is sorted by alt_threshold and last alt_threshold is float('inf').
        Assumes altitude_ft is non-negative.
        """
        # profile format: [(upper_alt_boundary_exclusive, value), ...]
        for alt_threshold, value in profile:
            if altitude_ft < alt_threshold:
                return value
        # Should not be reached if profile is correctly formatted and altitude_ft is finite,
        # as the last alt_threshold is float('inf').
        # This implies an issue if this point is reached, possibly with an empty or malformed profile
        # not caught by initial checks, or non-finite altitude (though inf should be caught by loop).
        # However, initial checks should prevent empty profiles being passed here.
        # And profiles must end in float('inf').
        # If profile = [(inf, val)], alt_ft < inf is true, returns val.
        # This part is a safeguard but ideally unreachable with validated inputs.
        if profile:  # If profile was not empty
            return profile[-1][
                1
            ]  # Fallback to last value if somehow loop didn't catch (e.g. alt_ft = inf)
        raise ValueError(
            f"Could not determine value for altitude {altitude_ft} from profile. Profile might be empty or malformed unexpectedly."
        )

    def _create_combined_profile(
        self,
        speed_profile: List[Tuple[float, float]],
        vs_profile: List[Tuple[float, float]],
    ) -> List[Tuple[float, float, float]]:
        """
        Combines separate speed and vertical speed profiles into a single profile.
        Each entry in the returned list is (altitude_upper_boundary_ft, speed_kts, vs_fpm).
        Assumes input profiles are sorted by altitude and end with float('inf').
        """
        # Collect all unique altitude thresholds from both profiles, ensuring 0.0 is present.
        alt_thresholds_set = {0.0}
        for alt_thresh, _ in speed_profile:
            alt_thresholds_set.add(alt_thresh)
        for alt_thresh, _ in vs_profile:
            alt_thresholds_set.add(alt_thresh)

        # Sort and deduplicate boundaries to define segments
        # e.g. [0.0, 5000.0, 10000.0, 10000.0, float('inf')] -> [0.0, 5000.0, 10000.0, float('inf')]
        unique_sorted_boundaries = sorted(list(alt_thresholds_set))

        processed_boundaries = []
        if unique_sorted_boundaries:
            processed_boundaries.append(unique_sorted_boundaries[0])
            for i in range(1, len(unique_sorted_boundaries)):
                if (
                    unique_sorted_boundaries[i] > unique_sorted_boundaries[i - 1]
                ):  # Strict increase
                    processed_boundaries.append(unique_sorted_boundaries[i])

        if not processed_boundaries or (
            len(processed_boundaries) == 1 and processed_boundaries[0] != float("inf")
        ):
            raise ValueError(
                f"Profile boundaries do not extend to infinity properly or are insufficient: {processed_boundaries}"
            )

        combined_profile_data = []
        # Iterate through pairs of boundaries to define segments [lower_bound, upper_bound)
        # If processed_boundaries is just [0.0, float('inf')], this loop runs once for that segment.
        # If processed_boundaries is just [float('inf')], means effective range 0 to inf with single values.
        if len(processed_boundaries) == 1 and processed_boundaries[0] == float("inf"):
            # Special case: profile essentially flat from 0 to infinity
            query_alt = 0.0  # Query at the start of the implicit [0, inf) segment
            segment_speed = self._get_value_at_alt(query_alt, speed_profile)
            segment_vs = self._get_value_at_alt(query_alt, vs_profile)
            combined_profile_data.append((float("inf"), segment_speed, segment_vs))
            return combined_profile_data

        for i in range(len(processed_boundaries) - 1):
            lower_alt_bound = processed_boundaries[i]
            upper_alt_bound = processed_boundaries[i + 1]

            # Query altitude for this segment [lower_alt_bound, upper_alt_bound).
            # Using lower_alt_bound as the query point is correct as per profile definition.
            query_alt = lower_alt_bound

            segment_speed = self._get_value_at_alt(query_alt, speed_profile)
            segment_vs = self._get_value_at_alt(query_alt, vs_profile)

            combined_profile_data.append((upper_alt_bound, segment_speed, segment_vs))

        return combined_profile_data

    def _get_values_from_combined_profile(
        self, altitude_ft: float, combined_profile: List[Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """
        Helper to get (speed, vs) from a combined profile at a given altitude.
        Combined profile format: [(upper_alt_boundary_ft, speed_kts, vs_fpm), ...]
        """
        # Ensure altitude_ft is not negative for lookup
        current_altitude_ft = max(0.0, altitude_ft)

        for alt_thresh, speed, vs in combined_profile:
            if current_altitude_ft < alt_thresh:
                return speed, vs

        # This fallback should ideally not be reached if combined_profile is correctly formed
        # (i.e., covers all altitudes up to float('inf') and is non-empty).
        # If combined_profile is non-empty and ends with (float('inf'), ...), the loop should always find a value.
        if combined_profile:  # Should be non-empty due to constructor checks
            return (
                combined_profile[-1][1],
                combined_profile[-1][2],
            )  # Value from the last segment (ending in inf)

        raise Exception(
            f"Could not determine performance values for altitude {current_altitude_ft}. Combined profile might be misconfigured or empty."
        )

    def get_tas(self, altitude_ft: float, phase: str) -> float:
        """
        Returns the True Airspeed (TAS) in knots for a given altitude and flight phase.
        Phase can be 'climb', 'descent', or 'cruise'.
        """
        current_altitude_ft = max(0.0, altitude_ft)

        if phase.lower() == "climb":
            if current_altitude_ft >= self.cruise_altitude_ft:
                return self.cruise_speed_kts
            speed, _ = self._get_values_from_combined_profile(
                current_altitude_ft, self._combined_climb_profile
            )
            return speed
        elif phase.lower() == "descent":
            speed, _ = self._get_values_from_combined_profile(
                current_altitude_ft, self._combined_descent_profile
            )
            return speed
        elif phase.lower() == "cruise":
            return self.cruise_speed_kts
        else:
            raise ValueError(
                f"Invalid phase: {phase}. Must be 'climb', 'descent', or 'cruise'."
            )

    def get_vertical_speed(self, altitude_ft: float, phase: str) -> float:
        """
        Returns the vertical speed (VS) in feet per minute (fpm) for a given altitude and flight phase.
        Positive for climb, negative for descent (assuming input profile provides this), zero for cruise.
        """
        current_altitude_ft = max(0.0, altitude_ft)

        if phase.lower() == "climb":
            if current_altitude_ft >= self.cruise_altitude_ft:
                return 0.0  # Reached or exceeded cruise altitude
            _, vs = self._get_values_from_combined_profile(
                current_altitude_ft, self._combined_climb_profile
            )
            return vs
        elif phase.lower() == "descent":
            # If at or below 0 ft (e.g. on ground), VS is 0.
            # Assumes descent profile targets ground or a minimum altitude where VS becomes 0.
            # The profile itself should ideally reflect VS=0 at target minimums if not ground.
            if current_altitude_ft <= 0.001:  # Using a small epsilon for ground check
                # Check if the profile itself dictates a non-zero VS at 0.
                # For simplicity, if at effectively 0 altitude, assume 0 VS for descent.
                # A more robust model might have the profile define VS at 0 explicitly.
                # For now, if at/near ground, descent stops.
                return 0.0
            _, vs = self._get_values_from_combined_profile(
                current_altitude_ft, self._combined_descent_profile
            )
            return (
                vs  # Assumes descent_vertical_speed_profile provides negative values.
            )
        elif phase.lower() == "cruise":
            return 0.0
        else:
            raise ValueError(
                f"Invalid phase: {phase}. Must be 'climb', 'descent', or 'cruise'."
            )

    def get_climb_eta(
        self, origin_airport_elevation_ft: float
    ) -> List[Tuple[float, float]]:
        """
        Calculates the elapsed time (in seconds) since takeoff to reach each significant
        altitude level during climb, up to cruise altitude.

        A "significant altitude" is an altitude at which there is a change in target
        true airspeed or vertical speed, as defined in the climb profiles, or the
        cruise altitude itself.

        Args:
            origin_airport_elevation_ft: The elevation of the origin airport in feet.

        Returns:
            A list of tuples (altitude_ft, elapsed_time_seconds). The first entry
            is always (origin_airport_elevation_ft, 0.0). Subsequent entries
            represent significant altitudes reached and the time taken.
        """
        results: List[Tuple[float, float]] = []
        current_alt_ft = float(origin_airport_elevation_ft)
        elapsed_time_sec = 0.0

        results.append((current_alt_ft, elapsed_time_sec))

        if current_alt_ft >= self.cruise_altitude_ft:
            return results

        for segment_upper_alt, _speed_kts, vs_fpm in self._combined_climb_profile:
            if current_alt_ft >= self.cruise_altitude_ft:
                break

            if vs_fpm <= 0:
                break

            climb_target_for_this_segment_step = min(
                segment_upper_alt, self.cruise_altitude_ft
            )

            if current_alt_ft < climb_target_for_this_segment_step:
                alt_delta_ft = climb_target_for_this_segment_step - current_alt_ft

                # print(f'alt_delta_ft: {alt_delta_ft}, vs_fpm: {vs_fpm}')

                time_delta_min = alt_delta_ft / vs_fpm
                time_delta_sec = time_delta_min * 60.0

                elapsed_time_sec += time_delta_sec
                current_alt_ft = climb_target_for_this_segment_step

                if not results or results[-1][0] != current_alt_ft:
                    results.append((current_alt_ft, elapsed_time_sec))
                elif results[-1][0] == current_alt_ft:
                    results[-1] = (current_alt_ft, elapsed_time_sec)

        return results
    
    def get_along_track_wind_adjusted_distance_for_climb(
        self, origin_airport_elevation_ft: float
    ) -> List[Tuple[float, float]]:
        """
        Calculates the cumulative along-track distance (in nautical miles) covered
        during climb at each significant altitude level, up to cruise altitude.
        The distance is calculated based on True Airspeed (TAS) and time spent in each
        climb segment. The term "wind-adjusted" in the method name suggests this
        is the air distance, which forms a basis before ground speed calculations
        incorporating actual wind effects.

        A "significant altitude" (or "knot point") is an altitude at which there is a 
        change in target true airspeed or vertical speed as defined in the climb profiles, 
        or the cruise altitude itself.

        Args:
            origin_airport_elevation_ft: The elevation of the origin airport in feet.

        Returns:
            A list of tuples (altitude_ft, cumulative_distance_nm). The first entry
            is always (origin_airport_elevation_ft, 0.0). Subsequent entries
            represent significant altitudes reached and the cumulative air distance covered.
        """
        results: List[Tuple[float, float]] = []
        current_alt_ft = float(origin_airport_elevation_ft)
        cumulative_distance_nm = 0.0

        results.append((current_alt_ft, cumulative_distance_nm))

        # If already at or above cruise altitude, no further climb distance is covered.
        if current_alt_ft >= self.cruise_altitude_ft:
            return results

        # Iterate through the segments defined in the combined climb profile.
        # Each tuple is (segment_upper_alt_ft, segment_tas_kts, segment_vs_fpm)
        # where segment_tas_kts and segment_vs_fpm apply for the climb up to segment_upper_alt_ft.
        for segment_upper_alt, segment_tas_kts, segment_vs_fpm in self._combined_climb_profile:
            # Stop if we've effectively reached or climbed past the cruise altitude.
            if current_alt_ft >= self.cruise_altitude_ft:
                break

            # If vertical speed is not positive, we cannot climb further using this segment's parameters.
            # This implies an issue with the profile for climb or that the aircraft is "stuck".
            if segment_vs_fpm <= 0:
                break 

            # Determine the target altitude for this specific processing step.
            climb_target_for_this_segment_step = min(segment_upper_alt, self.cruise_altitude_ft)

            # Only calculate if there's altitude to gain to reach climb_target_for_this_segment_step.
            if current_alt_ft < climb_target_for_this_segment_step:
                alt_delta_ft = climb_target_for_this_segment_step - current_alt_ft
                
                # Time to climb this altitude delta in hours.
                # segment_vs_fpm is in ft/min. (segment_vs_fpm * 60) is ft/hr.
                # alt_delta_ft / (ft/hr) = hours.
                # segment_vs_fpm is guaranteed positive due to the check above.
                time_delta_hr = alt_delta_ft / (segment_vs_fpm * 60.0)

                # Distance covered in this step (TAS is in knots, i.e., nm/hr).
                distance_step_nm = segment_tas_kts * time_delta_hr
                cumulative_distance_nm += distance_step_nm

                # Update current altitude to the altitude reached in this step.
                current_alt_ft = climb_target_for_this_segment_step

                # Add the new point (altitude, cumulative_distance) to results.
                # If current_alt_ft is the same as the last recorded altitude, update its distance.
                if not results or results[-1][0] != current_alt_ft:
                    results.append((current_alt_ft, cumulative_distance_nm))
                elif results[-1][0] == current_alt_ft:
                    results[-1] = (current_alt_ft, cumulative_distance_nm)
        
        return results

    def get_along_track_wind_adjusted_distance_for_descent(
        self, destination_airport_elevation_ft: float
    ) -> List[Tuple[float, float]]:
        """
        Calculates the cumulative along-track distance (in nautical miles) covered
        during descent from cruise altitude to destination airport elevation.
        The distance is calculated based on True Airspeed (TAS) and time spent in each
        descent segment. This is the air distance.

        A "significant altitude" is an altitude at which there is a change in target
        true airspeed or vertical speed as defined in the descent profiles, or the
        destination airport elevation itself.

        Args:
            destination_airport_elevation_ft: The elevation of the destination airport in feet.

        Returns:
            A list of tuples (altitude_ft, cumulative_distance_nm_from_TOD). The first entry
            is always (cruise_altitude_ft, 0.0). Subsequent entries represent
            significant altitudes reached and the cumulative air distance covered from TOD.
        """
        results: List[Tuple[float, float]] = []
        current_alt_ft = float(self.cruise_altitude_ft)
        cumulative_distance_nm = 0.0
        destination_alt_ft = float(destination_airport_elevation_ft)

        results.append((current_alt_ft, cumulative_distance_nm))

        # If already at or below destination altitude, no descent distance is covered.
        if current_alt_ft <= destination_alt_ft:
            return results

        # Collect all unique altitude boundaries from the descent profile that are
        # below the current altitude and above or at the destination altitude.
        # Also include the destination altitude itself.
        descent_profile_boundaries = [p[0] for p in self._combined_descent_profile]
        target_altitudes_to_pass = sorted(
            list(
                set(
                    [
                        b
                        for b in descent_profile_boundaries
                        if b < current_alt_ft and b >= destination_alt_ft
                    ]
                    + [destination_alt_ft]
                )
            ),
            reverse=True,  # Process from higher to lower altitudes
        )
        
        # Ensure we only consider altitudes strictly below the current starting altitude
        target_altitudes_to_pass = [
            alt for alt in target_altitudes_to_pass if alt < current_alt_ft
        ]


        for next_target_alt in target_altitudes_to_pass:
            if current_alt_ft <= destination_alt_ft:
                break  # Reached or passed destination

            # Get TAS and VS for the segment *from* current_alt_ft *towards* next_target_alt.
            # The TAS and VS are determined by the current_alt_ft.
            segment_tas_kts, segment_vs_fpm = self._get_values_from_combined_profile(
                current_alt_ft, self._combined_descent_profile
            )

            if segment_vs_fpm >= 0:
                segment_vs_fpm = -segment_vs_fpm # Convert to negative for descent if the original VS is positive (in case the vnav_profiles.py gives a positive VS for a descent profile)
            
            # Ensure we don't "descend" below the final destination_alt_ft in this step
            actual_descent_target_for_step = max(next_target_alt, destination_alt_ft)


            if current_alt_ft > actual_descent_target_for_step:
                alt_delta_ft = current_alt_ft - actual_descent_target_for_step
                
                # Time to descend this altitude delta in hours.
                # segment_vs_fpm is negative. (abs(segment_vs_fpm) * 60) is ft/hr.
                time_delta_hr = alt_delta_ft / (abs(segment_vs_fpm) * 60.0)

                # Distance covered in this step (TAS is in knots, i.e., nm/hr).
                distance_step_nm = segment_tas_kts * time_delta_hr
                cumulative_distance_nm += distance_step_nm

                current_alt_ft = actual_descent_target_for_step

                if not results or results[-1][0] != current_alt_ft:
                    results.append((current_alt_ft, cumulative_distance_nm))
                elif results[-1][0] == current_alt_ft: # Should not happen if target_altitudes are unique and sorted
                    results[-1] = (current_alt_ft, cumulative_distance_nm)
            
            if current_alt_ft <= destination_alt_ft: # Check again after update
                break
        
        # Ensure the final destination altitude is in the results if not already last.
        # This can happen if the loop terminates early or if destination_alt_ft wasn't a profile boundary.
        if results[-1][0] > destination_alt_ft and current_alt_ft > destination_alt_ft :
             # This case implies we did not reach destination_alt_ft exactly through target_altitudes_to_pass
             # We need one final segment calculation from current_alt_ft down to destination_alt_ft
            segment_tas_kts, segment_vs_fpm = self._get_values_from_combined_profile(
                current_alt_ft, self._combined_descent_profile
            )
            if segment_vs_fpm < 0: #Proceed only if VS is negative
                alt_delta_ft = current_alt_ft - destination_alt_ft
                time_delta_hr = alt_delta_ft / (abs(segment_vs_fpm) * 60.0)
                distance_step_nm = segment_tas_kts * time_delta_hr
                cumulative_distance_nm += distance_step_nm
                current_alt_ft = destination_alt_ft
                # Append or update the destination point
                if not results or results[-1][0] != current_alt_ft:
                     results.append((current_alt_ft, cumulative_distance_nm))
                elif results[-1][0] == current_alt_ft : # update if alt exists
                     results[-1] = (current_alt_ft, cumulative_distance_nm)

        # If the loop finished and current_alt_ft is not exactly destination_alt_ft,
        # but destination_alt_ft was the target, ensure it's represented.
        # This primarily handles the case where the loop correctly iterated down to destination_alt_ft
        # and it was added. The check `results[-1][0] != destination_alt_ft` might be redundant
        # if logic correctly adds it.
        # The final point should always be (destination_alt_ft, final_cumulative_distance)
        # if destination is reached.
        # If the results list's last altitude is not the destination_alt_ft,
        # but current_alt_ft became destination_alt_ft, ensure it's the last entry.
        if current_alt_ft == destination_alt_ft and (not results or results[-1][0] != destination_alt_ft):
            # This condition means current_alt_ft is now destination_alt_ft
            # but it's not the last entry in results or results is empty (though first entry is cruise alt)
             if not results or results[-1][0] > destination_alt_ft : # Append if last recorded alt is higher
                results.append((destination_alt_ft, cumulative_distance_nm))
             elif results[-1][0] == destination_alt_ft: # Update if exists
                results[-1] = (destination_alt_ft, cumulative_distance_nm)
        
        # results right now is a list of tuples (alt_ft, distance_nm), but Top of Descent is with distance_nm = 0
        # but we need the distance_nm at landing to be 0, thus we have to subtract the distance_nm at cruise altitude from the distance_nm at landing
        # note that we also inverse the sign of distance_nm to make it **positive**, this is the distance to go to the landing location, thus it should be positive.
        distance_at_landing = results[-1][1]
        results = [(alt, -distance + distance_at_landing) for alt, distance in results]

        return results

    def get_descent_eta(
        self, destination_airport_elevation_ft: float
    ) -> List[Tuple[float, float]]:
        """
        Calculates the elapsed time (in seconds) from Top of Descent (TOD) to reach
        each significant altitude level during descent, down to destination airport elevation.
        Descent starts from the aircraft's cruise altitude.

        A "significant altitude" is an altitude at which there is a change in target
        true airspeed or vertical speed, as defined in the descent profiles, or the
        destination airport elevation itself.

        Args:
            destination_airport_elevation_ft: The elevation of the destination airport in feet.

        Returns:
            A list of tuples (altitude_ft, elapsed_time_seconds_from_TOD). The first entry
            is always (cruise_altitude_ft, 0.0). Subsequent entries represent
            significant altitudes reached and the time taken from TOD.
        """
        results: List[Tuple[float, float]] = []
        current_alt_ft = float(self.cruise_altitude_ft)
        elapsed_time_sec = 0.0
        destination_alt_ft = float(destination_airport_elevation_ft)

        results.append((current_alt_ft, elapsed_time_sec))

        if current_alt_ft <= destination_alt_ft:
            return results

        descent_profile_boundaries = [p[0] for p in self._combined_descent_profile]

        target_altitudes_to_pass = sorted(
            list(
                set(
                    [
                        b
                        for b in descent_profile_boundaries
                        if b < current_alt_ft and b >= destination_alt_ft
                    ]
                    + [destination_alt_ft]
                )
            ),
            reverse=True,
        )

        target_altitudes_to_pass = [
            alt for alt in target_altitudes_to_pass if alt < current_alt_ft
        ]

        for next_target_alt in target_altitudes_to_pass:
            if current_alt_ft <= destination_alt_ft:
                break

            vs_fpm_for_segment = None
            # Find VS for the segment defined by current_alt_ft as its upper boundary,
            # or if current_alt_ft is within a segment, use that segment's VS.
            # First, check if current_alt_ft is an explicit upper boundary in the profile.
            # The VS from such an entry (upper_alt, speed, vs) applies to the segment *below* upper_alt.
            for prof_upper_alt, _s, prof_vs in self._combined_descent_profile:
                if prof_upper_alt == current_alt_ft:
                    vs_fpm_for_segment = prof_vs
                    break

            if vs_fpm_for_segment is None:
                # If current_alt_ft (e.g. initial cruise_altitude_ft) is not an explicit boundary,
                # it means it's within a larger segment (e.g. between 10k and inf).
                # _get_values_from_combined_profile(current_alt_ft, ...) will give the VS for this segment.
                _unused_speed, vs_fpm_for_segment = (
                    self._get_values_from_combined_profile(
                        current_alt_ft, self._combined_descent_profile
                    )
                )

            if vs_fpm_for_segment is None:  # Descent VS must be negative
                break

            if vs_fpm_for_segment > 0:
                vs_fpm_for_segment = (
                    -vs_fpm_for_segment
                )  # Convert to negative for descent if the original VS is positive (in case the vnav_profiles.py gives a positive VS for a descent profile)

            actual_descent_target_for_step = next_target_alt

            if (
                current_alt_ft > actual_descent_target_for_step
            ):  # Should always be true by loop construction
                alt_delta_ft = current_alt_ft - actual_descent_target_for_step

                # print(f'alt_delta_ft: {alt_delta_ft}, vs_fpm_for_segment: {vs_fpm_for_segment}')

                time_delta_min = alt_delta_ft / abs(vs_fpm_for_segment)
                time_delta_sec = time_delta_min * 60.0

                elapsed_time_sec += time_delta_sec
                current_alt_ft = actual_descent_target_for_step

                if not results or results[-1][0] != current_alt_ft:
                    results.append((current_alt_ft, elapsed_time_sec))
                elif results[-1][0] == current_alt_ft:
                    results[-1] = (current_alt_ft, elapsed_time_sec)

            if current_alt_ft <= destination_alt_ft:
                break

        # results right now is a list of tuples (alt_ft, ETATO), but Top of Descent is with ETATO = 0
        # but we need the ETATO at landing to be 0, thus we have to subtract the ETATO at cruise altitude from the ETATO at landing

        ETATO_at_landing = results[-1][1]
        results = [(alt, ETATO - ETATO_at_landing) for alt, ETATO in results]

        return results 


# Some helper functions for type conversion

# Define the merge helper function at the module level
def _merge_eta_and_distance_profiles(
    eta_profile: list, distance_profile: list,
    reverse_order: bool = False
) -> list:
    """
    Merge two profiles: eta_profile [(alt_ft, eta_sec)], distance_profile [(alt_ft, dist_nm)]
    into [(alt_ft, eta_sec, dist_nm)].

    If an altitude exists in only one, use the most recent value for the other.
    The altitude threshold is valid for altitudes above it, up to the next knot point.
    Profiles are sorted by altitude internally.
    """
    # Defensive: sort by altitude
    eta_profile = sorted(eta_profile, key=lambda x: x[0])
    distance_profile = sorted(distance_profile, key=lambda x: x[0])

    # Collect all unique knot points
    altitudes = sorted(set([a for a, _ in eta_profile] + [a for a, _ in distance_profile]))

    # Build dicts for fast lookup
    eta_dict = {a: v for a, v in eta_profile}
    dist_dict = {a: v for a, v in distance_profile}

    merged = []
    last_eta = None
    last_dist = None

    for alt in altitudes:
        if alt in eta_dict:
            last_eta = eta_dict[alt]
        if alt in dist_dict:
            last_dist = dist_dict[alt]
        merged.append((alt, last_eta, last_dist))

    if reverse_order:
        # Reverse the list by altitude
        merged = sorted(merged, key=lambda x: x[0], reverse=True)
    return merged

def get_eta_and_distance_climb(perf: Performance, origin_airport_elevation_ft: float):
    eta_climb = perf.get_climb_eta(origin_airport_elevation_ft) # (alt_ft, time_sec)
    along_track_wind_adjusted_distance = perf.get_along_track_wind_adjusted_distance_for_climb(origin_airport_elevation_ft) # (alt_ft, distance_nm)

    # Call the module-level merge function
    return _merge_eta_and_distance_profiles(eta_climb, along_track_wind_adjusted_distance)

def get_eta_and_distance_descent(perf: Performance, destination_airport_elevation_ft: float):
    # Example:
    #   Altitude (ft) |  ETATO (s) |    Distance (nm)
    # ------------------------------------------------
    #          35,000 |    -1300.0 |           141.50
    #          28,000 |     -880.0 |            89.00
    #          24,000 |     -640.0 |            59.00
    #          20,000 |     -560.0 |            35.00
    #          10,000 |     -360.0 |            15.00
    #           1,000 |        0.0 |             0.00
    eta_descent = perf.get_descent_eta(destination_airport_elevation_ft) # (alt_ft, time_sec)
    along_track_wind_adjusted_distance = perf.get_along_track_wind_adjusted_distance_for_descent(destination_airport_elevation_ft) # (alt_ft, distance_nm)

    # Call the module-level merge function
    return _merge_eta_and_distance_profiles(eta_descent, along_track_wind_adjusted_distance, reverse_order = True)

# Example Usage (for testing - not part of the class itself):
if __name__ == "__main__":
    # Example profiles (simplified from vnav_profiles.py for brevity)
    LIGHT_AIRCRAFT_CLIMB_PROFILE: List[Tuple[float, float]] = [
        (5000, 80),
        (float("inf"), 90),
    ]
    LIGHT_AIRCRAFT_DESCENT_PROFILE: List[Tuple[float, float]] = [
        (5000, 95),
        (float("inf"), 85),  # Simplified, vnav has more segments
    ]
    LIGHT_AIRCRAFT_CLIMB_VS_PROFILE: List[Tuple[float, float]] = [
        (5000, 600),
        (float("inf"), 400),
    ]
    # For descent VS, values should be negative
    LIGHT_AIRCRAFT_DESCENT_VS_PROFILE: List[Tuple[float, float]] = [
        (5000, -500),
        (float("inf"), -300),  # Example negative VS values
    ]

    # Test case 1: Light Aircraft
    print("--- Test Case: Light Aircraft ---")
    try:
        light_perf = Performance(
            climb_speed_profile=LIGHT_AIRCRAFT_CLIMB_PROFILE,
            descent_speed_profile=LIGHT_AIRCRAFT_DESCENT_PROFILE,
            climb_vertical_speed_profile=LIGHT_AIRCRAFT_CLIMB_VS_PROFILE,
            descent_vertical_speed_profile=LIGHT_AIRCRAFT_DESCENT_VS_PROFILE,
            cruise_altitude_ft=7000,
            cruise_speed_kts=85,
        )

        print(f"Combined Climb Profile: {light_perf._combined_climb_profile}")
        # Expected: [(5000.0, 80.0, 600.0), (float('inf'), 90.0, 400.0)]

        print(f"Combined Descent Profile: {light_perf._combined_descent_profile}")
        # Expected: [(5000.0, 95.0, -500.0), (float('inf'), 85.0, -300.0)]

        # Climb phase
        print(
            f"Climb TAS at 3000 ft: {light_perf.get_tas(3000, 'climb')} kts"
        )  # Expected: 80
        print(
            f"Climb VS at 3000 ft: {light_perf.get_vertical_speed(3000, 'climb')} fpm"
        )  # Expected: 600

        print(
            f"Climb TAS at 5000 ft: {light_perf.get_tas(5000, 'climb')} kts"
        )  # Expected: 90 (since 5000 is start of next segment)
        print(
            f"Climb VS at 5000 ft: {light_perf.get_vertical_speed(5000, 'climb')} fpm"
        )  # Expected: 400

        print(
            f"Climb TAS at 6000 ft: {light_perf.get_tas(6000, 'climb')} kts"
        )  # Expected: 90
        print(
            f"Climb VS at 6000 ft: {light_perf.get_vertical_speed(6000, 'climb')} fpm"
        )  # Expected: 400

        # Reaching cruise altitude
        print(
            f"Climb TAS at 7000 ft: {light_perf.get_tas(7000, 'climb')} kts"
        )  # Expected: 85 (cruise_speed_kts)
        print(
            f"Climb VS at 7000 ft: {light_perf.get_vertical_speed(7000, 'climb')} fpm"
        )  # Expected: 0.0
        print(
            f"Climb TAS at 8000 ft: {light_perf.get_tas(8000, 'climb')} kts"
        )  # Expected: 85 (cruise_speed_kts)
        print(
            f"Climb VS at 8000 ft: {light_perf.get_vertical_speed(8000, 'climb')} fpm"
        )  # Expected: 0.0

        # Cruise phase
        print(
            f"Cruise TAS at 7000 ft: {light_perf.get_tas(7000, 'cruise')} kts"
        )  # Expected: 85
        print(
            f"Cruise VS at 7000 ft: {light_perf.get_vertical_speed(7000, 'cruise')} fpm"
        )  # Expected: 0.0

        # Descent phase
        print(
            f"Descent TAS at 8000 ft (above cruise, descending from higher): {light_perf.get_tas(8000, 'descent')} kts"
        )  # Expected: 85
        print(
            f"Descent VS at 8000 ft: {light_perf.get_vertical_speed(8000, 'descent')} fpm"
        )  # Expected: -300

        print(
            f"Descent TAS at 6000 ft: {light_perf.get_tas(6000, 'descent')} kts"
        )  # Expected: 85
        print(
            f"Descent VS at 6000 ft: {light_perf.get_vertical_speed(6000, 'descent')} fpm"
        )  # Expected: -300

        print(
            f"Descent TAS at 5000 ft: {light_perf.get_tas(5000, 'descent')} kts"
        )  # Expected: 85 (start of [5000, inf) segment for descent)
        print(
            f"Descent VS at 5000 ft: {light_perf.get_vertical_speed(5000, 'descent')} fpm"
        )  # Expected: -300

        print(
            f"Descent TAS at 3000 ft: {light_perf.get_tas(3000, 'descent')} kts"
        )  # Expected: 95
        print(
            f"Descent VS at 3000 ft: {light_perf.get_vertical_speed(3000, 'descent')} fpm"
        )  # Expected: -500

        print(
            f"Descent TAS at 0 ft: {light_perf.get_tas(0, 'descent')} kts"
        )  # Expected: 95
        print(
            f"Descent VS at 0 ft: {light_perf.get_vertical_speed(0, 'descent')} fpm"
        )  # Expected: 0.0

        # Test with more complex profiles (where thresholds don't align)
        NB_JET_CLIMB_SPEED: List[Tuple[float, float]] = [
            (10000, 250),
            (28000, 300),
            (float("inf"), 450),
        ]
        NB_JET_CLIMB_VS_MODIFIED: List[Tuple[float, float]] = [
            (5000, 3000),
            (15000, 2000),
            (float("inf"), 1200),
        ]

        print(
            "\\n--- Test Case: Narrow Body Jet with Mismatched Thresholds (Climb) ---"
        )
        nb_perf = Performance(
            climb_speed_profile=NB_JET_CLIMB_SPEED,
            descent_speed_profile=LIGHT_AIRCRAFT_DESCENT_PROFILE,  # dummy
            climb_vertical_speed_profile=NB_JET_CLIMB_VS_MODIFIED,
            descent_vertical_speed_profile=LIGHT_AIRCRAFT_DESCENT_VS_PROFILE,  # dummy
            cruise_altitude_ft=35000,
            cruise_speed_kts=450,
        )
        print(f"NB Jet Combined Climb Profile: {nb_perf._combined_climb_profile}")
        # Expected segments based on boundaries 0, 5k, 10k, 15k, 28k, inf
        # [0, 5k): TAS(0)=250, VS(0)=3000 -> (5000.0, 250, 3000)
        # [5k, 10k): TAS(5k)=250, VS(5k)=2000 -> (10000.0, 250, 2000)
        # [10k, 15k): TAS(10k)=300, VS(10k)=2000 -> (15000.0, 300, 2000)
        # [15k, 28k): TAS(15k)=300, VS(15k)=1200 -> (28000.0, 300, 1200)
        # [28k, inf): TAS(28k)=450, VS(28k)=1200 -> (inf, 450, 1200)

        # Querying some values
        print(f"NB Climb TAS at 4000 ft: {nb_perf.get_tas(4000, 'climb')}")  # Exp: 250
        print(
            f"NB Climb VS at 4000 ft: {nb_perf.get_vertical_speed(4000, 'climb')}"
        )  # Exp: 3000

        print(f"NB Climb TAS at 7000 ft: {nb_perf.get_tas(7000, 'climb')}")  # Exp: 250
        print(
            f"NB Climb VS at 7000 ft: {nb_perf.get_vertical_speed(7000, 'climb')}"
        )  # Exp: 2000

        print(
            f"NB Climb TAS at 12000 ft: {nb_perf.get_tas(12000, 'climb')}"
        )  # Exp: 300
        print(
            f"NB Climb VS at 12000 ft: {nb_perf.get_vertical_speed(12000, 'climb')}"
        )  # Exp: 2000

        print(
            f"NB Climb TAS at 20000 ft: {nb_perf.get_tas(20000, 'climb')}"
        )  # Exp: 300
        print(
            f"NB Climb VS at 20000 ft: {nb_perf.get_vertical_speed(20000, 'climb')}"
        )  # Exp: 1200

        print(
            f"NB Climb TAS at 30000 ft: {nb_perf.get_tas(30000, 'climb')}"
        )  # Exp: 450
        print(
            f"NB Climb VS at 30000 ft: {nb_perf.get_vertical_speed(30000, 'climb')}"
        )  # Exp: 1200

    except ValueError as e:
        print(f"Error initializing Performance class: {e}")
    except Exception as e:
        print(f"An error occurred during testing: {e}")

    print("\\n--- Testing ETA Methods ---")
    try:
        # Use light_perf from previous tests (cruise_alt=7000, CS=85)
        # Climb Profile: [(5000, 80sp, 600vs), (inf, 90sp, 400vs)]
        # Descent Profile (Combined): [(5000.0, 95.0, -500.0), (float('inf'), 85.0, -300.0)]
        #   This means: VS=-300 for alt >= 5000, VS=-500 for alt < 5000.

        if "light_perf" not in locals():  # Ensure light_perf is defined
            light_perf = Performance(
                climb_speed_profile=LIGHT_AIRCRAFT_CLIMB_PROFILE,
                descent_speed_profile=LIGHT_AIRCRAFT_DESCENT_PROFILE,
                climb_vertical_speed_profile=LIGHT_AIRCRAFT_CLIMB_VS_PROFILE,
                descent_vertical_speed_profile=LIGHT_AIRCRAFT_DESCENT_VS_PROFILE,
                cruise_altitude_ft=7000,
                cruise_speed_kts=85,
            )

        print("\\n--- Test get_climb_eta ---")
        # Test 1: Simple climb from 0 ft
        eta_climb1 = light_perf.get_climb_eta(origin_airport_elevation_ft=0)
        print(f"Climb ETA from 0ft to 7000ft: {eta_climb1}")
        # Expected: [(0.0, 0.0), (5000.0, 500.0), (7000.0, 800.0)]
        # 0 to 5000ft: 5000ft / 600fpm * 60s/min = 500s
        # 5000 to 7000ft: 2000ft / 400fpm * 60s/min = 300s. Total: 500+300=800s

        # Test 2: Origin at a profile boundary
        eta_climb2 = light_perf.get_climb_eta(origin_airport_elevation_ft=5000)
        print(f"Climb ETA from 5000ft to 7000ft: {eta_climb2}")
        # Expected: [(5000.0, 0.0), (7000.0, 300.0)]

        # Test 3: Origin above cruise altitude
        eta_climb3 = light_perf.get_climb_eta(origin_airport_elevation_ft=8000)
        print(f"Climb ETA from 8000ft (cruise 7000ft): {eta_climb3}")
        # Expected: [(8000.0, 0.0)]

        # Test 4: Cruise altitude within a segment
        # Modify cruise alt for this test case locally if needed, or create new perf object
        perf_cruise_in_segment = Performance(
            climb_speed_profile=LIGHT_AIRCRAFT_CLIMB_PROFILE,
            descent_speed_profile=LIGHT_AIRCRAFT_DESCENT_PROFILE,
            climb_vertical_speed_profile=LIGHT_AIRCRAFT_CLIMB_VS_PROFILE,
            descent_vertical_speed_profile=LIGHT_AIRCRAFT_DESCENT_VS_PROFILE,
            cruise_altitude_ft=6000,  # Cruise at 6000 ft
            cruise_speed_kts=85,
        )
        eta_climb4 = perf_cruise_in_segment.get_climb_eta(origin_airport_elevation_ft=0)
        print(f"Climb ETA from 0ft to 6000ft: {eta_climb4}")
        # Expected: [(0.0, 0.0), (5000.0, 500.0), (6000.0, 650.0)]
        # 5000 to 6000ft: 1000ft / 400fpm * 60s/min = 150s. Total: 500+150=650s

        print("\\n--- Test get_descent_eta ---")
        # Test 1: Simple descent to 0 ft (from cruise 7000ft)
        # Descent profile combined: [(5000.0, 95.0, -500.0), (inf, 85.0, -300.0)]
        # Segment [5000, inf) has VS -300. Segment [0, 5000) has VS -500.
        eta_descent1 = light_perf.get_descent_eta(destination_airport_elevation_ft=0)
        print(f"Descent ETA from 7000ft to 0ft: {eta_descent1}")
        # Expected: [(7000.0, 0.0), (5000.0, 400.0), (0.0, 1000.0)]
        # 7000 to 5000ft: 2000ft / 300fpm * 60s/min = 400s
        # 5000 to 0ft: 5000ft / 500fpm * 60s/min = 600s. Total from 5000ft level = 600. Total from TOD = 400+600=1000s.

        # Test 2: Destination at a profile boundary
        eta_descent2 = light_perf.get_descent_eta(destination_airport_elevation_ft=5000)
        print(f"Descent ETA from 7000ft to 5000ft: {eta_descent2}")
        # Expected: [(7000.0, 0.0), (5000.0, 400.0)]

        # Test 3: Cruise altitude below destination (no descent)
        # Create specific perf object for this
        perf_cruise_low = Performance(
            climb_speed_profile=LIGHT_AIRCRAFT_CLIMB_PROFILE,  # dummy
            descent_speed_profile=LIGHT_AIRCRAFT_DESCENT_PROFILE,
            climb_vertical_speed_profile=LIGHT_AIRCRAFT_CLIMB_VS_PROFILE,  # dummy
            descent_vertical_speed_profile=LIGHT_AIRCRAFT_DESCENT_VS_PROFILE,
            cruise_altitude_ft=4000,  # Cruise at 4000 ft
            cruise_speed_kts=85,
        )
        eta_descent3 = perf_cruise_low.get_descent_eta(
            destination_airport_elevation_ft=5000
        )
        print(f"Descent ETA from 4000ft to 5000ft: {eta_descent3}")
        # Expected: [(4000.0, 0.0)]

        # Test 4: Destination within a segment
        eta_descent4 = light_perf.get_descent_eta(destination_airport_elevation_ft=2000)
        print(f"Descent ETA from 7000ft to 2000ft: {eta_descent4}")
        # Expected: [(7000.0, 0.0), (5000.0, 400.0), (2000.0, 760.0)]
        # 7000 to 5000ft: 400s (as above) -> (5000.0, 400.0)
        # 5000 to 2000ft: (5000-2000)=3000ft / 500fpm * 60s/min = 360s.
        # Total time to 2000ft: 400s (to 5k) + 360s (from 5k to 2k) = 760s.

    except ValueError as e:
        print(f"Error initializing Performance class for ETA tests: {e}")
    except Exception as e:
        print(f"An error occurred during ETA testing: {e}")
