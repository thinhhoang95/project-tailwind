
import pandas as pd
import datetime


def assign_delays(
    flights_df: pd.DataFrame,
    hotspot_flights: list,
    window_end: datetime.datetime,
    rate: int,
):
    """
    Assigns delays to flights in a hotspot.

    Args:
        flights_df (pd.DataFrame): The main DataFrame of all flights.
        hotspot_flights (list): A list of flight identifiers in the hotspot.
        window_end (datetime.datetime): The end time of the hotspot window.
        rate (int): The maximum allowed number of flights in the window.

    Returns:
        pd.DataFrame: The updated flights DataFrame with new revised_takeoff_time.
    """
    if not hotspot_flights:
        return flights_df

    # Sort flights in the hotspot by their current revised takeoff time
    hotspot_flights_df = flights_df[
        flights_df["flight_identifier"].isin(hotspot_flights)
    ].sort_values("revised_takeoff_time")

    # The flights to be delayed are those exceeding the rate
    flights_to_delay = hotspot_flights_df.iloc[rate:]

    if not flights_to_delay.empty:
        # The reference flight for delay calculation is the one at the 'rate' position
        reference_flight = hotspot_flights_df.iloc[rate - 1]
        
        # All subsequent flights in the hotspot are delayed
        for flight_id in flights_to_delay["flight_identifier"]:
            current_flight = flights_df[flights_df["flight_identifier"] == flight_id].iloc[0]
            
            # The delay is calculated to move the flight just after the window
            # relative to the reference flight's takeoff time.
            time_to_end_of_window = (window_end - reference_flight["revised_takeoff_time"]).total_seconds() / 60
            
            # Add a small buffer to ensure it's outside the window
            delay_minutes = time_to_end_of_window + 1 

            new_takeoff_time = current_flight["initial_takeoff_time"] + datetime.timedelta(
                minutes=delay_minutes
            )

            # Update the takeoff time only if the new delay is greater
            if new_takeoff_time > current_flight["revised_takeoff_time"]:
                flights_df.loc[
                    flights_df["flight_identifier"] == flight_id, "revised_takeoff_time"
                ] = new_takeoff_time

    return flights_df
