# Delay Assigner Documentation

## Overview

The `assign_delays` function is a core component of the CASA (Congestion-Aware Scheduling Algorithm) system. Its primary purpose is to manage airport capacity by assigning delays to flights scheduled to depart from a congested airport "hotspot" within a specific time window. When the number of flights in a hotspot exceeds a defined rate, this function strategically delays the excess flights to alleviate congestion.

## Algorithm Description

The algorithm operates as follows:

1.  **Identify Hotspot Flights**: The function receives a list of `flight_identifier`s that are part of a congestion hotspot. It filters the main flights DataFrame to isolate these specific flights.

2.  **Sort by Takeoff Time**: The identified hotspot flights are sorted in ascending order based on their `revised_takeoff_time`. This ensures that flights are processed in their scheduled departure sequence.

3.  **Select Flights for Delay**: The function determines which flights to delay based on the `rate` parameter, which represents the maximum number of flights the airport can handle in the given window. Any flight in the sorted list beyond the position defined by `rate` is marked for a potential delay. For example, if the rate is 10, the 11th flight and all subsequent flights in the hotspot are candidates for delay.

4.  **Establish a Delay Reference**: The last flight that is *not* scheduled for a delay (i.e., the flight at the `rate - 1` index in the sorted list) is used as the "reference flight". This flight's takeoff time serves as the baseline for calculating the necessary delays for all subsequent flights.

5.  **Calculate and Assign Delays**: For each flight that needs to be delayed:
    a. The time difference between the `reference_flight`'s `revised_takeoff_time` and the `window_end` is calculated in minutes.
    b. A small buffer (1 minute) is added to this duration. This ensures the delayed flight is pushed just outside the congested window.
    c. The total calculated delay is added to the flight's **`initial_takeoff_time`** to determine the `new_takeoff_time`.
    d. This `new_takeoff_time` is compared to the flight's existing `revised_takeoff_time`. The function will only update the flight's departure time if the newly calculated time is later. This prevents overwriting a potentially longer delay that might have been assigned by a previous process.

6.  **Return Updated DataFrame**: The function returns the main `flights_df` with the updated `revised_takeoff_time` for the delayed flights.

## Inputs

-   `flights_df` (pandas.DataFrame): A DataFrame containing all flight data. It must include the following columns:
    -   `flight_identifier`: A unique identifier for each flight.
    -   `initial_takeoff_time`: The original, scheduled takeoff time (`datetime.datetime`).
    -   `revised_takeoff_time`: The current takeoff time, which may have been modified by previous operations (`datetime.datetime`).
-   `hotspot_flights` (list): A list of `flight_identifier` strings for the flights within the congested window.
-   `window_end` (datetime.datetime): The timestamp marking the end of the hotspot window.
-   `rate` (int): The maximum number of flights allowed to depart within the window without being delayed.

## Outputs

-   `pd.DataFrame`: The function returns the modified `flights_df`, where the `revised_takeoff_time` for the affected flights has been updated.

## Usage Example

Here is a code snippet demonstrating how to use the `assign_delays` function.

```python
import pandas as pd
import datetime
from project_tailwind.casa.delay_assigner import assign_delays

# 1. Create a sample flights DataFrame
flight_data = {
    'flight_identifier': [f'FL{i:03}' for i in range(1, 7)],
    'initial_takeoff_time': [
        datetime.datetime(2023, 1, 1, 10, 0),
        datetime.datetime(2023, 1, 1, 10, 2),
        datetime.datetime(2023, 1, 1, 10, 4),
        datetime.datetime(2023, 1, 1, 10, 6),
        datetime.datetime(2023, 1, 1, 10, 8),
        datetime.datetime(2023, 1, 1, 10, 10),
    ],
    'revised_takeoff_time': [
        datetime.datetime(2023, 1, 1, 10, 0),
        datetime.datetime(2023, 1, 1, 10, 2),
        datetime.datetime(2023, 1, 1, 10, 4),
        datetime.datetime(2023, 1, 1, 10, 6),
        datetime.datetime(2023, 1, 1, 10, 8),
        datetime.datetime(2023, 1, 1, 10, 10),
    ]
}
flights_df = pd.DataFrame(flight_data)

# 2. Define the hotspot parameters
# All flights are in the hotspot for this example
hotspot_flights = [f'FL{i:03}' for i in range(1, 7)]
window_end = datetime.datetime(2023, 1, 1, 10, 15)
# The airport can only handle 3 flights in this window
rate = 3

print("Original DataFrame:")
print(flights_df)

# 3. Assign delays
updated_flights_df = assign_delays(
    flights_df=flights_df.copy(),
    hotspot_flights=hotspot_flights,
    window_end=window_end,
    rate=rate
)

print("\nUpdated DataFrame:")
print(updated_flights_df)

# Expected Output Explanation:
# - FL001, FL002, FL003 are within the rate and are not delayed.
# - FL003 is the reference flight (rate - 1 = 2nd index). Its takeoff is 10:04.
# - The window ends at 10:15. The time to the end of the window is 11 minutes.
# - A 1-minute buffer is added, so the delay is 12 minutes.
# - FL004's new time = initial_takeoff_time (10:06) + 12 mins = 10:18.
# - FL005's new time = initial_takeoff_time (10:08) + 12 mins = 10:20.
# - FL006's new time = initial_takeoff_time (10:10) + 12 mins = 10:22.
```