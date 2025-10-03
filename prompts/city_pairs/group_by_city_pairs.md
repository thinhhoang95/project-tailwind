Please help me write the Python code to build the list of city pairs from historical data.

# Inputs
1. The directory containing historical route data (csv files). Each CSV looks like following:
    ```csv
    flight_id,real_waypoints,pass_times,speeds,alts,real_full_waypoints,full_pass_times,full_speeds,full_alts
    0200AFRAM650E,GMFO VIBAS BLN BINVA BASIM_84 ABRIX DIBES LFPQ LFPO,1680333942 1680335699 1680336119 1680336479 1680337679 1680338879 1680340979 1680342119 1680342539,0.0944 0.2169 0.2533 0.2208 0.2335 0.2351 0.2348 0.1015 0.0000,1044 11918 11895 11857 11750 11590 8016 1745 236,OJD _rKTz5NPa _IY6nwpxO VIBAS BLN BINVA BASIM_84 ABRIX DIBES _nupgHjcL _YOMHXtrL _ULWw9op1 LFPQ _3pV0F70F _Jygd1MRR LFPO,1680333942 1680333959 1680334739 1680335699 1680336119 1680336479 1680337679 1680338879 1680340979 1680341459 1680341639 1680341999 1680342119 1680342299 1680342479 1680342539,0.0944 0.0944 0.2215 0.2169 0.2533 0.2208 0.2335 0.2351 0.2348 0.1861 0.1966 0.1334 0.1015 0.0879 0.0663 0.0000,1044 1044 9891 11918 11895 11857 11750 11590 8016 3315 3307 2393 1745 1196 450 236
    ```
    The origin airport is the first element in the `real_waypoints` column, and the destination airport is the last element in the `real_waypoints` column.

    The path defaults to `D:/project_akrav/matched_filtered_data`.
2. The output_dir for the csv output files. Check if the directory exists, if not create the directory.

# Instructions
1. As we iterate through all the CSV files in the historical data directory, we shall write out the routes into the output files in categories:

    For example: a flight from LFPG to EGLL will be written to the file called LFEG.csv.

    The content of the file includes: the `flight_identifier`, and `route` (which is the exact copy of the `real_waypoints` column).

