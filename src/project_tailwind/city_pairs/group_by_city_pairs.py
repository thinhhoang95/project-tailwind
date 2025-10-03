import argparse
import glob
import logging
import os
from typing import Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def group_city_pairs(file_list: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Groups flight routes by city pairs from a list of CSV files using pandas.

    Args:
        file_list: A list of paths to CSV files containing route data.

    Returns:
        A dictionary where keys are output filenames (e.g., 'EBEG.csv') and
        values are pandas DataFrames containing the corresponding route data.
    """
    if not file_list:
        return {}

    # Read all CSVs into a single DataFrame
    df_list = []
    for f in file_list:
        try:
            # Add error handling for empty or malformed files
            df_list.append(pd.read_csv(f, on_bad_lines="warn"))
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            logging.warning(f"Skipping file {f}: {e}")
            continue

    if not df_list:
        logging.warning("No valid data found in the provided files.")
        return {}

    df = pd.concat(df_list, ignore_index=True)

    # Basic data cleaning
    df.dropna(subset=["flight_id", "real_waypoints"], inplace=True)
    df = df[df["real_waypoints"].str.contains(" ", na=False)]

    # Vectorized operations to get origin and destination
    waypoints_split = df["real_waypoints"].str.split()
    df["origin"] = waypoints_split.str[0]
    df["destination"] = waypoints_split.str[-1]

    # Filter out invalid ICAO-like codes (optional, but good practice)
    df = df[df["origin"].str.len() >= 4]
    df = df[df["destination"].str.len() >= 4]

    if df.empty:
        logging.info("No valid routes found after filtering.")
        return {}

    # Create the output filename key
    df["output_filename"] = df["origin"].str[:2].str.upper() + df["destination"].str[:2].str.upper() + ".csv"

    # Rename columns for the final output
    df.rename(columns={"flight_id": "flight_identifier", "real_waypoints": "route"}, inplace=True)

    # Group by the target filename
    grouped_data = df.groupby("output_filename")

    # Create a dictionary of DataFrames
    output_dfs = {name: group[["flight_identifier", "route"]] for name, group in grouped_data}

    return output_dfs


def main():
    """Main function to run the script from the command line for single-threaded processing."""
    parser = argparse.ArgumentParser(
        description="Group flight routes by city pairs from historical CSV data using pandas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=False,
        default="D:\\project-akrav\\matched_filtered_data",
        help="Directory containing historical route data as CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="output/city_pairs/grouped_flights_by_cpairs",
        help="Directory to save the grouped city pair CSV files.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all CSV files recursively
    csv_pattern = os.path.join(args.input_dir, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)

    if not csv_files:
        logging.warning(f"No CSV files found in {args.input_dir}")
        return

    logging.info(f"Processing {len(csv_files)} files from {args.input_dir}")

    # Process files and get the dictionary of DataFrames
    output_dataframes = group_city_pairs(csv_files)

    # Write the results to disk
    for filename, df in output_dataframes.items():
        output_path = os.path.join(args.output_dir, filename)

        # Append to existing files or create new ones
        is_new_file = not os.path.exists(output_path)
        df.to_csv(output_path, mode="a", header=is_new_file, index=False)

    logging.info(f"Finished processing. Output files are in {args.output_dir}")


if __name__ == "__main__":
    main()
