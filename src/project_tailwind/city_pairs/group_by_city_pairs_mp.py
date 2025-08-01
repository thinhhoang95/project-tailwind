import argparse
import glob
import logging
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import pandas as pd
from project_tailwind.city_pairs.group_by_city_pairs import \
    group_city_pairs as process_files_to_df_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def group_city_pairs_mp(input_dir: str, output_dir: str, num_processes: int = None):
    """
    Multiprocessing wrapper for grouping city pairs using pandas.

    Args:
        input_dir: The directory containing historical route data as CSV files.
        output_dir: The directory to save the output CSV files.
        num_processes: Number of processes to use. Defaults to cpu_count().
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    logging.info(f"Starting multiprocessing with {num_processes} processes from {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    csv_pattern = os.path.join(input_dir, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)

    if not csv_files:
        logging.warning(f"No CSV files found in {input_dir}")
        return

    # Split files into chunks for each process
    chunk_size = max(1, len(csv_files) // num_processes)
    file_chunks = [
        csv_files[i : i + chunk_size] for i in range(0, len(csv_files), chunk_size)
    ]

    logging.info(f"Processing {len(csv_files)} files in {len(file_chunks)} chunks")

    # This will hold the aggregated data from all processes
    aggregated_data: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit file chunks to be processed
        future_to_chunk = {
            executor.submit(process_files_to_df_dict, chunk): chunk
            for chunk in file_chunks
        }

        for future in as_completed(future_to_chunk):
            try:
                # The result is a dictionary of DataFrames
                df_dict = future.result()
                for filename, df in df_dict.items():
                    aggregated_data[filename].append(df)
                logging.info(f"Completed a chunk of {len(future_to_chunk[future])} files.")
            except Exception as e:
                logging.error(f"A chunk generated an exception: {e}")

    logging.info("All chunks processed. Now consolidating and writing to disk.")

    # Consolidate and write aggregated results to CSV files
    for filename, df_list in aggregated_data.items():
        if not df_list:
            continue
        
        output_path = os.path.join(output_dir, filename)
        
        # Concatenate all DataFrames for a given city-pair file
        final_df = pd.concat(df_list, ignore_index=True)
        
        # Check if file exists to decide whether to write header
        is_new_file = not os.path.exists(output_path)
        
        try:
            # Use 'a' mode to append to existing files if any
            final_df.to_csv(output_path, mode="a", header=is_new_file, index=False)
        except IOError as e:
            logging.error(f"Could not write to file {output_path}: {e}")


    logging.info(f"Finished multiprocessing. Output files are in {output_dir}")


def main():
    """Main function to run the multiprocessing script from the command line."""
    parser = argparse.ArgumentParser(
        description="Group flight routes by city pairs from historical CSV data using multiprocessing and pandas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=False,
        default="D:/project-akrav/matched_filtered_data",
        help="Directory containing historical route data as CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="output/city_pairs/grouped_flights_by_cpairs",
        help="Directory to save the grouped city pair CSV files.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes to use. Defaults to the number of CPU cores.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        return

    group_city_pairs_mp(args.input_dir, args.output_dir, args.num_processes)


if __name__ == "__main__":
    main()
