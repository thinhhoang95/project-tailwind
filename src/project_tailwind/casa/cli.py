
import argparse
from cirrus.casa.casa_mp import run_casa

def main():
    """
    Command-line interface for the C-CASA simulation.
    """
    parser = argparse.ArgumentParser(
        description="Run the Continuous Computer Assisted Slot Allocation (C-CASA) simulation."
    )

    parser.add_argument(
        "--flights_path",
        type=str,
        default="cases/flights_20230801_0612.csv",
        help="Path to the flights CSV file.",
    )
    parser.add_argument(
        "--tv_path",
        type=str,
        default="cases/traffic_volumes_simplified.geojson",
        help="Path to the traffic volumes GeoJSON file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/ccasa.csv",
        help="Path to save the output CSV file with flight delays.",
    )
    parser.add_argument(
        "--time_begin",
        type=str,
        default="2023-08-01 00:00:00",
        help="Start of the simulation time window (YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--time_end",
        type=str,
        default="2023-08-01 23:59:59",
        help="End of the simulation time window (YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--window_length_min",
        type=int,
        default=20,
        help="Length of the rolling window in minutes.",
    )
    parser.add_argument(
        "--window_stride_min",
        type=int,
        default=10,
        help="Stride of the rolling window in minutes.",
    )

    args = parser.parse_args()

    run_casa(
        flights_path=args.flights_path,
        tv_path=args.tv_path,
        output_path=args.output_path,
        time_begin=args.time_begin,
        time_end=args.time_end,
        window_length_min=args.window_length_min,
        window_stride_min=args.window_stride_min,
    )

if __name__ == "__main__":
    main()
