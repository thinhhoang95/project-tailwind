import pandas as pd
import networkx as nx
import multiprocessing as mp
from multiprocessing.pool import Pool
import os
import numpy as np
from functools import partial
from typing import List, Tuple, Dict
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from project_tailwind.city_pairs.ap_grouper.compute_distance_frdist import (
    frechet_distance,
)
from project_tailwind.city_pairs.ap_grouper.community_detection import (
    build_graph_from_distance_matrix,
    detect_communities,
    get_representative_trajectories,
)

# Constants
MIN_FLIGHTS = 6
MIN_COMMUNITIES = 2
DISTANCE_THRESHOLD = 1.8
GML_FILE_PATH = "D:/project-akrav/data/graphs/ats_fra_nodes_only.gml"

# Get number of CPUs for multiprocessing
N_PROCESSES = max(1, mp.cpu_count() - 1)

# Set multiprocessing start method for Windows compatibility
if os.name == 'nt':  # Windows
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Initialize console for rich output
console = Console()

class WorkerInitializer:
    """Initializes worker processes with a shared graph resource."""
    _waypoint_graph = None

    @classmethod
    def initialize(cls):
        """Loads the waypoint graph if it hasn't been loaded."""
        if cls._waypoint_graph is None:
            console.print("[yellow]Loading waypoint graph for worker...[/yellow]")
            cls._waypoint_graph = nx.read_gml(GML_FILE_PATH)
            console.print(f"[green]✓ Worker loaded graph with {len(cls._waypoint_graph.nodes)} waypoints[/green]")

    @classmethod
    def get_graph(cls):
        """Returns the loaded waypoint graph."""
        if cls._waypoint_graph is None:
            cls.initialize()  # Should not happen if pool is initialized correctly
        return cls._waypoint_graph

def trajectory_to_coordinates(trajectory_str: str) -> List[Tuple[float, float]]:
    """
    Convert a trajectory string to a list of (lat, lon) coordinate tuples.
    
    Args:
        trajectory_str (str): Space-separated waypoint names
        
    Returns:
        List[Tuple[float, float]]: List of (lat, lon) coordinates
    """
    graph = WorkerInitializer.get_graph()
    waypoints = trajectory_str.strip().split()
    coordinates = []
    
    for waypoint in waypoints:
        if waypoint not in graph.nodes:
            print(f"Warning: Waypoint '{waypoint}' not found in graph")
            continue
        
        waypoint_node = graph.nodes[waypoint]
        
        if waypoint_node and 'lat' in waypoint_node and 'lon' in waypoint_node:
            coordinates.append((waypoint_node['lat'], waypoint_node['lon']))
        else:
            print(f"Warning: Waypoint '{waypoint}' not found in graph")
            
    return coordinates


def _compute_distance_chunk(chunk_data: Tuple[List[Tuple[int, str]], List[str]]) -> List[Tuple[int, int, float]]:
    """
    Worker function to compute distances for a chunk of trajectory pairs.
    
    Args:
        chunk_data: Tuple containing (pairs_with_indices, trajectories)
            - pairs_with_indices: List of (i, j) indices for trajectory pairs to compute
            - trajectories: List of all trajectory strings
            
    Returns:
        List of (i, j, distance) tuples
    """
    pairs_with_indices, trajectories = chunk_data
    results = []
    
    for i, j in pairs_with_indices:
        t1, t2 = trajectories[i], trajectories[j]
        
        coords1 = trajectory_to_coordinates(t1)
        coords2 = trajectory_to_coordinates(t2)
        
        if not coords1 or not coords2:
            distance = float('inf')
        else:
            distance = frechet_distance(coords1, coords2)
        
        results.append((i, j, distance))
    
    return results


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'departure_airport' and 'arrival_airport' columns to the dataframe
    if they do not exist, extracted from the 'route' column.

    Args:
        df (pd.DataFrame): The input dataframe with a 'route' column.

    Returns:
        pd.DataFrame: The dataframe with added columns.
    """
    if "departure_airport" not in df.columns and "route" in df.columns:
        route_splits = df["route"].str.split(" ")
        df["departure_airport"] = route_splits.str[0]
        df["arrival_airport"] = route_splits.str[-1]
    return df


def get_trajectories_for_airport_pair(
    df: pd.DataFrame, origin: str, destination: str
) -> List[str]:
    """
    Filters the dataframe for a specific origin-destination pair.

    Args:
        df (pd.DataFrame): The input dataframe with flight data.
        origin (str): The origin airport code.
        destination (str): The destination airport code.

    Returns:
        List[str]: A list of routes as strings.
    """
    df = _prepare_data(df)
    return df[
        (df["departure_airport"] == origin) & (df["arrival_airport"] == destination)
    ]["route"].tolist()


def compute_trajectory_distances(trajectories: List[str], pool: Pool, show_progress: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Computes the distance matrix for a list of trajectories using a multiprocessing pool.

    Args:
        trajectories (List[str]): A list of trajectory strings.
        pool (mp.Pool): An existing multiprocessing pool.
        show_progress (bool): Whether to display a progress bar.

    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing:
            - distance_matrix: numpy array of shape (n, n) with pairwise distances
            - trajectory_list: ordered list of trajectories corresponding to matrix indices
    """
    n_trajectories = len(trajectories)
    distance_matrix = np.zeros((n_trajectories, n_trajectories), dtype=float)
    
    pairs = []
    for i in range(n_trajectories):
        for j in range(i, n_trajectories):
            pairs.append((i, j))
    
    total_pairs = len(pairs)
    console.print(f"[blue]Using {N_PROCESSES} processes to compute {total_pairs} unique distances[/blue]")
    
    chunk_size = max(1, total_pairs // N_PROCESSES)
    chunks = []
    for i in range(0, total_pairs, chunk_size):
        chunk_pairs = pairs[i:i + chunk_size]
        chunks.append((chunk_pairs, trajectories))

    def process_results(p):
        chunk_results = [p.apply_async(_compute_distance_chunk, (chunk,)) for chunk in chunks]
        for result in chunk_results:
            chunk_distances = result.get()
            for i, j, distance in chunk_distances:
                distance_matrix[i, j] = distance
                if i != j:
                    distance_matrix[j, i] = distance

    def process_results_with_progress(p, progress):
        distance_task = progress.add_task("Computing trajectory distances...", total=total_pairs)
        chunk_results = [p.apply_async(_compute_distance_chunk, (chunk,)) for chunk in chunks]
        
        completed_pairs = 0
        for result in chunk_results:
            chunk_distances = result.get()
            for i, j, distance in chunk_distances:
                distance_matrix[i, j] = distance
                if i != j:
                    distance_matrix[j, i] = distance
            completed_pairs += len(chunk_distances)
            progress.update(distance_task, completed=completed_pairs)

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            process_results_with_progress(pool, progress)
    else:
        process_results(pool)

    return distance_matrix, trajectories


def group_trajectories_into_communities(
    distance_matrix: np.ndarray, trajectories: List[str]
) -> Tuple[int, Dict[str, int], Dict[int, str]]:
    """
    Groups trajectories into communities based on the distance matrix.

    Args:
        distance_matrix (np.ndarray): The distance matrix of trajectories.
        trajectories (List[str]): List of trajectory strings corresponding to matrix indices.

    Returns:
        Tuple[int, Dict[str, int], Dict[int, str]]: A tuple containing:
            - number of communities
            - dictionary mapping trajectory to community id
            - dictionary mapping community id to representative trajectory
    """
    graph = build_graph_from_distance_matrix(distance_matrix, threshold=DISTANCE_THRESHOLD)
    
    partition, communities_dict = detect_communities(graph)
    
    representatives_indices = get_representative_trajectories(communities_dict, distance_matrix)
    
    num_communities = len(communities_dict)
    trajectory_to_community = {}
    community_representatives = {}
    
    for community_id, member_indices in communities_dict.items():
        for member_idx in member_indices:
            trajectory_to_community[trajectories[member_idx]] = community_id
        
        rep_idx = representatives_indices[community_id]
        community_representatives[community_id] = trajectories[rep_idx]
    
    return num_communities, trajectory_to_community, community_representatives


def format_community_output(
    communities: Dict[str, int], trajectories: List[str], community_representatives: Dict[int, str]
) -> pd.DataFrame:
    """
    Formats the community detection output into a DataFrame.

    Args:
        communities (Dict[str, int]): Dictionary mapping trajectory to community id.
        trajectories (List[str]): The original list of trajectories.
        community_representatives (Dict[int, str]): Dictionary mapping community id to representative trajectory.

    Returns:
        pd.DataFrame: A DataFrame with trajectories, their community ids, and representative flags.
    """
    result_data = []
    for trajectory in trajectories:
        community_id = communities[trajectory]
        is_representative = community_representatives[community_id] == trajectory
        result_data.append({
            "trajectory": trajectory,
            "community_id": community_id,
            "is_representative": is_representative,
            "representative_trajectory": community_representatives[community_id]
        })
    
    return pd.DataFrame(result_data)


def process_airport_pair(
    df: pd.DataFrame, origin: str, destination: str, pool: Pool, max_trajectories_kept: int = 256, show_progress: bool = True
) -> pd.DataFrame | None:
    """
    Processes a single airport pair to find trajectory communities.

    Args:
        df (pd.DataFrame): The input dataframe with flight data.
        origin (str): The origin airport code.
        destination (str): The destination airport code.
        pool (mp.Pool): An existing multiprocessing pool.
        show_progress (bool): Whether to display a progress bar for distance calculation.

    Returns:
        pd.DataFrame | None: A DataFrame with community information, or None if processing fails.
    """
    import random
    random.seed(42)
    console.print(f"[cyan]Processing airport pair: {origin} → {destination}[/cyan]")
    
    console.print("[yellow]Extracting trajectories...[/yellow]")
    trajectories = get_trajectories_for_airport_pair(df, origin, destination)
    console.print(f"[green]✓ Found {len(trajectories)} trajectories[/green]")

    if len(trajectories) > max_trajectories_kept:
        console.print(f"[yellow]Keeping only {max_trajectories_kept} trajectories[/yellow]")
        import random
        trajectories = random.sample(trajectories, max_trajectories_kept)

    if len(trajectories) < MIN_FLIGHTS:
        console.print(f"[red]✗ Insufficient trajectories ({len(trajectories)} < {MIN_FLIGHTS})[/red]")
        return None

    console.print("[yellow]Computing distance matrix...[/yellow]")
    distance_matrix, trajectory_list = compute_trajectory_distances(trajectories, pool, show_progress=show_progress)
    console.print(f"[green]✓ Distance matrix computed ({len(trajectories)}x{len(trajectories)})[/green]")
    
    console.print("[yellow]Detecting communities...[/yellow]")
    num_communities, communities, community_representatives = group_trajectories_into_communities(distance_matrix, trajectory_list)
    console.print(f"[green]✓ Found {num_communities} communities[/green]")

    if num_communities < MIN_COMMUNITIES:
        console.print(f"[red]✗ Insufficient communities ({num_communities} < {MIN_COMMUNITIES})[/red]")
        return None

    # console.print("[blue]Community representatives:[/blue]")
    # for community_id, representative in community_representatives.items():
    #     community_size = sum(1 for comm_id in communities.values() if comm_id == community_id)
    #     console.print(f"  Community {community_id} ({community_size} trajectories): {representative}")

    console.print("[green]✓ Processing completed successfully[/green]")
    return format_community_output(communities, trajectories, community_representatives)


def find_unique_airport_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Finds all unique airport pairs from the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        List[Tuple[str, str]]: A list of unique (origin, destination) pairs.
    """
    df = _prepare_data(df)
    return (
        df[["departure_airport", "arrival_airport"]]
        .drop_duplicates()
        .to_records(index=False)
        .tolist()
    )


def process_all_pairs_and_get_representatives(df: pd.DataFrame, output_path: str, pool: Pool) -> None:
    """
    Processes all airport pairs from a DataFrame, extracts representative trajectories
    through community detection, and saves them to a CSV file incrementally.
    For pairs with too few flights, unique trajectories are kept instead.

    Args:
        df (pd.DataFrame): The input DataFrame with flight data.
        output_path (str): Path to save the output CSV file.
        pool (Pool): The multiprocessing pool to use for computation.
    """
    unique_pairs = find_unique_airport_pairs(df)
    console.print(f"[bold blue]Found {len(unique_pairs)} unique airport pairs to process.[/bold blue]")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write header to the CSV file
    pd.DataFrame(columns=['origin', 'destination', 'route']).to_csv(output_path, index=False)
    
    total_kept_count = 0
    all_kept_routes = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        pairs_task = progress.add_task("Processing all airport pairs...", total=len(unique_pairs))
        
        for origin, destination in unique_pairs:
            progress.update(pairs_task, advance=1, description=f"Processing {origin} -> {destination}")
            
            trajectories = get_trajectories_for_airport_pair(df, origin, destination)
            
            if not trajectories:
                continue

            kept_trajectories_data = []

            if len(trajectories) < MIN_FLIGHTS:
                # console.print(f"[{origin} -> {destination}] Too few flights ({len(trajectories)}). Keeping unique trajectories.")
                unique_trajectories = sorted(list(set(trajectories)))
                for traj in unique_trajectories:
                    if traj not in all_kept_routes:
                        kept_trajectories_data.append({
                            "origin": origin,
                            "destination": destination,
                            "route": traj,
                        })
                        all_kept_routes.add(traj)
            else:
                community_df = process_airport_pair(df, origin, destination, pool, show_progress=False)

                if community_df is not None and not community_df.empty:
                    representatives = community_df[community_df["is_representative"]]
                    for _, row in representatives.iterrows():
                        traj = row["trajectory"]
                        if traj not in all_kept_routes:
                            kept_trajectories_data.append({
                                "origin": origin,
                                "destination": destination,
                                "route": traj,
                            })
                            all_kept_routes.add(traj)
            
            if kept_trajectories_data:
                df_to_append = pd.DataFrame(kept_trajectories_data)
                df_to_append.to_csv(output_path, mode='a', header=False, index=False)
                total_kept_count += len(kept_trajectories_data)
    
    console.print()
    console.print(f"[bold green]✔ Processing complete.[/bold green]")
    console.print(f"Saved {total_kept_count} representative trajectories to [cyan]{output_path}[/cyan]")



def main():
    """
    Main function to run the trajectory grouping analysis.
    """
    # Define directory path for all CSV files
    input_dir = "output/city_pairs/grouped_flights_by_cpairs"
    output_dir = "output/city_pairs/representatives"
    
    # Ensure output directory exists    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    total_files = len(csv_files)
    
    console.print(f"[bold blue]🛫 Trajectory Grouping Analysis for All City Pair Files[/bold blue]")
    console.print(f"[blue]Input directory: {input_dir}[/blue]")
    console.print(f"[blue]Output directory: {output_dir}[/blue]")
    console.print(f"[blue]Found {total_files} CSV files to process[/blue]")
    console.print()

    with mp.Pool(processes=N_PROCESSES, initializer=WorkerInitializer.initialize) as pool:
        for file_idx, csv_file in enumerate(csv_files, 1):
            file_to_load = os.path.join(input_dir, csv_file)
            output_file = os.path.join(output_dir, csv_file.replace('.csv', '_representatives.csv'))
            
            if os.path.exists(output_file):
                console.print(f"[yellow]Skipping {csv_file}: output file already exists.[/yellow]")
                continue
            
            console.print(f"[bold cyan]Processing file {file_idx}/{total_files}: {csv_file}[/bold cyan]")
            
            try:
                # Load the dataset
                flights_df = pd.read_csv(file_to_load)
                console.print(f"[green]✓ Loaded {len(flights_df)} flight records[/green]")

                # Process all airport pairs to get representative trajectories and save to file
                process_all_pairs_and_get_representatives(flights_df, output_file, pool)
                
            except FileNotFoundError:
                console.print(f"[bold red]❌ File Not Found: {csv_file}[/bold red]")
            except Exception as e:
                console.print(f"[bold red]❌ Error processing {csv_file}: {str(e)}[/bold red]")
                continue
            
            console.print(f"[green]✓ Completed {csv_file}[/green]")
            console.print()
    
    console.print(f"[bold green]🎉 All {total_files} files processed successfully![/bold green]")


if __name__ == "__main__":
    main()
