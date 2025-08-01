import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from shapely.geometry import Point, LineString
import numpy as np
from scipy.sparse import lil_matrix, save_npz, csr_matrix
import datetime
import os
import yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
import json
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich import box
from rich.text import Text
from geopy.distance import geodesic
from geopy.point import Point as GeoPoint
from shapely.geometry import box as shapely_box

warnings.filterwarnings('ignore')

console = Console()

ECAC_BBOX = {
    "lat_min": 34,
    "lat_max": 72,
    "lon_min": -25,
    "lon_max": 50
}
ecac_polygon = shapely_box(ECAC_BBOX["lon_min"], ECAC_BBOX["lat_min"], ECAC_BBOX["lon_max"], ECAC_BBOX["lat_max"])

def is_in_ecac_bbox(lat, lon):
    """Check if a coordinate is within the approximate ECAC bounding box."""
    return ECAC_BBOX["lat_min"] <= lat <= ECAC_BBOX["lat_max"] and \
           ECAC_BBOX["lon_min"] <= lon <= ECAC_BBOX["lon_max"]

def create_time_window_mapping(time_bin_minutes: int = 30):
    """
    Creates a mapping from time window indices to human-readable time ranges.
    
    Returns:
        dict: A dictionary mapping time bin indices to time ranges.
    """
    if 1440 % time_bin_minutes != 0:
        raise ValueError("time_bin_minutes must be a divisor of 1440.")
    
    num_bins = 1440 // time_bin_minutes
    time_windows = {}
    for i in range(num_bins):
        start_minute_of_day = i * time_bin_minutes
        end_minute_of_day = start_minute_of_day + time_bin_minutes
        
        start_hour, start_minute = divmod(start_minute_of_day, 60)
        end_hour, end_minute = divmod(end_minute_of_day -1, 60)
        
        start_time = f"{start_hour:02d}:{start_minute:02d}"
        end_time = f"{end_hour:02d}:{end_minute + 1:02d}"

        time_windows[i] = f"{start_time}-{end_time}"
    return time_windows

def process_flight_batch(flight_ids_batch, flights_df, airblocks_gdf_dict,
                        num_airblocks, num_time_bins, time_begin, time_end,
                        debug=False, flight_log_path=None, sampling_dist_nm: float = 15.0,
                        time_bin_minutes: int = 30):
    """
    Process a batch of flights and return their counts.
    This function is designed to be run in parallel.
    """
    # Reconstruct GeoDataFrame from dictionary.
    # The spatial index will be rebuilt on first access of .sindex
    tv_gdf = gpd.GeoDataFrame(airblocks_gdf_dict)
    
    # Initialize local data structures
    local_flight_counts = lil_matrix((num_airblocks, num_time_bins), dtype=int)
    local_log_entries = []
    
    # Process each flight in the batch
    for flight_id in flight_ids_batch:
        flight_segments = flights_df[flights_df['flight_identifier'] == flight_id]
        log_entries = process_flight(flight_segments, tv_gdf,
                      local_flight_counts, time_begin, time_end, 
                      debug, flight_log_path=flight_log_path, sampling_dist_nm=sampling_dist_nm,
                      time_bin_minutes=time_bin_minutes)
        if log_entries:
            local_log_entries.extend(log_entries)
    
    # Convert to CSR format for efficient storage and return
    return local_flight_counts.tocsr(), local_log_entries

def flight_counter_tv(flights_path: str, tv_path: str, output_path: str,
                  time_begin: str, time_end: str, flight_log_path: str = None,
                  debug: bool = False, n_jobs: int = None, sampling_dist_nm: float = 15.0,
                  time_bin_minutes: int = 30):
    """
    Counts flights in traffic volumes within a given time window using parallel processing.
    Optionally generates a flight log.

    Args:
        flights_path (str): Path to the flights CSV file.
        tv_path (str): Path to the traffic volumes GeoJSON file.
        output_path (str): Path to save the output sparse matrix.
        time_begin (str): Start of the time window (YYYY-MM-DD HH:MM:SS).
        time_end (str): End of the time window (YYYY-MM-DD HH:MM:SS).
        flight_log_path (str, optional): Path to save the flight log CSV. Defaults to None.
        n_jobs (int): Number of parallel jobs. If None, uses CPU count - 1.
        sampling_dist_nm (float): Sampling distance in nautical miles for long segments.
    """
    # Display startup information with rich formatting
    startup_panel = Panel(
        f"[bold cyan]Time Window:[/bold cyan] [yellow]{time_begin}[/yellow] to [yellow]{time_end}[/yellow]",
        title="[bold]✈️  Flight Counting Process Started[/bold]",
        border_style="blue"
    )
    console.print(startup_panel)
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    workers_panel = Panel(
        f"[bold green]Workers:[/bold green] [bold magenta]{n_jobs}[/bold magenta] parallel processes",
        title="[bold]🛠️  Processing Configuration[/bold]",
        border_style="green"
    )
    console.print(workers_panel)

    # 1. Load and preprocess traffic volumes
    console.print("[bold blue]🗺️  Loading and preprocessing traffic volumes...[/bold blue]")
    tv_gdf, tv_map = load_traffic_volumes(tv_path)
    
    # Prepare data for pickling. The GeoDataFrame is converted to a dict.
    # The spatial index is not pickled but will be regenerated in each worker process.
    tv_gdf_dict = tv_gdf.to_dict('list')

    # 2. Initialize data structures
    num_tvs = len(tv_gdf)
    num_time_bins = 1440 // time_bin_minutes
    
    # 3. Load flight data
    console.print("[bold blue]✈️  Loading flight data...[/bold blue]")
    flights_df = pd.read_csv(flights_path)
    
    # Get unique flight IDs
    unique_flight_ids = flights_df['flight_identifier'].unique()
    n_flights = len(unique_flight_ids)
    flights_info_table = Table(show_header=False, box=box.ROUNDED)
    flights_info_table.add_column("Metric", style="cyan")
    flights_info_table.add_column("Value", style="bold white")
    flights_info_table.add_row("Unique Flights", f"[bold green]{n_flights:,}[/bold green]")
    flights_info_table.add_row("Batch Size", f"[bold yellow]{max(1, n_flights // (n_jobs * 4)):,}[/bold yellow]")
    
    console.print(Panel(flights_info_table, title="[bold]📊 Flight Processing Info[/bold]", border_style="cyan"))
    
    # 4. Split flights into batches for parallel processing
    batch_size = max(1, n_flights // (n_jobs * 4))  # Create more batches than workers for better load balancing
    flight_batches = [unique_flight_ids[i:i + batch_size] for i in range(0, n_flights, batch_size)]
    
    # 5. Process flights in parallel
    console.print("[bold magenta]🚀 Processing flights in parallel...[/bold magenta]")
    process_func = partial(process_flight_batch, 
                           flights_df=flights_df,
                           airblocks_gdf_dict=tv_gdf_dict,
                           num_airblocks=num_tvs,
                           num_time_bins=num_time_bins,
                           time_begin=time_begin,
                           time_end=time_end,
                           debug=debug,
                           flight_log_path=flight_log_path,
                           sampling_dist_nm=sampling_dist_nm,
                           time_bin_minutes=time_bin_minutes)
    
    with Pool(processes=n_jobs) as pool:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[green]Processing flight batches", total=len(flight_batches))
            results = []
            for result in pool.imap(process_func, flight_batches):
                results.append(result)
                progress.update(task, advance=1)
    
    # 6. Combine results from all workers
    console.print("[bold blue]🔄 Combining results from all workers...[/bold blue]")
    combined_flight_counts = csr_matrix((num_tvs, num_time_bins), dtype=int)
    all_log_entries = []
    
    for local_counts, local_log_entries in results:
        combined_flight_counts += local_counts
        if flight_log_path:
            all_log_entries.extend(local_log_entries)

    # Save flight log if path is provided
    if flight_log_path:
        if all_log_entries:
            log_df = pd.DataFrame(all_log_entries)
            log_df.to_csv(flight_log_path, index=False)
            console.print(f"[green]📊 Flight log with [bold]{len(log_df):,}[/bold] entries saved to [dim]{flight_log_path}[/dim][/green]")
        else:
            console.print("[yellow]⚠️  No flight log entries were generated.[/yellow]")
    
    # 7. Convert to DataFrame and calculate overload
    console.print("[bold blue]📈 Calculating overload and generating CSV...[/bold blue]")
    counts_df = pd.DataFrame(combined_flight_counts.toarray())
    
    # Create a map from index to traffic_volume_id
    idx_to_tv = {i: name for name, i in tv_map.items()}
    counts_df['traffic_volume_id'] = counts_df.index.map(idx_to_tv)
    
    # Set traffic_volume_id as the first column
    counts_df = counts_df[['traffic_volume_id'] + [c for c in counts_df.columns if c != 'traffic_volume_id']]
    
    # Name time window columns
    time_windows_map = create_time_window_mapping(time_bin_minutes)
    counts_df.columns = ['traffic_volume_id'] + list(time_windows_map.values())
    
    # Add capacity and calculate overload
    tv_gdf.set_index('id', inplace=True)
    
    # The capacity column from GeoJSON might be a string, so we need to parse it.
    # Let's handle both dict and string cases
    def parse_capacity(capacity):
        if isinstance(capacity, str):
            try:
                return json.loads(capacity)
            except json.JSONDecodeError:
                return {}
        elif isinstance(capacity, dict):
            return capacity
        return {}

    tv_gdf['capacity_dict'] = tv_gdf['capacity'].apply(parse_capacity)
    
    # For each hourly window, calculate if there was an overload.
    overload_cols = [f'overload_{h:02d}' for h in range(24)]
    for col in overload_cols:
        counts_df[col] = 0

    for tv_id_int, tv_row in counts_df.iterrows():
        # Get capacity for this TV from the GeoDataFrame
        capacity_info = tv_gdf.loc[tv_id_int]['capacity_dict']
        if not capacity_info:
            continue
            
        bins_per_hour = 60 // time_bin_minutes
        for hour in range(24):
            # Find the corresponding capacity value for the current hour
            capacity_key_found = None
            for key in capacity_info.keys():
                # Capacity keys are strings like '8:00-9:00'
                if key.startswith(f"{hour}:"):
                    capacity_key_found = key
                    break
            
            if capacity_key_found and capacity_info[capacity_key_found] is not None:
                capacity_val = capacity_info[capacity_key_found]
                
                # Sum counts from the bins that make up the hour
                start_bin = hour * bins_per_hour
                end_bin = start_bin + bins_per_hour
                
                # +1 to account for the 'traffic_volume_id' column at the beginning
                total_hourly_count = sum(tv_row.iloc[b + 1] for b in range(start_bin, end_bin))

                # If the total count exceeds capacity, set the overload flag for that hour
                if total_hourly_count > capacity_val:
                    counts_df.loc[tv_id_int, f'overload_{hour:02d}'] = 1

    # 8. Save results to CSV
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    console.print(f"[bold blue]💾 Saving results to [dim]{output_path}[/dim]...[/bold blue]")
    counts_df.to_csv(output_path, index=False)
    
    completion_panel = Panel(
        "[bold green]✅ Flight counting process completed successfully![/bold green]",
        title="[bold green]Success[/bold green]",
        border_style="green"
    )
    console.print(completion_panel)

def load_traffic_volumes(tv_path: str):
    """
    Loads traffic volumes from a GeoJSON file, creates a spatial index, and a name-to-index map.
    
    Returns:
        tuple: A tuple containing:
            - gpd.GeoDataFrame: GeoDataFrame with traffic volume data and a spatial index.
            - dict: A dictionary mapping unique traffic volume names to indices.
    """
    tv_gdf = gpd.read_file(tv_path)
    
    # The unique ID is traffic_volume_id
    tv_gdf['unique_id'] = tv_gdf['traffic_volume_id']
    
    # Create a mapping from unique_id to a zero-based index
    tv_map = {name: i for i, name in enumerate(tv_gdf['unique_id'])}
    tv_gdf['id'] = tv_gdf['unique_id'].map(tv_map)

    if 'capacity' not in tv_gdf.columns:
        raise ValueError("GeoJSON file must have a 'capacity' property in each feature.")
        
    if 'min_fl' not in tv_gdf.columns or 'max_fl' not in tv_gdf.columns:
        raise ValueError("GeoJSON file must have 'min_fl' and 'max_fl' properties in each feature.")

    # A spatial index will be created automatically by GeoPandas when needed,
    # e.g., when accessing tv_gdf.sindex.
    return tv_gdf, tv_map

def find_tvs_for_point(point: Point, fl: float, tv_gdf: gpd.GeoDataFrame):
    """
    Find all traffic volumes that contain a given point at a specific flight level.
    """
    if point.is_empty:
        return []
    
    # Use spatial index to find candidate polygons
    possible_matches_idx = list(tv_gdf.sindex.query(point, predicate='intersects'))
    
    if not possible_matches_idx:
        return []
    
    possible_matches = tv_gdf.iloc[possible_matches_idx]
    
    # Filter by actual geometry contains and flight level
    precise_matches = possible_matches[possible_matches.geometry.covers(point)]
    vertically_relevant_tvs = precise_matches[
        (precise_matches['min_fl'] <= fl) & (precise_matches['max_fl'] >= fl)
    ]
    
    return vertically_relevant_tvs.index.tolist()


def get_closest_point(geom, ref_point):
    """
    From a geometry, find the single Point on it that is closest to the reference point.
    """
    if geom.is_empty:
        return None
    
    if geom.geom_type == 'Point':
        return geom
    
    if geom.geom_type == 'LineString':
        p_start = Point(geom.coords[0])
        p_end = Point(geom.coords[-1])
        return p_start if p_start.distance(ref_point) < p_end.distance(ref_point) else p_end

    if hasattr(geom, 'geoms'): # Multi-geometries or GeometryCollection
        # Find the closest point from all sub-geometries.
        all_points = []
        for sub_geom in geom.geoms:
            p = get_closest_point(sub_geom, ref_point)
            if p:
                all_points.append(p)
        
        if not all_points:
            return None
        
        return min(all_points, key=lambda p: p.distance(ref_point))
    
    # Fallback for other geometry types (e.g. Polygon). Should not happen here.
    return geom.centroid


def update_counts(flight_counts: lil_matrix, tv_id: int, t_start: datetime.datetime, 
                  t_end: datetime.datetime, visited_tv_bins: set, 
                  entry_point: Point = None, exit_point: Point = None,
                  flight_id: str = None, origin: str = None, destination: str = None,
                  fl: float = None, tv_gdf: gpd.GeoDataFrame = None, debug: bool = False,
                  flight_log_path: str = None, time_bin_minutes: int = 30):
    """
    Increments counts and generates detailed log entries for a flight's passage
    through a traffic volume, clamping times and coordinates to each time bin.
    """
    if t_start >= t_end:
        return []

    all_log_entries = []
    
    current_time = t_start
    while current_time < t_end:
        # Determine the bin for the current time
        day_of_year = current_time.timetuple().tm_yday
        minute_of_day = current_time.hour * 60 + current_time.minute
        bin_of_day = minute_of_day // time_bin_minutes
        
        # We need a unique identifier for a bin across days if the flight spans more than one day
        # For this implementation, we assume flights are within a reasonable timeframe and use day_of_year
        # A more robust solution for very long flights might need a different approach.
        unique_bin_id = (tv_id, day_of_year, bin_of_day)

        if unique_bin_id not in visited_tv_bins:
            matrix_bin_idx = bin_of_day
            flight_counts[tv_id, matrix_bin_idx] += 1
            visited_tv_bins.add(unique_bin_id)

            if flight_log_path:
                time_windows = create_time_window_mapping(time_bin_minutes)
                time_window_str = time_windows.get(matrix_bin_idx, "Unknown")

                start_minute_of_day = bin_of_day * time_bin_minutes
                start_h, start_m = divmod(start_minute_of_day, 60)

                bin_start_time = current_time.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
                bin_end_time = bin_start_time + datetime.timedelta(minutes=time_bin_minutes)
                
                log_entry_time = max(t_start, bin_start_time)
                log_exit_time = min(t_end, bin_end_time)

                # Interpolate entry/exit points for this specific bin
                duration_in_tv = (t_end - t_start).total_seconds()
                
                entry_lon, entry_lat, exit_lon, exit_lat = '', '', '', ''

                if duration_in_tv > 0:
                    # Entry point for the bin
                    if log_entry_time > t_start:
                        ratio = (log_entry_time - t_start).total_seconds() / duration_in_tv
                        if entry_point and exit_point:
                           entry_lon = round(entry_point.x + (exit_point.x - entry_point.x) * ratio, 6)
                           entry_lat = round(entry_point.y + (exit_point.y - entry_point.y) * ratio, 6)
                    elif entry_point:
                        entry_lon = round(entry_point.x, 6)
                        entry_lat = round(entry_point.y, 6)

                    # Exit point for the bin
                    if log_exit_time < t_end:
                        ratio = (log_exit_time - t_start).total_seconds() / duration_in_tv
                        if entry_point and exit_point:
                            exit_lon = round(entry_point.x + (exit_point.x - entry_point.x) * ratio, 6)
                            exit_lat = round(entry_point.y + (exit_point.y - entry_point.y) * ratio, 6)
                    elif exit_point:
                        exit_lon = round(exit_point.x, 6)
                        exit_lat = round(exit_point.y, 6)

                log_entry = {
                    'flight_identifier': flight_id,
                    'origin_aerodrome': origin,
                    'destination_aerodrome': destination,
                    'fl': fl,
                    'traffic_volume_name': tv_gdf.loc[tv_id]['traffic_volume_id'],
                    'entry_lon': entry_lon,
                    'entry_lat': entry_lat,
                    'entry_time': log_entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_lon': exit_lon,
                    'exit_lat': exit_lat,
                    'exit_time': log_exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'time_window': time_window_str
                }
                all_log_entries.append(log_entry)
        
        # Advance to the start of the next bin
        start_minute_of_day = bin_of_day * time_bin_minutes
        start_h, start_m = divmod(start_minute_of_day, 60)
        bin_start_time = current_time.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        next_bin_start_time = bin_start_time + datetime.timedelta(minutes=time_bin_minutes)
        current_time = next_bin_start_time

    return all_log_entries


def process_flight(flight_segments: pd.DataFrame, tv_gdf: gpd.GeoDataFrame,
                   flight_counts: lil_matrix,
                   time_begin: str, time_end: str, debug: bool = False,
                   flight_log_path: str = None, sampling_dist_nm: float = 15.0,
                   time_bin_minutes: int = 30):
    """
    Processes a single flight's segments to determine which traffic volumes it crosses.
    Returns a list of log entries for this flight.
    """
    if flight_segments.empty:
        return []

    flight_id = flight_segments['flight_identifier'].iloc[0]
    origin = flight_segments['origin_aerodrome'].iloc[0]
    destination = flight_segments['destination_aerodrome'].iloc[0]

    log_entries = []
    visited_tv_bins = set()

    def parse_datetime(row, date_col, time_col):
        time_str = str(row[time_col]).zfill(6)
        return datetime.datetime.strptime(f"{row[date_col]} {time_str}", '%y%m%d %H%M%S')

    flight_segments['segment_start_time'] = flight_segments.apply(
        lambda row: parse_datetime(row, 'date_begin_segment', 'time_begin_segment'), axis=1
    )
    flight_segments = flight_segments.sort_values(by='segment_start_time').reset_index(drop=True)
    
    tv_entry_info = {}

    # Handle the first point of the flight
    if not flight_segments.empty:
        p1_data = flight_segments.iloc[0]
        t1 = p1_data['segment_start_time']
        point1_geom = Point(p1_data['longitude_begin'], p1_data['latitude_begin'])
        if is_in_ecac_bbox(point1_geom.y, point1_geom.x):
            fl = p1_data.get('fl', p1_data.get('flight_level_begin', 100))
            tvs_at_p1 = find_tvs_for_point(point1_geom, fl, tv_gdf)
            for tv_id in tvs_at_p1:
                if tv_id not in tv_entry_info:
                    tv_entry_info[tv_id] = {'time': t1, 'point': point1_geom}

    for i in range(len(flight_segments)):
        p1_data = flight_segments.iloc[i]
        t1 = p1_data['segment_start_time']
        point1_geom = Point(p1_data['longitude_begin'], p1_data['latitude_begin'])
        fl = p1_data.get('fl', p1_data.get('flight_level_begin', 100))
        
        t2 = parse_datetime(p1_data, 'date_end_segment', 'time_end_segment')
        point2_geom = Point(p1_data['longitude_end'], p1_data['latitude_end'])

        if t1 >= t2:
            continue

        p1_in_ecac = is_in_ecac_bbox(point1_geom.y, point1_geom.x)
        p2_in_ecac = is_in_ecac_bbox(point2_geom.y, point2_geom.x)

        segment_line = LineString([point1_geom, point2_geom]) if point1_geom.distance(point2_geom) > 1e-6 else None
        
        # Only process segments that are at least partially in the ECAC box
        if not p1_in_ecac and not p2_in_ecac:
            if not segment_line or not segment_line.intersects(ecac_polygon):
                continue
        
        if segment_line:
            distance_nm = geodesic((point1_geom.y, point1_geom.x), (point2_geom.y, point2_geom.x)).nm
            if distance_nm > sampling_dist_nm:
                num_samples = int(distance_nm / sampling_dist_nm)
                
                for j in range(1, num_samples + 1):
                    dist_along_segment = j * sampling_dist_nm
                    
                    # Interpolate along the segment line to find the sample point.
                    # This is more accurate than using a fixed bearing on a geodesic path.
                    fraction = dist_along_segment / distance_nm
                    sample_point = segment_line.interpolate(fraction, normalized=True)

                    if segment_line and not segment_line.buffer(1e-6).contains(sample_point):
                        warnings.warn(f"Sample outside segment (flight {flight_id}, idx {i})")
                    
                    # Only process samples inside the ECAC box
                    if not is_in_ecac_bbox(sample_point.y, sample_point.x):
                        continue
                    
                    time_ratio = dist_along_segment / distance_nm
                    sample_time = t1 + (t2 - t1) * time_ratio
                    
                    tvs_at_sample = set(find_tvs_for_point(sample_point, fl, tv_gdf))
                    
                    current_tvs = set(tv_entry_info.keys())
                    

                    # Debug 2:
                    
                    # tvs_at_sample_names = [tv_gdf.iloc[tv_id]['traffic_volume_id'] for tv_id in tvs_at_sample]
                    # tv_entry_info_names = [tv_gdf.iloc[tv_id]['traffic_volume_id'] for tv_id in tv_entry_info.keys()]
                    # print(f"{flight_id} sample @ {sample_time}: tvs_at_sample={tvs_at_sample_names}, "
                    #     f"tv_entry_info={tv_entry_info_names}, sample_point = {sample_point}")

                    # New entries
                    for tv_id in tvs_at_sample - current_tvs:
                        if tv_id not in tv_entry_info:
                            tv_entry_info[tv_id] = {'time': sample_time, 'point': sample_point}
                    
                    # Exits
                    for tv_id in current_tvs - tvs_at_sample:
                        if tv_id in tv_entry_info:
                            entry_info = tv_entry_info.pop(tv_id)
                            log_entries.extend(update_counts(
                                flight_counts, tv_id, entry_info['time'], sample_time, visited_tv_bins,
                                entry_point=entry_info['point'], exit_point=sample_point,
                                flight_id=flight_id, origin=origin, destination=destination, fl=fl,
                                tv_gdf=tv_gdf, debug=debug, flight_log_path=flight_log_path,
                                time_bin_minutes=time_bin_minutes
                            ))

        # We only care about TV transitions if the point is inside ECAC
        tvs_at_p1 = set(find_tvs_for_point(point1_geom, fl, tv_gdf)) if p1_in_ecac else set()
        tvs_at_p2 = set(find_tvs_for_point(point2_geom, fl, tv_gdf)) if p2_in_ecac else set()

        # Debug: Print traffic volume IDs at p1 and p2
        # if (tvs_at_p1 or tvs_at_p2):
        #     tv_names_at_p1 = [tv_gdf.iloc[tv_id]['traffic_volume_id'] for tv_id in tvs_at_p1]
        #     tv_names_at_p2 = [tv_gdf.iloc[tv_id]['traffic_volume_id'] for tv_id in tvs_at_p2]
        #     print(f"Flight {flight_id}: TVs at p1: {tv_names_at_p1}, TVs at p2: {tv_names_at_p2}")

        # New entries detected at the end of the segment
        for tv_id in tvs_at_p2 - tvs_at_p1:
            if tv_id not in tv_entry_info:
                entry_time, entry_point = t2, point2_geom
                tv_entry_info[tv_id] = {'time': entry_time, 'point': entry_point}

        # Exits detected at the end of the segment
        for tv_id in tvs_at_p1 - tvs_at_p2:
            if tv_id in tv_entry_info:
                entry_info = tv_entry_info.pop(tv_id)
                exit_time, exit_point = t2, point2_geom
                log_entries.extend(update_counts(
                    flight_counts, tv_id, entry_info['time'], exit_time, visited_tv_bins,
                    entry_point=entry_info['point'], exit_point=exit_point,
                    flight_id=flight_id, origin=origin, destination=destination, fl=fl,
                    tv_gdf=tv_gdf, debug=debug, flight_log_path=flight_log_path,
                    time_bin_minutes=time_bin_minutes
                ))

    # Handle flights that are still inside a TV at the end of their trajectory
    for tv_id, entry_info in tv_entry_info.items():
        last_segment = flight_segments.iloc[-1]
        exit_time = parse_datetime(last_segment, 'date_end_segment', 'time_end_segment')
        exit_point = Point(last_segment['longitude_end'], last_segment['latitude_end'])
        
        log_entries.extend(update_counts(
            flight_counts, tv_id, entry_info['time'], exit_time, visited_tv_bins,
            entry_point=entry_info['point'], exit_point=exit_point,
            flight_id=flight_id, origin=origin, destination=destination,
            fl=last_segment.get('fl', last_segment.get('flight_level_end', 100)),
            tv_gdf=tv_gdf, debug=debug, flight_log_path=flight_log_path,
            time_bin_minutes=time_bin_minutes
        ))

    return log_entries

if __name__ == '__main__':
    # Example usage:
    flight_counter_tv(
        flights_path='cases/flights_20230801d.csv',
        tv_path='cases/traffic_volumes_simplified.geojson',
        output_path='cases/flight_counts_tv.csv',
        time_begin='2023-08-01 00:00:00',
        time_end='2023-08-01 23:59:59',
        flight_log_path='cases/flight_log_tv.csv',
        debug=False,
        n_jobs=None,  # Will use CPU count - 1
        sampling_dist_nm=15.0,
        time_bin_minutes=30
    )