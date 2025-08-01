"""
This module provides functions for computing the distance between trajectories.
"""

from typing import List, Tuple

import numpy as np
from frechetdist import frdist
from shapely.geometry import LineString
from scipy.interpolate import interp1d


def _resample_trajectory(
    path: List[Tuple[float, float]], n_samples: int
) -> List[Tuple[float, float]]:
    """
    Resample a trajectory to have exactly n_samples points using linear interpolation.
    
    Args:
        path (List[Tuple[float, float]]): Original trajectory points
        n_samples (int): Number of samples in the resampled trajectory
        
    Returns:
        List[Tuple[float, float]]: Resampled trajectory with n_samples points
    """
    if len(path) < 2:
        # If path has fewer than 2 points, duplicate the point(s) to reach n_samples
        if len(path) == 1:
            return [path[0]] * n_samples
        else:
            return [(0.0, 0.0)] * n_samples
    
    if len(path) == n_samples:
        return path
    
    # Convert to numpy arrays for easier manipulation
    path_array = np.array(path)
    
    # Create parameter t along the path (0 to 1)
    t_original = np.linspace(0, 1, len(path))
    t_resampled = np.linspace(0, 1, n_samples)
    
    # Interpolate latitude and longitude separately
    interp_lat = interp1d(t_original, path_array[:, 0], kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
    interp_lon = interp1d(t_original, path_array[:, 1], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
    
    # Generate resampled points
    resampled_lats = interp_lat(t_resampled)
    resampled_lons = interp_lon(t_resampled)
    
    return list(zip(resampled_lats, resampled_lons))


def great_circle_distance(
    path1: List[Tuple[float, float]], path2: List[Tuple[float, float]]
) -> float:
    """
    Computes the great-circle distance between two paths.
    This is not a true trajectory distance metric, but a simple way to
    compare two paths.

    Args:
        path1 (List[Tuple[float, float]]): A list of (latitude, longitude) tuples.
        path2 (List[Tuple[float, float]]): A list of (latitude, longitude) tuples.

    Returns:
        float: The great-circle distance between the two paths.
    """
    line1 = LineString(path1)
    line2 = LineString(path2)
    return line1.distance(line2)


def frechet_distance(
    path1: List[Tuple[float, float]], path2: List[Tuple[float, float]], resampling_n: int = 8
) -> float:
    """
    Computes the Frechet distance between two paths.

    Args:
        path1 (List[Tuple[float, float]]): A list of (latitude, longitude) tuples.
        path2 (List[Tuple[float, float]]): A list of (latitude, longitude) tuples.
        resampling_n (int): Number of points to resample each trajectory to. Defaults to 50.

    Returns:
        float: The Frechet distance between the two paths.
    """
    # Resample both paths to have the same number of points
    resampled_path1 = _resample_trajectory(path1, resampling_n)
    resampled_path2 = _resample_trajectory(path2, resampling_n)
    
    return frdist(resampled_path1, resampled_path2)


def compute_distance_matrix(
    trajectories: List[List[Tuple[float, float]]], distance_metric: str = "frechet", resampling_n: int = 50
) -> np.ndarray:
    """
    Computes the pairwise distance matrix for a list of trajectories.

    Args:
        trajectories (List[List[Tuple[float, float]]]): A list of trajectories.
        distance_metric (str, optional): The distance metric to use.
            Defaults to "frechet".
        resampling_n (int): Number of points to resample each trajectory to. Defaults to 50.

    Returns:
        np.ndarray: The pairwise distance matrix.
    """
    n = len(trajectories)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if distance_metric == "frechet":
                distance = frechet_distance(trajectories[i], trajectories[j], resampling_n)
            else:
                distance = great_circle_distance(trajectories[i], trajectories[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix
