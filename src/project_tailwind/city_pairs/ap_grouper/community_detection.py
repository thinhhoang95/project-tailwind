"""
This module provides functions for community detection on a set of trajectories.
"""

from typing import Any, Dict, List, Tuple

import igraph as ig
import leidenalg as la
import numpy as np


def build_graph_from_distance_matrix(
    distance_matrix: np.ndarray, threshold: float = 0.1
) -> ig.Graph:
    """
    Builds a graph from a distance matrix, connecting nodes with a distance
    below a given threshold.

    Args:
        distance_matrix (np.ndarray): The pairwise distance matrix.
        threshold (float, optional): The distance threshold for creating an edge.
            Defaults to 0.1.

    Returns:
        ig.Graph: The resulting graph.
    """
    n = distance_matrix.shape[0]
    graph = ig.Graph()
    graph.add_vertices(n)
    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] < threshold:
                edges.append((i, j))
                weights.append(1.0 / (distance_matrix[i, j] + 1e-6))
    graph.add_edges(edges)
    graph.es["weight"] = weights
    return graph


def detect_communities(
    graph: ig.Graph,
) -> Tuple[la.VertexPartition, Dict[int, List[int]]]:
    """
    Detects communities in a graph using the Leiden algorithm.

    Args:
        graph (ig.Graph): The input graph.

    Returns:
        Tuple[la.VertexPartition, Dict[int, List[int]]]: A tuple containing
            the partition and a dictionary of communities.
    """
    partition = la.find_partition(graph, la.ModularityVertexPartition, weights="weight")
    
    # Create a dictionary mapping community id to list of vertex indices
    communities: Dict[int, List[int]] = {}
    for i, community_id in enumerate(partition.membership):
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(i)
    return partition, communities


def find_representative_trajectory(
    community: List[int], distance_matrix: np.ndarray
) -> int:
    """
    Finds the most representative trajectory in a community.
    The representative is the one with the minimum total distance to all other
    trajectories in the community.

    Args:
        community (List[int]): A list of trajectory indices in the community.
        distance_matrix (np.ndarray): The pairwise distance matrix.

    Returns:
        int: The index of the representative trajectory.
    """
    if not community:
        raise ValueError("Community cannot be empty.")
    if len(community) == 1:
        return community[0]

    submatrix = distance_matrix[np.ix_(community, community)]
    total_distances = submatrix.sum(axis=1)
    representative_index_in_community = np.argmin(total_distances)
    return community[representative_index_in_community]


def get_representative_trajectories(
    communities: Dict[int, List[int]], distance_matrix: np.ndarray
) -> Dict[int, int]:
    """
    Gets the representative trajectory for each community.

    Args:
        communities (Dict[int, List[int]]): A dictionary of communities.
        distance_matrix (np.ndarray): The pairwise distance matrix.

    Returns:
        Dict[int, int]: A dictionary mapping community index to the
            representative trajectory index.
    """
    representatives = {}
    for community_id, community_members in communities.items():
        representatives[community_id] = find_representative_trajectory(
            community_members, distance_matrix
        )
    return representatives
