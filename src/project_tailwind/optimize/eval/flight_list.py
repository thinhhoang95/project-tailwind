"""
FlightList class for loading and managing occupancy matrix data from SO6 flight data.
"""

import json
from typing import Dict, List, Any, Optional
import numpy as np
from scipy import sparse
from datetime import datetime, timedelta
from copy import deepcopy


class FlightList:
    """
    A class to load and manage flight occupancy data from SO6 format.
    
    This class loads the occupancy matrix where each row represents a flight
    and columns represent Traffic Volume Time Windows (TVTWs). The matrix
    is sparse to handle large datasets efficiently.
    """
    
    def __init__(self, occupancy_file_path: str, tvtw_indexer_path: str):
        """
        Initialize FlightList by loading occupancy data and TVTW indexer.
        
        Args:
            occupancy_file_path: Path to so6_occupancy_matrix_with_times.json
            tvtw_indexer_path: Path to tvtw_indexer.json
        """
        self.occupancy_file_path = occupancy_file_path
        self.tvtw_indexer_path = tvtw_indexer_path
        
        # Load TVTW indexer for dimension info
        with open(tvtw_indexer_path, 'r') as f:
            self.tvtw_indexer = json.load(f)
        
        # Get time bin duration and total number of TVTWs
        self.time_bin_minutes = self.tvtw_indexer['time_bin_minutes']
        self.tv_id_to_idx = self.tvtw_indexer['tv_id_to_idx']
        self.num_traffic_volumes = len(self.tv_id_to_idx)
        
        # Load flight data
        self._load_flight_data()
        
    def _load_flight_data(self):
        """Load flight occupancy data and build sparse matrix."""
        with open(self.occupancy_file_path, 'r') as f:
            self.flight_data = json.load(f)
        
        self.flight_ids = list(self.flight_data.keys())
        self.num_flights = len(self.flight_ids)
        
        # Create mapping from flight_id to row index
        self.flight_id_to_row = {fid: i for i, fid in enumerate(self.flight_ids)}
        
        # Determine maximum TVTW index to size the matrix
        max_tvtw_index = 0
        for flight_data in self.flight_data.values():
            for interval in flight_data['occupancy_intervals']:
                max_tvtw_index = max(max_tvtw_index, interval['tvtw_index'])
        
        self.num_tvtws = max_tvtw_index + 1
        
        # Build sparse occupancy matrix
        self._build_occupancy_matrix()
        
        # Store additional flight metadata
        self._extract_flight_metadata()
    
    def _build_occupancy_matrix(self):
        """Build sparse occupancy matrix from flight data."""
        row_indices = []
        col_indices = []
        data = []
        
        for row_idx, flight_id in enumerate(self.flight_ids):
            flight_info = self.flight_data[flight_id]
            
            for interval in flight_info['occupancy_intervals']:
                tvtw_index = interval['tvtw_index']
                
                row_indices.append(row_idx)
                col_indices.append(tvtw_index)
                data.append(1.0)  # Binary occupancy
        
        # Create sparse matrix in LIL format for efficient updates, keep CSR for reads
        self._occupancy_matrix_lil = sparse.lil_matrix(
            (self.num_flights, self.num_tvtws),
            dtype=np.float32
        )
        
        # Populate the LIL matrix
        for row_idx, flight_id in enumerate(self.flight_ids):
            flight_info = self.flight_data[flight_id]
            
            for interval in flight_info['occupancy_intervals']:
                tvtw_index = interval['tvtw_index']
                self._occupancy_matrix_lil[row_idx, tvtw_index] = 1.0
        
        # Convert to CSR for efficient operations
        self.occupancy_matrix = self._occupancy_matrix_lil.tocsr()
        
        # Track if LIL matrix has been modified
        self._lil_matrix_dirty = False
        
        # Create temporary buffer for occupancy vector operations (reused to avoid allocations)
        self._temp_occupancy_buffer = np.zeros(self.num_tvtws, dtype=np.float32)
    
    def _extract_flight_metadata(self):
        """Extract and store flight metadata for quick access."""
        self.flight_metadata = {}
        
        for flight_id in self.flight_ids:
            flight_info = self.flight_data[flight_id]
            
            # Parse takeoff time
            takeoff_time = datetime.fromisoformat(flight_info['takeoff_time'].replace('T', ' '))
            
            # Extract entry and exit times for each TVTW
            occupancy_intervals = []
            for interval in flight_info['occupancy_intervals']:
                occupancy_intervals.append({
                    'tvtw_index': interval['tvtw_index'],
                    'entry_time_s': interval['entry_time_s'],
                    'exit_time_s': interval['exit_time_s']
                })
            
            self.flight_metadata[flight_id] = {
                'takeoff_time': takeoff_time,
                'origin': flight_info['origin'],
                'destination': flight_info['destination'],
                'distance': flight_info['distance'],
                'occupancy_intervals': occupancy_intervals
            }
    
    def get_occupancy_vector(self, flight_id: str) -> np.ndarray:
        """
        Get the occupancy vector for a specific flight.
        
        Args:
            flight_id: The flight identifier
            
        Returns:
            1D numpy array representing the flight's TVTW occupancy
        """
        if flight_id not in self.flight_id_to_row:
            raise ValueError(f"Flight ID {flight_id} not found")
        
        row_idx = self.flight_id_to_row[flight_id]
        return self.occupancy_matrix[row_idx].toarray().flatten()
    
    def get_total_occupancy_by_tvtw(self) -> np.ndarray:
        """
        Get total flight count for each TVTW across all flights.
        
        Returns:
            1D numpy array with flight counts per TVTW
        """
        return np.array(self.occupancy_matrix.sum(axis=0)).flatten()
    
    def get_flight_metadata(self, flight_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific flight.
        
        Args:
            flight_id: The flight identifier
            
        Returns:
            Dictionary with flight metadata
        """
        if flight_id not in self.flight_metadata:
            raise ValueError(f"Flight ID {flight_id} not found")
        
        return self.flight_metadata[flight_id].copy()
    
    def get_flights_in_tvtw(self, tvtw_index: int) -> List[str]:
        """
        Get all flight IDs that occupy a specific TVTW.
        
        Args:
            tvtw_index: The TVTW index
            
        Returns:
            List of flight IDs
        """
        if tvtw_index >= self.num_tvtws:
            raise ValueError(f"TVTW index {tvtw_index} out of range")
        
        # Get column from sparse matrix
        col_data = self.occupancy_matrix[:, tvtw_index]
        flight_indices = col_data.nonzero()[0]
        
        return [self.flight_ids[i] for i in flight_indices]
    
    def shift_flight_occupancy(self, flight_id: str, delay_minutes: int) -> np.ndarray:
        """
        Shift a flight's occupancy vector by a delay in minutes.
        
        Args:
            flight_id: The flight identifier
            delay_minutes: Delay in minutes (positive for delay)
            
        Returns:
            New occupancy vector with shifted times
        """
        if flight_id not in self.flight_id_to_row:
            raise ValueError(f"Flight ID {flight_id} not found")
        
        # Calculate shift in time bins
        shift_bins = delay_minutes // self.time_bin_minutes
        if delay_minutes % self.time_bin_minutes != 0:
            shift_bins += 1  # Round up to next bin
        
        # Get original occupancy vector
        original_vector = self.get_occupancy_vector(flight_id)
        
        # Create shifted vector
        shifted_vector = np.zeros_like(original_vector)
        
        if shift_bins > 0 and shift_bins < len(original_vector):
            # Shift forward (delay)
            shifted_vector[shift_bins:] = original_vector[:-shift_bins]
        elif shift_bins <= 0:
            # Shift backward (early) - clip to valid range
            shift_back = abs(shift_bins)
            if shift_back < len(original_vector):
                shifted_vector[:-shift_back] = original_vector[shift_back:]
        
        return shifted_vector
    
    def update_flight_occupancy(self, flight_id: str, new_occupancy_vector: np.ndarray):
        """
        Update a flight's occupancy vector in the matrix.
        
        Args:
            flight_id: The flight identifier
            new_occupancy_vector: New occupancy vector
        """
        if flight_id not in self.flight_id_to_row:
            raise ValueError(f"Flight ID {flight_id} not found")
        
        row_idx = self.flight_id_to_row[flight_id]
        
        # Update LIL matrix (efficient for modifications)
        self._occupancy_matrix_lil[row_idx, :] = new_occupancy_vector
        self._lil_matrix_dirty = True
    
    def clear_flight_occupancy(self, flight_id: str):
        """
        Clear all occupancy data for a specific flight by setting its row to zero.
        
        Args:
            flight_id: The flight identifier
        """
        if flight_id not in self.flight_id_to_row:
            raise ValueError(f"Flight ID {flight_id} not found")
        
        row_idx = self.flight_id_to_row[flight_id]
        
        # Clear the row in LIL matrix (efficient for modifications)
        self._occupancy_matrix_lil[row_idx, :] = 0
        self._lil_matrix_dirty = True
    
    def add_flight_occupancy(self, flight_id: str, tvtw_indices: np.ndarray):
        """
        Add occupancy data for a flight at specific TVTW indices.
        This method assumes the flight's occupancy has been cleared first.
        
        Args:
            flight_id: The flight identifier
            tvtw_indices: Array of TVTW indices where the flight should be marked as occupying
        """
        if flight_id not in self.flight_id_to_row:
            raise ValueError(f"Flight ID {flight_id} not found")
        
        row_idx = self.flight_id_to_row[flight_id]
        
        if len(tvtw_indices) == 0:
            # No occupancy to add, keep the row empty
            return
        
        # Validate indices
        if np.any(tvtw_indices >= self.num_tvtws) or np.any(tvtw_indices < 0):
            raise ValueError(f"TVTW indices out of range: {tvtw_indices}")
        
        # Update LIL matrix (efficient for modifications)
        self._occupancy_matrix_lil[row_idx, tvtw_indices] = 1.0
        self._lil_matrix_dirty = True
    
    def _sync_occupancy_matrix(self):
        """Sync the CSR matrix with the LIL matrix if it's been modified."""
        if self._lil_matrix_dirty:
            self.occupancy_matrix = self._occupancy_matrix_lil.tocsr()
            self._lil_matrix_dirty = False
    
    def get_matrix_shape(self) -> tuple:
        """Get the shape of the occupancy matrix."""
        self._sync_occupancy_matrix()
        return self.occupancy_matrix.shape
    
    def finalize_occupancy_updates(self):
        """
        Finalize all pending occupancy updates by converting LIL to CSR.
        Call this method after batch updates for optimal performance.
        """
        self._sync_occupancy_matrix()
    
    def copy(self):
        """Create a deep copy of the FlightList."""
        # Create new instance with same file paths
        new_flight_list = FlightList(self.occupancy_file_path, self.tvtw_indexer_path)
        
        # Copy the occupancy matrix
        new_flight_list.occupancy_matrix = self.occupancy_matrix.copy()
        
        # Copy flight metadata
        new_flight_list.flight_metadata = deepcopy(self.flight_metadata)
        new_flight_list.flight_data = deepcopy(self.flight_data)
        
        return new_flight_list
    
    def update_flight(self, flight_id: str, updates: Dict[str, Any]):
        """Update flight data with new values."""
        if flight_id not in self.flight_id_to_row:
            raise ValueError(f"Flight ID {flight_id} not found")
        
        # Update occupancy vector if provided
        if "occupancy_vector" in updates:
            new_occupancy_vector = updates["occupancy_vector"]
            if hasattr(new_occupancy_vector, '__len__'):
                # Convert list to numpy array if needed
                if not isinstance(new_occupancy_vector, np.ndarray):
                    new_occupancy_vector = np.array(new_occupancy_vector)
                self.update_flight_occupancy(flight_id, new_occupancy_vector)
            
        # Update metadata if provided
        if flight_id in self.flight_metadata:
            for key, value in updates.items():
                if key != "occupancy_vector":
                    self.flight_metadata[flight_id][key] = value
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the flight data."""
        total_occupancy = self.get_total_occupancy_by_tvtw()
        
        return {
            'num_flights': self.num_flights,
            'num_tvtws': self.num_tvtws,
            'num_traffic_volumes': self.num_traffic_volumes,
            'time_bin_minutes': self.time_bin_minutes,
            'matrix_sparsity': 1 - (self.occupancy_matrix.nnz / (self.num_flights * self.num_tvtws)),
            'total_tvtw_occupancies': self.occupancy_matrix.nnz,
            'max_occupancy_per_tvtw': float(np.max(total_occupancy)) if len(total_occupancy) > 0 else 0,
            'avg_occupancy_per_tvtw': float(np.mean(total_occupancy)) if len(total_occupancy) > 0 else 0
        }