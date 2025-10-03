"""
FlightList class for loading and managing occupancy matrix data from SO6 flight data.
"""

import json
from typing import Dict, List, Any, Optional, Sequence, Iterable, Tuple, Union
import numpy as np
from scipy import sparse
from datetime import datetime, timedelta, timezone
from copy import deepcopy


def _parse_naive_utc(dt_str: str) -> datetime:
    """Parse a datetime string into a timezone-naive UTC datetime.

    Handles trailing 'Z' and common ISO formats; falls back to fromisoformat.
    """
    s = str(dt_str or "").strip()
    if s.endswith("Z"):
        s = s[:-1]
    s = s.replace("T", " ")
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class _IndexerProxy:
    """Lightweight proxy exposing minimal TVTW indexer API used by FlightList.

    Exposes:
      - tv_id_to_idx: Dict[str, int]
      - idx_to_tv_id: Dict[int, str]
      - num_time_bins: int
      - get_tvtw_from_index(tvtw_index) -> Optional[Tuple[str, int]]
    """

    def __init__(self, tv_id_to_idx: Dict[str, int], idx_to_tv_id: Dict[int, str], num_time_bins: int):
        self.tv_id_to_idx = tv_id_to_idx
        self.idx_to_tv_id = idx_to_tv_id
        self._num_time_bins = int(num_time_bins)

    @property
    def num_time_bins(self) -> int:
        return self._num_time_bins

    def get_tvtw_from_index(self, tvtw_index: int) -> Optional[Tuple[str, int]]:
        try:
            tvtw_index = int(tvtw_index)
        except Exception:
            return None
        if tvtw_index < 0:
            return None
        tv_index = tvtw_index // self._num_time_bins
        time_idx = tvtw_index % self._num_time_bins
        tv_id = self.idx_to_tv_id.get(int(tv_index))
        if tv_id is None:
            return None
        return (tv_id, int(time_idx))


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
        # Derived helpers
        # Number of time bins for each traffic volume (expected: 24 * 60 / time_bin_minutes)
        self.num_time_bins_per_tv: int = (24 * 60) // int(self.time_bin_minutes)
        # Reverse mapping from contiguous tv index -> traffic_volume_id
        self.idx_to_tv_id: Dict[int, str] = {idx: tv_id for tv_id, idx in self.tv_id_to_idx.items()}
        # Lightweight indexer proxy for compatibility helpers
        self._indexer = _IndexerProxy(self.tv_id_to_idx, self.idx_to_tv_id, self.num_time_bins_per_tv)
        
        # Load flight data
        self._load_flight_data()
        
        # Cache for per-flight TV index sequences (after de-duplicating consecutive repeats)
        self._flight_tv_sequence_cache: Dict[str, np.ndarray] = {}

    # --- Indexer proxies / helpers ---
    @property
    def indexer(self):
        """Return a lightweight proxy exposing TVTW index helpers.

        Provides tv_id_to_idx, idx_to_tv_id, num_time_bins, and get_tvtw_from_index.
        """
        return self._indexer

    @property
    def num_time_bins(self) -> int:
        """Number of time bins per day derived from time_bin_minutes."""
        return int(self.num_time_bins_per_tv)

    # --- Alternative loader: streaming JSON (ijson fallback) ---
    @classmethod
    def from_json(cls, path: str, indexer: Any) -> "FlightList":
        """Build a FlightList by streaming a large JSON mapping flight_id -> data.

        Prefer ijson for streaming; falls back to json.load. takeoff_time is parsed
        to naive UTC. The resulting instance is fully initialized with an occupancy
        matrix and metadata, similar to the standard constructor.
        """
        # Create uninitialized instance
        self = cls.__new__(cls)

        # Derive indexer mappings
        tv_id_to_idx: Dict[str, int] = {}
        idx_to_tv_id: Dict[int, str] = {}
        num_time_bins = None
        time_bin_minutes_val = None
        try:
            tv_id_to_idx = dict(getattr(indexer, 'tv_id_to_idx'))
        except Exception:
            tv_id_to_idx = {}
        try:
            idx_to_tv_id = dict(getattr(indexer, 'idx_to_tv_id'))
        except Exception:
            if tv_id_to_idx:
                idx_to_tv_id = {int(v): str(k) for k, v in tv_id_to_idx.items()}
            else:
                idx_to_tv_id = {}
        try:
            num_time_bins = int(getattr(indexer, 'num_time_bins'))
        except Exception:
            num_time_bins = None
        try:
            time_bin_minutes_val = int(getattr(indexer, 'time_bin_minutes'))
        except Exception:
            time_bin_minutes_val = None
        if num_time_bins is None:
            if time_bin_minutes_val is not None and time_bin_minutes_val > 0:
                num_time_bins = (24 * 60) // int(time_bin_minutes_val)
            else:
                num_time_bins = 1
        if time_bin_minutes_val is None or time_bin_minutes_val <= 0:
            time_bin_minutes_val = (24 * 60) // int(num_time_bins)

        # Set basic attributes to mirror standard constructor
        self.occupancy_file_path = path
        self.tvtw_indexer_path = getattr(indexer, 'source_path', None) or ''
        self.tvtw_indexer = {
            'time_bin_minutes': int(time_bin_minutes_val),
            'tv_id_to_idx': tv_id_to_idx,
        }
        self.time_bin_minutes = int(time_bin_minutes_val)
        self.tv_id_to_idx = tv_id_to_idx
        self.num_traffic_volumes = len(self.tv_id_to_idx)
        self.num_time_bins_per_tv = int(num_time_bins)
        self.idx_to_tv_id = idx_to_tv_id if idx_to_tv_id else {idx: tv for tv, idx in tv_id_to_idx.items()}
        self._indexer = _IndexerProxy(self.tv_id_to_idx, self.idx_to_tv_id, self.num_time_bins_per_tv)

        # Streaming ingest
        self.flight_data = {}
        self.flight_metadata = {}
        max_tvtw_index = 0

        def _ingest(fid: str, obj: Dict[str, Any]):
            nonlocal max_tvtw_index
            fid = str(fid)
            intervals_in = obj.get('occupancy_intervals', []) or []
            intervals: List[Dict[str, Any]] = []
            for it in intervals_in:
                try:
                    tvtw_idx = int(it.get('tvtw_index'))
                except Exception:
                    continue
                entry_s = it.get('entry_time_s', 0)
                exit_s = it.get('exit_time_s', entry_s)
                try:
                    entry_s = float(entry_s)
                except Exception:
                    entry_s = 0.0
                try:
                    exit_s = float(exit_s)
                except Exception:
                    exit_s = entry_s
                intervals.append({
                    'tvtw_index': tvtw_idx,
                    'entry_time_s': entry_s,
                    'exit_time_s': exit_s,
                })
                if tvtw_idx > max_tvtw_index:
                    max_tvtw_index = tvtw_idx
            # Minimal flight_data used for matrix construction
            self.flight_data[fid] = {
                'occupancy_intervals': intervals,
                'takeoff_time': obj.get('takeoff_time', ''),
                'origin': obj.get('origin'),
                'destination': obj.get('destination'),
                'distance': obj.get('distance'),
            }
            # Store normalized metadata
            self.flight_metadata[fid] = {
                'takeoff_time': _parse_naive_utc(str(obj.get('takeoff_time', '1970-01-01T00:00:00'))),
                'origin': obj.get('origin'),
                'destination': obj.get('destination'),
                'distance': obj.get('distance'),
                'occupancy_intervals': intervals,
            }

        try:
            import ijson  # type: ignore
            with open(path, 'rb') as f:
                for fid, obj in ijson.kvitems(f, ''):
                    if isinstance(obj, dict):
                        _ingest(fid, obj)
        except Exception:
            with open(path, 'r') as f:
                data = json.load(f)
            for fid, obj in data.items():
                if isinstance(obj, dict):
                    _ingest(fid, obj)

        # Finalize internal structures
        self.flight_ids = list(self.flight_data.keys())
        self.num_flights = len(self.flight_ids)
        self.flight_id_to_row = {fid: i for i, fid in enumerate(self.flight_ids)}
        self.num_tvtws = int(max_tvtw_index) + 1

        # Build sparse matrices
        self._build_occupancy_matrix()

        # Track state flags and buffers
        self._lil_matrix_dirty = False
        self._temp_occupancy_buffer = np.zeros(self.num_tvtws, dtype=np.float32)
        self._flight_tv_sequence_cache = {}

        return self
        
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

    # === Subflows/Footprint helpers ===
    def get_flight_tv_sequence_indices(self, flight_id: str) -> np.ndarray:
        """
        Return the sequence of traffic-volume indices visited by a flight, ordered
        by entry time and with consecutive duplicates compressed.

        Args:
            flight_id: The flight identifier.

        Returns:
            1D numpy array of integer TV indices in chronological order with
            consecutive repeats removed.
        """
        if flight_id not in self.flight_metadata:
            raise ValueError(f"Flight ID {flight_id} not found")

        # Serve from cache if available
        cached = self._flight_tv_sequence_cache.get(flight_id)
        if cached is not None:
            return cached

        meta = self.flight_metadata[flight_id]
        intervals = meta['occupancy_intervals']
        if not intervals:
            seq = np.empty(0, dtype=np.int64)
            self._flight_tv_sequence_cache[flight_id] = seq
            return seq

        # Sort by entry_time_s to ensure chronological order
        entry_times = np.fromiter((int(iv['entry_time_s']) for iv in intervals), dtype=np.int64)
        order = np.argsort(entry_times, kind='mergesort')
        # Map tvtw_index -> traffic volume index via integer division
        tvtw_indices = np.fromiter((int(intervals[i]['tvtw_index']) for i in order), dtype=np.int64)
        tv_indices = tvtw_indices // int(self.num_time_bins_per_tv)

        if tv_indices.size == 0:
            seq = tv_indices
        else:
            # Compress consecutive duplicates (stay within same TV over multiple bins)
            keep_mask = np.ones(tv_indices.shape[0], dtype=bool)
            keep_mask[1:] = tv_indices[1:] != tv_indices[:-1]
            seq = tv_indices[keep_mask]

        # Cache and return
        self._flight_tv_sequence_cache[flight_id] = seq
        return seq

    def get_flight_tv_footprint_indices(self, flight_id: str, hotspot_tv_index: Optional[int] = None) -> np.ndarray:
        """
        Compute a flight's TV footprint as the set of unique TV indices visited,
        optionally trimmed to the prefix up to and including the first occurrence
        of the hotspot TV index.

        Args:
            flight_id: The flight identifier.
            hotspot_tv_index: Optional TV index indicating where to prefix-trim.

        Returns:
            1D numpy array of unique TV indices (sorted ascending) representing
            the footprint.
        """
        seq = self.get_flight_tv_sequence_indices(flight_id)
        if seq.size == 0:
            return seq

        if hotspot_tv_index is not None:
            # Find first occurrence; include it in the prefix if present
            where = np.where(seq == int(hotspot_tv_index))[0]
            if where.size > 0:
                cut_idx = int(where[0]) + 1
                seq = seq[:cut_idx]

        # Return unique TV indices; order is irrelevant for Jaccard
        return np.unique(seq)

    def get_footprints_for_flights(
        self,
        flight_ids: Sequence[str],
        hotspot_tv_index: Optional[int] = None,
        *,
        trim_policy: Union[str, int, None] = None,
        hotspots: Optional[Sequence[Union[str, int]]] = None,
    ) -> List[np.ndarray]:
        """Return per-flight unique TV index arrays, optionally trimmed.

        Compatibility:
          - When trim_policy is None (default), behavior matches previous
            signature using hotspot_tv_index for prefix trimming.

        Extended behavior:
          - trim_policy: "none" or "earliest_hotspot". If an int is provided,
            it is treated as a single TV index to trim to (legacy support).
          - hotspots: sequence of TV ids or indices when using
            trim_policy="earliest_hotspot".
        """
        # If no explicit trim policy, use legacy behavior
        if trim_policy is None:
            footprints: List[np.ndarray] = []
            for fid in flight_ids:
                footprints.append(self.get_flight_tv_footprint_indices(fid, hotspot_tv_index))
            return footprints

        # Normalize trim_policy if provided as int for backward-compat
        if not isinstance(trim_policy, str) and trim_policy is not None:
            try:
                hotspot_tv_index = int(trim_policy)
                trim_policy = "earliest_hotspot"
                hotspots = [hotspot_tv_index]
            except Exception:
                trim_policy = "none"

        # Normalize hotspot set to TV indices
        hotspot_tv_indices: Optional[set[int]] = None
        if hotspots is not None:
            hotspot_tv_indices = set()
            for h in hotspots:
                if isinstance(h, str):
                    idx = self.tv_id_to_idx.get(h)
                    if idx is not None:
                        hotspot_tv_indices.add(int(idx))
                else:
                    try:
                        hotspot_tv_indices.add(int(h))
                    except Exception:
                        pass
            if len(hotspot_tv_indices) == 0:
                hotspot_tv_indices = None

        out: List[np.ndarray] = []
        for fid in flight_ids:
            seq = self.get_flight_tv_sequence_indices(fid)
            if seq.size == 0:
                out.append(seq)
                continue
            if isinstance(trim_policy, str) and trim_policy.lower() in {"earliest_hotspot", "earliest_hotspot(h)"} and hotspot_tv_indices:
                hit = None
                for i, v in enumerate(seq.tolist()):
                    if int(v) in hotspot_tv_indices:
                        hit = i
                        break
                if hit is not None:
                    seq = seq[: hit + 1]
            out.append(np.unique(seq))
        return out

    def iter_hotspot_crossings(
        self,
        hotspot_ids: Sequence[str],
        active_windows: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None,
    ) -> Iterable[Tuple[str, str, datetime, int]]:
        """Iterate over crossings of the provided hotspots within active windows.

        Yields (flight_id, hotspot_id, entry_dt, time_idx), where time_idx is in
        [0, self.num_time_bins).

        active_windows may be:
          - None: all windows
          - Sequence[int]: global set applied to all hotspots
          - Dict[str, Sequence[int]]: per-hotspot windows
        """
        hotspot_set = set(str(h) for h in hotspot_ids)
        global_windows: Optional[set[int]] = None
        per_hotspot: Optional[Dict[str, set[int]]] = None
        if active_windows is None:
            pass
        elif isinstance(active_windows, dict):
            per_hotspot = {str(k): set(int(x) for x in v) for k, v in active_windows.items()}
        else:
            global_windows = set(int(x) for x in active_windows)

        for fid, meta in self.flight_metadata.items():
            takeoff = meta.get('takeoff_time')
            if not isinstance(takeoff, datetime):
                continue
            for iv in meta.get('occupancy_intervals', []) or []:
                try:
                    tvtw_idx = int(iv['tvtw_index'])
                except Exception:
                    continue
                decoded = self.indexer.get_tvtw_from_index(tvtw_idx)
                if not decoded:
                    continue
                tv_id, time_idx = decoded
                if tv_id not in hotspot_set:
                    continue
                allowed = True
                if per_hotspot is not None:
                    allowed_set = per_hotspot.get(tv_id)
                    allowed = allowed_set is not None and int(time_idx) in allowed_set
                elif global_windows is not None:
                    allowed = int(time_idx) in global_windows
                if not allowed:
                    continue
                entry_s = iv.get('entry_time_s', 0)
                try:
                    entry_s = float(entry_s)
                except Exception:
                    entry_s = 0.0
                entry_dt = takeoff + timedelta(seconds=entry_s)
                yield (fid, tv_id, entry_dt, int(time_idx))
    
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