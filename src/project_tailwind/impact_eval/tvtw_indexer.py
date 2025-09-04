import json
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta

def create_time_window_mapping(time_bin_minutes: int = 30) -> Dict[int, str]:
    """
    Creates a mapping from time window indices to human-readable time ranges.
    """
    if 1440 % time_bin_minutes != 0:
        raise ValueError("time_bin_minutes must be a divisor of 1440.")
    
    num_bins = 1440 // time_bin_minutes
    time_windows = {}
    for i in range(num_bins):
        start_minute_of_day = i * time_bin_minutes
        end_minute_of_day = start_minute_of_day + time_bin_minutes
        
        start_hour, start_minute = divmod(start_minute_of_day, 60)
        end_hour, end_minute = divmod(end_minute_of_day, 60)
        
        if end_minute == 0:
            end_hour -=1
            end_minute = 60
        
        start_time = f"{start_hour:02d}:{start_minute:02d}"
        end_time = f"{end_hour:02d}:{end_minute:02d}"
        
        time_windows[i] = f"{start_time}-{end_time}"
    return time_windows

class TVTWIndexer:
    """
    Manages the mapping between TVTW (Traffic Volume Time Window) and a unique integer index.
    """
    def __init__(self, time_bin_minutes: int = 30):
        self.time_bin_minutes = int(time_bin_minutes)
        if 1440 % self.time_bin_minutes != 0:
            raise ValueError("time_bin_minutes must divide 1440")
        self._num_time_bins = 1440 // self.time_bin_minutes
        self._tv_id_to_idx: Dict[str, int] = {}
        self._idx_to_tv_id: Dict[int, str] = {}
        self._tvtw_to_idx: Dict[Tuple[str, int], int] = {}
        self._idx_to_tvtw: Dict[int, Tuple[str, int]] = {}
        self.time_window_map: Dict[int, str] = create_time_window_mapping(self.time_bin_minutes)

    # --- Index summary helpers -------------------------------------------------
    @property
    def num_time_bins(self) -> int:
        """Number of time bins per day for the configured bin length."""
        return self._num_time_bins

    @property
    def tv_id_to_idx(self) -> Dict[str, int]:
        return self._tv_id_to_idx

    @property
    def idx_to_tv_id(self) -> Dict[int, str]:
        return self._idx_to_tv_id

    @property
    def tvtw_to_idx(self) -> Dict[Tuple[str, int], int]:
        return self._tvtw_to_idx

    @property
    def idx_to_tvtw(self) -> Dict[int, Tuple[str, int]]:
        return self._idx_to_tvtw

    def build_from_tv_geojson(self, tv_geojson_path: str):
        """
        Builds the TVTW index from a traffic volume GeoJSON file.
        """
        import geopandas as gpd
        tv_gdf = gpd.read_file(tv_geojson_path)
        
        if 'traffic_volume_id' not in tv_gdf.columns:
            raise ValueError("GeoJSON file must have a 'traffic_volume_id' property in each feature.")
            
        traffic_volumes = sorted(tv_gdf['traffic_volume_id'].unique())
        
        self._tv_id_to_idx = {tv_id: i for i, tv_id in enumerate(traffic_volumes)}
        self._idx_to_tv_id = {i: tv_id for i, tv_id in enumerate(traffic_volumes)}
        
        self._populate_tvtw_mappings()

    def _populate_tvtw_mappings(self):
        """Populates the TVTW mappings based on the loaded traffic volumes."""
        self._tvtw_to_idx = {}
        self._idx_to_tvtw = {}

        num_tvs = len(self._tv_id_to_idx)

        for tv_name, tv_idx in self._tv_id_to_idx.items():
            for time_idx in range(self.num_time_bins):
                # The global index is calculated based on the traffic volume's index and the time bin's index.
                global_idx = tv_idx * self.num_time_bins + time_idx
                tvtw_tuple = (tv_name, time_idx)
                self._tvtw_to_idx[tvtw_tuple] = global_idx
                self._idx_to_tvtw[global_idx] = tvtw_tuple

    def get_tvtw_index(self, tv_id: str, time_window_idx: int) -> Optional[int]:
        """
        Get the unique index for a given TV and time window index.
        """
        return self._tvtw_to_idx.get((tv_id, time_window_idx))

    def get_tvtw_from_index(self, index: int) -> Optional[Tuple[str, int]]:
        """
        Get the TV name and time window index from a unique TVTW index.
        """
        return self._idx_to_tvtw.get(index)

    def get_human_readable_tvtw(self, index: int) -> Optional[Tuple[str, str]]:
        """
        Get the human-readable TV name and time window string from a unique TVTW index.
        """
        tvtw = self.get_tvtw_from_index(index)
        if tvtw:
            tv_name, time_idx = tvtw
            time_window_str = self.time_window_map.get(time_idx, "Unknown")
            return (tv_name, time_window_str)
        return None

    def save(self, file_path: str):
        """
        Saves the indexer's state to a JSON file.
        """
        state = {
            'time_bin_minutes': self.time_bin_minutes,
            'tv_id_to_idx': self._tv_id_to_idx,
        }
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)

    @classmethod
    def load(cls, file_path: str) -> 'TVTWIndexer':
        """
        Loads the indexer's state from a JSON file.
        """
        with open(file_path, 'r') as f:
            state = json.load(f)

        indexer = cls(time_bin_minutes=state['time_bin_minutes'])
        indexer._tv_id_to_idx = state['tv_id_to_idx']
        indexer._idx_to_tv_id = {int(v): k for k, v in indexer._tv_id_to_idx.items()}
        indexer._populate_tvtw_mappings()

        return indexer

    # --- Time helpers ----------------------------------------------------------
    def bin_of_datetime(self, dt: datetime) -> int:
        """
        Return the time-bin index for a datetime within a day given the
        configured `time_bin_minutes`.

        The first bin starts at 00:00 inclusive. The returned value satisfies
        0 <= bin < num_time_bins.
        """
        # Floor to the minute for binning
        minute_of_day = dt.hour * 60 + dt.minute
        bin_idx = minute_of_day // self.time_bin_minutes
        if bin_idx >= self.num_time_bins:
            # Guard for edge case 24:00 (should not occur for normal datetimes)
            bin_idx = self.num_time_bins - 1
        return int(bin_idx)

    def bin_range_for_interval(self, start_dt: datetime, end_dt: datetime) -> List[int]:
        """
        Return a list of bin indices overlapped by the half-open interval
        [start_dt, end_dt). If end_dt <= start_dt, returns an empty list.

        Note: This helper assumes start and end fall on the same day boundary
        for simplicity. For multi-day intervals, only the bins within the start
        day are returned.
        """
        if end_dt <= start_dt:
            return []
        start_bin = self.bin_of_datetime(start_dt)
        # For half-open interval, subtract a tiny epsilon from end
        # Use half-open interval: include bin containing start, exclude bin starting at end
        end_adj = end_dt - timedelta(microseconds=1)
        end_bin = self.bin_of_datetime(end_adj)
        if end_bin < start_bin:
            return []
        return list(range(start_bin, end_bin + 1))

    def rolling_window_size(self) -> int:
        """
        Return K, the number of bins in a rolling-hour window, i.e.,
        K = 60 / time_bin_minutes. Raises if 60 is not divisible by bin size.
        """
        if 60 % self.time_bin_minutes != 0:
            raise ValueError("60 is not divisible by time_bin_minutes; cannot form hour window")
        return 60 // self.time_bin_minutes
