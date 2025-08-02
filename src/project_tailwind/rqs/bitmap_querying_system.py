import orjson
import numpy as np
from pyroaring import BitMap
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
import os
import json

class RouteQuerySystem:
    """
    A route information querying system using ID-based posting lists and roaring bitmaps.
    This class builds a set of indexes from a route data file and provides
    efficient query methods.
    """

    def __init__(self, route_file_path: str):
        """
        Initializes and builds the query system from a route data file.
        
        Args:
            route_file_path: Path to the JSON file containing route data.
                             The file should be a dictionary where keys are route strings
                             and values are objects with 'impact_vectors' and 'distance'.
        """
        self.route_file_path = route_file_path
        self._build_indexes()

    def _build_indexes(self):
        """
        Parses the route data, builds all necessary data structures and indexes.
        """
        # 1) Parse JSON
        with open(self.route_file_path, "rb") as f:
            data = orjson.loads(f.read())

        self.routes: List[str] = list(data.keys())
        n_routes = len(self.routes)

        # 2) Assign route IDs and prepare data structures
        self.route_str_to_id: Dict[str, int] = {r: i for i, r in enumerate(self.routes)}
        self.route_id_to_str: List[str] = self.routes # Direct mapping is fine for now
        
        origins = np.empty(n_routes, dtype=object)
        dests = np.empty(n_routes, dtype=object)
        self.distances = np.zeros(n_routes, dtype=np.float64)

        # Occupancy vectors to contiguous buffer
        self.vec_off = np.zeros(n_routes + 1, dtype=np.int64)
        vec_chunks = []

        for i, r_str in enumerate(self.routes):
            route_info = data[r_str]
            toks = r_str.split()
            origins[i], dests[i] = toks[0], toks[-1]
            
            # Store distance
            self.distances[i] = route_info['distance']
            
            # Store impact vector
            v = route_info['impact_vectors']
            self.vec_off[i+1] = self.vec_off[i] + len(v)
            vec_chunks.append(np.asarray(v, dtype=np.uint32))
        
        self.vec_data = np.concatenate(vec_chunks) if vec_chunks else np.array([], dtype=np.uint32)

        # 3) Build OD index
        self.od_index: Dict[Tuple[str, str], BitMap] = defaultdict(BitMap)
        for i in range(n_routes):
            self.od_index[(origins[i], dests[i])].add(i)

        # 4) Build TVTW index
        self.tvtw_index: Dict[int, BitMap] = defaultdict(BitMap)
        for i in range(n_routes):
            start, end = self.vec_off[i], self.vec_off[i+1]
            for tv in self.vec_data[start:end]:
                self.tvtw_index[int(tv)].add(i)

    def get_vector(self, route_str: str) -> Optional[np.ndarray]:
        """
        Given a route string, return its occupancy vector.
        This returns a view of the data, not a copy.
        """
        route_id = self.route_str_to_id.get(route_str)
        if route_id is None:
            return None
        start, end = self.vec_off[route_id], self.vec_off[route_id+1]
        return self.vec_data[start:end]

    def get_distance(self, route_str: str) -> Optional[float]:
        """
        Given a route string, return its distance.
        """
        route_id = self.route_str_to_id.get(route_str)
        if route_id is None:
            return None
        return self.distances[route_id]
        
    def _format_output(self, bitmap: BitMap) -> List[Tuple[str, float]]:
        """Helper to format output from a bitmap of route IDs."""
        return [(self.route_id_to_str[i], self.distances[i]) for i in bitmap]

    def get_routes_by_OD(self, origin: str, dest: str) -> List[Tuple[str, float]]:
        """
        Get all routes and their distances for a given origin-destination pair.
        """
        bm = self.od_index.get((origin, dest))
        if not bm:
            return []
        return self._format_output(bm)

    def get_routes_avoiding_OD(self, origin: str, dest: str, banned_tvtws: List[int]) -> List[Tuple[str, float]]:
        """
        Get all (O,D) routes that avoid all specified TVTW indices.
        """
        cand_bm = self.od_index.get((origin, dest))
        if not cand_bm:
            return []
            
        banned_bm = BitMap()
        for tv in banned_tvtws:
            b = self.tvtw_index.get(tv)
            if b:
                banned_bm |= b
        
        ok_bm = cand_bm - banned_bm
        return self._format_output(ok_bm)

    def get_routes_matching_OD(self, origin: str, dest: str, tvtws: List[int], require_all: bool = False) -> List[Tuple[str, float]]:
        """
        Get (O,D) routes that contain some or all specified TVTWs.
        - require_all=False: Routes containing ANY of the tvtws (union).
        - require_all=True: Routes containing ALL of the tvtws (intersection).
        """
        cand_bm = self.od_index.get((origin, dest))
        if not cand_bm:
            return []
            
        if not tvtws:
            return self._format_output(cand_bm)

        if require_all:
            # Intersection: start with all candidates and narrow down
            match_bm = cand_bm.copy()
            for tv in tvtws:
                # Get bitmap for tvtw, if it doesn't exist, use an empty bitmap
                # The intersection with an empty bitmap will result in an empty one.
                tv_bm = self.tvtw_index.get(tv, BitMap())
                match_bm &= tv_bm
        else:
            # Union: start with an empty set and add routes that match
            match_bm = BitMap()
            for tv in tvtws:
                tv_bm = self.tvtw_index.get(tv)
                if tv_bm:
                    match_bm |= tv_bm
            # Finally, intersect with the OD candidates
            match_bm &= cand_bm
            
        return self._format_output(match_bm)

if __name__ == '__main__':
    file_path = "D:/project-tailwind/output/route_distances.json"

    # --- Initialize and use the RouteQuerySystem ---
    print("\n--- Initializing RouteQuerySystem ---")
    rqs = RouteQuerySystem(file_path)
    print("Initialization complete.")

    # --- Example Queries ---
    print("\n--- Running Example Queries ---")

    # 1. Get vector and distance for a specific route
    route_str = "BKPR ALELU EDDM"
    print(f"\n1. Vector for '{route_str}': {rqs.get_vector(route_str)}")
    print(f"   Distance for '{route_str}': {rqs.get_distance(route_str)}")

    # 2. Get all routes for an (O,D) pair
    origin, dest = "BKPR", "EDJA"
    print(f"\n2. All routes from {origin} to {dest}:")
    routes_od = rqs.get_routes_by_OD(origin, dest)
    for r, d in routes_od:
        print(f"   - {r} (Distance: {d:.2f})")
        
    # 3. Get (O,D) routes avoiding a TVTW
    banned_tvtw = 2474
    print(f"\n3. Routes from {origin} to {dest} avoiding TVTW {banned_tvtw}:")
    routes_avoiding = rqs.get_routes_avoiding_OD(origin, dest, [banned_tvtw])
    for r, d in routes_avoiding:
        tvtws = rqs.get_vector(r)
        print(f"   - {r} (Distance: {d:.2f}) TVTWs: {tvtws}")
        
    # 4. Get (O,D) routes matching ANY of a list of TVTWs
    tvtws_any = [1706, 3001]
    origin_any, dest_any = "BKPR", "EDDM"
    print(f"\n4. Routes from {origin_any} to {dest_any} matching ANY of {tvtws_any}:")
    routes_any = rqs.get_routes_matching_OD(origin_any, dest_any, tvtws_any, require_all=False)
    for r, d in routes_any:
        print(f"   - {r} (Distance: {d:.2f})")
        
    # 5. Get (O,D) routes matching ALL of a list of TVTWs
    tvtws_all = [1706, 2522]
    origin_all, dest_all = "BKPR", "EDDM"
    print(f"\n5. Routes from {origin_all} to {dest_all} matching ALL of {tvtws_all}:")
    routes_all = rqs.get_routes_matching_OD(origin_all, dest_all, tvtws_all, require_all=True)
    for r, d in routes_all:
        print(f"   - {r} (Distance: {d:.2f})")
        
    # 6. Sanity check from prompt
    print("\n--- Sanity check from prompt ---")
    print("Route 'BKPR ALELU EDDM' -> vector:", rqs.get_vector("BKPR ALELU EDDM"))
    routes_avoiding_3461 = rqs.get_routes_avoiding_OD("BKPR", "EDJA", [3461])
    print("Routes from BKPR to EDJA avoiding {3461}:", routes_avoiding_3461)
    should_be_excluded = "BKPR ERKIR EDNL EDJA"
    is_excluded = not any(r == should_be_excluded for r,d in routes_avoiding_3461)
    print(f"Is '{should_be_excluded}' excluded? {is_excluded}")

