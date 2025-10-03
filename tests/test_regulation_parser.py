import unittest
import json
from unittest.mock import MagicMock
from project_tailwind.optimize.parser.regulation_parser import Regulation, RegulationParser
from project_tailwind.impact_eval.tvtw_indexer import TVTWIndexer

class TestRegulationParser(unittest.TestCase):
    def setUp(self):
        # Create a mock TVTWIndexer
        self.indexer = TVTWIndexer(time_bin_minutes=30)
        
        # Manually populate the indexer for testing purposes
        traffic_volumes = ["TV1", "TV2", "EBBUELS1", "LFPPLW1"]
        self.indexer._tv_id_to_idx = {tv_id: i for i, tv_id in enumerate(traffic_volumes)}
        self.indexer._idx_to_tv_id = {i: tv_id for i, tv_id in enumerate(traffic_volumes)}
        self.indexer._populate_tvtw_mappings()

        # Mock flight data
        self.flights_data = {
            "flight1": {
                "occupancy_vector": [self.indexer.get_tvtw_index("EBBUELS1", 36)],
                "origin": "LIMC",
                "destination": "EGLL"
            },
            "flight2": {
                "occupancy_vector": [self.indexer.get_tvtw_index("EBBUELS1", 37)],
                "origin": "LIRF",
                "destination": "EGGW"
            },
            "flight3": {
                "occupancy_vector": [self.indexer.get_tvtw_index("LFPPLW1", 36)],
                "origin": "LFPG",
                "destination": "LIMC"
            },
            "flight4": {
                "occupancy_vector": [self.indexer.get_tvtw_index("TV1", 10), self.indexer.get_tvtw_index("TV2", 12)],
                "origin": "LFBO",
                "destination": "EDDM"
            }
        }
        
        # Create a temporary flights file
        self.flights_file = "temp_flights.json"
        with open(self.flights_file, 'w') as f:
            json.dump(self.flights_data, f)
            
        self.parser = RegulationParser(self.flights_file, self.indexer)

    def tearDown(self):
        import os
        os.remove(self.flights_file)

    def test_airport_pair_filtering(self):
        regulation_str = "TV_EBBUELS1 IC_LIMC_EGLL 60 36,37"
        regulation = Regulation(regulation_str)
        matched_flights = self.parser.parse(regulation)
        self.assertEqual(matched_flights, ["flight1"])

    def test_country_pair_filtering(self):
        # This will require a more sophisticated matching logic
        regulation_str = "TV_EBBUELS1 IC_LI>_EG> 60 36,37"
        regulation = Regulation(regulation_str)
        matched_flights = self.parser.parse(regulation)
        self.assertIn("flight1", matched_flights)
        self.assertIn("flight2", matched_flights)

    def test_tv_pair_filtering(self):
        regulation_str = "TV_TV1 TV_TV1_TV2 10 10"
        regulation = Regulation(regulation_str)
        matched_flights = self.parser.parse(regulation)
        self.assertEqual(matched_flights, ["flight4"])

    def test_no_match(self):
        regulation_str = "TV_LFPPLW1 IC_AAAA_BBBB 60 36"
        regulation = Regulation(regulation_str)
        matched_flights = self.parser.parse(regulation)
        self.assertEqual(matched_flights, [])

if __name__ == '__main__':
    unittest.main()

