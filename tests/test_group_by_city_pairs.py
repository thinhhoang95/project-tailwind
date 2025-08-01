import unittest
from pathlib import Path
import csv
import shutil
from project_tailwind.city_pairs.group_by_city_pairs import group_city_pairs

class TestGroupByCityPairs(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory structure for testing."""
        self.base_dir = Path("./test_temp_data")
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "output"

        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Remove the temporary directory structure after tests."""
        shutil.rmtree(self.base_dir)

    def _create_csv(self, filename, headers, data):
        """Helper function to create a CSV file."""
        file_path = self.input_dir / filename
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)
        return file_path

    def test_group_city_pairs(self):
        """Test the main functionality of grouping city pairs."""
        # 1. Prepare test data
        headers = ['flight_id', 'real_waypoints', 'pass_times', 'speeds', 'alts',
                   'real_full_waypoints', 'full_pass_times', 'full_speeds', 'full_alts']

        data1 = [
            ['flight1', 'LFPG EGLL', '1', '1', '1', 'LFPG EGLL', '1', '1', '1'],
            ['flight2', 'LFPO KJFK', '2', '2', '2', 'LFPO KJFK', '2', '2', '2']
        ]
        self._create_csv("routes1.csv", headers, data1)

        data2 = [
            ['flight3', 'LFPG EGLL EXTRA', '3', '3', '3', 'LFPG EGLL EXTRA', '3', '3', '3'],
            ['flight4', 'EDDF OMDB', '4', '4', '4', 'EDDF OMDB', '4', '4', '4']
        ]
        self._create_csv("routes2.csv", headers, data2)

        # 2. Run the function
        group_city_pairs(self.input_dir, self.output_dir)

        # 3. Assert the results
        # Check for LFEG.csv (LFPG -> EGLL)
        lfeg_path = self.output_dir / "LFEG.csv"
        self.assertTrue(lfeg_path.exists())
        with open(lfeg_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['flight_identifier'], 'flight1')
            self.assertEqual(rows[0]['route'], 'LFPG EGLL')

        # Check for LFKJ.csv (LFPO -> KJFK) - Note KJFK -> KJ
        lfki_path = self.output_dir / "LFKJ.csv"
        self.assertTrue(lfki_path.exists())
        with open(lfki_path, 'r', newline='', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 1)
            self.assertEqual(reader[0]['flight_identifier'], 'flight2')

        # Check for EDOM.csv (EDDF -> OMDB)
        edom_path = self.output_dir / "EDOM.csv"
        self.assertTrue(edom_path.exists())
        with open(edom_path, 'r', newline='', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 1)
            self.assertEqual(reader[0]['flight_identifier'], 'flight4')

        # Check for LFEX.csv (LFPG -> EXTRA)
        lfex_path = self.output_dir / "LFEX.csv"
        self.assertTrue(lfex_path.exists())
        with open(lfex_path, 'r', newline='', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 1)
            self.assertEqual(reader[0]['flight_identifier'], 'flight3')
            self.assertEqual(reader[0]['route'], 'LFPG EGLL EXTRA')
            
    def test_empty_input_directory(self):
        """Test behavior with an empty input directory."""
        group_city_pairs(self.input_dir, self.output_dir)
        # Check that the output directory is still empty
        self.assertEqual(len(list(self.output_dir.iterdir())), 0)
        
    def test_malformed_csv(self):
        """Test with CSVs that have missing columns or malformed rows."""
        # File with missing 'real_waypoints' column
        self._create_csv("bad1.csv", ['flight_id', 'other_data'], [['flight5', 'some data']])
        
        # File with not enough waypoints
        headers = ['flight_id', 'real_waypoints']
        self._create_csv("bad2.csv", headers, [['flight6', 'LFPG']])
        
        # File with empty rows or missing values
        self._create_csv("bad3.csv", headers, [['', 'LFPG EGLL'], ['flight7', '']])
        
        group_city_pairs(self.input_dir, self.output_dir)
        
        # No files should be created because all rows are faulty
        self.assertEqual(len(list(self.output_dir.iterdir())), 0)

    def test_empty_csv_file(self):
        """Test with an empty CSV file."""
        (self.input_dir / "empty.csv").touch()
        group_city_pairs(self.input_dir, self.output_dir)
        self.assertEqual(len(list(self.output_dir.iterdir())), 0)


if __name__ == '__main__':
    unittest.main()
