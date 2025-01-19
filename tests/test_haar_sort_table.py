import struct
import unittest
from unittest.mock import mock_open, patch
from qowi.haar_sort_table import HaarSortTable

class TestHaarSortTable(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.valid_grid = [3, 2, 1, 0]
        self.valid_index = 10
        self.bit_depth = 4
        self.table_name = "test_table"
        self.table = HaarSortTable(self.bit_depth, self.table_name)

    @patch("builtins.open", new_callable=mock_open)
    @patch("struct.calcsize", return_value=2)
    def test_grid_to_haar_sort_index(self, mock_calcsize, mock_file):
        """Test grid to Haar sort index conversion."""
        mock_file().read.return_value = struct.pack("H", self.valid_index)
        index = self.table.grid_to_haar_sort_index(self.valid_grid)
        self.assertEqual(index, self.valid_index)

    @patch("builtins.open", new_callable=mock_open)
    @patch("struct.calcsize", return_value=2)
    def test_haar_sort_index_to_grid(self, mock_calcsize, mock_file):
        """Test Haar sort index to grid conversion."""
        packed_data = sum(val << (i * self.bit_depth) for i, val in enumerate(reversed(self.valid_grid)))
        mock_file().read.return_value = struct.pack("H", packed_data)
        grid = self.table.haar_sort_index_to_grid(self.valid_index)
        self.assertEqual(grid, tuple(self.valid_grid))

    def test_invalid_bit_depth(self):
        """Test initialization with an invalid bit depth."""
        with self.assertRaises(ValueError):
            HaarSortTable(3, self.table_name)

    def test_grid_to_binary_position(self):
        """Test calculation of binary position from grid."""
        position = self.table._calculate_binary_position(self.valid_grid)
        expected_position = sum(val << (i * self.bit_depth) for i, val in enumerate(self.valid_grid))
        self.assertEqual(position, expected_position)

    def test_invalid_grid(self):
        """Test handling of invalid grids."""
        invalid_grid = [16, 0, 0, 0]  # Out of range for bit depth 4
        with self.assertRaises(ValueError):
            self.table._calculate_binary_position(invalid_grid)

    def test_grid_to_haar_sort_components(self):
        """Test conversion from grid to Haar sort components."""
        with patch.object(self.table, "grid_to_haar_sort_index", return_value=42):
            components = self.table.grid_to_haar_sort_components(self.valid_grid)
            self.assertEqual(components, (10, 1, 0, 0))

    def test_haar_sort_components_to_grid(self):
        """Test conversion from Haar sort components to grid."""
        with patch.object(self.table, "haar_sort_index_to_grid", return_value=tuple(self.valid_grid)):
            grid = self.table.haar_sort_components_to_grid(10, 1, 0, 0)
            self.assertEqual(grid, tuple(self.valid_grid))

    def test_missing_files(self):
        """Test handling of missing table files."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with self.assertRaises(ValueError):
                self.table.grid_to_haar_sort_index(self.valid_grid)

            with self.assertRaises(ValueError):
                self.table.haar_sort_index_to_grid(self.valid_index)

if __name__ == "__main__":
    unittest.main()
