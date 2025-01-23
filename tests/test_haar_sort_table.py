import unittest
from unittest.mock import mock_open, patch
import struct
import numpy as np
from qowi.haar_sort_table import HaarSortTable

class TestHaarSortTable8Bit(unittest.TestCase):
    def setUp(self):
        self.bit_depth = 8
        self.table_name = "/home/ctaylor/haar_sort/haar_sort_8bit"
        self.table = HaarSortTable(self.bit_depth, self.table_name)
        self.valid_grid = np.array([255, 128, 64, 0], dtype=np.uint8)
        self.valid_index = 1234

    @patch("builtins.open", new_callable=mock_open)
    def test_pixels_to_haar_sort_index(self, mock_file):
        """Test conversion of a valid grid to Haar sort index."""
        packed_index = struct.pack("I", self.valid_index)
        mock_file().read.return_value = packed_index

        index = self.table.pixels_to_haar_sort_index(self.valid_grid)
        self.assertEqual(index, self.valid_index)

    @patch("builtins.open", new_callable=mock_open)
    def test_haar_sort_index_to_pixels(self, mock_file):
        """Test conversion of a Haar sort index back to a grid."""
        packed_grid = sum(val << (i * self.bit_depth) for i, val in enumerate(reversed(self.valid_grid)))
        mock_file().read.return_value = struct.pack("I", packed_grid)

        grid = self.table.haar_sort_index_to_pixels(self.valid_index)
        expected_grid = [packed_grid >> (8 * i) & 0xFF for i in range(4)][::-1]
        self.assertEqual(grid.tolist(), expected_grid)

    def test_grid_out_of_range(self):
        """Test handling of grids with out-of-range values."""
        invalid_grids = [
            [256, 128, 64, 0],  # Value too large for 8-bit
            [-1, 128, 64, 0],   # Negative value
            [255, 255]          # Incorrect length
        ]
        for grid in invalid_grids:
            with self.assertRaises(ValueError):
                self.table.pixels_to_haar_sort_index(grid)

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_missing_forward_table(self, mock_open):
        """Test behavior when the forward table file is missing."""
        with self.assertRaises(ValueError):
            self.table.pixels_to_haar_sort_index(self.valid_grid)

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_missing_reverse_table(self, mock_open):
        """Test behavior when the reverse table file is missing."""
        with self.assertRaises(ValueError):
            self.table.haar_sort_index_to_pixels(self.valid_index)

    def test_min_max_grid_values(self):
        """Test handling of grids with minimum and maximum values."""
        min_grid = np.array([0, 0, 0, 0], dtype=np.uint8)
        max_grid = np.array([255, 255, 255, 255], dtype=np.uint8)

        # Mocking table files for both scenarios
        with patch("builtins.open", new_callable=mock_open) as mock_file:
            min_index = 0
            max_index = (255 << 24) | (255 << 16) | (255 << 8) | 255
            mock_file().read.side_effect = [struct.pack("I", min_index), struct.pack("I", max_index)]

            self.assertEqual(self.table.pixels_to_haar_sort_index(min_grid), min_index)
            self.assertEqual(self.table.pixels_to_haar_sort_index(max_grid), max_index)

    def test_invalid_bit_depth(self):
        """Test initialization with invalid bit depth."""
        with self.assertRaises(ValueError):
            HaarSortTable(16, self.table_name)  # 16 is not supported

if __name__ == "__main__":
    unittest.main()
