import os
import unittest
import numpy as np
from qowi.haar_sort_table import HaarSortTable

class TestHaarSortTableExhaustive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set lookup table paths for 2-bit, 4-bit, and 8-bit grids."""
        cls.table_prefix_2bit = os.path.expanduser("~/haar_sort/haar_sort_2bit")
        cls.table_prefix_4bit = os.path.expanduser("~/haar_sort/haar_sort_4bit")
        cls.table_prefix_8bit = os.path.expanduser("~/haar_sort/haar_sort_8bit")


    def test_round_trip_2bit(self):
        """Exhaustive round-trip test for 2-bit grids."""
        haar_sort_table = HaarSortTable(pixel_bit_depth=2, table_name=self.table_prefix_2bit)

        total_grids = 1 << (2 * 4)  # 2-bit depth, 4 pixels
        grids = np.arange(total_grids, dtype=np.uint32)
        pixels = haar_sort_table.haar_sort_indices_to_pixels(grids)
        indices = haar_sort_table.pixels_to_haar_sort_indices(pixels)

        np.testing.assert_array_equal(grids, indices, "Round-trip failed for 2-bit grids.")

    def test_round_trip_4bit(self):
        """Exhaustive round-trip test for 4-bit grids."""
        haar_sort_table = HaarSortTable(pixel_bit_depth=4, table_name=self.table_prefix_4bit)

        total_grids = 1 << (4 * 4)  # 4-bit depth, 4 pixels
        grids = np.arange(total_grids, dtype=np.uint32)
        pixels = haar_sort_table.haar_sort_indices_to_pixels(grids)
        indices = haar_sort_table.pixels_to_haar_sort_indices(pixels)

        np.testing.assert_array_equal(grids, indices, "Round-trip failed for 4-bit grids.")

    def test_round_trip_8bit(self):
        """Sampled round-trip test for 8-bit grids."""
        haar_sort_table = HaarSortTable(pixel_bit_depth=8, table_name=self.table_prefix_8bit)

        num_samples = 10000  # Test a subset instead of exhaustive validation
        grids = np.random.randint(0, 1 << (8 * 4), size=num_samples, dtype=np.uint32)

        # Include edge cases (min/max values)
        grids = np.concatenate([grids, np.array([0, (1 << (8 * 4)) - 1], dtype=np.uint32)])

        pixels = haar_sort_table.haar_sort_indices_to_pixels(grids)
        indices = haar_sort_table.pixels_to_haar_sort_indices(pixels)

        np.testing.assert_array_equal(grids, indices, "Round-trip failed for 8-bit grids.")


if __name__ == "__main__":
    unittest.main()
