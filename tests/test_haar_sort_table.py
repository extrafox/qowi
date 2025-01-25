import unittest
import numpy as np
from qowi.haar_sort_table import HaarSortTable
from haar_sort_generator import HaarSortGenerator

class TestHaarSortTableExhaustive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generate lookup tables for 2-bit and 4-bit grids."""
        cls.table_prefix_2bit = "/tmp/haar_sort_2bit"
        cls.table_prefix_4bit = "/tmp/haar_sort_4bit"

        # Generate 2-bit lookup tables
        generator_2bit = HaarSortGenerator(pixel_bit_depth=2, table_name=cls.table_prefix_2bit)
        generator_2bit.haar_sort()
        generator_2bit.generate_reverse_lookup_table(f"{cls.table_prefix_2bit}_grids.bin",
                                                     f"{cls.table_prefix_2bit}_index.bin")

        # Generate 4-bit lookup tables
        generator_4bit = HaarSortGenerator(pixel_bit_depth=4, table_name=cls.table_prefix_4bit)
        generator_4bit.haar_sort()
        generator_4bit.generate_reverse_lookup_table(f"{cls.table_prefix_4bit}_grids.bin",
                                                     f"{cls.table_prefix_4bit}_index.bin")

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

if __name__ == "__main__":
    unittest.main()
