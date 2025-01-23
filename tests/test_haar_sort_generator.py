import os
import unittest
import tempfile
import struct
import numpy as np
from itertools import product
from scripts.haar_sort_generator import HaarSortGenerator
from qowi.grids import calculate_haar_coefficients

class TestHaarSortGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.gettempdir()

    def test_generate_all_grids_to_file(self):
        pixel_bit_depth = 2
        max_value = (1 << pixel_bit_depth) - 1
        expected_grids = list(product(range(max_value + 1), repeat=4))

        generator = HaarSortGenerator(pixel_bit_depth=pixel_bit_depth, temp_dir=self.temp_dir)
        output_file = f"{self.temp_dir}/test_all_grids.bin"
        generator.generate_all_grids_to_file(output_file)

        with open(output_file, "rb") as f:
            grids = [struct.unpack(generator.struct_format, f.read(generator.entry_size))[0]
                     for _ in range(len(expected_grids))]

        self.assertEqual(len(expected_grids), len(grids), "Generated grid count mismatch.")

    def test_round_trip(self):
        for pixel_bit_depth in [2, 4]:
            max_value = (1 << pixel_bit_depth) - 1
            all_grids = np.array(list(product(range(max_value + 1), repeat=4)), dtype=np.uint8)

            generator = HaarSortGenerator(pixel_bit_depth=pixel_bit_depth, temp_dir=self.temp_dir, table_name=f"test_round_trip_{pixel_bit_depth}bit")

            # Generate forward and reverse lookup tables
            forward_table = generator.haar_sort()
            reverse_table = f"{self.temp_dir}/test_round_trip_{pixel_bit_depth}bit_index.bin"
            generator.generate_reverse_lookup_table(forward_table, reverse_table)

            # Convert grids to Haar sort indices and back
            pixel_indices = np.array([struct.unpack(generator.struct_format, struct.pack(generator.struct_format, idx))[0] for idx in range(len(all_grids))], dtype=np.uint32)
            reconstructed_grids = np.array([tuple(((val >> (pixel_bit_depth * (3 - i))) & max_value) for i in range(4)) for val in pixel_indices], dtype=np.uint8)

            # Assert that the original and reconstructed grids are identical
            np.testing.assert_array_equal(all_grids, reconstructed_grids,
                                          f"Round trip failed for {pixel_bit_depth}-bit pixel depth.")

    def test_temp_file_cleanup(self):
        pixel_bit_depth = 2
        generator = HaarSortGenerator(pixel_bit_depth=pixel_bit_depth, temp_dir=self.temp_dir, table_name="test_cleanup")

        generator.haar_sort()
        temp_files = [f for f in os.listdir(self.temp_dir) if "temp" in f]
        self.assertEqual(len(temp_files), 0, "Temporary files were not cleaned up.")

if __name__ == "__main__":
    unittest.main()
