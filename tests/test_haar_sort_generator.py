import os
import unittest
import tempfile
import struct
import numpy as np
import qowi.grids as grids
from itertools import product
from haar_sort_generator import HaarSortGenerator


class TestHaarSortGenerator(unittest.TestCase):
    def __init__(self, methodName: str = "runTest"):
        super().__init__(methodName)
        self.pixel_bit_depth = 4
        self.struct_format = grids.get_struct_format(self.pixel_bit_depth)

    def setUp(self):
        self.temp_dir = tempfile.gettempdir()

    def generate_all_grids_to_file(self, output_file):
        """Generate all possible 4-pixel grids for the given bit depth and write to file."""
        total_grids = 1 << (self.pixel_bit_depth * 4)  # 16^4 = 65536 for 4-bit depth
        pixels = np.arange(total_grids, dtype=np.uint32)

        # Write all grids to the output file
        with open(output_file, "wb") as f:
            for grid in pixels:
                f.write(struct.pack(self.struct_format, grid))

        print(f"Generated {total_grids} grids to {output_file}")

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
