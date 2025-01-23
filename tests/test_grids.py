import unittest
import numpy as np
import qowi.grids as grids

class TestGrids(unittest.TestCase):

    def test_validate_bit_depth(self):
        valid_depths = [2, 4, 8]
        for depth in valid_depths:
            try:
                grids.validate_bit_depth(depth)
            except ValueError:
                self.fail(f"validate_bit_depth raised ValueError for valid depth {depth}")

        with self.assertRaises(ValueError):
            grids.validate_bit_depth(1)

    def test_get_struct_format(self):
        self.assertEqual(grids.get_struct_format(2), "B")
        self.assertEqual(grids.get_struct_format(4), "H")
        self.assertEqual(grids.get_struct_format(8), "I")
        with self.assertRaises(KeyError):
            grids.get_struct_format(1)

    def test_calculate_haar_coefficients(self):
        pixels = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int16)
        coefficients = grids.calculate_haar_coefficients(pixels)
        expected = np.array([
            [10, -2, -4, 0],
            [10, 2, 4, 0]
        ], dtype=np.int16)
        np.testing.assert_array_equal(coefficients, expected)

    def test_grid_to_index(self):
        pixels = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.uint8)
        index = grids.grids_to_indices(pixels, 8)
        # Update expected values based on bit shifts in the `grids_to_indices` method
        expected = np.array([0x01020304, 0x04030201], dtype=np.uint32)
        np.testing.assert_array_equal(index, expected)

    def test_calculate_haar_keys(self):
        coefficients = np.array([
            [10, -2, -4, 0],
            [10, 2, 4, 0]
        ], dtype=np.int16)
        keys = grids.calculate_haar_keys(coefficients)
        expected = np.array([0x0AFEFC00, 0x0A020400], dtype=np.uint32)
        np.testing.assert_array_equal(keys, expected)

    def test_indices_to_grids(self):
        index = np.array([0x01020304], dtype=np.uint32)
        observed = grids.indices_to_grids(index, 8)
        expected = np.array([[1, 2, 3, 4]], dtype=np.uint8)  # Update expected to 2D array
        np.testing.assert_array_equal(observed, expected)

    def test_end_to_end_grid_conversion(self):
        pixels = np.array([[15, 7, 3, 1], [255, 128, 64, 32]], dtype=np.uint8)
        pixel_bit_depth = 8
        indices = grids.grids_to_indices(pixels, pixel_bit_depth)
        reconstructed_grid = grids.indices_to_grids(indices, pixel_bit_depth)
        np.testing.assert_array_equal(reconstructed_grid, pixels)

    def test_haar_coefficients_consistency(self):
        """Test consistency between coefficients and reconstructed values."""
        pixels = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int16)
        coefficients = grids.calculate_haar_coefficients(pixels)

        # Ensure LL coefficients are consistent with the sum of grids
        ll_coefficients = coefficients[:, 0]
        expected_ll = np.sum(pixels, axis=1)
        np.testing.assert_array_equal(ll_coefficients, expected_ll)

if __name__ == "__main__":
    unittest.main()
