import unittest
import numpy as np
from qowi.grids import (
    validate_bit_depth,
    get_struct_format,
    calculate_haar_coefficients,
    grid_to_index,
    calculate_haar_keys,
    index_to_grid
)

class TestGrids(unittest.TestCase):

    def test_validate_bit_depth(self):
        valid_depths = [2, 4, 8]
        for depth in valid_depths:
            try:
                validate_bit_depth(depth)
            except ValueError:
                self.fail(f"validate_bit_depth raised ValueError for valid depth {depth}")

        with self.assertRaises(ValueError):
            validate_bit_depth(1)

    def test_get_struct_format(self):
        self.assertEqual(get_struct_format(2), "B")
        self.assertEqual(get_struct_format(4), "H")
        self.assertEqual(get_struct_format(8), "I")
        with self.assertRaises(KeyError):
            get_struct_format(1)

    def test_calculate_haar_coefficients(self):
        grids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int16)
        coefficients = calculate_haar_coefficients(grids)
        expected = np.array([
            [10, -2, -4, 0],
            [10, 2, 4, 0]
        ], dtype=np.int16)
        np.testing.assert_array_equal(coefficients, expected)

    def test_grid_to_index(self):
        grids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.uint8)
        index = grid_to_index(grids, 8)
        expected = np.array([0x01020304, 0x04030201], dtype=np.uint32)
        np.testing.assert_array_equal(index, expected)

    def test_calculate_haar_keys(self):
        coefficients = np.array([
            [10, -2, -4, 0],
            [10, 2, 4, 0]
        ], dtype=np.int16)
        keys = calculate_haar_keys(coefficients)
        expected = np.array([0x0AFEFC00, 0x0A020400], dtype=np.uint32)
        np.testing.assert_array_equal(keys, expected)

    def test_index_to_grid(self):
        index = 0x01020304
        grid = index_to_grid(index, 8)
        expected = np.array([1, 2, 3, 4], dtype=np.uint8)
        np.testing.assert_array_equal(grid, expected)

    def test_end_to_end_grid_conversion(self):
        """Test end-to-end conversion from grid to index and back."""
        grids = np.array([[15, 7, 3, 1], [255, 128, 64, 32]], dtype=np.uint8)
        pixel_bit_depth = 8

        for grid in grids:
            index = grid_to_index(grid[np.newaxis, :], pixel_bit_depth)
            reconstructed_grid = index_to_grid(index[0], pixel_bit_depth)
            np.testing.assert_array_equal(grid, reconstructed_grid)

    def test_haar_coefficients_consistency(self):
        """Test consistency between coefficients and reconstructed values."""
        grids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int16)
        coefficients = calculate_haar_coefficients(grids)

        # Ensure LL coefficients are consistent with the sum of grids
        ll_coefficients = coefficients[:, 0]
        expected_ll = np.sum(grids, axis=1)
        np.testing.assert_array_equal(ll_coefficients, expected_ll)

if __name__ == "__main__":
    unittest.main()
