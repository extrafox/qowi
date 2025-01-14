import numpy as np
import unittest
from itertools import product
import argparse

class HaarSortTable:
    def __init__(self, bit_depth):
        self.bit_depth = bit_depth
        self.ordered_grids = None

    def _calculate_haar_coefficients(self, grid):
        """Calculate Haar coefficients LL, HL, LH, HH without dividing by 4."""
        a, b, c, d = grid.astype(np.int32)  # Cast to signed integer for safe arithmetic
        LL = a + b + c + d
        HL = a - b + c - d
        LH = a + b - c - d
        HH = a - b - c + d
        return LL, HL, LH, HH

    def grid_to_haar_sort_index(self, grid):
        """Get the Haar sort index of the grid in the ordered list."""
        for index, g in enumerate(self.ordered_grids):
            if np.array_equal(grid, g):
                return index
        raise ValueError(f"Grid {grid} not found in the ordered list.")

    def haar_sort_index_to_grid(self, haar_sort_index):
        """Map a Haar sort index back to the corresponding grid."""
        if haar_sort_index < 0 or haar_sort_index >= len(self.ordered_grids):
            raise ValueError(f"Invalid Haar sort index: {haar_sort_index}")
        return self.ordered_grids[haar_sort_index]

    def generate_all_possible_grids(self):
        """Generate, sort, and flatten all possible 4-pixel grids for the given bit depth."""
        max_value = (1 << self.bit_depth) - 1
        grids = np.array(list(product(range(max_value + 1), repeat=4)), dtype=np.uint32)

        # Sort by LL coefficients
        grids = self._sort_grids_by_haar(grids, index=0)

        # Sort by HL coefficients within LL groups
        grids = self._sort_grids_by_haar(grids, index=1)

        # Sort by LH coefficients within HL groups
        grids = self._sort_grids_by_haar(grids, index=2)

        # Sort by HH coefficients within LH groups
        grids = self._sort_grids_by_haar(grids, index=3)

        # Flatten the sorted grids
        self.ordered_grids = grids

        return grids

    def _sort_grids_by_haar(self, grids, index):
        """Sort grids by a specified Haar coefficient and lexicographical order."""
        grids = sorted(grids, key=lambda grid: (self._calculate_haar_coefficients(grid)[index], tuple(grid)))
        return np.array(grids, dtype=np.uint32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haar Sorting Algorithm Tool")
    parser.add_argument("--bit_depth", type=int, required=True, help="Set the bit depth for the grids.")
    parser.add_argument("--grid", nargs=4, type=int, help="Provide four pixel values to get the Haar sort index.")
    parser.add_argument("--index", type=int, help="Provide a Haar sort index to get the corresponding pixel values.")
    args = parser.parse_args()

    # Initialize HaarSortTable with the given bit depth
    table = HaarSortTable(args.bit_depth)
    table.generate_all_possible_grids()

    if args.grid:
        # Convert grid to Haar sort index
        grid = np.array(args.grid, dtype=np.uint32)
        try:
            index = table.grid_to_haar_sort_index(grid)
            print(f"The Haar sort index for grid {grid.tolist()} is {index}.")
        except ValueError as e:
            print(e)

    if args.index is not None:
        # Convert Haar sort index to grid
        try:
            grid = table.haar_sort_index_to_grid(args.index)
            print(f"The grid for Haar sort index {args.index} is {grid.tolist()}.")
        except ValueError as e:
            print(e)

class TestHaarSorting(unittest.TestCase):
    def test_generate_all_possible_grids(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        grids = table.generate_all_possible_grids()
        self.assertEqual(len(grids), 2 ** (bit_depth * 4))  # 4 bits per pixel, 4 pixels -> 256 combinations

    def test_sort_grids_by_haar(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        max_value = (1 << bit_depth) - 1

        # Generate all grids
        grids = np.array(list(product(range(max_value + 1), repeat=4)), dtype=np.uint32)

        # Validate LL sorting
        grids = table._sort_grids_by_haar(grids, index=0)
        previous_LL = None
        for grid in grids:
            LL, _, _, _ = table._calculate_haar_coefficients(grid)
            if previous_LL is not None:
                self.assertLessEqual(previous_LL, LL)
            previous_LL = LL

        # Validate HL sorting within LL groups
        grids = table._sort_grids_by_haar(grids, index=1)
        previous_HL = None
        for grid in grids:
            _, HL, _, _ = table._calculate_haar_coefficients(grid)
            if previous_HL is not None:
                self.assertLessEqual(previous_HL, HL)
            previous_HL = HL

        # Validate LH sorting within HL groups
        grids = table._sort_grids_by_haar(grids, index=2)
        previous_LH = None
        for grid in grids:
            _, _, LH, _ = table._calculate_haar_coefficients(grid)
            if previous_LH is not None:
                self.assertLessEqual(previous_LH, LH)
            previous_LH = LH

        # Validate HH sorting within LH groups
        grids = table._sort_grids_by_haar(grids, index=3)
        previous_HH = None
        for grid in grids:
            _, _, _, HH = table._calculate_haar_coefficients(grid)
            if previous_HH is not None:
                self.assertLessEqual(previous_HH, HH)
            previous_HH = HH

    def test_grid_to_haar_sort_index_and_back(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        grids = table.generate_all_possible_grids()

        for index, grid in enumerate(grids):
            retrieved_index = table.grid_to_haar_sort_index(grid)
            self.assertEqual(index, retrieved_index)
            retrieved_grid = table.haar_sort_index_to_grid(retrieved_index)
            self.assertTrue(np.array_equal(grid, retrieved_grid))

    def test_invalid_inputs(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        table.generate_all_possible_grids()

        with self.assertRaises(ValueError):
            table.grid_to_haar_sort_index(np.array([999, 999, 999, 999]))  # Grid not in the ordered list

        with self.assertRaises(ValueError):
            table.haar_sort_index_to_grid(-1)  # Negative index

        with self.assertRaises(ValueError):
            table.haar_sort_index_to_grid(len(table.ordered_grids))  # Out-of-range index

# Note: To run tests, use the command: python -m unittest <script_name>.py
