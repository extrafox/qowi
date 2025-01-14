import numpy as np
import unittest
from itertools import product

class HaarSortTable:
    def __init__(self, bit_depth):
        self.bit_depth = bit_depth
        self.grouped_grids = None

    def _calculate_haar_coefficients(self, grid):
        """Calculate Haar coefficients LL, HL, LH, HH without dividing by 4."""
        a, b, c, d = grid
        LL = a + b + c + d
        HL = a - b + c - d
        LH = a + b - c - d
        HH = a - b - c + d
        return LL, HL, LH, HH

    def grid_to_haar_sort_index(self, grid):
        """Generate the sort index based on Haar LL coefficients and group assignment."""
        for group_index, group in enumerate(self.grouped_grids):
            if any((grid == g).all() for g in group):
                return group_index
        raise ValueError("Grid not found in precomputed groups.")

    def haar_sort_index_to_grid(self, haar_sort_index):
        """Map a Haar sort index back to the corresponding grid."""
        if 0 <= haar_sort_index < len(self.grouped_grids):
            return self.grouped_grids[haar_sort_index]
        raise ValueError("Invalid Haar sort index.")

    def generate_all_possible_grids(self):
        """Generate, sort, and group all possible 4-pixel grids for the given bit depth."""
        max_value = (1 << self.bit_depth) - 1
        grids = np.array(list(product(range(max_value + 1), repeat=4)), dtype=np.uint32)
        grids = self._sort_grids_by_haar(grids)
        self._split_into_groups(grids)
        return grids

    def _sort_grids_by_haar(self, grids):
        """Sort grids by Haar LL coefficients and lexicographical order."""
        grids = sorted(grids, key=lambda grid: (self._calculate_haar_coefficients(grid)[0], tuple(grid)))
        return np.array(grids, dtype=np.uint32)

    def _split_into_groups(self, sorted_grids):
        """Split sorted grids into 2 ** bit_depth groups."""
        num_groups = 1 << self.bit_depth
        group_size = len(sorted_grids) // num_groups
        self.grouped_grids = [sorted_grids[i * group_size: (i + 1) * group_size] for i in range(num_groups)]

class TestHaarSorting(unittest.TestCase):
    def test_generate_all_possible_grids(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        grids = table.generate_all_possible_grids()
        self.assertEqual(len(grids), 2 ** (bit_depth * 4))  # 4 bits per pixel, 4 pixels -> 256 combinations

    def test_sort_grids_by_haar(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        grids = table.generate_all_possible_grids()

        # Validate sorted order by LL coefficient
        previous_LL = None
        for grid in grids:
            LL, _, _, _ = table._calculate_haar_coefficients(grid)
            if previous_LL is not None:
                self.assertLessEqual(previous_LL, LL)
            previous_LL = LL

    def test_grid_to_haar_sort_index_and_back(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        grids = table.generate_all_possible_grids()

        for group_index, group in enumerate(table.grouped_grids):
            for grid in group:
                index = table.grid_to_haar_sort_index(grid)
                self.assertEqual(index, group_index)
                self.assertTrue(np.array_equal(table.haar_sort_index_to_grid(index), group))

    def test_group_consistency(self):
        bit_depth = 3
        table = HaarSortTable(bit_depth)
        grids = table.generate_all_possible_grids()

        all_grouped_grids = np.concatenate(table.grouped_grids)
        self.assertEqual(len(grids), len(all_grouped_grids))
        self.assertEqual(len(set(map(tuple, all_grouped_grids))), len(grids))  # No duplicates

    def test_invalid_inputs(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        table.generate_all_possible_grids()

        with self.assertRaises(ValueError):
            table.grid_to_haar_sort_index(np.array([999, 999, 999, 999]))  # Grid not in groups

        with self.assertRaises(ValueError):
            table.haar_sort_index_to_grid(-1)  # Negative index

        with self.assertRaises(ValueError):
            table.haar_sort_index_to_grid(1 << bit_depth)  # Out-of-range index

if __name__ == "__main__":
    bit_depth = 4  # Example bit depth
    table = HaarSortTable(bit_depth)
    grids = table.generate_all_possible_grids()

    # Save sorted grids to validate the LL_index generation
    print(f"Generated {len(table.grouped_grids)} groups for bit depth {bit_depth}.")

    unittest.main()
