import unittest
import numpy as np
from qowi.grids import calculate_haar_coefficients, indices_to_grids, grids_to_indices

class TestHaarSort(unittest.TestCase):
    def setUp(self):
        self.pixel_bit_depth = 4  # Example bit depth
        self.total_grids = 1 << (self.pixel_bit_depth * 4)  # Total number of 4-pixel grids
        self.grids_array = indices_to_grids(np.arange(self.total_grids, dtype=np.uint32), self.pixel_bit_depth)

    def test_sort_and_preserve_indices(self):
        # Initialize indices to track
        tracked_indices = {}

        # Step 1: Sort by LL coefficient
        coefficients = calculate_haar_coefficients(self.grids_array)
        sorted_indices_LL = np.argsort(coefficients[:, 0], kind="stable")
        sorted_grids_LL = self.grids_array[sorted_indices_LL]

        # Record LL index of (0, 0, 0, 0)
        target_grid = np.array([0, 0, 0, 0], dtype=np.uint8)
        target_index_LL = np.where((sorted_grids_LL == target_grid).all(axis=1))[0][0]
        tracked_indices['LL'] = target_index_LL

        # Step 2: Sort groups of size 2^4 by HL coefficient
        group_size = 1 << self.pixel_bit_depth
        num_groups = len(sorted_grids_LL) // group_size
        sorted_grids_HL = np.zeros_like(sorted_grids_LL)

        for i in range(num_groups):
            group_start = i * group_size
            group_end = group_start + group_size
            group = sorted_grids_LL[group_start:group_end]

            # Sort within group by HL coefficient
            group_coefficients = calculate_haar_coefficients(group)
            group_sorted_indices = np.argsort(group_coefficients[:, 1], kind="stable")
            sorted_grids_HL[group_start:group_end] = group[group_sorted_indices]

        # Record HL index of (0, 0, 0, 0)
        target_index_HL = np.where((sorted_grids_HL == target_grid).all(axis=1))[0][0]
        tracked_indices['HL'] = target_index_HL

        # Step 3: Sort groups of size 2^4 by LH coefficient
        sorted_grids_LH = np.zeros_like(sorted_grids_HL)
        for i in range(num_groups):
            group_start = i * group_size
            group_end = group_start + group_size
            group = sorted_grids_HL[group_start:group_end]

            # Sort within group by LH coefficient
            group_coefficients = calculate_haar_coefficients(group)
            group_sorted_indices = np.argsort(group_coefficients[:, 2], kind="stable")
            sorted_grids_LH[group_start:group_end] = group[group_sorted_indices]

        # Record LH index of (0, 0, 0, 0)
        target_index_LH = np.where((sorted_grids_LH == target_grid).all(axis=1))[0][0]
        tracked_indices['LH'] = target_index_LH

        # Step 4: Sort groups of size 2^4 by HH coefficient
        sorted_grids_HH = np.zeros_like(sorted_grids_LH)
        for i in range(num_groups):
            group_start = i * group_size
            group_end = group_start + group_size
            group = sorted_grids_LH[group_start:group_end]

            # Sort within group by HH coefficient
            group_coefficients = calculate_haar_coefficients(group)
            group_sorted_indices = np.argsort(group_coefficients[:, 3], kind="stable")
            sorted_grids_HH[group_start:group_end] = group[group_sorted_indices]

        # Record HH index of (0, 0, 0, 0)
        target_index_HH = np.where((sorted_grids_HH == target_grid).all(axis=1))[0][0]
        tracked_indices['HH'] = target_index_HH

        # Validation: Ensure indices are preserved through all sorting steps
        self.assertEqual(tracked_indices['LL'], target_index_LL, "LL index not preserved after HL sorting")
        self.assertEqual(tracked_indices['HL'], target_index_HL, "HL index not preserved after LH sorting")
        self.assertEqual(tracked_indices['LH'], target_index_LH, "LH index not preserved after HH sorting")
        self.assertEqual(tracked_indices['HH'], target_index_HH, "HH index not preserved after final sorting")

if __name__ == "__main__":
    unittest.main()
