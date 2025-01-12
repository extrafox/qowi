import numpy as np
import itertools
import struct
import os
import unittest


class HaarSorting:
    def __init__(self, bit_depth=8):
        self.bit_depth = bit_depth
        self.pixel_space_size = 2 ** (4 * bit_depth)
        self.grid_size = 4
        self.max_value = 2 ** bit_depth - 1
        self.group_size = 2 ** bit_depth

    def haar_coefficients(self, grid):
        a, b, c, d = grid
        LL = a + b + c + d
        HL = a - b + c - d
        LH = a + b - c - d
        HH = a - b - c + d
        return LL, HL, LH, HH

    def generate_lookup_table(self):
        pixel_space = range(0, self.max_value + 1)
        grids = itertools.product(pixel_space, repeat=self.grid_size)

        # Step 1: Sort grids by LL coefficient
        sorted_grids = sorted(grids, key=lambda grid: (self.haar_coefficients(grid)[0], grid))

        # Step 2: Assign LL index
        ll_groups = np.array_split(sorted_grids, self.group_size)
        ll_indices = {tuple(grid): i for i, group in enumerate(ll_groups) for grid in group}

        # Step 3: Sort by HL within each LL group
        hl_sorted = sorted(
            sorted_grids,
            key=lambda grid: (ll_indices[tuple(grid)], self.haar_coefficients(grid)[1], grid)
        )
        hl_groups = np.array_split(hl_sorted, self.group_size)
        hl_indices = {tuple(grid): i for i, group in enumerate(hl_groups) for grid in group}

        # Step 4: Sort by LH within each HL group
        lh_sorted = sorted(
            hl_sorted,
            key=lambda grid: (hl_indices[tuple(grid)], self.haar_coefficients(grid)[2], grid)
        )
        lh_groups = np.array_split(lh_sorted, self.group_size)
        lh_indices = {tuple(grid): i for i, group in enumerate(lh_groups) for grid in group}

        # Step 5: Sort by HH within each LH group
        hh_sorted = sorted(
            lh_sorted,
            key=lambda grid: (lh_indices[tuple(grid)], self.haar_coefficients(grid)[3], grid)
        )
        hh_groups = np.array_split(hh_sorted, self.group_size)
        lookup_table = {tuple(grid): (ll_indices[tuple(grid)], hl_indices[tuple(grid)], lh_indices[tuple(grid)], i)
                        for i, grid in enumerate(hh_sorted)}

        return lookup_table

    def save_to_disk(self, lookup_table, filename):
        with open(filename, 'wb') as f:
            for grid, indices in lookup_table.items():
                packed_grid = struct.pack(f'{self.grid_size}B', *grid)
                packed_indices = struct.pack('4B', *indices)
                f.write(packed_grid + packed_indices)

    def load_from_disk(self, filename):
        lookup_table = {}
        with open(filename, 'rb') as f:
            while chunk := f.read(self.grid_size + 4):
                grid = struct.unpack(f'{self.grid_size}B', chunk[:self.grid_size])
                indices = struct.unpack('4B', chunk[self.grid_size:])
                lookup_table[grid] = indices
        return lookup_table

    def haar_sort_encode(self, pixels):
        lookup_table = self.generate_lookup_table()
        return lookup_table[tuple(pixels)]

    def haar_sort_decode(self, haar_sort_index):
        lookup_table = self.generate_lookup_table()
        reverse_lookup = {v: k for k, v in lookup_table.items()}
        return np.array(reverse_lookup[haar_sort_index])


class TestHaarSorting(unittest.TestCase):
    def test_generate_lookup_table_size(self):
        hs = HaarSorting(bit_depth=2)
        lookup_table = hs.generate_lookup_table()
        self.assertEqual(len(lookup_table), hs.pixel_space_size, "Lookup table size mismatch")

    def test_save_and_load_lookup_table(self):
        hs = HaarSorting(bit_depth=2)
        lookup_table = hs.generate_lookup_table()
        filename = "test_lookup_table.bin"
        hs.save_to_disk(lookup_table, filename)
        loaded_table = hs.load_from_disk(filename)
        self.assertEqual(lookup_table, loaded_table, "Loaded table does not match saved table")
        os.remove(filename)

    def test_haar_coefficients(self):
        hs = HaarSorting(bit_depth=2)
        grid = (3, 1, 2, 0)
        LL, HL, LH, HH = hs.haar_coefficients(grid)
        self.assertEqual(LL, sum(grid), "Incorrect LL coefficient")
        self.assertEqual(HL, grid[0] - grid[1] + grid[2] - grid[3], "Incorrect HL coefficient")
        self.assertEqual(LH, grid[0] + grid[1] - grid[2] - grid[3], "Incorrect LH coefficient")
        self.assertEqual(HH, grid[0] - grid[1] - grid[2] + grid[3], "Incorrect HH coefficient")

    def test_haar_sort_encode_decode(self):
        hs = HaarSorting(bit_depth=2)
        pixels = np.array([3, 1, 2, 0])
        haar_sort_index = hs.haar_sort_encode(pixels)
        decoded_pixels = hs.haar_sort_decode(haar_sort_index)
        np.testing.assert_array_equal(pixels, decoded_pixels, "Encode and decode mismatch")


if __name__ == "__main__":
    unittest.main()
