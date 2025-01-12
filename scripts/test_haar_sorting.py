import numpy as np
import itertools
import struct
import os
import tempfile
import heapq
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

    def external_sort(self, grids, key):
        temp_files = []
        chunk_size = 10_000  # Number of grids per chunk
        grids_iter = iter(grids)

        while True:
            chunk = list(itertools.islice(grids_iter, chunk_size))
            if not chunk:
                break
            chunk.sort(key=key)
            temp_file = tempfile.TemporaryFile()
            np.save(temp_file, np.array(chunk, dtype=object))
            temp_file.seek(0)
            temp_files.append(temp_file)

        return self.merge_sorted_files(temp_files, key)

    def merge_sorted_files(self, temp_files, key):
        def file_iterator(temp_file):
            temp_file.seek(0)
            chunk = np.load(temp_file, allow_pickle=True)
            yield from chunk

        iterators = [file_iterator(f) for f in temp_files]
        return heapq.merge(*iterators, key=key)

    def generate_lookup_table(self, output_filename):
        pixel_space = range(0, self.max_value + 1)
        grids = itertools.product(pixel_space, repeat=self.grid_size)

        # Step 1: Sort grids by LL coefficient
        sorted_grids = self.external_sort(grids, key=lambda grid: (self.haar_coefficients(grid)[0], grid))

        # Step 2: Assign LL index
        ll_groups = []
        ll_indices = {}
        group_count = 0

        for i, grid in enumerate(sorted_grids):
            if i % (self.pixel_space_size // self.group_size) == 0:
                ll_groups.append([])
            ll_groups[-1].append(grid)
            ll_indices[tuple(grid)] = group_count // (self.pixel_space_size // self.group_size)
            group_count += 1

        # Save lookup table to disk
        with open(output_filename, 'wb') as f:
            for group in ll_groups:
                for grid in group:
                    packed_grid = struct.pack(f'{self.grid_size}B', *grid)
                    ll_index = ll_indices[tuple(grid)]
                    f.write(packed_grid + struct.pack('B', ll_index))

    def query_lookup_table(self, pixels, filename):
        with open(filename, 'rb') as f:
            while chunk := f.read(self.grid_size + 1):
                grid = struct.unpack(f'{self.grid_size}B', chunk[:self.grid_size])
                if tuple(pixels) == grid:
                    return struct.unpack('B', chunk[self.grid_size:])[0]
        return None

class TestHaarSorting(unittest.TestCase):
    def test_generate_lookup_table(self):
        hs = HaarSorting(bit_depth=2)
        filename = "test_lookup_table.bin"
        hs.generate_lookup_table(filename)
        self.assertTrue(os.path.exists(filename), "Lookup table file not created")
        os.remove(filename)

    def test_query_lookup_table(self):
        hs = HaarSorting(bit_depth=2)
        filename = "test_lookup_table.bin"
        hs.generate_lookup_table(filename)
        pixels = [3, 1, 2, 0]
        index = hs.query_lookup_table(pixels, filename)
        self.assertIsNotNone(index, "Pixel grid not found in lookup table")
        os.remove(filename)

if __name__ == "__main__":
    unittest.main()
