import numpy as np
import os
import tempfile
import argparse
from itertools import islice, product
import heapq
import struct
import unittest

class HaarSortTable:
    def __init__(self, bit_depth, temp_dir=None):
        self.bit_depth = bit_depth
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.sorted_file = None

    def _calculate_haar_coefficients(self, grid):
        a, b, c, d = grid.astype(np.int32)
        LL = a + b + c + d
        HL = a - b + c - d
        LH = a + b - c - d
        HH = a - b - c + d
        return LL, HL, LH, HH

    def sort_and_save_chunks(self, chunk_size=10**6):
        max_value = (1 << self.bit_depth) - 1
        all_grids = product(range(max_value + 1), repeat=4)
        temp_files = []

        while True:
            chunk = list(islice(all_grids, chunk_size))
            if not chunk:
                break
            chunk = sorted(chunk, key=lambda grid: (*self._calculate_haar_coefficients(np.array(grid)), grid))
            temp_file = os.path.join(self.temp_dir, f"sorted_chunk_{len(temp_files)}.bin")
            with open(temp_file, "wb") as f:
                for grid in chunk:
                    f.write(struct.pack("4I", *grid))
            temp_files.append(temp_file)

        return temp_files

    def merge_sorted_chunks(self, temp_files, output_file):
        with open(output_file, "wb") as f_out:
            sorted_streams = [self._read_binary_file(file) for file in temp_files]
            for grid in heapq.merge(*sorted_streams, key=lambda grid: (*self._calculate_haar_coefficients(np.array(grid)), grid)):
                f_out.write(struct.pack("4I", *grid))
        self.sorted_file = output_file
        for file in temp_files:
            os.remove(file)

    def _read_binary_file(self, file_path):
        with open(file_path, "rb") as f:
            while True:
                data = f.read(16)  # Each grid is 4 uint32s (4 x 4 bytes = 16 bytes)
                if not data:
                    break
                yield struct.unpack("4I", data)

    def grid_to_haar_sort_index(self, grid, table_file):
        with open(table_file, "rb") as f:
            index = 0
            while True:
                data = f.read(16)
                if not data:
                    break
                current_grid = struct.unpack("4I", data)
                if np.array_equal(current_grid, grid):
                    return index
                index += 1
        raise ValueError("Grid not found.")

    def haar_sort_index_to_grid(self, index, table_file):
        with open(table_file, "rb") as f:
            # Calculate the maximum valid index
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            max_index = file_size // 16 - 1

            if index < 0 or index > max_index:
                raise ValueError(f"Index out of range. Valid range: 0 to {max_index}.")

            # Seek to the correct position
            f.seek(index * 16)
            data = f.read(16)
            return struct.unpack("4I", data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haar Sorting Algorithm Tool")
    parser.add_argument("--bit_depth", type=int, required=False, help="Set the bit depth for the grids.")
    parser.add_argument("--generate", action="store_true", help="Generate and sort a Haar sort table.")
    parser.add_argument("--chunk_size", type=int, default=10**6, help="Set the chunk size for sorting.")
    parser.add_argument("--output", type=str, help="Output file for the Haar sort table.")
    parser.add_argument("--grid", nargs=4, type=int, help="Provide four pixel values to get the Haar sort index.")
    parser.add_argument("--index", type=int, help="Provide a Haar sort index to get the corresponding pixel values.")
    parser.add_argument("--table", type=str, help="Path to the Haar sort table file.")
    args = parser.parse_args()

    if args.generate:
        if not args.bit_depth or not args.output:
            print("--bit_depth and --output are required for table generation.")
            exit(1)
        table = HaarSortTable(args.bit_depth)
        temp_files = table.sort_and_save_chunks(chunk_size=args.chunk_size)
        table.merge_sorted_chunks(temp_files, args.output)
        print(f"Haar sort table generated and saved to {args.output}.")

    if args.grid:
        if not args.table:
            print("--table is required to convert a grid to an index.")
            exit(1)
        grid = np.array(args.grid, dtype=np.uint32)
        table = HaarSortTable(args.bit_depth or 8)  # Bit depth is not critical for this operation
        try:
            index = table.grid_to_haar_sort_index(grid, args.table)
            print(f"The Haar sort index for grid {grid.tolist()} is {index}.")
        except ValueError as e:
            print(e)

    if args.index is not None:
        if not args.table:
            print("--table is required to convert an index to a grid.")
            exit(1)
        table = HaarSortTable(args.bit_depth or 8)  # Bit depth is not critical for this operation
        try:
            grid = table.haar_sort_index_to_grid(args.index, args.table)
            print(f"The grid for Haar sort index {args.index} is {grid}.")
        except ValueError as e:
            print(e)

class TestHaarSorting(unittest.TestCase):
    def test_sort_and_merge(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        temp_files = table.sort_and_save_chunks(chunk_size=10)
        output_file = os.path.join(table.temp_dir, "test_sorted_table.bin")
        table.merge_sorted_chunks(temp_files, output_file)

        grids = []
        with open(output_file, "rb") as f:
            while True:
                data = f.read(16)
                if not data:
                    break
                grids.append(struct.unpack("4I", data))

        # Ensure all grids are sorted
        previous = None
        for grid in grids:
            if previous is not None:
                self.assertLessEqual(
                    (*table._calculate_haar_coefficients(np.array(previous)), previous),
                    (*table._calculate_haar_coefficients(np.array(grid)), grid)
                )
            previous = grid

        os.remove(output_file)

    def test_grid_to_index_and_back(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        temp_files = table.sort_and_save_chunks(chunk_size=10)
        output_file = os.path.join(table.temp_dir, "test_sorted_table.bin")
        table.merge_sorted_chunks(temp_files, output_file)

        test_grid = (0, 1, 2, 3)
        index = table.grid_to_haar_sort_index(test_grid, output_file)
        recovered_grid = table.haar_sort_index_to_grid(index, output_file)

        self.assertEqual(test_grid, recovered_grid)
        os.remove(output_file)

    def test_invalid_inputs(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth)
        temp_files = table.sort_and_save_chunks(chunk_size=10)
        output_file = os.path.join(table.temp_dir, "test_sorted_table.bin")
        table.merge_sorted_chunks(temp_files, output_file)

        with self.assertRaises(ValueError):
            table.grid_to_haar_sort_index((999, 999, 999, 999), output_file)

        with self.assertRaises(ValueError):
            table.haar_sort_index_to_grid(-1, output_file)

        with self.assertRaises(ValueError):
            table.haar_sort_index_to_grid(10**6, output_file)

        os.remove(output_file)

# To run tests, use: python -m unittest <script_name>.py
