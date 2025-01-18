import numpy as np
import os
import tempfile
import argparse
from itertools import islice, product
import heapq
import struct
import unittest
from tqdm import tqdm

class HaarSortTable:
    def __init__(self, bit_depth=None, temp_dir=None):
        self.bit_depth = bit_depth
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.sorted_file = None

    def _get_struct_format(self):
        if self.bit_depth <= 2:
            return "B"  # 8-bit unsigned integer for 2-bit grids
        elif self.bit_depth <= 4:
            return "H"  # 16-bit unsigned integer for 4-bit grids
        elif self.bit_depth <= 8:
            return "I"  # 32-bit unsigned integer for 8-bit grids
        else:
            raise ValueError("Unsupported bit depth: must be 2, 4, or 8 bits.")

    def _grid_to_index(self, grid):
        """Convert a grid into a packed index based on the bit depth."""
        struct_format = self._get_struct_format()
        bit_depth = struct.calcsize(struct_format) * 8
        index = 0
        for value in grid:
            index = (index << (bit_depth // 4)) | value
        return index

    def _index_to_grid(self, index):
        """Convert a packed index back into a grid based on the bit depth."""
        struct_format = self._get_struct_format()
        bit_depth = struct.calcsize(struct_format) * 8
        grid = []
        for _ in range(4):
            grid.append(index & ((1 << (bit_depth // 4)) - 1))
            index >>= (bit_depth // 4)
        return tuple(reversed(grid))

    def _calculate_haar_coefficients(self, grid):
        a, b, c, d = grid.astype(np.int32)
        LL = a + b + c + d
        HL = a - b + c - d
        LH = a + b - c - d
        HH = a - b - c + d
        return LL, HL, LH, HH

    def _haar_sort_key(self, grid):
        """Generate a Haar Sort key for the given grid."""
        coefficients = self._calculate_haar_coefficients(np.array(grid))
        LL, HL, LH, HH = coefficients
        return (LL << 24) | ((HL & 0xFF) << 16) | ((LH & 0xFF) << 8) | (HH & 0xFF)

    def sort_and_save_chunks(self, chunk_size=10**6):
        if self.bit_depth is None:
            raise ValueError("Bit depth must be set for generating the table.")

        max_value = (1 << self.bit_depth) - 1
        total_grids = (max_value + 1) ** 4
        all_grids = product(range(max_value + 1), repeat=4)
        temp_files = []
        struct_format = self._get_struct_format()

        with tqdm(total=total_grids, desc="Sorting and Saving Chunks") as pbar:
            while True:
                chunk = list(islice(all_grids, chunk_size))
                if not chunk:
                    break
                chunk = sorted(chunk, key=lambda grid: self._haar_sort_key(grid))
                temp_file = os.path.join(self.temp_dir, f"sorted_chunk_{len(temp_files)}.bin")
                with open(temp_file, "wb") as f:
                    for grid in chunk:
                        index = self._grid_to_index(grid)
                        f.write(struct.pack(struct_format, index))
                temp_files.append(temp_file)
                pbar.update(len(chunk))

        return temp_files

    def merge_sorted_chunks(self, temp_files, table_file):
        struct_format = self._get_struct_format()
        total_grids = sum(os.path.getsize(file) // struct.calcsize(struct_format) for file in temp_files)

        def stream_file(file):
            with open(file, "rb") as f:
                while True:
                    data = f.read(struct.calcsize(struct_format))
                    if not data:
                        break
                    yield struct.unpack(struct_format, data)[0]

        with open(table_file, "wb") as f_out:
            with tqdm(total=total_grids, desc="Merging and Writing Main Table") as pbar:
                for index in heapq.merge(*(stream_file(f) for f in temp_files)):
                    f_out.write(struct.pack(struct_format, index))
                    pbar.update(1)

        for file in temp_files:
            os.remove(file)

    def generate_reverse_lookup_table(self, table_file, reverse_table_file):
        struct_format = self._get_struct_format()
        entry_size = struct.calcsize(struct_format)

        with open(table_file, "rb") as f_table, open(reverse_table_file, "wb") as f_reverse:
            index = 0
            while True:
                data = f_table.read(entry_size)
                if not data:
                    break
                f_reverse.write(struct.pack(struct_format, index))
                index += 1

    def validate_table_sizes(self, forward_table, reverse_table_file):
        grid_size = os.path.getsize(forward_table)
        reverse_size = os.path.getsize(reverse_table_file)

        if grid_size != reverse_size:
            raise ValueError(
                f"Size mismatch: Forward table ({grid_size} bytes) does not match "
                f"Reverse table ({reverse_size} bytes)."
            )
        print(f"Validation passed: Both tables are consistent in size ({grid_size} bytes).")

class TestHaarSortTable(unittest.TestCase):
    def test_round_trip_2bit(self):
        bit_depth = 2
        table = HaarSortTable(bit_depth=bit_depth)
        max_value = (1 << bit_depth) - 1

        for grid in product(range(max_value + 1), repeat=4):
            index = table._grid_to_index(grid)
            reverse_grid = table._index_to_grid(index)
            self.assertEqual(grid, reverse_grid, f"Mismatch for grid {grid}")

    def test_round_trip_4bit(self):
        bit_depth = 4
        table = HaarSortTable(bit_depth=bit_depth)
        max_value = (1 << bit_depth) - 1

        for grid in product(range(max_value + 1), repeat=4):
            index = table._grid_to_index(grid)
            reverse_grid = table._index_to_grid(index)
            self.assertEqual(grid, reverse_grid, f"Mismatch for grid {grid}")

    def test_table_file_sizes(self):
        bit_depths = [2, 4]
        for bit_depth in bit_depths:
            table = HaarSortTable(bit_depth=bit_depth)
            forward_table = tempfile.NamedTemporaryFile(delete=False)
            reverse_table = tempfile.NamedTemporaryFile(delete=False)

            try:
                max_value = (1 << bit_depth) - 1
                total_entries = (max_value + 1) ** 4

                with open(forward_table.name, "wb") as f:
                    for index in range(total_entries):
                        f.write(struct.pack(table._get_struct_format(), index))

                table.generate_reverse_lookup_table(forward_table.name, reverse_table.name)

                forward_size = os.path.getsize(forward_table.name)
                reverse_size = os.path.getsize(reverse_table.name)
                expected_size = total_entries * struct.calcsize(table._get_struct_format())

                self.assertEqual(forward_size, expected_size, "Forward table size mismatch")
                self.assertEqual(reverse_size, expected_size, "Reverse table size mismatch")
            finally:
                os.unlink(forward_table.name)
                os.unlink(reverse_table.name)

    def test_struct_format_sizes(self):
        bit_depth_to_format = {
            2: "B",  # 8 bits
            4: "H",  # 16 bits
            8: "I",  # 32 bits
        }
        for bit_depth, expected_format in bit_depth_to_format.items():
            table = HaarSortTable(bit_depth=bit_depth)
            struct_format = table._get_struct_format()
            self.assertEqual(struct_format, expected_format, f"Incorrect struct format for bit depth {bit_depth}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haar Sorting Algorithm Tool")
    parser.add_argument("--generate", action="store_true", help="Generate and sort a Haar sort table.")
    parser.add_argument("--bit_depth", type=int, help="Set the bit depth for the grids (required for generation).")
    parser.add_argument("--chunk_size", type=int, default=250_000_000, help="Set the chunk size for sorting.")
    parser.add_argument("-t", "--table", type=str, help="Table prefix for Haar sort files.")
    args = parser.parse_args()

    if args.generate:
        if not args.bit_depth or not args.table:
            print("--bit_depth and --table are required for table generation.")
            exit(1)
        table = HaarSortTable(bit_depth=args.bit_depth)
        temp_files = table.sort_and_save_chunks(chunk_size=args.chunk_size)
        forward_table = f"{args.table}_grids.bin"
        reverse_table = f"{args.table}_index.bin"
        table.merge_sorted_chunks(temp_files, forward_table)
        table.generate_reverse_lookup_table(forward_table, reverse_table)
        table.validate_table_sizes(forward_table, reverse_table)
        print(f"Haar sort table generated and saved to {forward_table}. Reverse lookup table saved to {reverse_table}.")
