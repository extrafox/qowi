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
        # Determine the struct format based on pixel bit depth
        if self.bit_depth == 2:
            return "4B"  # 1 byte for 4 pixels, each 2 bits wide
        elif self.bit_depth == 4:
            return "4H"  # 2 bytes for 4 pixels, each 4 bits wide
        elif self.bit_depth == 8:
            return "4I"  # 4 bytes for 4 pixels, each 8 bits wide
        else:
            raise ValueError("Unsupported bit depth: must be 2, 4, or 8.")

    def _calculate_haar_coefficients(self, grid):
        a, b, c, d = grid.astype(np.int32)
        LL = a + b + c + d
        HL = a - b + c - d
        LH = a + b - c - d
        HH = a - b - c + d
        return LL, HL, LH, HH

    def _binary_position(self, grid):
        """Calculate the binary position for a grid."""
        return (
            grid[0] << (3 * self.bit_depth) |
            grid[1] << (2 * self.bit_depth) |
            grid[2] << self.bit_depth |
            grid[3]
        )

    def _convert_haar_coefficients_to_index(self, coefficients):
        """Convert (LL, HL, LH, HH) coefficients to a single Haar sort index."""
        bit_depth = self.bit_depth
        LL_index = coefficients[0] & ((1 << bit_depth) - 1)
        HL_index = coefficients[1] & ((1 << bit_depth) - 1)
        LH_index = coefficients[2] & ((1 << bit_depth) - 1)
        HH_index = coefficients[3] & ((1 << bit_depth) - 1)

        return (
            (LL_index << (3 * bit_depth)) |
            (HL_index << (2 * bit_depth)) |
            (LH_index << bit_depth) |
            HH_index
        )

    def _convert_index_to_haar_coefficients(self, index):
        """Convert a Haar sort index back to (LL, HL, LH, HH) coefficients."""
        bit_depth = self.bit_depth
        mask = (1 << bit_depth) - 1

        LL_index = (index >> (3 * bit_depth)) & mask
        HL_index = (index >> (2 * bit_depth)) & mask
        LH_index = (index >> bit_depth) & mask
        HH_index = index & mask

        return LL_index, HL_index, LH_index, HH_index

    def sort_and_save_chunks(self, chunk_size=10**6):
        if self.bit_depth is None:
            raise ValueError("Bit depth must be set for generating the table.")

        max_value = (1 << self.bit_depth) - 1
        all_grids = product(range(max_value + 1), repeat=4)
        temp_files = []
        struct_format = self._get_struct_format()

        while True:
            chunk = list(islice(all_grids, chunk_size))
            if not chunk:
                break
            chunk = sorted(chunk, key=lambda grid: (*self._calculate_haar_coefficients(np.array(grid)), grid))
            temp_file = os.path.join(self.temp_dir, f"sorted_chunk_{len(temp_files)}.bin")
            with open(temp_file, "wb") as f:
                for grid in chunk:
                    packed_grid = self._pack_grid(grid, struct_format)
                    f.write(packed_grid)
            temp_files.append(temp_file)

        return temp_files

    def merge_sorted_chunks(self, temp_files, table_file):
        total_grids = sum(os.path.getsize(file) // struct.calcsize(self._get_struct_format()) for file in temp_files)

        with open(table_file, "wb") as f_out:
            sorted_streams = [self._read_binary_file(file) for file in temp_files]
            struct_format = self._get_struct_format()

            with tqdm(total=total_grids, desc="Merging and Writing Main Table") as pbar:
                for grid in heapq.merge(*sorted_streams, key=lambda grid: (*self._calculate_haar_coefficients(np.array(grid)), grid)):
                    packed_grid = self._pack_grid(grid, struct_format)
                    f_out.write(packed_grid)
                    pbar.update(1)

        self.sorted_file = table_file
        for file in temp_files:
            os.remove(file)

    def generate_reverse_lookup_table(self, table_file, reverse_table_file):
        struct_format = self._get_struct_format()
        entry_size = struct.calcsize(struct_format)

        with open(table_file, "rb") as f_table, open(reverse_table_file, "wb") as f_reverse:
            while True:
                data = f_table.read(entry_size)  # Read one grid
                if not data:
                    break
                grid = self._unpack_grid(data, struct_format)
                haar_coefficients = self._calculate_haar_coefficients(np.array(grid))
                # Decompose Haar sort index into 4 components
                LL_index, HL_index, LH_index, HH_index = self._convert_index_to_haar_coefficients(
                    self._convert_haar_coefficients_to_index(haar_coefficients)
                )
                packed_index = struct.pack(struct_format, LL_index, HL_index, LH_index, HH_index)
                f_reverse.write(packed_index)

    def _pack_grid(self, grid, struct_format):
        # Packs a grid based on the bit depth format
        max_value = (1 << self.bit_depth) - 1
        normalized_grid = [min(max_value, val) for val in grid]  # Ensure values fit in bit depth
        return struct.pack(struct_format, *normalized_grid)

    def _unpack_grid(self, packed_grid, struct_format):
        # Unpacks a grid based on the bit depth format
        return struct.unpack(struct_format, packed_grid)

    def _read_binary_file(self, file_path):
        struct_format = self._get_struct_format()
        with open(file_path, "rb") as f:
            while True:
                data = f.read(struct.calcsize(struct_format))
                if not data:
                    break
                yield self._unpack_grid(data, struct_format)

    def grid_to_haar_sort_index(self, grid, table_file):
        """Convert a grid to its Haar sort index."""
        struct_format = self._get_struct_format()
        binary_position = self._binary_position(grid)
        entry_size = struct.calcsize(struct_format)

        with open(table_file, "rb") as f:
            f.seek(binary_position * entry_size)
            data = f.read(entry_size)
            if not data:
                raise ValueError(f"Grid {grid} not found in the forward table.")
            found_grid = self._unpack_grid(data, struct_format)
            if tuple(found_grid) != tuple(grid):
                raise ValueError(f"Grid {grid} does not match the binary position in the table.")
            return binary_position

    def haar_sort_index_to_grid(self, index, table_file):
        """Convert a Haar sort index to its corresponding grid."""
        struct_format = self._get_struct_format()
        entry_size = struct.calcsize(struct_format)

        with open(table_file, "rb") as f:
            f.seek(index * entry_size)
            data = f.read(entry_size)
            if len(data) < entry_size:
                raise ValueError("Index out of range.")
            return self._unpack_grid(data, struct_format)

    def validate_table_sizes(self, forward_table, reverse_table_file):
        grid_size = os.path.getsize(forward_table)
        reverse_size = os.path.getsize(reverse_table_file)

        if grid_size != reverse_size:
            raise ValueError(
                f"Size mismatch: Forward table ({grid_size} bytes) does not match "
                f"Reverse table ({reverse_size} bytes)."
            )
        print(f"Validation passed: Both tables are consistent in size ({grid_size} bytes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haar Sorting Algorithm Tool")
    parser.add_argument("--generate", action="store_true", help="Generate and sort a Haar sort table.")
    parser.add_argument("--bit_depth", type=int, help="Set the bit depth for the grids (required for generation).")
    parser.add_argument("--chunk_size", type=int, default=10**6, help="Set the chunk size for sorting.")
    parser.add_argument("-t", "--table", type=str, help="Table prefix for Haar sort files.")
    parser.add_argument("-g", "--grid", nargs=4, type=int, help="Provide four pixel values to get the Haar sort index.")
    parser.add_argument("-i", "--index", type=int, help="Provide a Haar sort index to get the corresponding pixel values.")
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

    if args.grid:
        if not args.table:
            print("--table is required to convert a grid to an index.")
            exit(1)
        grid = np.array(args.grid, dtype=np.int32)  # Convert grid to match signed integer indices
        table = HaarSortTable(bit_depth=args.bit_depth)  # Bit depth is required for lookups
        forward_table = f"{args.table}_grids.bin"  # Use forward table
        try:
            index = table.grid_to_haar_sort_index(grid, forward_table)
            print(f"The Haar sort index for grid {grid.tolist()} is {index}.")
        except ValueError as e:
            print(e)

    if args.index is not None:
        if not args.table:
            print("--table is required to convert an index to a grid.")
            exit(1)
        table = HaarSortTable(bit_depth=args.bit_depth)  # Bit depth is required for lookups
        forward_table = f"{args.table}_grids.bin"
        try:
            grid = table.haar_sort_index_to_grid(args.index, forward_table)
            print(f"The grid for Haar sort index {args.index} is {grid}.")
        except ValueError as e:
            print(e)
