import struct
import argparse

class HaarSortQuery:
    def __init__(self, bit_depth, table_name=None):
        self.bit_depth = bit_depth
        self.forward_table_file = f"{table_name}_grids.bin" if table_name else None
        self.reverse_table_file = f"{table_name}_index.bin" if table_name else None

    def _get_struct_format(self):
        if self.bit_depth == 2:
            return "B"  # 8-bit unsigned integer for each 2-bit grid entry
        elif self.bit_depth == 4:
            return "H"  # 16-bit unsigned integer for each 4-bit grid entry
        elif self.bit_depth == 8:
            return "I"  # 32-bit unsigned integer for each 8-bit grid entry
        else:
            raise ValueError("Unsupported bit depth. Only 2, 4, or 8 are allowed.")

    def _calculate_binary_position(self, grid):
        """Calculate the binary position of the grid in the file."""
        shift_amount = self.bit_depth  # Number of bits per value
        return sum((value << (shift_amount * i)) for i, value in enumerate(grid))

    def grid_to_haar_sort_index(self, grid):
        """Finds the Haar sort index for a given grid in O(1) time."""
        if not self.forward_table_file:
            raise ValueError("Forward table file is required for grid lookup.")

        struct_format = self._get_struct_format()
        entry_size = struct.calcsize(struct_format)

        binary_position = self._calculate_binary_position(grid)

        with open(self.forward_table_file, "rb") as f:
            f.seek(binary_position * entry_size)
            data = f.read(entry_size)
            if not data:
                raise ValueError(f"Grid {grid} not found in the forward table.")

        return struct.unpack(struct_format, data)[0]

    def haar_sort_index_to_grid(self, index):
        """Finds the grid corresponding to a Haar sort index in O(1) time."""
        if not self.reverse_table_file:
            raise ValueError("Reverse table file is required for index lookup.")

        struct_format = self._get_struct_format()
        entry_size = struct.calcsize(struct_format)

        with open(self.reverse_table_file, "rb") as f:
            f.seek(index * entry_size)
            data = f.read(entry_size)
            if not data:
                raise ValueError(f"Index {index} out of range.")

        packed_value = struct.unpack(struct_format, data)[0]

        # Unpack the value into a 4-element tuple
        grid = []
        mask = (1 << self.bit_depth) - 1
        for _ in range(4):
            grid.append(packed_value & mask)
            packed_value >>= self.bit_depth

        return tuple(reversed(grid))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Haar Sort Lookup Tables")
    parser.add_argument("--bit_depth", type=int, required=True, help="Bit depth for the grids.")
    parser.add_argument("--table_name", type=str, required=True, help="Base name of the Haar sort table files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--grid", type=int, nargs=4, help="Query the Haar sort index for a grid.")
    group.add_argument("--index", type=int, help="Query the grid for a Haar sort index.")
    args = parser.parse_args()

    query = HaarSortQuery(
        bit_depth=args.bit_depth,
        table_name=args.table_name
    )

    if args.grid:
        try:
            index = query.grid_to_haar_sort_index(args.grid)
            print(f"The Haar sort index for grid {args.grid} is {index}.")
        except ValueError as e:
            print(e)

    if args.index is not None:
        try:
            grid = query.haar_sort_index_to_grid(args.index)
            print(f"The grid for Haar sort index {args.index} is {grid}.")
        except ValueError as e:
            print(e)
