import struct

from typing import List, Tuple


class HaarSortTable:
    def __init__(self, bit_depth: int, table_name: str = None):
        """
        Initializes the HaarSortTable with a given bit depth and optional table name.

        Args:
            bit_depth (int): The bit depth of the grid values. Supported values are 2, 4, or 8.
            table_name (str, optional): Base name for the forward and reverse table files. Defaults to None.
        """
        self.bit_depth = bit_depth
        self.validate_bit_depth()
        self.forward_table_file = f"{table_name}_grids.bin" if table_name else "default_grids.bin"
        self.reverse_table_file = f"{table_name}_index.bin" if table_name else "default_index.bin"
        self._struct_format = self._get_struct_format()

    def _get_struct_format(self) -> str:
        """
        Determines the struct format string based on the bit depth.

        Returns:
            str: The struct format string.
        """
        if self.bit_depth == 2:
            return "B"  # 8-bit unsigned integer for each 2-bit grid entry
        elif self.bit_depth == 4:
            return "H"  # 16-bit unsigned integer for each 4-bit grid entry
        elif self.bit_depth == 8:
            return "I"  # 32-bit unsigned integer for each 8-bit grid entry
        else:
            raise ValueError("Unsupported bit depth. Only 2, 4, or 8 are allowed. Please provide a valid bit depth.")

    def _calculate_binary_position(self, grid: Tuple[int, int, int, int]) -> int:
        """
        Calculate the binary position of the grid in the file.

        Args:
            grid (list[int]): A list of integers representing the grid values.

        Returns:
            int: The binary position in the table.

        Raises:
            ValueError: If the grid does not have exactly 4 elements or contains out-of-range values.
        """
        shift_amount = self.bit_depth  # Number of bits per value
        if len(grid) != 4:
            raise ValueError("Grid must have exactly 4 elements.")
        if any(value < 0 or value >= (1 << self.bit_depth) for value in grid):
            raise ValueError(f"Grid values must be within the range [0, {(1 << self.bit_depth) - 1}].")
        return sum((value << (shift_amount * i)) for i, value in enumerate(grid))

    def grid_to_haar_sort_index(self, grid: Tuple[int, int, int, int]) -> int:
        """
        Finds the Haar sort index for a given grid in O(1) time.

        Args:
            grid (list[int]): A list of integers representing the grid values.

        Returns:
            int: The Haar sort index corresponding to the grid.

        Raises:
            ValueError: If the forward table file is missing or the grid is not found.
        """
        if not self.forward_table_file:
            raise ValueError("Forward table file is required for grid lookup.")

        binary_position = self._calculate_binary_position(grid)

        try:
            with open(self.forward_table_file, "rb") as f:
                entry_size = struct.calcsize(self._struct_format)
                f.seek(binary_position * entry_size)
                data = f.read(entry_size)
                if not data:
                    raise ValueError(f"Grid {grid} not found in the forward table.")
        except FileNotFoundError:
            raise ValueError(f"Forward table file {self.forward_table_file} not found.")

        return struct.unpack(self._struct_format, data)[0]

    def haar_sort_index_to_grid(self, index: int) -> Tuple[int, int, int, int]:
        """
        Finds the grid corresponding to a Haar sort index in O(1) time.

        Args:
            index (int): The Haar sort index.

        Returns:
            tuple[int, int, int, int]: The grid values corresponding to the index.

        Raises:
            ValueError: If the reverse table file is missing or the index is out of range.
        """
        if not self.reverse_table_file:
            raise ValueError("Reverse table file is required for index lookup.")

        try:
            with open(self.reverse_table_file, "rb") as f:
                entry_size = struct.calcsize(self._struct_format)
                f.seek(index * entry_size)
                data = f.read(entry_size)
                if not data:
                    raise ValueError(f"Index {index} out of range.")
        except FileNotFoundError:
            raise ValueError(f"Reverse table file {self.reverse_table_file} not found.")

        packed_value = struct.unpack(self._struct_format, data)[0]

        # Unpack the value into a 4-element tuple
        grid = []
        mask = (1 << self.bit_depth) - 1
        for _ in range(4):
            grid.append(packed_value & mask)
            packed_value >>= self.bit_depth

        return tuple(reversed(grid))

    def grid_to_haar_sort_components(self, grid: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Converts a grid to Haar sort components (LL_index, HL_index, LH_index, HH_index).

        Args:
            grid (list[int]): A list of integers representing the grid values.

        Returns:
            tuple[int, int, int, int]: The Haar sort components (LL_index, HL_index, LH_index, HH_index).
        """
        index = self.grid_to_haar_sort_index(grid)
        LL_index = index // 4
        HL_index = (index % 4) // 2
        LH_index = (index % 2)
        HH_index = 0  # HH_index is redundant as it's always zero in this logic
        return LL_index, HL_index, LH_index, HH_index

    def haar_sort_components_to_grid(self, LL_index: int, HL_index: int, LH_index: int, HH_index: int) -> Tuple[int, int, int, int]:
        """
        Converts Haar sort components back to the grid.

        Args:
            LL_index (int): LL component index.
            HL_index (int): HL component index.
            LH_index (int): LH component index.
            HH_index (int): HH component index (expected to be 0).

        Returns:
            tuple[int, int, int, int]: The grid corresponding to the provided components.

        Raises:
            ValueError: If any component index is out of range.
        """
        max_index = (1 << self.bit_depth) - 1
        if HH_index != 0:
            raise ValueError("HH_index must be 0.")

        index = LL_index * 4 + HL_index * 2 + LH_index
        return self.haar_sort_index_to_grid(index)

    def validate_bit_depth(self) -> None:
        """
        Validates the provided bit depth and raises an error if it is unsupported.

        Raises:
            ValueError: If the bit depth is not supported.
        """
        supported_bit_depths = [2, 4, 8]
        if self.bit_depth not in supported_bit_depths:
            raise ValueError(
                f"Unsupported bit depth: {self.bit_depth}. Supported bit depths are {supported_bit_depths}."
            )
