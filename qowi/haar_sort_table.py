import qowi.grids as grids
import struct
import numpy as np

class HaarSortTable:
    def __init__(self, pixel_bit_depth: int, table_name: str):
        self.pixel_bit_depth = pixel_bit_depth
        grids.validate_bit_depth(pixel_bit_depth)
        self.forward_table_file = f"{table_name}_grids.bin"
        self.reverse_table_file = f"{table_name}_index.bin"
        self._struct_format = grids.get_struct_format(pixel_bit_depth)

    def _lookup_index(self, key: int, table_file: str) -> int:
        try:
            with open(table_file, "rb") as f:
                entry_size = struct.calcsize(self._struct_format)
                f.seek(key * entry_size)
                data = f.read(entry_size)
                if not data:
                    raise ValueError(f"Key {key} not found in the table {table_file}.")
        except FileNotFoundError:
            raise ValueError(f"Table file {table_file} not found.")

        return struct.unpack(self._struct_format, data)[0]

    def pixels_to_haar_sort_index(self, pixels: np.ndarray) -> int:
        pixel_index = grids.grid_to_index(pixels, self.pixel_bit_depth)
        return self._lookup_index(pixel_index, self.forward_table_file)

    def haar_sort_index_to_pixels(self, wavelet_index: int) -> np.ndarray:
        pixel_index = self._lookup_index(wavelet_index, self.reverse_table_file)
        return grids.index_to_grid(pixel_index, self.pixel_bit_depth).astype(np.uint8)

    def pixels_to_wavelet(self, pixels: np.ndarray) -> np.ndarray:
        pixel_index = self.pixels_to_haar_sort_index(pixels)
        return grids.index_to_grid(pixel_index, self.pixel_bit_depth).astype(np.uint8)

    def wavelet_to_pixels(self, wavelet: np.ndarray) -> np.ndarray:
        wavelet_index = grids.grid_to_index(wavelet, self.pixel_bit_depth)
        return self.haar_sort_index_to_pixels(wavelet_index).astype(np.uint8)

