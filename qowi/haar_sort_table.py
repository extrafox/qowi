import struct
import numpy as np
import qowi.grids as grids


class HaarSortTable:
    def __init__(self, pixel_bit_depth: int, table_name: str):
        self.pixel_bit_depth = pixel_bit_depth
        grids.validate_bit_depth(pixel_bit_depth)
        self.forward_table_file = f"{table_name}_grids.bin"
        self.reverse_table_file = f"{table_name}_index.bin"
        self._struct_format = grids.get_struct_format(pixel_bit_depth)

    def _lookup_index(self, keys: np.ndarray, table_file: str) -> np.ndarray:
        try:
            with open(table_file, "rb") as f:
                entry_size = struct.calcsize(self._struct_format)
                results = np.empty(len(keys), dtype=np.uint32)  # Ensure uint32

                for idx, key in enumerate(keys):
                    f.seek(key * entry_size)
                    data = f.read(entry_size)
                    if not data:
                        raise ValueError(f"Key {key} not found in {table_file}.")
                    results[idx] = struct.unpack(self._struct_format, data)[0]

        except FileNotFoundError:
            raise ValueError(f"Table file {table_file} not found.")

        return results.astype(np.uint32)  # Ensure uint32 before returning

    def pixels_to_haar_sort_indices(self, pixels: np.ndarray) -> np.ndarray:
        pixel_indices = grids.grids_to_indices(pixels, self.pixel_bit_depth)
        return self._lookup_index(pixel_indices, self.forward_table_file)

    def haar_sort_indices_to_pixels(self, wavelet_indices: np.ndarray) -> np.ndarray:
        pixel_indices = self._lookup_index(wavelet_indices, self.reverse_table_file)
        return grids.indices_to_grids(pixel_indices, self.pixel_bit_depth).astype(np.uint8)

    def pixels_to_wavelets(self, pixels: np.ndarray) -> np.ndarray:
        pixel_indices = self.pixels_to_haar_sort_indices(pixels)
        return grids.indices_to_grids(pixel_indices, self.pixel_bit_depth).astype(np.uint8)

    def wavelets_to_pixels(self, wavelet: np.ndarray) -> np.ndarray:
        wavelet_indices = grids.grids_to_indices(wavelet, self.pixel_bit_depth)
        return self.haar_sort_indices_to_pixels(wavelet_indices).astype(np.uint8)
