import numpy as np
import os
import tempfile
import argparse
from itertools import islice, product
import struct
from tqdm import tqdm

class HaarSortTable:
    def __init__(self, bit_depth=None, temp_dir=None):
        if bit_depth not in {2, 4, 8}:
            raise ValueError("bit_depth must be precisely 2, 4, or 8.")
        self.bit_depth = bit_depth
        self.temp_dir = temp_dir or tempfile.gettempdir()

    def _get_struct_format(self):
        if self.bit_depth == 2:
            return "B"
        elif self.bit_depth == 4:
            return "H"
        elif self.bit_depth == 8:
            return "I"

    def _calculate_haar_coefficients_vectorized(self, grids):
        grids = np.array(grids, dtype=np.int16)
        LL = np.sum(grids, axis=1)
        HL = grids[:, 0] - grids[:, 1] + grids[:, 2] - grids[:, 3]
        LH = grids[:, 0] + grids[:, 1] - grids[:, 2] - grids[:, 3]
        HH = grids[:, 0] - grids[:, 1] - grids[:, 2] + grids[:, 3]
        return LL, HL, LH, HH

    def _grid_to_index_vectorized(self, grids):
        grids = np.array(grids, dtype=np.uint8)
        shifts = np.array([3, 2, 1, 0]) * (self.bit_depth // 4)
        return np.sum(grids << shifts, axis=1)

    def sort_and_save_chunks(self, chunk_size=10**6):
        if not self.bit_depth:
            raise ValueError("Bit depth must be set.")

        max_value = (1 << self.bit_depth) - 1
        total_grids = (max_value + 1) ** 4
        all_grids = product(range(max_value + 1), repeat=4)  # Streamed grid generation
        struct_format = self._get_struct_format()

        temp_files = []
        with tqdm(total=total_grids, desc="Sorting and Saving Chunks") as pbar:
            while True:
                chunk = list(islice(all_grids, chunk_size))
                if not chunk:
                    break

                chunk = np.array(chunk, dtype=np.uint8)
                LL, HL, LH, HH = self._calculate_haar_coefficients_vectorized(chunk)
                keys = (LL << 24) | ((HL & 0xFF) << 16) | ((LH & 0xFF) << 8) | (HH & 0xFF)
                sorted_indices = np.argsort(keys)
                sorted_chunk = chunk[sorted_indices]
                indices = self._grid_to_index_vectorized(sorted_chunk)

                temp_file = os.path.join(self.temp_dir, f"sorted_chunk_{len(temp_files)}.bin")
                with open(temp_file, "wb") as f:
                    f.write(struct.pack(f"{len(indices)}{struct_format}", *indices))
                temp_files.append(temp_file)
                pbar.update(len(chunk))

        return temp_files

    def merge_sorted_chunks(self, temp_files, table_file):
        struct_format = self._get_struct_format()

        def stream_file(file):
            with open(file, "rb") as f:
                while chunk := f.read(struct.calcsize(struct_format)):
                    yield struct.unpack(struct_format, chunk)[0]

        with open(table_file, "wb") as f_out:
            with tqdm(desc="Merging and Writing Main Table") as pbar:
                streams = [stream_file(f) for f in temp_files]
                for index in heapq.merge(*streams):
                    f_out.write(struct.pack(struct_format, index))
                    pbar.update(1)

        for file in temp_files:
            os.remove(file)

    def generate_reverse_lookup_table(self, table_file, reverse_table_file):
        struct_format = self._get_struct_format()
        entry_size = struct.calcsize(struct_format)
        total_entries = os.path.getsize(table_file) // entry_size

        with open(reverse_table_file, "wb") as f_reverse:
            f_reverse.write(b"\x00" * total_entries * entry_size)

        with open(table_file, "rb") as f_table, open(reverse_table_file, "r+b") as f_reverse:
            with tqdm(total=total_entries, desc="Generating Reverse Lookup Table") as pbar:
                for position, data in enumerate(iter(lambda: f_table.read(entry_size), b"")):
                    index = struct.unpack(struct_format, data)[0]
                    f_reverse.seek(index * entry_size)
                    f_reverse.write(struct.pack(struct_format, position))
                    pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haar Sorting Algorithm Tool")
    parser.add_argument("--generate-forward", action="store_true", help="Generate and sort a Haar sort forward table.")
    parser.add_argument("--generate-reverse", action="store_true", help="Generate a Haar sort reverse lookup table.")
    parser.add_argument("--generate", action="store_true", help="Generate both forward and reverse Haar sort tables.")
    parser.add_argument("--bit_depth", type=int, required=True, help="Set the bit depth for the grids.")
    parser.add_argument("--chunk_size", type=int, default=5_000_000, help="Set the chunk size for sorting.")
    parser.add_argument("-t", "--table", type=str, required=True, help="Table prefix for Haar sort files.")
    args = parser.parse_args()

    table = HaarSortTable(bit_depth=args.bit_depth)

    if args.generate_forward or args.generate:
        temp_files = table.sort_and_save_chunks(args.chunk_size)
        forward_table = f"{args.table}_grids.bin"
        table.merge_sorted_chunks(temp_files, forward_table)
        print(f"Forward Haar sort table generated at {forward_table}.")

    if args.generate_reverse or args.generate:
        reverse_table = f"{args.table}_index.bin"
        forward_table = f"{args.table}_grids.bin"
        table.generate_reverse_lookup_table(forward_table, reverse_table)
        print(f"Reverse lookup table generated at {reverse_table}.")

