import numpy as np
import os
import tempfile
import argparse
from itertools import islice, product
import struct
from tqdm import tqdm
import heapq
import qowi.grids as grids

DEFAULT_CHUNK_SIZE = 5_000_000

class HaarSortTable:
    def __init__(self, pixel_bit_depth=None, temp_dir=None):
        grids.validate_bit_depth(pixel_bit_depth)
        self.pixel_bit_depth = pixel_bit_depth
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.struct_format = grids.get_struct_format(self.pixel_bit_depth)
        self.entry_size = struct.calcsize(self.struct_format)

    def open_binary_file(self, filepath, mode):
        return open(filepath, mode)

    def stream_file(self, file):
        with self.open_binary_file(file, "rb") as f:
            while chunk := f.read(self.entry_size):
                yield struct.unpack(self.struct_format, chunk)[0]

    def generate_all_grids(self):
        max_value = (1 << self.pixel_bit_depth) - 1
        return product(range(max_value + 1), repeat=4)

    def calculate_and_sort_keys(self, grids):
        grids = np.array(grids, dtype=np.uint8)
        coefficients = grids.calculate_haar_coefficients(grids)
        keys = grids.calculate_haar_keys(coefficients)
        sorted_indices = np.argsort(keys)
        return grids[sorted_indices]

    def sort_and_save_chunks(self, chunk_size=DEFAULT_CHUNK_SIZE):
        if not self.pixel_bit_depth:
            raise ValueError("Bit depth must be set.")

        all_grids = self.generate_all_grids()
        temp_files = []
        total_grids = (1 << self.pixel_bit_depth) ** 4

        with tqdm(total=total_grids, desc="Sorting and Saving Chunks") as pbar:
            while True:
                chunk = list(islice(all_grids, chunk_size))
                if not chunk:
                    break

                sorted_chunk = self.calculate_and_sort_keys(chunk)
                indices = grids.grids_to_indices(sorted_chunk, self.pixel_bit_depth)

                temp_file = os.path.join(self.temp_dir, f"sorted_chunk_{len(temp_files)}.bin")
                with self.open_binary_file(temp_file, "wb") as f:
                    f.write(struct.pack(f"{len(indices)}{self.struct_format}", *indices))
                temp_files.append(temp_file)
                pbar.update(len(chunk))

        return temp_files

    def merge_sorted_chunks(self, temp_files, table_file):
        total_entries = sum(os.path.getsize(f) // self.entry_size for f in temp_files)
        with self.open_binary_file(table_file, "wb") as f_out:
            with tqdm(total=total_entries, desc="Merging and Writing Main Table") as pbar:
                streams = [self.stream_file(f) for f in temp_files]
                for index in heapq.merge(*streams):
                    f_out.write(struct.pack(self.struct_format, index))
                    pbar.update(1)

        for file in temp_files:
            os.remove(file)

    def generate_reverse_lookup_table(self, table_file, reverse_table_file):
        total_entries = os.path.getsize(table_file) // self.entry_size

        with tqdm(total=total_entries, desc="Initializing Reverse Lookup Table") as pbar:
            with self.open_binary_file(reverse_table_file, "wb") as f_reverse:
                for _ in range(total_entries):
                    f_reverse.write(b"\x00" * self.entry_size)
                    pbar.update(1)

        with self.open_binary_file(table_file, "rb") as f_table, self.open_binary_file(reverse_table_file, "r+b") as f_reverse:
            with tqdm(total=total_entries, desc="Generating Reverse Lookup Table") as pbar:
                for position, data in enumerate(iter(lambda: f_table.read(self.entry_size), b"")):
                    index = struct.unpack(self.struct_format, data)[0]
                    f_reverse.seek(index * self.entry_size)
                    f_reverse.write(struct.pack(self.struct_format, position))
                    pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haar Sorting Algorithm Tool")
    parser.add_argument("--generate-forward", action="store_true", help="Generate and sort a Haar sort forward table.")
    parser.add_argument("--generate-reverse", action="store_true", help="Generate a Haar sort reverse lookup table.")
    parser.add_argument("--generate", action="store_true", help="Generate both forward and reverse Haar sort tables.")
    parser.add_argument("--bit_depth", type=int, required=True, help="Set the bit depth for the grids.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Set the chunk size for sorting.")
    parser.add_argument("-t", "--table", type=str, required=True, help="Table prefix for Haar sort files.")
    args = parser.parse_args()

    table = HaarSortTable(pixel_bit_depth=args.bit_depth)

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
