import os
import struct
import tempfile
import numpy as np
from itertools import product
from tqdm import tqdm
import heapq
import argparse
import qowi.grids as grids

DEFAULT_CHUNK_SIZE = 5_000_000

class HaarSortGenerator:
    def __init__(self, pixel_bit_depth=None, temp_dir=None, table_name=None):
        grids.validate_bit_depth(pixel_bit_depth)
        self.pixel_bit_depth = pixel_bit_depth
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.table_name = table_name
        self.struct_format = grids.get_struct_format(pixel_bit_depth)
        self.entry_size = struct.calcsize(self.struct_format)

    def stream_file(self, file):
        with open(file, "rb") as f:
            while chunk := f.read(self.entry_size):
                yield struct.unpack(self.struct_format, chunk)[0]

    def generate_all_grids_to_file(self, output_file):
        max_value = (1 << self.pixel_bit_depth) - 1
        total_grids = (max_value + 1) ** 4

        with open(output_file, "wb") as f:
            with tqdm(total=total_grids, desc="Generating all grids") as pbar:
                for grid in product(range(max_value + 1), repeat=4):
                    packed_value = sum(grid[i] << (self.pixel_bit_depth * (3 - i)) for i in range(4))
                    f.write(struct.pack(self.struct_format, packed_value))
                    pbar.update(1)

    def merge_sort_to_file(self, input_file, output_file, coefficient_index):
        entry_size = self.entry_size
        temp_dir = self.temp_dir
        temp_file = os.path.join(temp_dir, "temp_merge_sort.bin")

        total_entries = os.path.getsize(input_file) // entry_size
        with tqdm(total=total_entries, desc=f"Sorting by coefficient {coefficient_index}") as pbar:
            with open(input_file, "rb") as in_f, open(temp_file, "wb") as temp_f:
                while chunk := in_f.read(DEFAULT_CHUNK_SIZE * entry_size):
                    grids_data = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, 4)
                    coefficients = grids.calculate_haar_coefficients(grids_data)
                    sorted_indices = np.argsort(coefficients[:, coefficient_index], kind='mergesort')
                    sorted_entries = grids_data[sorted_indices]
                    temp_f.write(sorted_entries.tobytes())
                    pbar.update(len(grids_data))

        with tqdm(total=total_entries, desc=f"Finalizing sort for coefficient {coefficient_index}") as pbar:
            with open(temp_file, "rb") as temp_f, open(output_file, "wb") as out_f:
                all_entries = temp_f.read()
                entries = np.frombuffer(all_entries, dtype=np.uint8).reshape(-1, 4)
                coefficients = grids.calculate_haar_coefficients(entries)
                sorted_indices = np.argsort(coefficients[:, coefficient_index], kind='mergesort')
                sorted_entries = entries[sorted_indices]
                out_f.write(sorted_entries.tobytes())
                pbar.update(len(entries))

        os.remove(temp_file)  # Cleanup temporary file

    def hierarchical_sort(self, input_file, output_file, coefficient_index, num_groups):
        entry_size = self.entry_size
        total_entries = os.path.getsize(input_file) // entry_size
        group_sizes = total_entries // num_groups

        with tqdm(total=num_groups, desc=f"Sorting groups by coefficient {coefficient_index}") as pbar:
            with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
                for group in range(num_groups):
                    group_start = group * group_sizes * entry_size
                    group_end = (group + 1) * group_sizes * entry_size

                    f_in.seek(group_start)
                    group_data = f_in.read(group_end - group_start)
                    grids_data = np.frombuffer(group_data, dtype=np.uint8).reshape(-1, 4)

                    coefficients = grids.calculate_haar_coefficients(grids_data)
                    sorted_indices = np.argsort(coefficients[:, coefficient_index], kind='mergesort')
                    sorted_entries = grids_data[sorted_indices]

                    f_out.write(sorted_entries.tobytes())
                    pbar.update(1)

    def haar_sort(self):
        temp_ll_sorted = os.path.join(self.temp_dir, "ll_sorted.bin")
        temp_hl_sorted = os.path.join(self.temp_dir, "hl_sorted.bin")
        temp_lh_sorted = os.path.join(self.temp_dir, "lh_sorted.bin")
        final_sorted = f"{self.temp_dir}/{self.table_name}_grids.bin"  # Use correct table name

        # Step 1: Generate all grids and sort by LL
        temp_input_file = os.path.join(self.temp_dir, "all_grids.bin")
        self.generate_all_grids_to_file(temp_input_file)
        self.merge_sort_to_file(temp_input_file, temp_ll_sorted, coefficient_index=0)
        os.remove(temp_input_file)  # Cleanup

        # Step 2: Sort by HL
        num_groups = 1 << self.pixel_bit_depth
        self.hierarchical_sort(temp_ll_sorted, temp_hl_sorted, coefficient_index=1, num_groups=num_groups)
        os.remove(temp_ll_sorted)  # Cleanup

        # Step 3: Sort by LH
        num_groups <<= self.pixel_bit_depth
        self.hierarchical_sort(temp_hl_sorted, temp_lh_sorted, coefficient_index=2, num_groups=num_groups)
        os.remove(temp_hl_sorted)  # Cleanup

        # Step 4: Sort by HH
        num_groups <<= self.pixel_bit_depth
        self.hierarchical_sort(temp_lh_sorted, final_sorted, coefficient_index=3, num_groups=num_groups)
        os.remove(temp_lh_sorted)  # Cleanup

        return final_sorted

    def generate_reverse_lookup_table(self, table_file, reverse_table_file):
        total_entries = os.path.getsize(table_file) // self.entry_size

        with tqdm(total=total_entries, desc="Initializing Reverse Lookup Table") as pbar:
            with open(reverse_table_file, "wb") as f_reverse:
                for _ in range(total_entries):
                    f_reverse.write(b"\x00" * self.entry_size)
                    pbar.update(1)

        with open(table_file, "rb") as f_table:
            with open(reverse_table_file, "r+b") as f_reverse:
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
    parser.add_argument("-d", "--bit_depth", type=int, required=True, help="Set the bit depth for the grids.")
    parser.add_argument("-t", "--table", type=str, required=True, help="Table prefix for Haar sort files.")
    args = parser.parse_args()

    table = HaarSortGenerator(pixel_bit_depth=args.bit_depth, table_name=args.table)  # Pass table_name

    if args.generate_forward or args.generate:
        forward_table = table.haar_sort()
        print(f"Forward Haar sort table generated at {forward_table}.")

    if args.generate_reverse or args.generate:
        reverse_table = os.path.join(table.temp_dir, f"{args.table}_index.bin")
        table.generate_reverse_lookup_table(forward_table, reverse_table)
        print(f"Reverse lookup table generated at {reverse_table}.")


