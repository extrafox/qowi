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
        total_grids = (1 << (self.pixel_bit_depth * 4))  # Total number of grids
        with open(output_file, "wb") as f:
            with tqdm(total=total_grids, desc="Generating all grids") as pbar:
                for value in range(total_grids):
                    f.write(struct.pack(self.struct_format, value))
                    pbar.update(1)

    def sort_data_by_coefficient(self, grids_data, coefficient_index):
        coefficients = grids.calculate_haar_coefficients(grids_data)
        sorted_indices = np.argsort(coefficients[:, coefficient_index], kind='stable')
        return grids_data[sorted_indices]

    def hierarchical_sort(self, input_file, output_file, coefficient_index, num_groups=1):
        entry_size = self.entry_size
        total_entries = os.path.getsize(input_file) // entry_size
        group_size = total_entries // num_groups

        with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
            with tqdm(total=num_groups, desc=f"Sorting by coefficient {coefficient_index}") as pbar:
                for group in range(num_groups):
                    group_start = group * group_size * entry_size
                    group_end = (group + 1) * group_size * entry_size if group < num_groups - 1 else None

                    f_in.seek(group_start)
                    group_data = f_in.read(group_size * entry_size if group_end else None)
                    grids_data = np.frombuffer(group_data, dtype=np.uint8).reshape(-1, 4)

                    sorted_data = self.sort_data_by_coefficient(grids_data, coefficient_index)
                    f_out.write(sorted_data.tobytes())
                    pbar.update(1)

    def haar_sort(self):
        temp_ll_sorted = os.path.join(self.temp_dir, "ll_sorted.bin")
        temp_hl_sorted = os.path.join(self.temp_dir, "hl_sorted.bin")
        temp_lh_sorted = os.path.join(self.temp_dir, "lh_sorted.bin")
        final_sorted = f"{self.table_name}_grids.bin"

        # Step 1: Generate all grids and sort by LL
        temp_input_file = os.path.join(self.temp_dir, "all_grids.bin")
        self.generate_all_grids_to_file(temp_input_file)
        self.hierarchical_sort(temp_input_file, temp_ll_sorted, coefficient_index=0, num_groups=1)
        os.remove(temp_input_file)

        # Step 2: Sort by HL
        num_groups = 1 << self.pixel_bit_depth
        self.hierarchical_sort(temp_ll_sorted, temp_hl_sorted, coefficient_index=1, num_groups=num_groups)
        os.remove(temp_ll_sorted)

        # Step 3: Sort by LH
        num_groups <<= self.pixel_bit_depth
        self.hierarchical_sort(temp_hl_sorted, temp_lh_sorted, coefficient_index=2, num_groups=num_groups)
        os.remove(temp_hl_sorted)

        # Step 4: Sort by HH
        num_groups <<= self.pixel_bit_depth
        self.hierarchical_sort(temp_lh_sorted, final_sorted, coefficient_index=3, num_groups=num_groups)
        os.remove(temp_lh_sorted)

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

    generator = HaarSortGenerator(pixel_bit_depth=args.bit_depth, table_name=args.table)

    if args.generate_forward or args.generate:
        forward_table = generator.haar_sort()
        print(f"Forward Haar sort table generated at {forward_table}.")

    if args.generate_reverse or args.generate:
        forward_table = f"{args.table}_grids.bin"
        reverse_table = f"{args.table}_index.bin"
        generator.generate_reverse_lookup_table(forward_table, reverse_table)
        print(f"Reverse lookup table generated at {reverse_table}.")


