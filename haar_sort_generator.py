import numpy as np
from tqdm import tqdm
import qowi.grids as grids
import os
import struct

class HaarSortGenerator:
    def __init__(self, pixel_bit_depth=None, table_name=None, temp_dir=None):
        grids.validate_bit_depth(pixel_bit_depth)
        self.pixel_bit_depth = pixel_bit_depth
        self.table_name = table_name
        self.temp_dir = temp_dir or "/tmp"
        self.struct_format = grids.get_struct_format(pixel_bit_depth)
        self.entry_size = struct.calcsize(self.struct_format)

    def sort_data_by_coefficient(self, grids_data, coefficient_index):
        coefficients = grids.calculate_haar_coefficients(grids_data)
        sorted_indices = np.argsort(coefficients[:, coefficient_index], kind='stable')
        return grids_data[sorted_indices]

    def hierarchical_sort(self, grids_data, coefficient_index, group_size):
        num_groups = len(grids_data) // group_size
        sorted_data = np.zeros_like(grids_data)

        for i in tqdm(range(num_groups), desc=f"Sorting by coefficient {coefficient_index}"):
            group_start = i * group_size
            group_end = group_start + group_size
            group = grids_data[group_start:group_end]

            group_sorted = self.sort_data_by_coefficient(group, coefficient_index)
            sorted_data[group_start:group_end] = group_sorted

        return sorted_data

    def haar_sort(self):
        # Step 1: Generate all grids
        total_grids = (1 << (self.pixel_bit_depth * 4))  # Total number of grids
        grids_array = grids.indices_to_grids(np.arange(total_grids, dtype=np.uint32), self.pixel_bit_depth)

        # Step 2: Sort by LL
        print("Sorting by LL coefficient...")
        sorted_grids_LL = self.sort_data_by_coefficient(grids_array, 0)

        # Step 3: Sort by HL
        print("Sorting by HL coefficient...")
        group_size = 1 << self.pixel_bit_depth
        sorted_grids_HL = self.hierarchical_sort(sorted_grids_LL, 1, group_size)

        # Step 4: Sort by LH
        print("Sorting by LH coefficient...")
        group_size <<= self.pixel_bit_depth  # 2^(2 * bit_depth)
        sorted_grids_LH = self.hierarchical_sort(sorted_grids_HL, 2, group_size)

        # Step 5: Sort by HH
        print("Sorting by HH coefficient...")
        group_size <<= self.pixel_bit_depth  # 2^(3 * bit_depth)
        sorted_grids_HH = self.hierarchical_sort(sorted_grids_LH, 3, group_size)

        # Step 6: Save the final sorted grids
        print("Saving final sorted grids...")
        final_sorted = f"{self.table_name}_grids.bin"
        with open(final_sorted, "wb") as f:
            for grid in sorted_grids_HH:
                packed_grid = grids.grids_to_indices(np.array([grid], dtype=np.uint8), self.pixel_bit_depth)[0]
                f.write(struct.pack(self.struct_format, packed_grid))

        return final_sorted

    def stream_file(self, file):
        with open(file, "rb") as f:
            while chunk := f.read(self.entry_size):
                yield struct.unpack(self.struct_format, chunk)

    def generate_all_grids_to_file(self, output_file):
        total_grids = (1 << (self.pixel_bit_depth * 4))
        with open(output_file, "wb") as f:
            for value in range(total_grids):
                f.write(struct.pack(self.struct_format, value))

    def generate_reverse_lookup_table(self, table_file, reverse_table_file):
        total_entries = os.path.getsize(table_file) // self.entry_size
        reverse_lookup = [0] * total_entries

        with open(table_file, "rb") as f:
            for position, data in enumerate(iter(lambda: f.read(self.entry_size), b"")):
                index = struct.unpack(self.struct_format, data)[0]
                reverse_lookup[index] = position

        with open(reverse_table_file, "wb") as f:
            for value in reverse_lookup:
                f.write(struct.pack(self.struct_format, value))

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Haar Sorting Algorithm Tool (In-Memory)")
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
