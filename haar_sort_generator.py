import numpy as np
import qowi.grids as grids
import os
import struct
import heapq

class HaarSortGenerator:
    def __init__(self, pixel_bit_depth=None, table_name=None, temp_dir=None):
        grids.validate_bit_depth(pixel_bit_depth)
        self.pixel_bit_depth = pixel_bit_depth
        self.table_name = table_name
        self.temp_dir = temp_dir or "/tmp"
        self.struct_format = grids.get_struct_format(pixel_bit_depth)
        self.entry_size = struct.calcsize(self.struct_format)

    def sort_chunk(self, grids_data, coefficient_index):
        coefficients = grids.calculate_haar_coefficients(grids_data)
        sorted_indices = np.argsort(coefficients[:, coefficient_index], kind='stable')
        return grids_data[sorted_indices]

    def save_chunk(self, sorted_data, chunk_file):
        with open(chunk_file, "wb") as f:
            for grid in sorted_data:
                packed_grid = grids.grids_to_indices(np.array([grid], dtype=np.uint8), self.pixel_bit_depth)[0]
                f.write(struct.pack(self.struct_format, packed_grid))

    def merge_chunks(self, chunk_files, output_file):
        streams = [open(chunk, "rb") for chunk in chunk_files]
        heap = []

        for i, stream in enumerate(streams):
            chunk = stream.read(self.entry_size)
            if chunk:
                heapq.heappush(heap, (struct.unpack(self.struct_format, chunk)[0], i, chunk))

        with open(output_file, "wb") as f:
            while heap:
                _, index, data = heapq.heappop(heap)
                f.write(data)
                next_chunk = streams[index].read(self.entry_size)
                if next_chunk:
                    heapq.heappush(heap, (struct.unpack(self.struct_format, next_chunk)[0], index, next_chunk))

        for stream in streams:
            stream.close()

    def haar_sort(self):
        total_grids = (1 << (self.pixel_bit_depth * 4))
        grids_per_chunk = 1 << 20  # Adjust chunk size as needed

        chunk_files = []

        # Step 1: Generate sorted chunks for LL coefficient
        print("Sorting chunks by LL coefficient...")
        for start in range(0, total_grids, grids_per_chunk):
            end = min(start + grids_per_chunk, total_grids)
            grids_array = grids.indices_to_grids(np.arange(start, end, dtype=np.uint32), self.pixel_bit_depth)
            sorted_chunk = self.sort_chunk(grids_array, 0)
            chunk_file = os.path.join(self.temp_dir, f"chunk_LL_{start}_{end}.bin")
            self.save_chunk(sorted_chunk, chunk_file)
            chunk_files.append(chunk_file)

        # Step 2: Merge LL sorted chunks
        print("Merging LL sorted chunks...")
        merged_LL_file = os.path.join(self.temp_dir, f"{self.table_name}_sorted_LL.bin")
        self.merge_chunks(chunk_files, merged_LL_file)

        # Repeat for HL, LH, and HH coefficients
        for coeff_index, coeff_name in enumerate(["HL", "LH", "HH"], start=1):
            print(f"Sorting chunks by {coeff_name} coefficient...")
            chunk_files = []

            with open(merged_LL_file if coeff_index == 1 else merged_file, "rb") as f:
                grids_array = np.zeros((grids_per_chunk, 4), dtype=np.uint8)
                chunk_index = 0

                while True:
                    chunk = f.read(grids_per_chunk * self.entry_size)
                    if not chunk:
                        break

                    for i, offset in enumerate(range(0, len(chunk), self.entry_size)):
                        grids_array[i] = grids.indices_to_grids(
                            np.array([struct.unpack(self.struct_format, chunk[offset:offset + self.entry_size])[0]],
                                     dtype=np.uint32),
                            self.pixel_bit_depth
                        )[0]

                    sorted_chunk = self.sort_chunk(grids_array[:i + 1], coeff_index)
                    chunk_file = os.path.join(self.temp_dir, f"chunk_{coeff_name}_{chunk_index}.bin")
                    self.save_chunk(sorted_chunk, chunk_file)
                    chunk_files.append(chunk_file)
                    chunk_index += 1

            print(f"Merging {coeff_name} sorted chunks...")
            merged_file = os.path.join(self.temp_dir, f"{self.table_name}_sorted_{coeff_name}.bin")
            self.merge_chunks(chunk_files, merged_file)

        print("Saving final sorted grids...")
        final_sorted = f"{self.table_name}_grids.bin"
        os.rename(merged_file, final_sorted)

        return final_sorted

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

    parser = ArgumentParser(description="Haar Sorting Algorithm Tool (Disk-Based Merge Sort)")
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
