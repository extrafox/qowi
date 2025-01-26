import numpy as np
from tqdm import tqdm

import qowi.grids as grids
import os
import struct
import heapq

CHUNK_SIZE = 1_000_000

class HaarSortGenerator:
    def __init__(self, pixel_bit_depth=None, table_name=None, temp_dir=None):
        grids.validate_bit_depth(pixel_bit_depth)
        self.pixel_bit_depth = pixel_bit_depth
        self.table_name = table_name
        self.temp_dir = temp_dir or "/tmp"
        self.struct_format = grids.get_struct_format(pixel_bit_depth)
        self.entry_size = struct.calcsize(self.struct_format)
        self.progress_bar = None

    def init_progress_bar(self, total, desc):
        if self.progress_bar:
            self.progress_bar.close()
        self.progress_bar = tqdm(total=total, desc=desc, leave=True)

    def update_progress(self, step):
        if self.progress_bar:
            self.progress_bar.update(step)

    def update_progress_desc(self, desc):
        if self.progress_bar:
            self.progress_bar.set_description(desc)

    def close_progress_bar(self):
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None

    def sort_chunk(self, grids_data, coefficient_index):
        coefficients = grids.calculate_haar_coefficients(grids_data)
        sorted_indices = np.argsort(coefficients[:, coefficient_index], kind='stable')
        return grids_data[sorted_indices]

    def save_chunk(self, sorted_data, chunk_file):
        with open(chunk_file, "wb") as f:
            for grid in sorted_data:
                packed_grid = grids.grids_to_indices(np.array([grid], dtype=np.uint8), self.pixel_bit_depth)[0]
                f.write(struct.pack(self.struct_format, packed_grid))

    def merge_chunks(self, chunk_files, output_file, max_open_files=50):
        with open(output_file, "wb") as f:  # Keep the output file open
            chunk_index = 0
            while chunk_index < len(chunk_files):
                # Open a batch of chunk files (limit the number of simultaneously open files)
                batch_files = chunk_files[chunk_index:chunk_index + max_open_files]
                streams = [open(chunk, "rb") for chunk in batch_files]
                heap = []

                # Initialize heap with the first entry of each chunk file
                for i, stream in enumerate(streams):
                    chunk = stream.read(self.entry_size)
                    if chunk:
                        heapq.heappush(heap, (struct.unpack(self.struct_format, chunk)[0], i, chunk))

                # Merge chunks from the current batch into the output file
                while heap:
                    _, index, data = heapq.heappop(heap)
                    f.write(data)
                    next_chunk = streams[index].read(self.entry_size)
                    if next_chunk:
                        heapq.heappush(heap, (struct.unpack(self.struct_format, next_chunk)[0], index, next_chunk))

                # Close the files after processing the batch
                for stream in streams:
                    stream.close()

                # Delete chunk files after merging to free up space
                for chunk_file in batch_files:
                    os.remove(chunk_file)

                # Move to the next batch
                chunk_index += max_open_files

    def haar_sort(self):
        total_grids = (1 << (self.pixel_bit_depth * 4))
        total_chunks = (total_grids + CHUNK_SIZE - 1) // CHUNK_SIZE
        self.init_progress_bar(total_chunks * 4, "Sorting grids...")

        chunk_files = []

        # Step 1: Generate sorted chunks for LL coefficient
        self.update_progress_desc("Sorting chunks by LL coefficient...")
        for start in range(0, total_grids, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total_grids)
            grids_array = grids.indices_to_grids(np.arange(start, end, dtype=np.uint32), self.pixel_bit_depth)
            sorted_chunk = self.sort_chunk(grids_array, 0)
            chunk_file = os.path.join(self.temp_dir, f"chunk_LL_{start}_{end}.bin")
            self.save_chunk(sorted_chunk, chunk_file)
            chunk_files.append(chunk_file)
            self.update_progress(1)

        # Step 2: Merge LL sorted chunks
        self.update_progress_desc("Merging LL sorted chunks...")
        merged_LL_file = os.path.join(self.temp_dir, f"{self.table_name}_sorted_LL.bin")
        self.merge_chunks(chunk_files, merged_LL_file)

        merged_file = merged_LL_file

        # Repeat for HL, LH, and HH coefficients
        for coeff_index, coeff_name in enumerate(["HL", "LH", "HH"], start=1):
            self.update_progress_desc(f"Sorting chunks by {coeff_name} coefficient...")
            chunk_files = []

            with open(merged_LL_file if coeff_index == 1 else merged_file, "rb") as f:
                grids_array = np.zeros((CHUNK_SIZE, 4), dtype=np.uint8)
                chunk_index = 0

                while True:
                    chunk = f.read(CHUNK_SIZE * self.entry_size)
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
                    self.update_progress(1)

            self.update_progress_desc(f"Merging {coeff_name} sorted chunks...")
            merged_file = os.path.join(self.temp_dir, f"{self.table_name}_sorted_{coeff_name}.bin")
            self.merge_chunks(chunk_files, merged_file)

        self.update_progress_desc("Saving final sorted grids...")
        final_sorted = f"{self.table_name}_grids.bin"
        os.rename(merged_file, final_sorted)

        self.close_progress_bar()
        return final_sorted

    def generate_reverse_lookup_table(self, table_file, reverse_table_file):
        total_entries = os.path.getsize(table_file) // self.entry_size

        # Initialize the progress bar for initializing the reverse lookup table
        self.init_progress_bar(total_entries * 2, "Initializing Reverse Lookup Table...")  # Two steps: initializing and generating

        with open(reverse_table_file, "wb") as f_reverse:
            for _ in range(total_entries):
                f_reverse.write(b"\x00" * self.entry_size)
                self.update_progress(1)

        self.update_progress_desc("Generating Reverse Lookup Table...")

        with open(table_file, "rb") as f_table:
            with open(reverse_table_file, "r+b") as f_reverse:
                for position, data in enumerate(iter(lambda: f_table.read(self.entry_size), b"")):
                    index = struct.unpack(self.struct_format, data)[0]
                    f_reverse.seek(index * self.entry_size)
                    f_reverse.write(struct.pack(self.struct_format, position))
                    self.update_progress(1)

        self.close_progress_bar()


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
