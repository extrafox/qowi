import os
import struct
import heapq
import argparse
from tqdm import tqdm

class HaarSortTable:
    def __init__(self, bit_depth=None):
        self.bit_depth = bit_depth

    def _get_struct_format(self):
        if self.bit_depth == 2:
            return "4B"
        elif self.bit_depth == 4:
            return "4H"
        elif self.bit_depth == 8:
            return "4I"
        else:
            raise ValueError("Unsupported bit depth: must be 2, 4, or 8.")

    def resume_merge_sorted_chunks(self, temp_dir, output_file, struct_format, completed_files):
        remaining_files = sorted(
            [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
             if f.startswith("sorted_chunk_") and os.path.join(temp_dir, f) not in completed_files]
        )
        if not remaining_files:
            print("No remaining files to merge. Ensure all chunks are available.")
            return

        total_grids = sum(
            os.path.getsize(file) // struct.calcsize(struct_format) for file in remaining_files
        )

        def stream_file(file):
            with open(file, "rb") as f:
                while True:
                    data = f.read(struct.calcsize(struct_format))
                    if not data:
                        break
                    yield struct.unpack(struct_format, data)

        # Limit the number of concurrently open files
        max_open_files = 100
        merged_chunks = []

        with open(output_file, "ab") as f_out:
            with tqdm(total=total_grids, desc="Resuming Merge") as pbar:
                for i in range(0, len(remaining_files), max_open_files):
                    chunk_files = remaining_files[i:i + max_open_files]
                    sorted_streams = (stream_file(f) for f in chunk_files)
                    for grid in heapq.merge(*sorted_streams):
                        packed_grid = struct.pack(struct_format, *grid)
                        f_out.write(packed_grid)
                        pbar.update(1)
                    merged_chunks.extend(chunk_files)

        # Remove processed files after merging
        for file in merged_chunks:
            os.remove(file)

        print(f"Merge completed. Output saved to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Haar Sorting Process")
    parser.add_argument("--bit_depth", type=int, required=True, help="Bit depth for the grids.")
    parser.add_argument("--temp_dir", type=str, required=True, help="Directory containing temporary sorted chunks.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output merged table file.")
    parser.add_argument("--completed_files", type=str, nargs="*", default=[], help="List of already merged chunk files.")
    args = parser.parse_args()

    table = HaarSortTable(bit_depth=args.bit_depth)
    struct_format = table._get_struct_format()
    table.resume_merge_sorted_chunks(args.temp_dir, args.output_file, struct_format, args.completed_files)
