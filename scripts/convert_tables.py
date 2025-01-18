import struct
import argparse
from tqdm import tqdm

def convert_table(input_file, output_file, input_format, output_format):
    """
    Converts a binary table file from one format to another.

    Args:
        input_file (str): Path to the input table file.
        output_file (str): Path to the output table file.
        input_format (str): Struct format of the input file (e.g., '4I').
        output_format (str): Struct format of the output file (e.g., 'I').
    """
    input_size = struct.calcsize(input_format)
    output_size = struct.calcsize(output_format)

    # Calculate total entries based on file size
    with open(input_file, "rb") as infile:
        infile.seek(0, 2)  # Move to the end of the file
        total_entries = infile.tell() // input_size

    with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
        with tqdm(total=total_entries, desc="Converting Table") as pbar:
            while True:
                data = infile.read(input_size)
                if not data:
                    break
                # Unpack the input data
                unpacked = struct.unpack(input_format, data)

                # Convert the unpacked data into a single integer (pack into output format)
                packed_value = 0
                for i, value in enumerate(unpacked):
                    packed_value |= (value & ((1 << 8) - 1)) << (i * 8)

                # Write the packed value in the output format
                outfile.write(struct.pack(output_format, packed_value))
                pbar.update(1)

def generate_reverse_table(forward_file, reverse_file, format):
    """
    Generates a reverse table from a forward table.

    Args:
        forward_file (str): Path to the forward table file.
        reverse_file (str): Path to the reverse table file.
        format (str): Struct format of the table (e.g., 'I').
    """
    entry_size = struct.calcsize(format)

    # Calculate total entries based on file size
    with open(forward_file, "rb") as f:
        f.seek(0, 2)  # Move to the end of the file
        total_entries = f.tell() // entry_size

    reverse_mapping = [None] * total_entries

    # Read forward table and populate reverse mapping
    with open(forward_file, "rb") as f:
        with tqdm(total=total_entries, desc="Generating Reverse Table") as pbar:
            for index in range(total_entries):
                data = f.read(entry_size)
                value = struct.unpack(format, data)[0]
                reverse_mapping[value] = index
                pbar.update(1)

    # Write reverse mapping to reverse table
    with open(reverse_file, "wb") as f:
        with tqdm(total=total_entries, desc="Writing Reverse Table") as pbar:
            for index in reverse_mapping:
                f.write(struct.pack(format, index))
                pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Haar Sort Tables and Generate Reverse Table.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for converting tables
    convert_parser = subparsers.add_parser("convert", help="Convert table format from 4I to I.")
    convert_parser.add_argument("--input_file", type=str, required=True, help="Path to the input table file.")
    convert_parser.add_argument("--output_file", type=str, required=True, help="Path to the output table file.")
    convert_parser.add_argument("--input_format", type=str, default="4I", help="Input table format (default: 4I).")
    convert_parser.add_argument("--output_format", type=str, default="I", help="Output table format (default: I).")

    # Subparser for generating reverse table
    reverse_parser = subparsers.add_parser("generate_reverse", help="Generate reverse table from forward table.")
    reverse_parser.add_argument("--forward_file", type=str, required=True, help="Path to the forward table file.")
    reverse_parser.add_argument("--reverse_file", type=str, required=True, help="Path to the reverse table file.")
    reverse_parser.add_argument("--format", type=str, default="I", help="Table format (default: I).")

    args = parser.parse_args()

    if args.command == "convert":
        convert_table(args.input_file, args.output_file, args.input_format, args.output_format)
    elif args.command == "generate_reverse":
        generate_reverse_table(args.forward_file, args.reverse_file, args.format)
