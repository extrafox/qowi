import argparse
from qowi.haar_sort_table import HaarSortTable

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Haar Sort Lookup Tables")
    parser.add_argument("-d", "--bit_depth", type=int, required=True, help="Pixel bit depth for the grid pixels.")
    parser.add_argument("-t", "--table_name", type=str, required=True, help="Base name of the Haar Sort table files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--pixels", type=int, nargs=4, help="Query the Haar Sort index for a pixel grid.")
    group.add_argument("-i", "--index", type=int, help="Query the pixel grid for a Haar Sort index.")
    group.add_argument("-w", "--wavelet", type=int, nargs=4, help="Query the pixel grid for a wavelet.")
    args = parser.parse_args()

    table = HaarSortTable(pixel_bit_depth=args.bit_depth, table_name=args.table_name)

    if args.pixels:
        try:
            index = table.pixels_to_haar_sort_indices([args.pixels])[0]
            wavelet = table.pixels_to_wavelets([args.pixels])[0]
            print(f"The Haar Sort index for pixels {args.pixels} is {index} / {wavelet}.")
        except ValueError as e:
            print(e)

    if args.index is not None:
        try:
            pixels = table.haar_sort_indices_to_pixels([args.index])[0]
            print(f"The pixels for Haar Sort index {args.index} is {pixels}.")
        except ValueError as e:
            print(e)

    if args.wavelet:
        try:
            pixels = table.wavelets_to_pixels([args.wavelet])[0]
            print(f"The pixels for wavelet {args.wavelet} is {pixels}.")
        except ValueError as e:
            print(e)
