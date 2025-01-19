import argparse
from qowi.haar_sort_table import HaarSortTable

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Haar Sort Lookup Tables")
    parser.add_argument("--bit_depth", type=int, required=True, help="Bit depth for the grids.")
    parser.add_argument("--table_name", type=str, required=True, help="Base name of the Haar sort table files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--grid", type=int, nargs=4, help="Query the Haar sort index for a grid.")
    group.add_argument("--index", type=int, help="Query the grid for a Haar sort index.")
    args = parser.parse_args()

    query = HaarSortTable(bit_depth=args.bit_depth, table_name=args.table_name)

    if args.grid:
        try:
            index = query.grid_to_haar_sort_index(args.grid)
            print(f"The Haar sort index for grid {args.grid} is {index}.")
        except ValueError as e:
            print(e)

    if args.index is not None:
        try:
            grid = query.haar_sort_index_to_grid(args.index)
            print(f"The grid for Haar sort index {args.index} is {grid}.")
        except ValueError as e:
            print(e)
