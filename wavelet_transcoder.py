#!/usr/bin/env python3

import argparse
import os
import sys
from qowi.haar_sort_wavelet import HaarSortWavelet
from skimage import io

SUPPORTED_IMAGE_FORMATS = {"png", "jpg", "jpeg", "bmp", "tiff"}

def wavelet_transform(source_path, dest_path, haar_sort_table):
    try:
        source_image = io.imread(source_path)
        wavelet = HaarSortWavelet(haar_sort_table=haar_sort_table)
        wavelet.prepare_from_image(source_image)
        wavelet_image = wavelet.wavelet

        # Temporarily save the file with a standard extension
        temp_path = dest_path.replace("ww", "")  # Remove "ww" prefix for temporary saving
        io.imsave(temp_path, wavelet_image.astype(source_image.dtype))

        # Rename the file to include the "ww" prefix
        os.rename(temp_path, dest_path)

        print(f"Wavelet transform completed: {dest_path}")
    except Exception as e:
        print(f"Error during wavelet transform: {e}")
        sys.exit(1)

def inverse_wavelet_transform(source_path, dest_path, haar_sort_table):
    try:
        wavelet_image = io.imread(source_path)

        wavelet = HaarSortWavelet(haar_sort_table=haar_sort_table)
        wavelet.prepare_from_wavelet(wavelet_image)
        decoded_image = wavelet.as_image()

        # Save reconstructed image
        io.imsave(dest_path, decoded_image.astype(wavelet_image.dtype))
        print(f"Inverse wavelet transform completed: {dest_path}")
    except Exception as e:
        print(f"Error during inverse wavelet transform: {e}")
        sys.exit(1)

def copy_image(source_path, dest_path):
    try:
        source_image = io.imread(source_path)
        io.imsave(dest_path, source_image)
        print(f"Image copied without transformation: {dest_path}")
    except Exception as e:
        print(f"Error during image copying: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Wavelet Transform Tool")
    parser.add_argument("source", type=str, help="Path to the source image file")
    parser.add_argument("dest", type=str, help="Path to save the transformed or reconstructed image")
    parser.add_argument("-t", "--haar-sort-table", type=str, required=True, help="Path to the Haar sort table file")

    args = parser.parse_args()

    # Validate file paths
    if not os.path.isfile(args.source):
        print("Error: Source file does not exist.")
        sys.exit(1)

    source_extension = os.path.splitext(args.source)[1].lower()[1:]
    dest_extension = os.path.splitext(args.dest)[1].lower()[1:]

    if source_extension.startswith("ww"):
        # Perform inverse wavelet transform if source is a wrapped wavelet
        inverse_wavelet_transform(args.source, args.dest, args.haar_sort_table)
    elif source_extension in SUPPORTED_IMAGE_FORMATS and dest_extension.startswith("ww"):
        # Perform wavelet transform if destination has a ww prefix
        wavelet_transform(args.source, args.dest, args.haar_sort_table)
    elif source_extension in SUPPORTED_IMAGE_FORMATS and dest_extension in SUPPORTED_IMAGE_FORMATS:
        # Copy image directly if no wavelet transformation is needed
        copy_image(args.source, args.dest)
    else:
        print("Error: Unsupported file format for source or destination.")
        sys.exit(1)

if __name__ == "__main__":
    main()
