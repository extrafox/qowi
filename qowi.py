#!/usr/bin/env python3

import argparse
import sys
from bitstring import BitStream
from skimage import io

from qowi.qowi_encoder import QOWIEncoder
from qowi.qowi_decoder import QOWIDecoder

DEFAULT_HARD_THRESHOLD = -1
DEFAULT_SOFT_THRESHOLD = -1
DEFAULT_WAVELET_LEVELS = 10
DEFAULT_WAVELET_PRECISION_DIGITS = 0

def encode(source_path, dest_path, hard_threshold, soft_threshold, wavelet_levels, wavelet_precision_digits):
    source_image = io.imread(source_path)

    encoder = QOWIEncoder(hard_threshold, soft_threshold, wavelet_levels, wavelet_precision_digits)
    encoder.from_array(source_image)
    bitstream = BitStream()
    encoder.to_bitstream(bitstream)
    encoder.encode()

    with open(dest_path, 'wb') as dest_file:
        dest_file.write(bitstream.bytes)

    print("Encoding completed successfully.")

def decode(source_path, dest_path):
    with open(source_path, 'rb') as source_file:
        input_stream = BitStream(source_file.read())

    decoder = QOWIDecoder()
    decoder.from_bitstream(input_stream)
    decoder.decode()
    decoded_image = decoder.as_array()

    io.imsave(dest_path, decoded_image)

    print("Decoding completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Quite OK Wavelet Image (QOWI) Encoder/Decoder")
    parser.add_argument("operation", type=str, choices=["encode", "decode"], help="Operation to perform: encode or decode")
    parser.add_argument("source", type=str, help="Path to the source file")
    parser.add_argument("destination", type=str, help="Path to the destination file")

    # Encoding-specific arguments
    parser.add_argument("-t", "--hard-threshold", type=int, default=DEFAULT_HARD_THRESHOLD, help="Wavelet hard threshold")
    parser.add_argument("-s", "--soft-threshold", type=int, default=DEFAULT_SOFT_THRESHOLD, help="Wavelet soft threshold")
    parser.add_argument("-w", "--wavelet-levels", type=int, default=DEFAULT_WAVELET_LEVELS, help="Number of wavelet levels to encode. Defaults to {}".format(DEFAULT_WAVELET_LEVELS))
    parser.add_argument("-p", "--wavelet-precision", type=int, default=DEFAULT_WAVELET_PRECISION_DIGITS, help="Precision in binary digits to round at each wavelet level. Defaults to {}".format(DEFAULT_WAVELET_PRECISION_DIGITS))

    args = parser.parse_args()

    # Validate file paths
    if not args.source or not args.destination:
        print("Error: Source and destination file paths must be provided.")
        sys.exit(1)

    if args.operation == "encode":
        encode(
            args.source,
            args.destination,
            args.hard_threshold,
            args.soft_threshold,
            args.wavelet_levels,
            args.wavelet_precision
        )
    elif args.operation == "decode":
        decode(
            args.source,
            args.destination)
    else:
        print("Error: Invalid operation specified.")
        sys.exit(1)

if __name__ == "__main__":
    main()
