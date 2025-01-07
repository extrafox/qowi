#!/usr/bin/env python3
import os
import argparse
from bitstring import BitStream
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio
import random
import csv
from qowi.qowi_decoder import QOWIDecoder
from qowi.qowi_encoder import QOWIEncoder

# Define parameter ranges
hard_threshold_range = (0.0001, 1)
soft_threshold_range = (0.0001, 1)
wavelet_levels_range = (1, 20)
wavelet_precision_range = (0, 24)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Optimize QOWI codec parameters.")
parser.add_argument("input_dir", type=str, help="Path to the directory containing training images.")
parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")
parser.add_argument("-t", "--hard-threshold", type=float, default=None, help="Set a fixed hard threshold.")
parser.add_argument("-s", "--soft-threshold", type=float, default=None, help="Set a fixed soft threshold.")
parser.add_argument("-w", "--wavelet-levels", type=int, default=None, help="Set a fixed number of wavelet levels.")
parser.add_argument("-p", "--wavelet-precision", type=int, default=None, help="Set a fixed wavelet precision.")
parser.add_argument("-n", "--num-samples", type=int, default=100, help="Number of samples to process.")
args = parser.parse_args()

# Training images directory
training_images_dir = args.input_dir

# Output results file
output_file = args.output_csv

# Function to generate settings (randomized or from arguments)
def generate_settings():
    hard_threshold = args.hard_threshold if args.hard_threshold is not None else round(random.uniform(*hard_threshold_range), 4)
    soft_threshold = args.soft_threshold if args.soft_threshold is not None else (round(random.uniform(*soft_threshold_range), 4) if hard_threshold is None else -1)
    wavelet_levels = args.wavelet_levels if args.wavelet_levels is not None else random.randint(*wavelet_levels_range)
    wavelet_precision = args.wavelet_precision if args.wavelet_precision is not None else random.randint(*wavelet_precision_range)
    return hard_threshold, soft_threshold, wavelet_levels, wavelet_precision

# Function to randomly pick an image file from the filesystem
def get_random_image(directory):
    valid_files = []
    for root, _, files in os.walk(directory):
        valid_files.extend([os.path.join(root, file) for file in files if file.endswith(".JPEG")])

    if not valid_files:
        raise FileNotFoundError("No valid image files found in the specified directory.")

    random_file = random.choice(valid_files)
    print("Processing file {}".format(random_file))
    return imread(random_file)


def process_image(source_image, wavelet_levels, wavelet_precision, soft_threshold, hard_threshold):
    original_image_size = source_image.shape[0] * source_image.shape[1] * source_image.shape[2] * 8
    print("Original image shape {} and size (bits): {}".format(source_image.shape, original_image_size))

    print("Encoding with wavelet levels {}, wavelet precision {}, soft threshold {} and hard threshold {}...".format(
        wavelet_levels, wavelet_precision, soft_threshold, hard_threshold))

    encoded_bitstream = BitStream()
    e = QOWIEncoder(hard_threshold, soft_threshold, wavelet_levels, wavelet_precision)
    e.from_array(source_image)
    e.to_bitstream(encoded_bitstream)
    e.encode()
    print("Finished encoding in {:.2f} seconds".format(e.encode_duration))

    encoded_size = len(encoded_bitstream)
    compression_percentage = round(encoded_size / original_image_size * 100, 2)
    print("Encoded (bits): {} ({}%)".format(encoded_size, compression_percentage))

    print()
    print("Decoding...")
    d = QOWIDecoder()
    d.from_bitstream(encoded_bitstream)
    d.decode()
    decoded_image = d.as_array()
    print("Finished decoding in {:.2f} seconds".format(d.decode_duration))

    print("Computing signal to noise ratio (SNR)...")
    psnr = peak_signal_noise_ratio(source_image, decoded_image)
    print("PSNR: {}".format(psnr))

    return encoded_size, psnr, compression_percentage

# Open output file and write results incrementally
with open(output_file, mode='w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Hard Threshold", "Soft Threshold", "Wavelet Levels", "Wavelet Precision", "PSNR", "Compressed File Size", "Compression Percentage"])
    file.flush()

    for _ in range(args.num_samples):
        hard_threshold, soft_threshold, wavelet_levels, wavelet_precision = generate_settings()

        source_image = get_random_image(training_images_dir)  # Randomly pick an image from the filesystem
        if len(source_image.shape) < 3:
            continue

        encoded_size, psnr, compression_percentage = process_image(source_image, wavelet_levels, wavelet_precision, soft_threshold, hard_threshold)

        writer.writerow([hard_threshold, soft_threshold, wavelet_levels, wavelet_precision, psnr, encoded_size, compression_percentage])
        file.flush()

print(f"Sampling complete. Results saved to {output_file}")
