import numpy as np
import pandas as pd
from bitstring import BitStream
from matplotlib import pyplot as plt

from qowi.qowi_decoder import QOWIDecoder
from qowi.qowi_encoder import QOWIEncoder
from qowi.wavelet import Wavelet
from skimage import io

TEST_IMAGE_PATH = "/home/ctaylor/media/imagenet-mini/train/n01443537/n01443537_10408.JPEG"
# TEST_IMAGE_PATH = "media/mango_32x32.jpg"
PRINT_STATS = True
HARD_THRESHOLD = 2
SOFT_THRESHOLD = -1
BIT_SHIFT = 0
NUM_CARRY_OVER_BITS = 1

def op_code_frequency(df):
    # Count occurrences of each op_code
    op_code_counts = df['op_code'].value_counts()

    # Create a simple table with op_code values and their counts
    ret = pd.DataFrame(op_code_counts).reset_index()
    ret.columns = ['op_code', 'count']

    return ret


def op_code_max_values(df):
    # Fill missing values with 0 and convert to integers
    df['run_length'] = df['run_length'].fillna(0).astype(int)
    df['index'] = df['index'].fillna(0).astype(int)

    # Calculate max run_length for RUN
    max_run_length = df[df['op_code'] == 'RUN']['run_length'].max()

    # Calculate max index for CACHE
    max_index_cache = df[df['op_code'] == 'CACHE']['index'].max()

    # Create a result table
    ret = pd.DataFrame({
        'Metric': ['Max Run Length (RUN)', 'Max Index (CACHE)'],
        'Value': [max_run_length, max_index_cache]
    })

    return ret


def rgb_frequency_histogram(df):
    # Convert diff columns to float
    df['uint10_r'] = df['uint10_r'].astype(float)
    df['uint10_g'] = df['uint10_g'].astype(float)
    df['uint10_b'] = df['uint10_b'].astype(float)

    # Filter rows where op_code is "VALUE"
    filtered_df = df[df['op_code'] == "VALUE"]

    # Define variable-sized bins
    bins = [0]
    multiplier = 1
    while bins[-1] < 256:
        bins.append(bins[-1] + multiplier)
        multiplier *= 2

    # Define labels for the bins
    labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]

    # Calculate histogram counts for each diff column
    hist_r = pd.cut(filtered_df['uint10_r'].abs(), bins=bins, labels=labels, right=False).value_counts().sort_index()
    hist_g = pd.cut(filtered_df['uint10_g'].abs(), bins=bins, labels=labels, right=False).value_counts().sort_index()
    hist_b = pd.cut(filtered_df['uint10_b'].abs(), bins=bins, labels=labels, right=False).value_counts().sort_index()

    # Create a histogram table
    ret = pd.DataFrame({
        'Range': labels,
        'uint10_r': hist_r.values,
        'uint10_g': hist_g.values,
        'uint10_b': hist_b.values
    })

    return ret


def op_code_num_bits_frequency_histgram(df):
    # Define bins and labels for the histogram
    bins = np.arange(2, 62, 2)  # Increments of 1 from 1 to 60

    # Calculate histogram counts for each op_code
    op_code_values = df['op_code'].unique()
    histogram_data = {}
    for op_code in op_code_values:
        filtered_df = df[df['op_code'] == op_code]
        hist_counts = pd.cut(filtered_df['num_bits'], bins=bins, right=False).value_counts().sort_index()
        histogram_data[op_code] = hist_counts.values

    # Create a histogram table
    ret = pd.DataFrame(histogram_data, index=[f"{int(bins[i])}" for i in range(len(bins) - 1)])
    ret.index.name = "Range"

    return ret


def signal_to_noise(source, compressed):
    signal_power = np.mean(source ** 2)
    noise_power = np.mean((source - compressed) ** 2)
    if noise_power == 0:
        return float('inf')  # Perfect reconstruction
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def display_images_side_by_side(image1, image2, title1="Source", title2="Compressed"):
    """
    Displays two images side by side using matplotlib.

    Parameters:
        image1 (numpy.ndarray): The first image to display.
        image2 (numpy.ndarray): The second image to display.
        title1 (str): Title for the first image.
        title2 (str): Title for the second image.
    """
    # Ensure both images are NumPy arrays
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image
    axes[0].imshow(image1, cmap="gray" if len(image1.shape) == 2 else None)
    axes[0].axis('off')  # Turn off axes
    axes[0].set_title(title1)

    # Display the second image
    axes[1].imshow(image2, cmap="gray" if len(image2.shape) == 2 else None)
    axes[1].axis('off')  # Turn off axes
    axes[1].set_title(title2)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

###
### Prepare image and intermediates
###

print("Processing image: {}".format(TEST_IMAGE_PATH))

source_image = io.imread(TEST_IMAGE_PATH)
original_image_size = source_image.shape[0] * source_image.shape[1] * 3 * 8
print("Original image shape {} and size (bits): {}".format(source_image.shape, original_image_size))

print("Encoding with bit shift {}, carry over {}, soft threshold {} and hard threshold {}...".format(BIT_SHIFT, NUM_CARRY_OVER_BITS, SOFT_THRESHOLD, HARD_THRESHOLD))
encoded_bitstream = BitStream()
e = QOWIEncoder(HARD_THRESHOLD, SOFT_THRESHOLD, BIT_SHIFT, NUM_CARRY_OVER_BITS)
e.from_array(source_image)
e.to_bitstream(encoded_bitstream)
e.encode()
print()
print("Finished encoding in {:.2f} seconds".format(e.encode_duration))

encoded_size = len(encoded_bitstream)
print("Encoded (bits): {} ({}%)".format(encoded_size, round(encoded_size / original_image_size * 100, 2)))

print()
print("Decoding...")
d = QOWIDecoder()
d.from_bitstream(encoded_bitstream)
d.decode()
decoded_image = d.as_array()
print("Finished decoding in {:.2f} seconds".format(d.decode_duration))

print("Computing signal to noise ratio (SNR)...")
print("SNR: {}".format(signal_to_noise(source_image, decoded_image)))

display_images_side_by_side(source_image, decoded_image)

###
### Output histogram of RGB values
###

if PRINT_STATS:
    df = pd.DataFrame(e.stats)

    print()
    print("Frequency of op_code usage")
    print(op_code_frequency(df))
    print()
    print("Max values for op_codes")
    print(op_code_max_values(df))
    print()
    # print("RGB Difference Value Frequencies")
    # print(rgb_frequency_histogram(df))
    # print()
    print("op_code num_bits frequencies")
    print(op_code_num_bits_frequency_histgram(df))

###
### Show Diagrams
###

# # Group by 'op_code' and plot histograms for 'num_bits'
# grouped = df.groupby('op_code')
#
# # Define the bin edges for smaller increments
# bin_edges = np.arange(4, 60, 2)  # Bins from 10 to 55 with an increment of 5
#
# # Create a figure and axes for multiple plots
# fig, axes = plt.subplots(len(grouped), 1, figsize=(8, 6), sharex=True)
#
# # Plot histograms for each op_code
# for (op_code, group), ax in zip(grouped, axes):
#     group['num_bits'].hist(ax=ax, bins=128, color='blue', edgecolor='black')
#     ax.set_title(f'Histogram for op_code: {op_code}')
#     ax.set_xlabel('num_bits')
#     ax.set_ylabel('Frequency')
#     ax.set_xticks(bin_edges)  # Set custom ticks on the x-axis
#
# # Adjust layout
# plt.tight_layout()
# plt.show()

print("Done")
