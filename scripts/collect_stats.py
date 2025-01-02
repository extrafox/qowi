import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io
from qowi.encoder import Encoder
from qowi.wavelet import Wavelet

TEST_IMAGE_PATH = "media/mango_512x512.jpg"

def op_code_frequency(df):
	# Count occurrences of each op_code
	op_code_counts = df['op_code'].value_counts()

	# Create a simple table with op_code values and their counts
	ret = pd.DataFrame(op_code_counts).reset_index()
	ret.columns = ['op_code', 'count']
	
	return ret
	
def rgb_frequency_histogram(df):
    # Convert diff columns to float
    df['diff_r'] = df['diff_r'].astype(float)
    df['diff_g'] = df['diff_g'].astype(float)
    df['diff_b'] = df['diff_b'].astype(float)

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
    hist_r = pd.cut(filtered_df['diff_r'].abs(), bins=bins, labels=labels, right=False).value_counts().sort_index()
    hist_g = pd.cut(filtered_df['diff_g'].abs(), bins=bins, labels=labels, right=False).value_counts().sort_index()
    hist_b = pd.cut(filtered_df['diff_b'].abs(), bins=bins, labels=labels, right=False).value_counts().sort_index()

    # Create a histogram table
    ret = pd.DataFrame({
        'Range': labels,
        'diff_r': hist_r.values,
        'diff_g': hist_g.values,
        'diff_b': hist_b.values
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

###
### Prepare image and intermediates
###

print("Processing image: {}".format(TEST_IMAGE_PATH))

image = io.imread(TEST_IMAGE_PATH)
original_image_size = image.shape[0] * image.shape[1] * 3 * 8
print("Original image size (bits): {}".format(original_image_size))

w = Wavelet().prepare_from_image(image)
print("Wavelet prepared.")

e = Encoder(w)

print("Encoding...")
bits = e.encode()
print("Encoded image size (bits): {}".format(len(bits)))
print("Compression: {}% of original size".format(round(len(bits) / original_image_size * 100, 2)))

###
### Output histogram of RGB values
###

df = pd.DataFrame(e.stats)

print()
print("Frequency of op_code usage")
print(op_code_frequency(df))
print()
print("RGB Difference Value Frequencies")
print(rgb_frequency_histogram(df))
print()
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
