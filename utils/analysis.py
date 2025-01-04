import numpy as np
import pandas as pd

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
