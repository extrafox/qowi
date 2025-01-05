import pandas as pd
import utils.analysis as analysis
from bitstring import BitStream
from qowi.qowi_decoder import QOWIDecoder
from qowi.qowi_encoder import QOWIEncoder
from utils.visualization import display_images_side_by_side
from skimage import io

TEST_IMAGE_PATH = "/home/ctaylor/media/imagenet-mini/train/n01443537/n01443537_10408.JPEG"
# TEST_IMAGE_PATH = "media/mango_32x32.jpg"
PRINT_STATS = True
HARD_THRESHOLD = 0
SOFT_THRESHOLD = -1

###
### Prepare image and intermediates
###

print("Processing image: {}".format(TEST_IMAGE_PATH))

source_image = io.imread(TEST_IMAGE_PATH)
original_image_size = source_image.shape[0] * source_image.shape[1] * 3 * 8
print("Original image shape {} and size (bits): {}".format(source_image.shape, original_image_size))

print("Encoding with soft threshold {} and hard threshold {}...".format(SOFT_THRESHOLD, HARD_THRESHOLD))
encoded_bitstream = BitStream()
e = QOWIEncoder(HARD_THRESHOLD, SOFT_THRESHOLD)
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
print()
print("Finished decoding in {:.2f} seconds".format(d.decode_duration))

print("Computing signal to noise ratio (SNR)...")
print("SNR: {}".format(analysis.signal_to_noise(source_image, decoded_image)))

display_images_side_by_side(source_image, decoded_image)

if PRINT_STATS:
    df = pd.DataFrame(e.stats)

    print()
    print("Frequency of op_code usage")
    print(analysis.op_code_frequency(df))
    print()
    print("Max values for op_codes")
    print(analysis.op_code_max_values(df))
    print()
    # print("RGB Difference Value Frequencies")
    # print(rgb_frequency_histogram(df))
    # print()
    print("op_code num_bits frequencies")
    print(analysis.op_code_num_bits_frequency_histgram(df))

print("Done")
