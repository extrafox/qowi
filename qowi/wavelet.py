import math
import numpy as np
from blosc import destroy
from numpy import ndarray

AVERAGE_PIXEL = (128, 128, 128)

def binary_round(a: ndarray, num_bits=2) -> ndarray:
    return np.round(a * 2 ** num_bits) / 2 ** num_bits

# TODO: right now I am working with square wavelets. generalize to different sizes

def gen_wavelet(image: np.ndarray, dtype=np.float64) -> np.ndarray:
    # generate wavelet with a size that is a factor of 2 in both dimensions
    num_horiz_levels = math.ceil(math.log2(image.shape[0]))
    num_vert_levels = math.ceil(math.log2(image.shape[1]))
    width = 2 ** num_horiz_levels
    height = 2 ** num_vert_levels

    source_wavelet = np.full((width, height, 3), AVERAGE_PIXEL, dtype=dtype)
    source_wavelet[:image.shape[0], :image.shape[1]] = image

    dest_width, dest_height = width // 2, height // 2
    dest_wavelet = None
    while dest_width >= 1 or dest_height >= 1:
        dest_wavelet = np.copy(source_wavelet)

        # iterate over only the LL positions
        for i in range(dest_width):
            for j in range(dest_height):
                source_i_min, source_j_min = 2 * i, 2 * j

                # NOTE: all values are stored in as non-averaged int values, do average now

                # a = binary_round(source_wavelet[source_i_min, source_j_min])
                # b = binary_round(source_wavelet[source_i_min, source_j_min + 1])
                # c = binary_round(source_wavelet[source_i_min + 1, source_j_min])
                # d = binary_round(source_wavelet[source_i_min + 1, source_j_min + 1])

                a = source_wavelet[source_i_min, source_j_min]
                b = source_wavelet[source_i_min, source_j_min + 1]
                c = source_wavelet[source_i_min + 1, source_j_min]
                d = source_wavelet[source_i_min + 1, source_j_min + 1]

                ll = (a + b + c + d) / 4
                hl = (a - b + c - d) / 4
                lh = (a + b - c - d) / 4
                hh = (a - b - c + d) / 4

                dest_wavelet[i, j] = ll
                dest_wavelet[i, dest_width + j] = hl
                dest_wavelet[dest_height + i, j] = lh
                dest_wavelet[dest_height + i, dest_width + j] = hh

        source_wavelet = dest_wavelet
        dest_width = dest_width // 2
        dest_height = dest_height // 2

    return dest_wavelet

def gen_spatial(wavelet: ndarray, width, height, dtype=np.uint8) -> ndarray:
    final_width, final_height = wavelet.shape[0] // 2, wavelet.shape[1] // 2
    this_width, this_height = 1, 1
    source_wavelet = wavelet
    dest_wavelet = wavelet.copy()
    level = 0
    while this_width <= final_width and this_height <= final_height:
        for i in range(this_height):
            for j in range(this_width):
                ll = source_wavelet[i, j]
                hl = source_wavelet[i, this_width + j]
                lh = source_wavelet[this_height + i, j]
                hh = source_wavelet[this_height + i, this_width + j]

                a = ll + hl + lh + hh
                b = ll - hl + lh - hh
                c = ll + hl - lh - hh
                d = ll - hl - lh + hh

                dest_wavelet[2 * i, 2 * j] = a
                dest_wavelet[2 * i, 2 * j + 1] = b
                dest_wavelet[2 * i + 1, 2 * j] = c
                dest_wavelet[2 * i + 1, 2 * j + 1] = d

        level += 1
        this_width, this_height = this_width * 2, this_height * 2
        source_wavelet = dest_wavelet
        dest_wavelet = source_wavelet.copy()

    spatial = dest_wavelet[:width, :height].astype(dtype)
    return spatial