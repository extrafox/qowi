import math
import numpy as np
from numpy import ndarray

def haar_encode(a, b, c, d):
    ll = a + b + c + d
    hl = a + b - c - d
    lh = a - b + c - d
    hh = a - b - c + d

    return ll, hl, lh, hh

def haar_decode(ll, hl, lh, hh):
    a = (ll + hl + lh + hh) // 4
    b = (ll + hl - lh - hh) // 4
    c = (ll - hl + lh - hh) // 4
    d = (ll - hl - lh + hh) // 4

    return a, b, c, d

class Wavelet:
    def __init__(self, width=0, height=0, encode_levels=10, round_bits=0):
        self.width = 0
        self.height = 0
        self.length = 0
        self.num_levels = 0
        self.encode_levels = encode_levels
        self.round_bits = round_bits
        self.wavelet = None
        self.carry_over = None

        self._initialize_from_shape(width, height)

    def _initialize_from_shape(self, width, height):
        self.width = width
        self.height = height
        if width == 0 and height == 0:
            self.num_levels = 0
            self.level = 0
        else:
            self.num_levels = max(math.ceil(math.log2(width)), math.ceil(math.log2(height)))
            self.length = 2 ** self.num_levels

        self.wavelet = np.zeros((self.length, self.length, 3), dtype=np.int64)

    def _gen_wavelet(self):
        for dest_level in reversed(range(self.num_levels)):
            dest_length = 2 ** dest_level
            dest_wavelets = np.zeros((2 * dest_length, 2 * dest_length, 3), dtype=np.int64)

            for i in range(dest_length):
                for j in range(dest_length):
                    a = self.wavelet[2 * i, 2 * j]
                    b = self.wavelet[2 * i, 2 * j + 1]
                    c = self.wavelet[2 * i + 1, 2 * j]
                    d = self.wavelet[2 * i + 1, 2 * j + 1]

                    ll, hl, lh, hh = haar_encode(a, b, c, d)

                    dest_wavelets[i, j] = ll
                    dest_wavelets[i, dest_length + j] = hl
                    dest_wavelets[dest_length + i, j] = lh
                    dest_wavelets[dest_length + i, dest_length + j] = hh

            # copy to main wavelets
            self.wavelet[:dest_wavelets.shape[1], :dest_wavelets.shape[1]] = dest_wavelets

    def prepare_from_image(self, image: ndarray):
        self._initialize_from_shape(image.shape[0], image.shape[1])

        # fill the empty area with zeros and copy source image to top left of wavelet
        self.wavelet[:self.width, :self.height] = image

        # generate the wavelets and carry-over
        self._gen_wavelet()

        return self

    def as_image(self):
        ret_wavelet = self.wavelet.copy()

        for source_level in range(self.num_levels):
            source_length = 2 ** source_level
            dest_wavelets = np.zeros((2 * source_length, 2 * source_length, 3), dtype=np.int64)

            for i in range(source_length):
                for j in range(source_length):
                    ll = ret_wavelet[i, j]
                    hl = ret_wavelet[i, source_length + j]
                    lh = ret_wavelet[source_length + i, j]
                    hh = ret_wavelet[source_length + i, source_length + j]

                    a, b, c, d = haar_decode(ll, hl, lh, hh)

                    dest_wavelets[2 * i, 2 * j] = a
                    dest_wavelets[2 * i, 2 * j + 1] = b
                    dest_wavelets[2 * i + 1, 2 * j] = c
                    dest_wavelets[2 * i + 1, 2 * j + 1] = d

            ret_wavelet[:dest_wavelets.shape[0], :dest_wavelets.shape[1]] = dest_wavelets

        return ret_wavelet[:self.width, :self.height].astype(np.uint8)

    def apply_hard_threshold(self, threshold: float):
        root_element = self.wavelet[0, 0]
        self.wavelet[np.abs(self.wavelet) < threshold] = 0
        self.wavelet[0, 0] = root_element

    def apply_soft_threshold(self, threshold: float):
        root_element = self.wavelet[0, 0]
        self.wavelet = np.where(np.abs(self.wavelet) < threshold,
                                np.sign(self.wavelet) * np.maximum(np.abs(self.wavelet) - threshold, 0),
                                self.wavelet)
        self.wavelet[0, 0] = root_element

