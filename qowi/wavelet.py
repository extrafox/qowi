import math
import numpy as np
from numpy import ndarray

class Wavelet:
    def __init__(self, width=0, height=0):
        self.width = 0
        self.height = 0
        self.length = 0
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

        self.wavelet = np.zeros((self.length, self.length, 3), dtype=np.float16)
        self._gen_carry_over()

    def _gen_carry_over(self):
        self.carry_over = []
        for level in range(self.num_levels):
            self.carry_over.append(np.zeros((2 ** level, 2 ** level, 3), dtype=np.float16))

    def _gen_wavelet(self):
        for dest_level in reversed(range(self.num_levels)):
            dest_length = 2 ** dest_level
            dest_wavelets = np.zeros((2 * dest_length, 2 * dest_length, 3), dtype=np.float16)

            for i in range(dest_length):
                for j in range(dest_length):
                    a10 = self.wavelet[2 * i, 2 * j]
                    b10 = self.wavelet[2 * i, 2 * j + 1]
                    c10 = self.wavelet[2 * i + 1, 2 * j]
                    d10 = self.wavelet[2 * i + 1, 2 * j + 1]

                    a8 = np.trunc(a10)
                    b8 = np.trunc(b10)
                    c8 = np.trunc(c10)
                    d8 = np.trunc(d10)

                    if dest_level < self.num_levels - 1:
                        a_co = a10 - a8
                        b_co = b10 - b8
                        c_co = c10 - c8
                        d_co = d10 - d8

                        self.carry_over[dest_level + 1][2 * i, 2 * j] = a_co
                        self.carry_over[dest_level + 1][2 * i, 2 * j + 1] = b_co
                        self.carry_over[dest_level + 1][2 * i + 1, 2 * j] = c_co
                        self.carry_over[dest_level + 1][2 * i + 1, 2 * j + 1] = d_co

                    ll = (a8 + b8 + c8 + d8) / 4
                    hl = (a8 - b8 + c8 - d8) / 4
                    lh = (a8 + b8 - c8 - d8) / 4
                    hh = (a8 - b8 - c8 + d8) / 4

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

    # TODO: move the lossy methods to where they modify the wavelet itself

    def as_image(self):
        ret_wavelet = self.wavelet.copy()

        for source_level in range(self.num_levels):
            source_length = 2 ** source_level
            dest_wavelets = np.zeros((2 * source_length, 2 * source_length, 3), dtype=np.float16)

            for i in range(source_length):
                for j in range(source_length):
                    ll = ret_wavelet[i, j]
                    hl = ret_wavelet[i, source_length + j]
                    lh = ret_wavelet[source_length + i, j]
                    hh = ret_wavelet[source_length + i, source_length + j]

                    a8 = ll + hl + lh + hh
                    b8 = ll - hl + lh - hh
                    c8 = ll + hl - lh - hh
                    d8 = ll - hl - lh + hh

                    if source_level < self.num_levels - 1:
                        a_co = self.carry_over[source_level + 1][2 * i, 2 * j]
                        b_co = self.carry_over[source_level + 1][2 * i, 2 * j + 1]
                        c_co = self.carry_over[source_level + 1][2 * i + 1, 2 * j]
                        d_co = self.carry_over[source_level + 1][2 * i + 1, 2 * j + 1]
                    else:
                        a_co, b_co, c_co, d_co = 0, 0, 0, 0

                    dest_wavelets[2 * i, 2 * j] = a8 + a_co
                    dest_wavelets[2 * i, 2 * j + 1] = b8 + b_co
                    dest_wavelets[2 * i + 1, 2 * j] = c8 + c_co
                    dest_wavelets[2 * i + 1, 2 * j + 1] = d8 + d_co

            ret_wavelet[:dest_wavelets.shape[0], :dest_wavelets.shape[1]] = dest_wavelets

        return np.round(ret_wavelet[:self.width, :self.height]).astype(np.uint8)

    def apply_hard_threshold(self, threshold: float):
        root_element = self.wavelet[0, 0]
        self.wavelet[self.wavelet < threshold] = 0
        self.wavelet[0, 0] = root_element

    def apply_soft_threshold(self, threshold: float):
        root_element = self.wavelet[0, 0]
        self.wavelet = np.sign(self.wavelet) * np.maximum(np.abs(self.wavelet) - threshold, 0)
        self.wavelet[0, 0] = root_element
