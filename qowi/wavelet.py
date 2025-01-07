import math
import numpy as np
from numpy import ndarray
from qowi import integers

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
    def __init__(self, width=0, height=0, color_depth=0, wavelet_levels=10, precision_digits=0):
        self.width = 0
        self.height = 0
        self.color_depth = 0
        self.length = 0
        self.num_levels = 0
        self.wavelet_levels = wavelet_levels
        self.precision_binary_digits = precision_digits
        self.wavelet = None
        self.carry_over = None

        self._initialize_from_shape(width, height, color_depth)

    def _initialize_from_shape(self, width, height, color_depth):
        self.width = width
        self.height = height
        self.color_depth = color_depth
        if width == 0 and height == 0:
            self.num_levels = 0
            self.level = 0
        else:
            self.num_levels = max(math.ceil(math.log2(width)), math.ceil(math.log2(height)))
            self.length = 2 ** self.num_levels

        self.wavelet = np.zeros((self.length, self.length, self.color_depth), dtype=np.int64)

    def _gen_wavelet(self):
        lowest_order_level = max(self.num_levels - self.wavelet_levels, 0)
        for dest_level in reversed(range(lowest_order_level, self.num_levels)):
            dest_length = 2 ** dest_level
            dest_wavelets = np.zeros((2 * dest_length, 2 * dest_length, self.color_depth), dtype=np.int64)

            for i in range(dest_length):
                for j in range(dest_length):
                    a = self.wavelet[2 * i, 2 * j]
                    b = self.wavelet[2 * i, 2 * j + 1]
                    c = self.wavelet[2 * i + 1, 2 * j]
                    d = self.wavelet[2 * i + 1, 2 * j + 1]

                    if self.precision_binary_digits > 0:
                        scaling_factor_digits = (self.num_levels - dest_level) * 2
                        rescale_digits = scaling_factor_digits - self.precision_binary_digits
                        if rescale_digits > 0:
                            a = integers.rescale_ndarray(a, -rescale_digits) # shift right
                            b = integers.rescale_ndarray(b, -rescale_digits)  # shift right
                            c = integers.rescale_ndarray(c, -rescale_digits)  # shift right
                            d = integers.rescale_ndarray(d, -rescale_digits)  # shift right

                    ll, hl, lh, hh = haar_encode(a, b, c, d)

                    dest_wavelets[i, j] = ll
                    dest_wavelets[i, dest_length + j] = hl
                    dest_wavelets[dest_length + i, j] = lh
                    dest_wavelets[dest_length + i, dest_length + j] = hh

            # copy to main wavelets
            self.wavelet[:dest_wavelets.shape[1], :dest_wavelets.shape[1]] = dest_wavelets

    def prepare_from_image(self, image: ndarray):
        self._initialize_from_shape(image.shape[0], image.shape[1], image.shape[2])

        # fill the empty area with zeros and copy source image to top left of wavelet
        self.wavelet[:self.width, :self.height] = image

        # generate the wavelets and carry-over
        self._gen_wavelet()

        return self

    def as_image(self):
        ret_wavelet = self.wavelet.copy()
        lowest_order_level = max(self.num_levels - self.wavelet_levels, 0)
        for source_level in range(lowest_order_level, self.num_levels):
            source_length = 2 ** source_level
            dest_wavelets = np.zeros((2 * source_length, 2 * source_length, self.color_depth), dtype=np.int64)

            for i in range(source_length):
                for j in range(source_length):
                    ll = ret_wavelet[i, j]
                    hl = ret_wavelet[i, source_length + j]
                    lh = ret_wavelet[source_length + i, j]
                    hh = ret_wavelet[source_length + i, source_length + j]

                    a, b, c, d = haar_decode(ll, hl, lh, hh)

                    if self.precision_binary_digits > 0:
                        scaling_factor_digits = (self.num_levels - source_level) * 2
                        rescale_digits = scaling_factor_digits - self.precision_binary_digits
                        if rescale_digits > 0:
                            a = integers.rescale_ndarray(a, rescale_digits) # shift left
                            b = integers.rescale_ndarray(b, rescale_digits) # shift left
                            c = integers.rescale_ndarray(c, rescale_digits) # shift left
                            d = integers.rescale_ndarray(d, rescale_digits) # shift left

                    dest_wavelets[2 * i, 2 * j] = a
                    dest_wavelets[2 * i, 2 * j + 1] = b
                    dest_wavelets[2 * i + 1, 2 * j] = c
                    dest_wavelets[2 * i + 1, 2 * j + 1] = d

            ret_wavelet[:dest_wavelets.shape[0], :dest_wavelets.shape[1]] = dest_wavelets

        return ret_wavelet[:self.width, :self.height].astype(np.uint8)

    def apply_hard_threshold(self, threshold: float):
        if threshold == 0:
            return

        lowest_order_level = max(self.num_levels - self.wavelet_levels, 0)
        this_threshold = None
        for this_level in range(lowest_order_level, self.num_levels):
            this_length = 2 ** this_level

            # rescale the threshold in order to perform the calculation in the int domain
            scaling_factor_digits = (self.num_levels - this_level) * 2
            this_threshold = int(round(threshold * 2 ** scaling_factor_digits))
            if self.precision_binary_digits > 0:
                rescale_digits = scaling_factor_digits - self.precision_binary_digits
                if rescale_digits > 0:
                    this_threshold = int(round(threshold * 2 ** rescale_digits))

            for i in range(this_length):
                for j in range(this_length):
                    hl = self.wavelet[i, this_length + j]
                    lh = self.wavelet[this_length + i, j]
                    hh = self.wavelet[this_length + i, this_length + j]

                    hl[np.abs(hl) < this_threshold] = 0
                    lh[np.abs(lh) < this_threshold] = 0
                    hh[np.abs(hh) < this_threshold] = 0

    def apply_soft_threshold(self, threshold: float):
        if threshold == -1:
            return

        lowest_order_level = max(self.num_levels - self.wavelet_levels, 0)
        this_threshold = None
        for this_level in range(lowest_order_level, self.num_levels):
            this_length = 2 ** this_level

            # rescale the threshold in order to perform the calculation in the int domain
            scaling_factor_digits = (self.num_levels - this_level) * 2
            this_threshold = int(round(threshold * 2 ** scaling_factor_digits))
            if self.precision_binary_digits > 0:
                rescale_digits = scaling_factor_digits - self.precision_binary_digits
                if rescale_digits > 0:
                    this_threshold = int(round(threshold * 2 ** rescale_digits))

            for i in range(this_length):
                for j in range(this_length):
                    hl = self.wavelet[i, this_length + j]
                    lh = self.wavelet[this_length + i, j]
                    hh = self.wavelet[this_length + i, this_length + j]

                    hl[:] = np.sign(hl) * np.maximum(np.abs(hl) - this_threshold, 0)
                    lh[:] = np.sign(lh) * np.maximum(np.abs(lh) - this_threshold, 0)
                    hh[:] = np.sign(hh) * np.maximum(np.abs(hh) - this_threshold, 0)
