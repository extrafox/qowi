import math
import numpy as np
from numpy import ndarray
from qowi.haar_sort_table import HaarSortTable

class HaarSortWavelet:
    def __init__(self, haar_sort_table, width=0, height=0, color_depth=0, haar_sort_depth=8):
        self.width = None
        self.height = None
        self.color_depth = None
        self.haar_sort_table = HaarSortTable(haar_sort_depth, haar_sort_table)
        self.length = 0
        self.num_levels = 0
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

        self.wavelet = np.zeros((self.length, self.length, self.color_depth), dtype=np.uint8)

    def _gen_wavelet(self):
        for dest_level in reversed(range(0, self.num_levels)):
            dest_length = 2 ** dest_level
            dest_wavelets = np.zeros((2 * dest_length, 2 * dest_length, self.color_depth), dtype=np.uint8)

            for i in range(dest_length):
                for j in range(dest_length):
                    a = self.wavelet[2 * i, 2 * j]
                    b = self.wavelet[2 * i, 2 * j + 1]
                    c = self.wavelet[2 * i + 1, 2 * j]
                    d = self.wavelet[2 * i + 1, 2 * j + 1]

                    # TODO: handle multiple haar sort bit depths

                    ll = np.empty(3, dtype=np.uint8)
                    hl = np.empty(3, dtype=np.uint8)
                    lh = np.empty(3, dtype=np.uint8)
                    hh = np.empty(3, dtype=np.uint8)

                    for p in range(3):
                        ll[p], hl[p], lh[p], hh[p] = self.haar_sort_table.grid_to_haar_sort_components((a[p], b[p], c[p], d[p]))

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

    def prepare_from_wavelet(self, wavelet: ndarray):
        self._initialize_from_shape(wavelet.shape[0], wavelet.shape[1], wavelet.shape[2])

        self.wavelet = wavelet

    def as_image(self):
        ret_wavelet = self.wavelet.copy()
        for source_level in range(0, self.num_levels):
            source_length = 2 ** source_level
            dest_wavelets = np.zeros((2 * source_length, 2 * source_length, self.color_depth), dtype=np.uint8)

            for i in range(source_length):
                for j in range(source_length):
                    ll = ret_wavelet[i, j]
                    hl = ret_wavelet[i, source_length + j]
                    lh = ret_wavelet[source_length + i, j]
                    hh = ret_wavelet[source_length + i, source_length + j]

                    a = np.empty(3, dtype=np.uint8)
                    b = np.empty(3, dtype=np.uint8)
                    c = np.empty(3, dtype=np.uint8)
                    d = np.empty(3, dtype=np.uint8)

                    for p in range(3):
                        a[p], b[p], c[p], d[p] = self.haar_sort_table.grid_to_haar_sort_components((ll[p], hl[p], lh[p], hh[p]))

                    dest_wavelets[2 * i, 2 * j] = a
                    dest_wavelets[2 * i, 2 * j + 1] = b
                    dest_wavelets[2 * i + 1, 2 * j] = c
                    dest_wavelets[2 * i + 1, 2 * j + 1] = d

            ret_wavelet[:dest_wavelets.shape[0], :dest_wavelets.shape[1]] = dest_wavelets

        return ret_wavelet[:self.width, :self.height].astype(np.uint8)
