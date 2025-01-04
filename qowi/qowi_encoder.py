import sys

import numpy as np
import qowi.entropy as entropy
from bitstring import Bits, BitStream
from qowi.uint10_encoder import Uint10Encoder
from skimage import io
from qowi.header import Header
from qowi.wavelet import Wavelet


def _progress_bar(progress, total, bar_width=80):
    filled_length = int(bar_width * progress // total)
    bar = '=' * filled_length + '-' * (bar_width - filled_length)
    sys.stdout.write(f'\r|{bar}| {progress}/{total} ({(progress / total) * 100:.2f}%)')
    sys.stdout.flush()

class QOWIEncoder:
    def __init__(self, hard_threshold=0, soft_threshold=0, bit_shift=0, num_carry_over_bits=2):
        self._hard_threshold = max(0, min(hard_threshold, 510))
        self._soft_threshold = max(0, min(soft_threshold, 510))

        self._header = Header()
        self._header.bit_shift = max(0, min(bit_shift, 7))
        self._header.num_carry_over_bits = max(0, min(num_carry_over_bits, 2))

        self._wavelet = Wavelet()
        self._uint10_array = None
        self._bitstream = None

        self._finished = False
        self.stats = {}

    def from_array(self, array: np.ndarray):
        self._wavelet.prepare_from_image(array)
        self._header.width = self._wavelet.width
        self._header.height = self._wavelet.height
        self._uint10_array = self._wavelet.as_uint10_array()

    def from_file(self, filename):
        self.from_array(io.imread(filename))

    def to_bitstream(self, bitstream: BitStream):
        self._bitstream = bitstream

    def to_file(self, filename):
        raise NotImplementedError

    def encode(self):
        if self._finished:
            return

        if self._bitstream is None:
            raise RuntimeError("Destination must be prepared to encode")
        if self._wavelet is None:
            raise RuntimeError("Source must be prepared to encode")

        self._header.write(self._bitstream)

        # encode the top value of the wavelet
        root_pixel = self._uint10_array[0, 0]
        self._bitstream.append(entropy.encode_array(root_pixel))

        self._write_coefficients()
        self._write_carry_over_bits()

        self._finished = True

    def _write_coefficients(self):
        stack = [(0, 'HH', 0, 0), (0, 'LH', 0, 0), (0, 'HL', 0, 0)]
        uint10_encoder = Uint10Encoder(self._bitstream, self._header.cache_size)

        number_of_tokens = self._wavelet.length ** 2 - 1
        counter = 1
        while len(stack) > 0:
            _progress_bar(counter, number_of_tokens)
            counter += 1

            level, filter, i, j = stack.pop()

            level_length = 2 ** level
            if filter == 'HL':
                i_offset = 0
                j_offset = level_length
            elif filter == 'LH':
                i_offset = level_length
                j_offset = 0
            elif filter == 'HH':
                i_offset = level_length
                j_offset = level_length
            else:
                raise ValueError("Unknown filter '{}'".format(filter))

            # encode this coefficient
            this_coefficient = self._uint10_array[i + i_offset][j + j_offset]
            uint10_encoder.encode_next(tuple(this_coefficient))

            # append children to the stack
            if level + 1 < self._wavelet.num_levels:
                stack.append((level + 1, filter, 2 * i, 2 * j))
                stack.append((level + 1, filter, 2 * i, 2 * j + 1))
                stack.append((level + 1, filter, 2 * i + 1, 2 * j))
                stack.append((level + 1, filter, 2 * i + 1, 2 * j + 1))

        uint10_encoder.finish()
        self.stats = uint10_encoder.stats

    def _write_carry_over_bits(self):
        if self._header.num_carry_over_bits == 0:
            return

        co_bits = self._wavelet.carry_over
        for level in range(len(co_bits)):
            level_bits = co_bits[level]
            for i in range(level_bits.shape[0]):
                for j in range(level_bits.shape[1]):
                    for v in level_bits[i, j]:

                        # processing one color channel at a time
                        v_uint = int((v % 1.0) * 4)
                        if v_uint < 0 or v_uint > 3:
                            raise ValueError("Carry over v_uint {} must be between 0 and 3".format(v))

                        if self._header.num_carry_over_bits == 2:
                            self._bitstream.append(Bits(uint=v_uint, length=2))
                        elif self._header.num_carry_over_bits == 1:
                            v_uint = v_uint >> 1
                            self._bitstream.append(Bits(uint=v_uint, length=1))