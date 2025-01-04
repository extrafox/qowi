import qowi.entropy as entropy
import numpy as np
import qowi.uint10 as uint10
from bitstring import BitStream
from qowi.header import Header
from qowi.uint10_decoder import Uint10Decoder
from qowi.wavelet import Wavelet


class QOWIDecoder:
    def __init__(self):
        self._header = Header()
        self._wavelet = None
        self._uint10_array = None
        self._bitstream = None
        self._finished = False

    def from_bitstream(self, bitstream: BitStream):
        self._bitstream = bitstream

    def from_file(self, filename):
        raise NotImplementedError

    def as_array(self) -> np.ndarray:
        if not self._finished:
            raise RuntimeError("Decoder must be finished")
        return self._wavelet.as_image()

    def to_file(self, filename):
        raise NotImplementedError

    def decode(self):
        if self._finished:
            return

        if self._bitstream is None:
            raise RuntimeError("Destination must be prepared to encode")

        self._header.read(self._bitstream)
        self._wavelet = Wavelet(self._header.width, self._header.height)
        self._uint10_array = self._wavelet.as_uint10_array()

        # decode the top value of the wavelet
        root_pixel = entropy.decode_array(self._bitstream, 3)
        self._uint10_array[0, 0] = root_pixel

        self._read_coefficients()
        self._read_carry_over_bits()

        self._finished = True

    def _read_coefficients(self):
        stack = [(0, 'HH', 0, 0), (0, 'LH', 0, 0), (0, 'HL', 0, 0)]
        uint10_decoder = Uint10Decoder(self._bitstream, self._header.cache_size)

        while len(stack) > 0:
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

            # decode this coefficient
            this_coefficient = uint10_decoder.decode_next()
            self._uint10_array[i + i_offset][j + j_offset] = this_coefficient

            # append children to the stack
            if level + 1 < self._wavelet.num_levels:
                stack.append((level + 1, filter, 2 * i, 2 * j))
                stack.append((level + 1, filter, 2 * i, 2 * j + 1))
                stack.append((level + 1, filter, 2 * i + 1, 2 * j))
                stack.append((level + 1, filter, 2 * i + 1, 2 * j + 1))

        self._wavelet.from_uint10_array(self._uint10_array)

    def _read_carry_over_bits(self):
        if self._header.num_carry_over_bits == 0:
            return

        co_bits = self._wavelet.carry_over
        for level in range(len(co_bits)):
            level_bits = co_bits[level]
            for i in range(level_bits.shape[0]):
                for j in range(level_bits.shape[1]):
                    for v_i in range(3):

                        # processing one color channel at a time
                        if self._header.num_carry_over_bits == 2:
                            value = self._bitstream.read(2).uint
                            level_bits[i, j, v_i] = value / 4

                        elif self._header.num_carry_over_bits == 1:
                            value = self._bitstream.read(1).uint
                            level_bits[i, j, v_i] = value / 2
