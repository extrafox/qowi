import qowi.entropy as entropy
import numpy as np
import qowi.integers as integers
import time
from bitstring import BitStream
from qowi.header import Header
from qowi.integer_decoder import IntegerDecoder
from qowi.wavelet import Wavelet
from utils.progress_bar import progress_bar


class QOWIDecoder:
    def __init__(self):
        self._header = Header()
        self._wavelet = None
        self._bitstream = None
        self._finished = False
        self.decode_duration = 0

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

        start_time = time.time()

        if self._bitstream is None:
            raise RuntimeError("Destination must be prepared to encode")

        self._header.read(self._bitstream)
        self._wavelet = Wavelet(self._header.width, self._header.height)

        # decode the top value of the wavelet
        root_zigzag = entropy.golomb_decode_tuple(self._bitstream, 3)
        root_integer = integers.zigzag_tuple_to_int_tuple(root_zigzag)
        self._wavelet.wavelet[0, 0] = root_integer

        self._read_coefficients()

        end_time = time.time()
        self.decode_duration = end_time - start_time
        self._finished = True

    def _read_coefficients(self):
        stack = [(0, 'HH', 0, 0), (0, 'LH', 0, 0), (0, 'HL', 0, 0)]
        integer_decoder = IntegerDecoder(self._bitstream, self._header.cache_size)

        number_of_tokens = self._wavelet.length ** 2 - 1
        counter = 1
        while len(stack) > 0:
            progress_bar(counter, number_of_tokens)
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

            # decode this coefficient
            this_integer = integer_decoder.decode_next()
            self._wavelet.wavelet[i + i_offset][j + j_offset] = this_integer

            # append children to the stack
            if level + 1 < self._wavelet.num_levels:
                stack.append((level + 1, filter, 2 * i, 2 * j))
                stack.append((level + 1, filter, 2 * i, 2 * j + 1))
                stack.append((level + 1, filter, 2 * i + 1, 2 * j))
                stack.append((level + 1, filter, 2 * i + 1, 2 * j + 1))
