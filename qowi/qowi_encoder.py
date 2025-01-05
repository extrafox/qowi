import numpy as np
import qowi.entropy as entropy
import time
from bitstring import Bits, BitStream
from qowi import integers
from qowi.integer_encoder import IntegerEncoder
from skimage import io
from qowi.header import Header
from qowi.wavelet import Wavelet
from utils.progress_bar import progress_bar

DEFAULT_CACHE_SIZE = 65533

class QOWIEncoder:
    def __init__(self, hard_threshold=-1, soft_threshold=-1):
        self._hard_threshold = max(-1, min(hard_threshold, 510))
        self._soft_threshold = max(-1, min(soft_threshold, 510))

        self._header = Header()
        self._header.cache_size = DEFAULT_CACHE_SIZE

        self._wavelet = Wavelet()
        self._bitstream = None

        self._finished = False
        self.stats = {}
        self.encode_duration = 0

    def from_array(self, array: np.ndarray):
        self._wavelet.prepare_from_image(array)
        self._header.width = self._wavelet.width
        self._header.height = self._wavelet.height

    def from_file(self, filename):
        self.from_array(io.imread(filename))

    def to_bitstream(self, bitstream: BitStream):
        self._bitstream = bitstream

    def to_file(self, filename):
        raise NotImplementedError

    def encode(self):
        if self._finished:
            return

        start_time = time.time()

        if self._bitstream is None:
            raise RuntimeError("Destination must be prepared to encode")
        if self._wavelet is None:
            raise RuntimeError("Source must be prepared to encode")

        if self._hard_threshold > -1:
            self._wavelet.apply_hard_threshold(self._hard_threshold)
        elif self._soft_threshold > -1:
            self._wavelet.apply_soft_threshold(self._soft_threshold)

        self._header.write(self._bitstream)

        # encode the top value of the wavelet
        root_integer = self._wavelet.wavelet[0, 0]
        root_zigzag = integers.int_tuple_to_zigzag_tuple(root_integer)
        self._bitstream.append(entropy.simple_encode_tuple(root_zigzag))

        self._write_coefficients()

        end_time = time.time()
        self.encode_duration = end_time - start_time
        self._finished = True

    def _write_coefficients(self):
        stack = [(0, 'HH', 0, 0), (0, 'LH', 0, 0), (0, 'HL', 0, 0)]
        integer_encoder = IntegerEncoder(self._bitstream, DEFAULT_CACHE_SIZE)

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

            # encode this coefficient
            this_integer = self._wavelet.wavelet[i + i_offset][j + j_offset]
            integer_encoder.encode_next(tuple(this_integer))

            # append children to the stack
            if level + 1 < self._wavelet.num_levels:
                stack.append((level + 1, filter, 2 * i, 2 * j))
                stack.append((level + 1, filter, 2 * i, 2 * j + 1))
                stack.append((level + 1, filter, 2 * i + 1, 2 * j))
                stack.append((level + 1, filter, 2 * i + 1, 2 * j + 1))

        integer_encoder.finish()
        self.stats = integer_encoder.stats
