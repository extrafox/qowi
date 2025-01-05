import warnings
import numpy as np
import time
from bitstring import Bits, BitStream
from qowi.integer_encoder import IntegerEncoder
from skimage import io
from qowi.header import Header
from utils.progress_bar import progress_bar

DEFAULT_CACHE_SIZE = 256

class SpatialEncoder:
    def __init__(self):
        self._header = Header()
        self._header.cache_size = DEFAULT_CACHE_SIZE

        if self._header.bit_shift > 0:
            warnings.warn("Bit shift is currently not implemented", category=UserWarning)

        self._source_image = None
        self._bitstream = None

        self._finished = False
        self.stats = {}
        self.encode_duration = 0

    def from_array(self, array: np.ndarray):
        self._source_image = array

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
        if self._source_image is None:
            raise RuntimeError("Source must be prepared to encode")

        self._header.width = self._source_image.shape[0]
        self._header.height = self._source_image.shape[1]
        self._header.write(self._bitstream)

        self._write_pixels()

        end_time = time.time()
        self.encode_duration = end_time - start_time
        self._finished = True

    def _write_pixels(self):
        integer_encoder = IntegerEncoder(self._bitstream, DEFAULT_CACHE_SIZE)
        self._header.cache_size = DEFAULT_CACHE_SIZE

        number_of_tokens = self._source_image.shape[0] * self._source_image.shape[1]
        counter = 1
        for i in range(self._header.width):
            for j in range(self._header.height):
                progress_bar(counter, number_of_tokens)
                counter += 1

                this_pixel = self._source_image[i, j]
                integer_encoder.encode_next(tuple(this_pixel))

        integer_encoder.finish()
        self.stats = integer_encoder.stats
