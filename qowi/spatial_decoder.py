import qowi.entropy as entropy
import numpy as np
import time
from bitstring import BitStream

from qowi import integers
from qowi.header import Header
from qowi.integer_decoder import IntegerDecoder
from utils.progress_bar import progress_bar

class SpatialDecoder:
    def __init__(self):
        self._header = Header()
        self._output_image = None
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
        return self._output_image

    def to_file(self, filename):
        raise NotImplementedError

    def decode(self):
        if self._finished:
            return

        start_time = time.time()

        if self._bitstream is None:
            raise RuntimeError("Destination must be prepared to encode")

        self._header.read(self._bitstream)
        self._output_image = np.empty((self._header.width, self._header.height, 3), dtype=np.uint8)

        self._read_pixels()

        end_time = time.time()
        self.decode_duration = end_time - start_time
        self._finished = True

    def _read_pixels(self):
        integer_decoder = IntegerDecoder(self._bitstream, self._header.cache_size)

        number_of_tokens = self._header.width * self._header.height
        counter = 1
        for i in range(self._header.width):
            for j in range(self._header.height):
                progress_bar(counter, number_of_tokens)
                counter += 1

                this_shifted = integer_decoder.decode_next()
                this_pixel = integers.shifted_tuple_to_int_tuple(this_shifted, 256)
                self._output_image[i, j] = this_pixel
