import numpy as np
import sys
from bitstring import BitArray, Bits, BitStream
from qowi.lru_cache import LRUCache
from qowi.wavelet import Wavelet
from qowi.entropy import entropy_decode, entropy_encode

CACHE_SIZE = 2048
ZERO_TOKEN = (0, 0, 0)

def difference_to_int(a: np.ndarray) -> np.ndarray:
    return (a * 4).astype(int)

def int_to_difference(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16) / 4

def difference_array_to_token(a: np.ndarray) -> tuple:
    return tuple(difference_to_int(a))

def token_to_difference_array(t: tuple):
    return int_to_difference(np.array(t))

class Decoder:
    def __init__(self, bitstream: BitStream):
        self._wavelet = None
        self._bitstream = bitstream
        self._run_length = 0
        self._last_token = None

        self._cache = LRUCache(CACHE_SIZE)
        self._cache.observe(ZERO_TOKEN)

        self._decoded = False

    def _decode_next_difference(self):
        if self._run_length > 0:
            self._run_length -= 1
            return token_to_difference_array(self._last_token)

        op_code = self._bitstream.read(1).uint

        if op_code == 0: # RUN op code
            self._run_length = entropy_decode(self._bitstream) # no point in incrementing, then decrementing
            return token_to_difference_array(self._last_token)

        op_code = (op_code << 1) + self._bitstream.read(1).uint

        if op_code == 2: # CACHE op code
            pos = entropy_decode(self._bitstream)
            this_token = self._cache[pos]
            self._last_token = this_token
            self._cache.observe(this_token)
            return token_to_difference_array(this_token)

        elif op_code == 3: # VALUE op code
            this_token = [None, None, None]
            for i in range(3):
                sign = self._bitstream.read(1).uint
                this_token[i] = entropy_decode(self._bitstream)
                if sign == 1:
                    this_token[i] = this_token[i] * -1
            this_token = tuple(this_token)
            self._last_token = this_token
            self._cache.observe(this_token)
            return token_to_difference_array(this_token)
        else:
            raise ValueError("Invalid op code value {}".format(op_code))

    def _decode_carry_over(self):
        carry_over = np.array([
            self._bitstream.read(2).uint,
            self._bitstream.read(2).uint,
            self._bitstream.read(2).uint,
        ], dtype=np.float16)
        return int_to_difference(carry_over)

    def decode(self) -> Wavelet:
        if self._decoded:
            return self._wavelet.as_image()

        # decode configuration values
        width = self._bitstream.read(16).uint
        height = self._bitstream.read(16).uint
        self._wavelet = Wavelet(width, height)

        # decode the top value of the wavelet
        r = self._bitstream.read(10).uint
        g = self._bitstream.read(10).uint
        b = self._bitstream.read(10).uint
        self._wavelet.wavelet[0, 0] = int_to_difference(np.array((r, g, b), dtype=np.float16))

        ### iterate over the wavelet encoding difference and carry-over values ###

        for source_level in range(self._wavelet.num_levels):
            source_length = 2 ** source_level

            for i in range(source_length):
                for j in range(source_length):

                    # TODO: read filters from stream and write to wavelet

                    hl = self._decode_next_difference()
                    lh = self._decode_next_difference()
                    hh = self._decode_next_difference()

                    self._wavelet.wavelet[i, source_length + j] = hl
                    self._wavelet.wavelet[source_length + i, j] = lh
                    self._wavelet.wavelet[source_length + i, source_length + j] = hh

                    # TODO: read carry_over bits from stream and write to wavelet

                    if source_level < self._wavelet.num_levels - 1:
                        a_co = self._decode_carry_over()
                        b_co = self._decode_carry_over()
                        c_co = self._decode_carry_over()
                        d_co = self._decode_carry_over()

                        self._wavelet.carry_over[source_level + 1][2 * i, 2 * j] = a_co
                        self._wavelet.carry_over[source_level + 1][2 * i, 2 * j + 1] = b_co
                        self._wavelet.carry_over[source_level + 1][2 * i + 1, 2 * j] = c_co
                        self._wavelet.carry_over[source_level + 1][2 * i + 1, 2 * j + 1] = d_co

        self._decoded = True
        return self._wavelet.as_image()