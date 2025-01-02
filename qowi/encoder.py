import numpy as np
import sys
from bitstring import BitArray, Bits
from qowi.lru_cache import LRUCache
from qowi.primitives import PList, PUnsignedInteger
from qowi.wavelet import Wavelet

CACHE_SIZE = 2048
ZERO_TOKEN = (0, 0, 0)
OP_CODE_RUN = Bits('0b0')
OP_CODE_CACHE = Bits('0b10')
OP_CODE_VALUE = Bits('0b11')

def min_non_zero(*values) -> int:
    smallest = sys.maxsize
    for v in values:
        if 0 < v < smallest:
            smallest = v
    return smallest

class Encoder:
    def __init__(self, wavelet: Wavelet):
        self._wavelet = wavelet
        self._run_length = 0
        self._last_token = None

        self._cache = LRUCache(CACHE_SIZE)
        self._cache.observe(ZERO_TOKEN)

        self._encoded = False
        self._response = BitArray()

    def _gen_run_encoding(self):
        ret = BitArray()
        ret.append(OP_CODE_RUN)
        ret.append(PUnsignedInteger(self._run_length - 1).entropy_coded)
        return ret

    def _gen_cache_encoding(self, difference: np.ndarray):
        ret = BitArray()

        try:
            this_token = PList.from_ndarray(difference).token
            pos = self._cache.index(this_token)
        except ValueError:
            return ret

        ret.append(OP_CODE_CACHE)
        ret.append(PUnsignedInteger(pos).entropy_coded)
        return ret

    def _gen_difference_value_encoding(self, difference: np.ndarray):
        ret = BitArray()
        ret.append(OP_CODE_VALUE)
        ret.append(PList.from_ndarray(difference).entropy_coded)
        return ret

    def _encode_difference(self, difference: np.ndarray):
        this_token = PList.from_ndarray(difference).token

        if this_token == self._last_token:
            self._run_length += 1
            return

        ### this_pixel does not equal last_pixel ###

        if self._run_length > 1:
            run = self._gen_run_encoding()
            self._response.append(run)
            self._run_length = 0

        cached = self._gen_cache_encoding(difference)
        value = self._gen_difference_value_encoding(difference)

        smallest_length = min_non_zero(len(cached), len(value))
        if len(cached) == smallest_length:
            self._response.append(cached)
            self._cache.observe(this_token)
        elif len(value) == smallest_length:
            self._response.append(value)
            self._cache.observe(this_token)
        else:
            raise ValueError("Both cached and value encodings were zero length")

    def _encode_carry_over(self, carry_over: np.ndarray):
        for v in carry_over:
            v_uint = int((v % 1.0) * 4)
            self._response.append(Bits(uint=v_uint, length=2))

    def encode(self) -> BitArray:
        if self._encoded:
            return self._response

        # encode configuration values
        self._response.append(Bits(uint=self._wavelet.width, length=16))
        self._response.append(Bits(uint=self._wavelet.height, length=16))

        # encode the top value of the wavelet
        root_pixel = PList.from_ndarray(self._wavelet.wavelet[0, 0])
        self._response.append(root_pixel.entropy_coded)

        ### iterate over the wavelet encoding difference and carry-over values ###

        for source_level in range(self._wavelet.num_levels):
            source_length = 2 ** source_level

            for i in range(source_length):
                for j in range(source_length):
                    hl = self._wavelet.wavelet[i, source_length + j]
                    lh = self._wavelet.wavelet[source_length + i, j]
                    hh = self._wavelet.wavelet[source_length + i, source_length + j]

                    self._encode_difference(hl)
                    self._encode_difference(lh)
                    self._encode_difference(hh)

                    if source_level < self._wavelet.num_levels - 1:
                        a_co = self._wavelet.carry_over[source_level + 1][2 * i, 2 * j]
                        b_co = self._wavelet.carry_over[source_level + 1][2 * i, 2 * j + 1]
                        c_co = self._wavelet.carry_over[source_level + 1][2 * i + 1, 2 * j]
                        d_co = self._wavelet.carry_over[source_level + 1][2 * i + 1, 2 * j + 1]

                        self._encode_carry_over(a_co)
                        self._encode_carry_over(b_co)
                        self._encode_carry_over(c_co)
                        self._encode_carry_over(d_co)

        # flush the run length
        if self._run_length > 1:
            run = self._gen_run_encoding()
            self._response.append(run)
            self._run_length = 0

        self._encoded = True
        return self._response
