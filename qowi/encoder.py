import numpy as np
import sys
from bitstring import BitArray, Bits, BitStream
from qowi.lru_cache import LRUCache
from qowi.wavelet import Wavelet
from qowi.entropy import entropy_encode

CACHE_SIZE = 2048
ZERO_TOKEN = (0, 0, 0)

def difference_to_int(a: np.ndarray) -> np.ndarray:
    return (a * 4).astype(int)

def difference_array_to_token(a: np.ndarray) -> tuple:
    return tuple(difference_to_int(a))

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
        entropy_coded = entropy_encode(self._run_length - 1)
        ret.append(Bits(uint=0, length=1)) # RUN op code
        ret.append(entropy_coded)
        return ret

    def _gen_cache_encoding(self, difference: np.ndarray):
        ret = BitArray()

        try:
            this_token = difference_array_to_token(difference)
            pos = self._cache.index(this_token)
        except ValueError:
            return ret

        entropy_coded = entropy_encode(pos)
        ret.append(BitArray(Bits(uint=2, length=2))) # CACHE op code
        ret.append(entropy_coded)
        return ret

    def _gen_difference_value_encoding(self, difference: np.ndarray):
        ret = BitArray()
        ret.append(Bits(uint=3, length=2)) # VALUE op code

        diff_int = difference_to_int(difference)
        for v in diff_int:
            if v > 0:
                ret.append(Bits(uint=0, length=1))
            else:
                ret.append(Bits(uint=1, length=1))

            entropy_coded = entropy_encode(abs(v))
            ret.append(entropy_coded)

        return ret

    def _encode_difference(self, difference: np.ndarray):
        this_token = difference_array_to_token(difference)

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
            v_uint = difference_to_int(v)
            self._response.append(Bits(uint=v_uint, length=2))

    def encode(self) -> BitArray:
        if self._encoded:
            return self._response

        # encode configuration values
        self._response.append(Bits(uint=self._wavelet.width, length=16))
        self._response.append(Bits(uint=self._wavelet.height, length=16))

        # encode the top value of the wavelet
        root_pixel = difference_to_int(self._wavelet.wavelet[0, 0])
        self._response.append(Bits(uint=root_pixel[0], length=10))
        self._response.append(Bits(uint=root_pixel[1], length=10))
        self._response.append(Bits(uint=root_pixel[2], length=10))

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
