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

def gen_difference_value_encoding(difference: np.ndarray):
    ret = BitArray()
    ret.append(OP_CODE_VALUE)
    entropy_coded = PList.from_ndarray(difference).entropy_coded
    ret.append(entropy_coded)
    return ret

def gen_cache_encoding(position):
    ret = BitArray()
    ret.append(OP_CODE_CACHE)
    ret.append(PUnsignedInteger(position).entropy_coded)
    return ret

def gen_run_encoding(run_length):
    ret = BitArray()
    ret.append(OP_CODE_RUN)
    ret.append(PUnsignedInteger(run_length - 1).entropy_coded)
    return ret

class Encoder:
    def __init__(self, wavelet: Wavelet):
        self._wavelet = wavelet
        self._run_length = 0
        self._last_token = None

        self._cache = LRUCache(CACHE_SIZE)
        self._cache.observe(ZERO_TOKEN)

        self._encoded = False
        self._response = BitArray()

        self.stats = []
        self._update_counter = 0

    def record(self, stats_record):
        self.stats.append(stats_record)

    def _encode_difference(self, difference: np.ndarray):
        this_token = PList.from_ndarray(difference).token

        if this_token == self._last_token:
            self._run_length += 1
            return

        ### this_pixel does not equal last_pixel ###

        if self._run_length > 0:
            run = gen_run_encoding(self._run_length)
            self._response.append(run)
            self._run_length = 0
            self.record({"op_code": "RUN", "run_length": self._run_length, "num_bits": len(run)})

        try:
            this_token = PList.from_ndarray(difference).token
            pos = self._cache.index(this_token)
            cached = gen_cache_encoding(pos)
        except ValueError:
            pos = -1
            cached = Bits()

        value = gen_difference_value_encoding(difference)

        if 0 < len(cached) < len(value):
            self._response.append(cached)
            self._cache.observe(this_token)
            self._last_token = this_token
            self.record({"op_code": "CACHE", "index": pos, "num_bits": len(cached)})
        elif 0 < len(value):
            self._response.append(value)
            self._cache.observe(this_token)
            self._last_token = this_token
            self.record({"op_code": "VALUE", "num_bits": len(value), "diff_r": format(difference[0], ".2f"),
                         "diff_g": format(difference[1], ".2f"), "diff_b": format(difference[2], ".2f")})
        else:
            raise ValueError("Both cached and value encodings were zero length")

    def _encode_carry_over(self, carry_over: np.ndarray):
        for v in carry_over:
            v_uint = int((v % 1.0) * 4)
            self._response.append(Bits(uint=v_uint, length=2))

    def update_progress(self, char):
        if self._update_counter % 80 == 0:
            print()
        print(char, end="")
        self._update_counter += 1

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

        progress_count = 0
        for source_level in range(self._wavelet.num_levels):
            self.update_progress("|")
            source_length = 2 ** source_level

            for i in range(source_length):
                self.update_progress(".")

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
            run = gen_run_encoding(self._run_length)
            self._response.append(run)
            self._run_length = 0

        print("|")
        self._encoded = True
        return self._response
