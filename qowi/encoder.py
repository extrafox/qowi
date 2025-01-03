import numpy as np
import qowi.entropy as entropy
import qowi.uint10 as uint10
from bitstring import BitArray, Bits
from qowi.lru_cache import LRUCache
from qowi.wavelet import Wavelet

CACHE_SIZE = 2048
ZERO_TOKEN = (0, 0, 0)
OP_CODE_RUN = Bits('0b0')
OP_CODE_CACHE = Bits('0b10')
OP_CODE_VALUE = Bits('0b11')


def gen_difference_value_encoding(difference: np.ndarray):
    ret = BitArray()
    ret.append(OP_CODE_VALUE)
    ret.append(entropy.encode_array(difference))
    return ret


def gen_cache_encoding(position):
    ret = BitArray()
    ret.append(OP_CODE_CACHE)
    ret.append(entropy.encode(position))
    return ret


def gen_run_encoding(run_length):
    ret = BitArray()
    ret.append(OP_CODE_RUN)
    ret.append(entropy.encode(run_length - 1))
    return ret


class Encoder:
    def __init__(self, wavelet: Wavelet, bit_shift=0, carry_over_bits=2):
        self._wavelet = wavelet
        self._uint10_array = wavelet.as_uint10_array()

        self._run_length = 0
        self._last_token = None

        self._bit_shift = max(0, min(bit_shift, 7))
        self._carry_over_bits = max(0, min(carry_over_bits, 2))

        self._cache = LRUCache(CACHE_SIZE)
        self._cache.observe(ZERO_TOKEN)

        self._encoded = False
        self._response = BitArray()

        self.stats = []
        self._update_counter = 0

    def record(self, stats_record):
        self.stats.append(stats_record)

    def _look_in_cache_and_encode(self, token):
        try:
            position = self._cache.index(token)
            cached = gen_cache_encoding(position)
            return position, cached
        except ValueError:
            return -1, Bits()

    def _encode_difference(self, difference: np.ndarray):
        shifted_difference = uint10.right_bit_shift(difference, self._bit_shift)
        this_token = tuple(shifted_difference)

        if this_token == self._last_token:
            self._run_length += 1
            return

        ### this_pixel does not equal last_pixel ###

        if self._run_length > 0:
            run = gen_run_encoding(self._run_length)
            self._response.append(run)
            self.record({"op_code": "RUN", "run_length": self._run_length, "num_bits": len(run)})
            self._run_length = 0

        position, cached = self._look_in_cache_and_encode(this_token)
        value = gen_difference_value_encoding(shifted_difference)

        if 0 < len(cached) < len(value):  # CACHED is shortest
            self._response.append(cached)
            self._cache.observe(this_token)
            self._last_token = this_token
            self.record({"op_code": "CACHE", "index": position, "num_bits": len(cached)})
        elif 0 < len(value):  # VALUE is shortest
            self._response.append(value)
            self._cache.observe(this_token)
            self._last_token = this_token
            self.record({"op_code": "VALUE", "num_bits": len(value), "diff_r": format(difference[0], ".2f"),
                         "diff_g": format(difference[1], ".2f"), "diff_b": format(difference[2], ".2f")})
        else:
            raise ValueError("Both cached and value encodings were zero length")

    def _encode_carry_over(self, carry_over: np.ndarray):
        if self._carry_over_bits == 0:
            return

        for v in carry_over:
            if v < 0 or v > 2:
                raise ValueError("Carry over {} must be between 0 and 2".format(v))

            v_uint = int((v % 1.0) * 4)

            if v_uint < 0 or v_uint > 3:
                raise ValueError("Carry over v_uint {} must be between 0 and 3".format(v))

            if self._carry_over_bits == 2:
                self._response.append(Bits(uint=v_uint, length=2))
            elif self._carry_over_bits == 1:
                v_uint = v_uint >> 1
                self._response.append(Bits(uint=v_uint, length=1))

    def update_progress(self, char):
        if self._update_counter % 80 == 0 and self._update_counter != 0:
            print()
        print(char, end="")
        self._update_counter += 1

    def encode(self) -> BitArray:
        if self._encoded:
            return self._response

        # encode configuration values
        self._response.append(Bits(uint=self._wavelet.width, length=16))
        self._response.append(Bits(uint=self._wavelet.height, length=16))
        self._response.append(Bits(uint=self._bit_shift, length=4))
        self._response.append(Bits(uint=self._carry_over_bits, length=2))

        # encode the top value of the wavelet
        root_pixel = self._uint10_array[0, 0]
        self._response.append(entropy.encode_array(root_pixel))

        ### iterate over the wavelet encoding difference and carry-over values ###

        progress_count = 0
        for source_level in range(self._wavelet.num_levels):
            self.update_progress("|")
            source_length = 2 ** source_level

            for i in range(source_length):
                self.update_progress(".")

                for j in range(source_length):
                    hl = self._uint10_array[i, source_length + j]
                    lh = self._uint10_array[source_length + i, j]
                    hh = self._uint10_array[source_length + i, source_length + j]

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
            self.record({"op_code": "RUN", "run_length": self._run_length, "num_bits": len(run)})
            self._run_length = 0

        print("|")
        self._encoded = True
        return self._response
