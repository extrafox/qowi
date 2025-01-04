import numpy as np
import qowi.entropy as entropy
import qowi.uint10 as uint10
from bitstring import BitArray, Bits, BitStream
from qowi.lru_cache import LRUCache

ZERO_TOKEN = (0, 0, 0)
OP_CODE_RUN = Bits('0b00')
OP_CODE_CACHE = Bits('0b01')
OP_CODE_DELTA = Bits('0b10')
OP_CODE_VALUE = Bits('0b11')


def gen_difference_value_encoding(this_token: tuple):
    ret = BitArray()
    ret.append(OP_CODE_VALUE)
    ret.append(entropy.encode_tuple(this_token))
    return ret


def gen_cache_encoding(position):
    ret = BitArray()
    ret.append(OP_CODE_CACHE)
    ret.append(entropy.encode(position))
    return ret


def gen_delta_encoding(last_token: tuple, this_token: tuple):
    ret = BitArray()
    ret.append(OP_CODE_DELTA)

    # NOTE: all operations here should be in uint10 domain; no conversions

    last_uint10 = np.array(last_token, dtype=np.uint16)
    this_uint10 = np.array(this_token, dtype=np.uint16)
    delta_uint = last_uint10 - this_uint10 + 510 # shift to convert the int to a uint
    ret.append(entropy.encode_array(delta_uint))
    return ret


def gen_run_encoding(run_length):
    ret = BitArray()
    ret.append(OP_CODE_RUN)
    ret.append(entropy.encode(run_length - 1))
    return ret


class Uint10Encoder:
    def __init__(self, bit_stream: BitStream, cache_size: int):
        self._response = bit_stream
        self._run_length = 0
        self._last_token = ZERO_TOKEN
        self._cache = LRUCache(cache_size)
        self._cache.observe(ZERO_TOKEN)
        self.stats = []
        self._finished = False

    def _record(self, stats_record):
        self.stats.append(stats_record)

    def _look_in_cache_and_encode(self, token):
        try:
            position = self._cache.index(token)
            cached = gen_cache_encoding(position)
            return position, cached
        except IndexError:
            return -1, Bits()

    def encode_next(self, this_token: tuple):
        if self._finished:
            raise RuntimeError("You cannot call encode_next after finished has been called")

        if this_token == self._last_token:
            self._run_length += 1
            return

        ### this_pixel does not equal last_pixel ###

        if self._run_length > 0:
            run = gen_run_encoding(self._run_length)
            self._response.append(run)
            self._record({"op_code": "RUN", "run_length": self._run_length, "num_bits": len(run)})
            self._run_length = 0

        position, cached = self._look_in_cache_and_encode(this_token)
        delta = gen_delta_encoding(self._last_token, this_token)
        value = gen_difference_value_encoding(this_token)

        # delta = Bits() # TODO: just for testing, but remove

        smallest_length = min(x for x in (cached.len, delta.len, value.len) if x > 0)
        if cached.len == smallest_length:  # CACHED is shortest
            self._response.append(cached)
            self._cache.observe(this_token)
            self._last_token = this_token
            self._record({"op_code": "CACHE", "index": position, "num_bits": len(cached)})
        elif delta.len == smallest_length:
            self._response.append(delta)
            self._cache.observe(this_token)
            self._last_token = this_token
            self._record({"op_code": "DELTA", "num_bits": len(delta)})
        elif value.len == smallest_length:  # VALUE is shortest
            self._response.append(value)
            self._cache.observe(this_token)
            self._last_token = this_token
            self._record({"op_code": "VALUE", "num_bits": len(value), "uint10_r": this_token[0], "uint10_g": this_token[1], "uint10_b": this_token})
        else:
            raise ValueError("Cached, delta and value encodings were zero length")

    def finish(self):
        if self._finished:
            return

        # flush the run length
        if self._run_length > 1:
            run = gen_run_encoding(self._run_length)
            self._response.append(run)
            self._record({"op_code": "RUN", "run_length": self._run_length, "num_bits": len(run)})
            self._run_length = 0

        self._finished = True