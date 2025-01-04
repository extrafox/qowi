import qowi.entropy as entropy
import qowi.integers as integers
from bitstring import BitArray, Bits, BitStream
from qowi.lru_cache import LRUCache

ZERO_INTEGER = (0, 0, 0)
OP_CODE_RUN = Bits('0b00')
OP_CODE_CACHE = Bits('0b01')
OP_CODE_DELTA = Bits('0b10')
OP_CODE_VALUE = Bits('0b11')


def gen_difference_value_encoding(this_integer: tuple):
    ret = BitArray()
    ret.append(OP_CODE_VALUE)

    zigzag = integers.int_tuple_to_zigzag_tuple(this_integer)
    ret.append(entropy.encode_tuple(zigzag))

    return ret


def gen_cache_encoding(position):
    ret = BitArray()
    ret.append(OP_CODE_CACHE)
    ret.append(entropy.encode(position))
    return ret


def gen_delta_encoding(last_integer: tuple, this_integer: tuple):
    ret = BitArray()
    ret.append(OP_CODE_DELTA)

    delta = integers.subtract_tuples(last_integer, this_integer)
    delta_zigzag = integers.int_tuple_to_zigzag_tuple(delta)
    ret.append(entropy.encode_tuple(delta_zigzag))

    return ret


def gen_run_encoding(run_length):
    ret = BitArray()
    ret.append(OP_CODE_RUN)
    ret.append(entropy.encode(run_length - 1))
    return ret


class IntegerEncoder:
    def __init__(self, bit_stream: BitStream, cache_size: int):
        self._response = bit_stream
        self._run_length = 0
        self._last_integer = ZERO_INTEGER
        self._cache = LRUCache(cache_size)
        self._cache.observe(ZERO_INTEGER)
        self.stats = []
        self._finished = False

    def _record(self, stats_record):
        self.stats.append(stats_record)

    def _look_in_cache_and_encode(self, token: tuple):
        try:
            position = self._cache.index(token)
            cached = gen_cache_encoding(position)
            return position, cached
        except ValueError:
            return -1, Bits()

    def encode_next(self, this_integer: tuple):
        if self._finished:
            raise RuntimeError("You cannot call encode_next after finished has been called")

        if this_integer == self._last_integer:
            self._run_length += 1
            return

        ### this_integer does not equal last_integer ###

        if self._run_length > 0:
            run = gen_run_encoding(self._run_length)
            self._response.append(run)
            self._record({"op_code": "RUN", "run_length": self._run_length, "num_bits": len(run)})
            self._run_length = 0

        position, cached = self._look_in_cache_and_encode(this_integer)
        delta = gen_delta_encoding(self._last_integer, this_integer)
        value = gen_difference_value_encoding(this_integer)

        smallest_length = min(x for x in (cached.len, delta.len, value.len) if x > 0)
        if cached.len == smallest_length:  # CACHED is shortest
            self._response.append(cached)
            self._cache.observe(this_integer)
            self._last_integer = this_integer
            self._record({"op_code": "CACHE", "index": position, "num_bits": len(cached)})
        elif delta.len == smallest_length:
            self._response.append(delta)
            self._cache.observe(this_integer)
            self._last_integer = this_integer
            self._record({"op_code": "DELTA", "num_bits": len(delta)})
        elif value.len == smallest_length:  # VALUE is shortest
            self._response.append(value)
            self._cache.observe(this_integer)
            self._last_integer = this_integer
            self._record({"op_code": "VALUE", "num_bits": len(value), "color_R": this_integer[0], "color_G": this_integer[1], "color_B": this_integer[2]})
        else:
            raise ValueError("Cached, delta and value encodings were zero length")

    def finish(self):
        if self._finished:
            return

        # flush the run length
        if self._run_length > 0:
            run = gen_run_encoding(self._run_length)
            self._response.append(run)
            self._record({"op_code": "RUN", "run_length": self._run_length, "num_bits": len(run)})
            self._run_length = 0

        self._finished = True