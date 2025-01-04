import qowi.entropy as entropy
import numpy as np
import qowi.uint10 as uint10
from bitstring import Bits, BitStream
from qowi.lru_cache import LRUCache

ZERO_TOKEN = (0, 0, 0)
OP_CODE_RUN = Bits('0b00')
OP_CODE_CACHE = Bits('0b01')
OP_CODE_DELTA = Bits('0b10')
OP_CODE_VALUE = Bits('0b11')

class Uint10Decoder:
    def __init__(self, bitstream: BitStream, cache_size):
        self._bitstream = bitstream
        self._run_length = 0
        self._last_token = ZERO_TOKEN
        self._cache = LRUCache(cache_size)
        self._cache.observe(ZERO_TOKEN)
        self._finished = False

    def decode_next(self) -> tuple:
        if self._run_length > 0:
            self._run_length -= 1
            return self._last_token

        op_code = self._bitstream.read(2).uint

        if op_code == OP_CODE_RUN.uint:
            # NOTE: no point in incrementing for the offset, then decrementing for the use
            self._run_length = entropy.decode(self._bitstream)
            return self._last_token

        elif op_code == OP_CODE_CACHE.uint:
            position = entropy.decode(self._bitstream)
            this_token = self._cache[position]
            self._last_token = this_token
            self._cache.observe(this_token)
            return this_token

        elif op_code == OP_CODE_DELTA.uint:
            last_uint10 = np.array(self._last_token, dtype=np.int16)
            delta = entropy.decode_array(self._bitstream, 3) - 510 # shift to to back to int
            this_token = tuple(last_uint10 - delta)
            self._last_token = this_token
            self._cache.observe(this_token)
            return this_token

        elif op_code == OP_CODE_VALUE.uint:
            value_a = entropy.decode_array(self._bitstream, 3)
            this_token = tuple(value_a)
            self._last_token = this_token
            self._cache.observe(this_token)
            return this_token
        else:
            raise ValueError("Invalid op code value {}".format(op_code))

