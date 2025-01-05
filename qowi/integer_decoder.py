import qowi.entropy as entropy
import numpy as np
from bitstring import Bits, BitStream

from qowi import integers
from qowi.mflru_cache import MFLRUCache

ZERO_INTEGER = (0, 0, 0)
OP_CODE_RUN = Bits('0b00')
OP_CODE_CACHE = Bits('0b01')
OP_CODE_DELTA = Bits('0b10')
OP_CODE_VALUE = Bits('0b11')

class IntegerDecoder:
    def __init__(self, bitstream: BitStream, cache_size):
        self._bitstream = bitstream
        self._run_length = 0
        self._last_integer = ZERO_INTEGER
        self._cache = MFLRUCache(cache_size)
        self._cache.observe(ZERO_INTEGER)
        self._finished = False

    def decode_next(self) -> tuple:
        if self._run_length > 0:
            self._run_length -= 1
            return self._last_integer

        op_code = self._bitstream.read(2).uint

        if op_code == OP_CODE_RUN.uint:
            # NOTE: no point in incrementing for the offset, then decrementing for the use
            self._run_length = entropy.simple_decode(self._bitstream)
            return self._last_integer

        elif op_code == OP_CODE_CACHE.uint:
            position = entropy.simple_decode(self._bitstream)
            this_integer = self._cache[position]
            self._last_integer = this_integer
            self._cache.observe(this_integer)
            return this_integer

        elif op_code == OP_CODE_DELTA.uint:
            delta_zigzag = entropy.simple_decode_tuple(self._bitstream, 3)
            delta = integers.zigzag_tuple_to_int_tuple(delta_zigzag)
            this_integer = integers.subtract_tuples(self._last_integer, delta)
            self._last_integer = this_integer
            self._cache.observe(this_integer)
            return this_integer

        elif op_code == OP_CODE_VALUE.uint:
            value_zigzag = entropy.simple_decode_tuple(self._bitstream, 3)
            this_integer = integers.zigzag_tuple_to_int_tuple(value_zigzag)
            self._last_integer = this_integer
            self._cache.observe(this_integer)
            return this_integer
        else:
            raise ValueError("Invalid op code value {}".format(op_code))

