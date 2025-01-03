import qowi.entropy as entropy
import numpy as np
import qowi.uint10 as uint10
from bitstring import Bits, BitStream
from qowi.lru_cache import LRUCache
from qowi.wavelet import Wavelet

CACHE_SIZE = 2048
ZERO_TOKEN = (0, 0, 0)
OP_CODE_RUN = Bits('0b0')
OP_CODE_CACHE = Bits('0b10')
OP_CODE_VALUE = Bits('0b11')

class Decoder:
    def __init__(self, bitstream: BitStream):
        self._wavelet = None
        self._uint10_array = None
        self._bitstream = bitstream
        self._run_length = 0
        self._last_token = None
        self._bit_shift = 0
        self._carry_over_bits = 0

        self._cache = LRUCache(CACHE_SIZE)
        self._cache.observe(ZERO_TOKEN)

        self._decoded = False

    def _decode_next_difference(self) -> np.ndarray:
        if self._run_length > 0:
            self._run_length -= 1
            return self._last_token

        op_code = self._bitstream.read(1).uint

        if op_code == OP_CODE_RUN.uint:
            # NOTE: no point in incrementing for the offset, then decrementing for the use
            self._run_length = entropy.decode(self._bitstream)
            return self._last_token

        op_code = (op_code << 1) + self._bitstream.read(1).uint

        if op_code == OP_CODE_CACHE.uint:
            position = entropy.decode(self._bitstream)
            this_token = self._cache[position]
            self._last_token = this_token
            self._cache.observe(this_token)
            return this_token

        elif op_code == OP_CODE_VALUE.uint:
            unshifted_values = entropy.decode_array(self._bitstream, 3)
            value = uint10.left_bit_shift(unshifted_values, self._bit_shift)
            this_token = tuple(value)
            self._last_token = this_token
            self._cache.observe(this_token)
            return this_token
        else:
            raise ValueError("Invalid op code value {}".format(op_code))

    def _decode_carry_over(self) -> np.ndarray:
        if self._carry_over_bits == 0:
            return np.zeros(3, dtype=np.float16)

        if self._carry_over_bits == 2:
            carry_over = np.array([
                self._bitstream.read(2).uint,
                self._bitstream.read(2).uint,
                self._bitstream.read(2).uint,
            ], dtype=np.float16)
            return carry_over.astype(np.float16) / 4
        elif self._carry_over_bits == 1:
            carry_over = np.array([
                self._bitstream.read(1).uint,
                self._bitstream.read(1).uint,
                self._bitstream.read(1).uint,
            ], dtype=np.float16)
            return carry_over.astype(np.float16) / 4

    def decode(self) -> Wavelet:
        if self._decoded:
            return self._wavelet

        # decode configuration values
        width = self._bitstream.read(16).uint
        height = self._bitstream.read(16).uint
        self._wavelet = Wavelet(width, height)
        self._uint10_array = np.zeros((width, height, 3), dtype=np.uint16)

        self._bit_shift = self._bitstream.read(4).uint
        self._carry_over_bits = self._bitstream.read(2).uint

        # decode the top value of the wavelet
        self._uint10_array[0, 0] = entropy.decode_array(self._bitstream, 3)

        ### iterate over the wavelet encoding difference and carry-over values ###

        for source_level in range(self._wavelet.num_levels):
            source_length = 2 ** source_level

            for i in range(source_length):
                for j in range(source_length):
                    hl = self._decode_next_difference()
                    lh = self._decode_next_difference()
                    hh = self._decode_next_difference()

                    self._uint10_array[i, source_length + j] = hl
                    self._uint10_array[source_length + i, j] = lh
                    self._uint10_array[source_length + i, source_length + j] = hh

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
        self._wavelet.from_uint10_array(self._uint10_array)
        return self._wavelet