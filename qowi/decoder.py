import numpy as np
import sys
from bitstring import BitArray, Bits, BitStream
from qowi.lru_cache import LRUCache
from qowi.primitives import PList, PUnsignedInteger, PFloat
from qowi.wavelet import Wavelet

CACHE_SIZE = 2048
ZERO_TOKEN = (0, 0, 0)
OP_CODE_RUN = Bits('0b0')
OP_CODE_CACHE = Bits('0b10')
OP_CODE_VALUE = Bits('0b11')

class Decoder:
    def __init__(self, bitstream: BitStream):
        self._wavelet = None
        self._bitstream = bitstream
        self._run_length = 0
        self._last_token = None

        self._cache = LRUCache(CACHE_SIZE)
        self._cache.observe(ZERO_TOKEN)

        self._decoded = False

    def _decode_next_difference(self) -> np.ndarray:
        if self._run_length > 0:
            self._run_length -= 1
            return PList.from_token(self._last_token).ndarray

        op_code = self._bitstream.read(1).uint

        if op_code == OP_CODE_RUN.uint:
            self._run_length = entropy_decode(self._bitstream, dtype=PUnsignedInteger).value # no point in incrementing, then decrementing
            return PList.from_token(self._last_token).ndarray

        op_code = (op_code << 1) + self._bitstream.read(1).uint

        if op_code == OP_CODE_CACHE.uint:
            pos = PList.from_bitstream(self._bitstream, 1, dtype=PUnsignedInteger)[0].value
            this_token = self._cache[pos]
            self._last_token = this_token
            self._cache.observe(this_token)
            return PList.from_token(this_token).ndarray

        elif op_code == OP_CODE_VALUE.uint:
            value = PList.from_bitstream(self._bitstream, 3)
            this_token = value.token
            self._last_token = this_token
            self._cache.observe(this_token)
            return value.ndarray
        else:
            raise ValueError("Invalid op code value {}".format(op_code))

    def _decode_carry_over(self) -> np.ndarray:
        # TODO: handle carry over in a cleaner way

        carry_over = np.array([
            self._bitstream.read(2).uint,
            self._bitstream.read(2).uint,
            self._bitstream.read(2).uint,
        ], dtype=np.float16)
        ret = np.array([0, 0, 0], dtype=np.float16)
        for i in range(3):
            ret[i] = carry_over[i] / 4
        return ret

    def decode(self) -> Wavelet:
        if self._decoded:
            return self._wavelet.as_image()

        # decode configuration values
        width = self._bitstream.read(16).uint
        height = self._bitstream.read(16).uint
        self._wavelet = Wavelet(width, height)

        # decode the top value of the wavelet
        self._wavelet.wavelet[0, 0] = PList.from_bitstream(self._bitstream, 3).ndarray

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