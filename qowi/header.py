from bitstring import BitStream, Bits

DEFAULT_CACHE_SIZE = 65534 # maximum size that fits can be referenced by a 54 bit code

WIDTH_NUM_BITS = 16
HEIGHT_NUM_BITS = 16
BIT_SHIFT_NUM_BITS = 4
NUM_CARRY_OVER_BITS_NUM_BITS = 2
CACHE_NUM_BITS = 16

class Header:

    def __init__(self):
        self.width = 0
        self.height = 0
        self.cache_size = DEFAULT_CACHE_SIZE
        self.num_carry_over_bits = 0
        self.bit_shift = 0

    def write(self, bitstream: BitStream):
        bitstream.append(Bits(uint=self.cache_size, length=CACHE_NUM_BITS))
        bitstream.append(Bits(uint=self.width, length=WIDTH_NUM_BITS))
        bitstream.append(Bits(uint=self.height, length=HEIGHT_NUM_BITS))
        bitstream.append(Bits(uint=self.bit_shift, length=BIT_SHIFT_NUM_BITS))
        bitstream.append(Bits(uint=self.num_carry_over_bits, length=NUM_CARRY_OVER_BITS_NUM_BITS))

    def read(self, bitstream: BitStream):
        self.cache_size = bitstream.read(CACHE_NUM_BITS).uint
        self.width = bitstream.read(WIDTH_NUM_BITS).uint
        self.height = bitstream.read(HEIGHT_NUM_BITS).uint
        self.bit_shift = bitstream.read(BIT_SHIFT_NUM_BITS).uint
        self.num_carry_over_bits = bitstream.read(NUM_CARRY_OVER_BITS_NUM_BITS).uint
