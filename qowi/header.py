from bitstring import BitStream, Bits

DEFAULT_CACHE_SIZE = 65534 # maximum size that fits can be referenced by a 54 bit code
WIDTH_SIZE = 16
HEIGHT_SIZE = 16
BIT_SHIFT_SIZE = 4
NUM_CARRY_OVER_BITS_SIZE = 2
CACHE_SIZE = 16

class Header:

    def __init__(self):
        self.width = 0
        self.height = 0
        self.cache_size = DEFAULT_CACHE_SIZE
        self.num_carry_over_bits = 0
        self.bit_shift = 0

    def write(self, bitstream: BitStream):
        bitstream.append(Bits(uint=self.cache_size, length=CACHE_SIZE))
        bitstream.append(Bits(uint=self.width, length=WIDTH_SIZE))
        bitstream.append(Bits(uint=self.height, length=HEIGHT_SIZE))
        bitstream.append(Bits(uint=self.bit_shift, length=BIT_SHIFT_SIZE))
        bitstream.append(Bits(uint=self.num_carry_over_bits, length=NUM_CARRY_OVER_BITS_SIZE))

    def read(self, bitstream: BitStream):
        self.cache_size = bitstream.read(CACHE_SIZE).uint
        self.width = bitstream.read(WIDTH_SIZE).uint
        self.height = bitstream.read(HEIGHT_SIZE).uint
        self.bit_shift = bitstream.read(BIT_SHIFT_SIZE).uint
        self.num_carry_over_bits = bitstream.read(NUM_CARRY_OVER_BITS_SIZE).uint
