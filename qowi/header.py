from bitstring import BitStream, Bits

WIDTH_NUM_BITS = 16
HEIGHT_NUM_BITS = 16
CACHE_NUM_BITS = 16
WAVELET_LEVELS_BITS = 4

class Header:

    def __init__(self):
        self.width = None
        self.height = None
        self.cache_size = None
        self.wavelet_levels = None

    def write(self, bitstream: BitStream):
        bitstream.append(Bits(uint=self.width, length=WIDTH_NUM_BITS))
        bitstream.append(Bits(uint=self.height, length=HEIGHT_NUM_BITS))
        bitstream.append(Bits(uint=self.cache_size, length=CACHE_NUM_BITS))
        bitstream.append(Bits(uint=self.wavelet_levels, length=WAVELET_LEVELS_BITS))

    def read(self, bitstream: BitStream):
        self.width = bitstream.read(WIDTH_NUM_BITS).uint
        self.height = bitstream.read(HEIGHT_NUM_BITS).uint
        self.cache_size = bitstream.read(CACHE_NUM_BITS).uint
        self.wavelet_levels = bitstream.read(WAVELET_LEVELS_BITS).uint
