from bitstring import BitStream, Bits, BitArray

WIDTH_NUM_BITS = 16
HEIGHT_NUM_BITS = 16
COLOR_DEPTH_BITS = 2
CACHE_NUM_BITS = 16
WAVELET_LEVELS_BITS = 4
WAVELET_PRECISION_DIGITS_BITS = 8

class Header:

    def __init__(self):
        self.width = None
        self.height = None
        self.color_depth = None
        self.cache_size = None
        self.wavelet_levels = None
        self.wavelet_precision_digits = None

    def header_bits(self) -> Bits:
        buffer = BitArray()
        buffer.append(Bits(uint=self.width, length=WIDTH_NUM_BITS))
        buffer.append(Bits(uint=self.height, length=HEIGHT_NUM_BITS))
        buffer.append(Bits(uint=self.color_depth - 1, length=COLOR_DEPTH_BITS))
        buffer.append(Bits(uint=self.cache_size, length=CACHE_NUM_BITS))
        buffer.append(Bits(uint=self.wavelet_levels, length=WAVELET_LEVELS_BITS))
        buffer.append(Bits(uint=self.wavelet_precision_digits, length=WAVELET_PRECISION_DIGITS_BITS))
        return buffer

    def read(self, bitstream: BitStream):
        self.width = bitstream.read(WIDTH_NUM_BITS).uint
        self.height = bitstream.read(HEIGHT_NUM_BITS).uint
        self.color_depth = bitstream.read(COLOR_DEPTH_BITS).uint + 1
        self.cache_size = bitstream.read(CACHE_NUM_BITS).uint
        self.wavelet_levels = bitstream.read(WAVELET_LEVELS_BITS).uint
        self.wavelet_precision_digits = bitstream.read(WAVELET_PRECISION_DIGITS_BITS).uint

    def __eq__(self, other):
        return self.width == other.width and self.height == other.height and self.color_depth == other.color_depth and self.cache_size == other.cache_size and self.wavelet_levels == other.wavelet_levels and self.wavelet_precision_digits == other.wavelet_precision_digits
