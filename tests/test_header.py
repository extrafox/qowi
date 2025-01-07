from bitstring import BitStream
from qowi.header import Header
import unittest

class TestHeader(unittest.TestCase):

    def test_round_trip(self):
        expected = Header()
        expected.width = 100
        expected.height = 150
        expected.color_depth = 3
        expected.cache_size = 200
        expected.wavelet_precision_digits = 0
        expected.wavelet_levels = 10

        encoded = expected.header_bits()

        observed = Header()
        observed.read(BitStream(encoded))

        self.assertEqual(expected, observed)

if __name__ == '__main__':
    unittest.main()
