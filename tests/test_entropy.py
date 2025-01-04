import numpy as np
import qowi.entropy as entropy
import unittest
from bitstring import BitStream

class EntropyTestCases(unittest.TestCase):

    def test_entropy_round_trip(self):
        for expected in range(2048):
            bits = entropy.golomb_encode(expected)
            bitstream = BitStream(bits)
            observed = entropy.golomb_decode(bitstream)
            num_unread = bitstream.len - bitstream.pos
            self.assertEqual(expected, observed)
            self.assertTrue(num_unread == 0)

if __name__ == '__main__':
    unittest.main()
