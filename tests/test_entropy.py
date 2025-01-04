import numpy as np
import qowi.entropy as entropy
import unittest
from bitstring import BitStream

class EntropyTestCases(unittest.TestCase):

    def test_entropy_round_trip(self):
        for expected in range(2048):
            bits = entropy.encode(expected)
            bitstream = BitStream(bits)
            observed = entropy.decode(bitstream)
            num_unread = bitstream.len - bitstream.pos
            self.assertEqual(expected, observed)
            self.assertTrue(num_unread == 0)

    def test_calculate_entropy(self):
        actual = []
        for v in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
            actual.append(entropy.calculate_order(v))
        expected = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        self.assertEqual(expected, actual)

    def test_array_round_trip(self):
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=np.uint16)
        encoded = entropy.encode_array(expected)
        observed = entropy.decode_array(BitStream(encoded), len(expected))
        self.assertTrue(all(observed == expected))

if __name__ == '__main__':
    unittest.main()
