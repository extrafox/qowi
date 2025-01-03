import unittest
from bitstring import BitStream
from qowi.entropy import entropy_decode, entropy_encode, calculate_order

class EntropyTestCases(unittest.TestCase):

    def test_entropy_round_trip(self):
        for v1 in range(2048):
            v2 = entropy_decode(BitStream(entropy_encode(v1)))
            self.assertEqual(v1, v2)

    def test_calculate_entropy(self):
        actual = []
        for v in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
            actual.append(calculate_order(v))
        expected = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
