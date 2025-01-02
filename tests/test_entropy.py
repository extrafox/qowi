import unittest
from bitstring import BitStream
from qowi.entropy import entropy_decode, entropy_encode

class EntropyTestCases(unittest.TestCase):

    def test_entropy_round_trip(self):
        for v1 in range(2048):
            v2 = entropy_decode(BitStream(entropy_encode(v1)))
            self.assertEqual(v1, v2)

if __name__ == '__main__':
    unittest.main()
