import numpy as np
from bitstring import Bits
import qowi.integers as integers
import unittest


class TestIntegers(unittest.TestCase):

    def test_round_trip_with_integers(self):
        for expected in range(-512, 512):
            observed = integers.zigzag_to_integer(integers.integer_to_zigzag(expected))
            self.assertEqual(expected, observed)

    def test_round_trip_with_integer_tuples(self):
        for i in range(-512, 512):
            expected_tuple = (i, i, i)
            observed_tuple = integers.zigzag_tuple_to_int_tuple(integers.int_tuple_to_zigzag_tuple(expected_tuple))
            self.assertEqual(expected_tuple, observed_tuple)

    def test_rescale(self):
        value = Bits('0b11111111').uint
        expected_minus_one = Bits('0b10000000').uint
        observed_minus_one = integers.rescale(value, -1)
        self.assertEqual(expected_minus_one, observed_minus_one)

        expected_plus_one = Bits('0b111111110').uint
        observed_plus_one = integers.rescale(value, 1)
        self.assertEqual(expected_plus_one, observed_plus_one)

if __name__ == '__main__':
    unittest.main()
