import numpy as np
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

    def test_round_to_fraction_decimal(self):
        input = int(3.4159 * 10000)
        rounded = integers.round_scaled_integer(input, 10000, 0.01)
        expected = int(3.42 * 10000)
        self.assertEqual(expected, rounded)

    def test_round_to_fraction_binary(self):
        input = (255 + 255) - (0 + 0) # integer representation of 127.5 with scaling factor 4
        rounded = integers.round_scaled_integer(input, 4, 1)
        expected = 128 * 4
        self.assertEqual(expected, rounded)


if __name__ == '__main__':
    unittest.main()
