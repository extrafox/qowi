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


    # def test_round_trip_with_floats(self):
    #     for expected in np.arange(-127.5, 128, 0.25):
    #         observed = uint10.uint10_to_float(uint10.float_to_uint10(expected))
    #         self.assertEqual(expected, observed)


    # def test_round_trip_with_float_array(self):
    #     given = np.array([-2., -1.75, -1.5, -1.25, -1.1, -1., 0., 1., 1.1, 1.25, 1.5, 1.75, 2.])
    #     computed = uint10.to_array(uint10.from_array(given))
    #     expected = np.array([-2., -1.75, -1.5, -1.25, -1., -1., 0., 1., 1., 1.25, 1.5, 1.75, 2.])
    #     self.assertTrue((computed == expected).all())


    # def test_round_trip_with_int_array(self):
    #     expected = np.array([-2, -1, 0, 1, 2])
    #     computed = uint10.zigzag_array_to_int16_array(uint10.int_array_to_zigzag_array(expected))
    #     self.assertTrue((computed == expected).all())


    # def test_round_trip_with_float_array(self):
    #     expected = np.array([-2., -1.75, -1.5, -1.25, -1., 0., 1., 1.25, 1.5, 1.75, 2.])
    #     computed = uint10.uint10_array_to_float16_array(uint10.float_array_to_uint10_array(expected))
    #     self.assertTrue((computed == expected).all())


    # def test_right_bit_shift(self):
    #     given = np.array([-2., -1.75, -1.5, -1.25, -1.1, -1., 0., 1., 1.1, 1.25, 1.5, 1.75, 2.])
    #     expected = np.array([-0.5, -0.25, -0.25, -0.25, -0.25, -0.25,  0., 0.25, .25,  0.25,  0.25,  0.25,  0.5])
    #     uint_array = uint10.float_array_to_uint10_array(given)
    #     shifted = uint10.right_bit_shift(uint_array, 2)
    #     observed = uint10_array_to_float16_array(shifted)
    #     self.assertTrue(all(expected == observed))


    # def test_right_bit_shift(self):
    #     given = np.array([-0.5, -0.25, 0., 0.25,  0.5])
    #     expected = np.array([-2, -1, 0, 1, 2])
    #     uint_array = uint10.float_array_to_uint10_array(given)
    #     shifted = uint10.left_bit_shift(uint_array, 2)
    #     observed = uint10_array_to_float16_array(shifted)
    #     self.assertTrue(all(expected == observed))


if __name__ == '__main__':
    unittest.main()
