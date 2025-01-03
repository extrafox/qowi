import numpy as np
import qowi.uint10 as uint10
import unittest

class TestUint10(unittest.TestCase):

    def test_round_trip_with_integers(self):
        for expected in range(-512, 512):
            observed = uint10.to_int(uint10.from_int(expected))
            self.assertEqual(expected, observed)

    def test_round_trip_with_floats(self):
        for expected in np.arange(-127.5, 128, 0.25):
            observed = uint10.to_float(uint10.from_float(expected))
            self.assertEqual(expected, observed)

    def test_round_trip_with_float_array(self):
        given = np.array([-2., -1.75, -1.5, -1.25, -1.1, -1., 0., 1., 1.1, 1.25, 1.5, 1.75, 2.])
        computed = uint10.to_array(uint10.from_array(given))
        expected = np.array([-2., -1.75, -1.5, -1.25, -1., -1., 0., 1., 1., 1.25, 1.5, 1.75, 2.])
        self.assertTrue((computed == expected).all())

    def test_round_trip_with_int_array(self):
        expected = np.array([-2, -1, 0, 1, 2])
        computed = uint10.to_int_array(uint10.from_int_array(expected))
        self.assertTrue((computed == expected).all())

    def test_round_trip_with_float_array(self):
        expected = np.array([-2., -1.75, -1.5, -1.25, -1., 0., 1., 1.25, 1.5, 1.75, 2.])
        computed = uint10.to_float_array(uint10.from_float_array(expected))
        self.assertTrue((computed == expected).all())

if __name__ == '__main__':
    unittest.main()
