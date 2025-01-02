import random
import numpy as np
import unittest
from qowi.primitives import PFloat, PInteger, PList, PUnsignedInteger

class TestPrimitives(unittest.TestCase):

    def test_pfloat_positive(self):
        f = PFloat(1.5)
        expected_uint10 = 12
        self.assertEqual(f.uint10, expected_uint10)

    def test_pfloat_negative(self):
        f = PFloat(-1.5)
        expected_uint10 = 13
        self.assertEqual(f.uint10, expected_uint10)

    def test_pinteger_positive(self):
        i = PInteger(1)
        expected_uint10 = 2
        self.assertEqual(i.uint10, expected_uint10)

    def test_pinteger_negative(self):
        i = PInteger(-1)
        expected_uint10 = 3
        self.assertEqual(i.uint10, expected_uint10)

    def test_plist_from_ndarray_int_token(self):
        array = np.array([2, -1, 4], dtype=np.int8)
        plist = PList.from_ndarray(array)
        expected_token = (4, 3, 8)
        self.assertEqual(plist.token, expected_token)

    def test_plist_from_ndarray_uint_token(self):
        array = np.array([2, 3, 4], dtype=np.uint8)
        plist = PList.from_ndarray(array)
        expected_token = (2, 3, 4)
        self.assertEqual(plist.token, expected_token)

    def test_plist_from_ndarray_float_token(self):
        array = np.array([1.5, 1.25, 1.75], dtype=np.float16)
        plist = PList.from_ndarray(array)
        expected_token = (12, 10, 14)
        self.assertEqual(plist.token, expected_token)

    def test_plist_from_tuple_uint_token(self):
        expected_token = (2, 3, 4)
        plist = PList.from_token(expected_token)
        self.assertEqual(plist.token, expected_token)

    def test_plist_from_tuple_float_token(self):
        array = (1.5, 1.25, 1.75)
        plist = PList.from_list(array)
        expected_token = (12, 10, 14)
        self.assertEqual(plist.token, expected_token)

    def test_plist_from_ndarray_uint_to_ndarray(self):
        expected_array = np.array([2, 3, 4], dtype=np.uint8)
        plist = PList.from_ndarray(expected_array)
        self.assertTrue((plist.ndarray == expected_array).all())

    def test_plist_from_ndarray_float_to_ndarray(self):
        expected_array = np.array([1.5, 1.25, 1.75], dtype=np.float16)
        plist = PList.from_ndarray(expected_array)
        self.assertTrue((plist.ndarray == expected_array).all())

    def test_plist_setitem(self):
        plist = PList.from_list((0, 0, 0))
        expected_value = PUnsignedInteger(3)
        plist[2] = expected_value
        self.assertEqual(plist[2], expected_value)

    def test_pfloat_random_uint10_round_trips(self):
        for expected_uint10 in range(1024):
            if expected_uint10 == 1: # 1 is not an allowed uint10 value
                continue
            a = PFloat.from_uint10(expected_uint10)
            test_uint10 = a.uint10
            self.assertEqual(test_uint10, expected_uint10)

if __name__ == '__main__':
    unittest.main()
