import unittest
import numpy as np

from qowi.haar_sort_table import HaarSortTable
from qowi.haar_sort_wavelet import HaarSortWavelet  # Adjust import path if necessary

HAAR_SORT_TABLE_PATH = "/home/ctaylor/haar_sort/haar_sort_8bit"

class TestHaarSortWavelet(unittest.TestCase):
    def setUp(self):
        haar_sort_table = HaarSortTable(8, HAAR_SORT_TABLE_PATH)
        self.wavelet = HaarSortWavelet(haar_sort_table, width=4, height=4, color_depth=3)

    def test_initialize_from_shape(self):
        self.wavelet._initialize_from_shape(4, 4, 3)
        self.assertEqual(self.wavelet.width, 4)
        self.assertEqual(self.wavelet.height, 4)
        self.assertEqual(self.wavelet.color_depth, 3)
        self.assertEqual(self.wavelet.num_levels, 2)
        self.assertEqual(self.wavelet.length, 4)

    def test_prepare_wavelet_0(self):
        input_image = np.array(
            [
                [[0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
            ], dtype=np.uint8)
        expected_wavelet = np.array(
            [
                [[0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
            ], dtype=np.uint8)
        self.wavelet.prepare_from_image(input_image)
        self.assertTrue(np.array_equal(expected_wavelet, self.wavelet.wavelet))
        observed_image = self.wavelet.as_image()
        self.assertTrue(np.array_equal(input_image, observed_image))

    def test_prepare_wavelet_255(self):
        input_image = np.array(
            [
                [[255, 255, 255], [255, 255, 255]],
                [[255, 255, 255], [255, 255, 255]],
            ], dtype=np.uint8)
        expected_wavelet = np.array(
            [
                [[255, 255, 255], [255, 255, 255]],
                [[255, 255, 255], [255, 255, 255]],
            ], dtype=np.uint8)
        self.wavelet.prepare_from_image(input_image)
        self.assertTrue(np.array_equal(expected_wavelet, self.wavelet.wavelet))
        observed_image = self.wavelet.as_image()
        self.assertTrue(np.array_equal(input_image, observed_image))

    def test_prepare_wavelet_2x2(self):
        input_image = np.array(
            [
                [[255, 127, 0], [211, 22, 48]],
                [[45, 200, 12], [230, 102, 54]],
            ], dtype=np.uint8)
        self.wavelet.prepare_from_image(input_image)
        observed_image = self.wavelet.as_image()
        self.assertTrue(np.array_equal(input_image, observed_image))

    def test_prepare_wavelet_4x4(self):
        input_image = np.array(
            [
                [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
                [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
                [[255, 255, 255], [255, 255, 255], [255, 255, 255], [0, 0, 0]],
                [[255, 255, 255], [255, 255, 255], [0, 0, 0], [255, 255, 255]],
            ], dtype=np.uint8)
        self.wavelet.prepare_from_image(input_image)
        observed_image = self.wavelet.as_image()
        self.assertTrue(np.array_equal(input_image, observed_image))

    def test_prepare_wavelet_zigzag(self):
        input_image = np.array(
            [
                [[255, 255, 255], [0, 0, 0]],
                [[0, 0, 0], [255, 255, 255]],
            ], dtype=np.uint8)
        expected_wavelet = np.array(
            [
                [[254, 254, 254], [213, 213, 213]],
                [[32, 32, 32], [227, 227, 227]],
            ], dtype=np.uint8)
        self.wavelet.prepare_from_image(input_image)
        self.assertTrue(np.array_equal(expected_wavelet, self.wavelet.wavelet))
        observed_image = self.wavelet.as_image()
        self.assertTrue(np.array_equal(input_image, observed_image))

    def test_prepare_from_image(self):
        expected_image = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        self.wavelet.prepare_from_image(expected_image)
        observed_image = self.wavelet.as_image()
        self.assertTrue(np.array_equal(expected_image, observed_image))

    def test_prepare_from_wavelet(self):
        wavelet = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        self.wavelet.prepare_from_wavelet(wavelet)
        self.assertTrue(np.array_equal(self.wavelet.wavelet, wavelet))

    def test_as_image(self):
        image = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        self.wavelet.prepare_from_image(image)
        reconstructed_image = self.wavelet.as_image()
        self.assertEqual(reconstructed_image.shape, image.shape)

    def test_gen_wavelet_structure(self):
        self.wavelet._initialize_from_shape(4, 4, 3)
        self.wavelet._gen_wavelet()
        self.assertEqual(self.wavelet.wavelet.shape, (4, 4, 3))

if __name__ == '__main__':
    unittest.main()
