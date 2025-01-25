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
        self.assertIsNotNone(self.wavelet.wavelet)

    def test_prepare_from_image(self):
        input_image = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        self.wavelet.prepare_from_image(input_image)
        self.assertEqual(self.wavelet.wavelet.shape, (4, 4, 3))
        reconstructed_image = self.wavelet.as_image()
        self.assertTrue(np.array_equal(input_image, reconstructed_image))

    def test_prepare_from_wavelet(self):
        wavelet_array = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        self.wavelet.prepare_from_wavelet(wavelet_array)
        self.assertTrue(np.array_equal(self.wavelet.wavelet, wavelet_array))

    def test_generate_wavelet(self):
        input_image = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        self.wavelet.prepare_from_image(input_image)
        self.wavelet._generate_wavelet()  # Ensure the private method runs without error
        self.assertEqual(self.wavelet.wavelet.shape, (4, 4, 3))

    def test_as_image(self):
        input_image = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        self.wavelet.prepare_from_image(input_image)
        reconstructed_image = self.wavelet.as_image()
        self.assertEqual(reconstructed_image.shape, input_image.shape)
        self.assertTrue(np.array_equal(input_image, reconstructed_image))

    def test_wavelet_transformation_consistency(self):
        input_image = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        self.wavelet.prepare_from_image(input_image)

        # Perform forward and reverse transformations
        transformed_wavelet = self.wavelet.wavelet.copy()
        reconstructed_image = self.wavelet.as_image()

        # Verify dimensions and data consistency
        self.assertEqual(transformed_wavelet.shape, (4, 4, 3))
        self.assertTrue(np.array_equal(input_image, reconstructed_image))

    def test_edge_case_empty_image(self):
        self.wavelet._initialize_from_shape(0, 0, 0)
        self.assertEqual(self.wavelet.num_levels, 0)
        self.assertIsNone(self.wavelet.wavelet)

    def test_edge_case_non_power_of_two(self):
        input_image = np.random.randint(0, 256, (3, 5, 3), dtype=np.uint8)
        self.wavelet.prepare_from_image(input_image)
        self.assertEqual(self.wavelet.wavelet.shape[0], 8)  # Padded to next power of two
        self.assertEqual(self.wavelet.wavelet.shape[1], 8)

if __name__ == "__main__":
    unittest.main()
