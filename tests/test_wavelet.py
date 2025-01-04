import pathlib

import numpy as np
import unittest
from qowi import wavelet
from qowi.wavelet import Wavelet
from skimage import io

from utils.test_images import TEST_WAVELETS

TEST_IMAGES = [
    np.array([
        [[255, 255, 255], [0, 0, 0]],
        [[0, 0, 0], [255, 255, 255]],
    ], dtype='uint8'),
    np.array([
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]],
        [[8, 8, 8], [8, 8, 8], [8, 8, 8], [8, 8, 8]],
        [[16, 16, 16], [16, 16, 16], [16, 16, 16], [16, 16, 16]],
    ], dtype='uint8'),
]

TEST_WAVELETS = [
    np.array([
        [[2., -2., 127.5], [-127.5, 0.5, -0.5]],
        [[0.25, -0.25, 1.], [-1., 1.25, -1.25]],
    ], dtype=np.float16),
    np.array([
        [[2., -2., 127.5], [-127.5, 0., 0.]],
        [[0., 0., 1.], [-1., 1.25, -1.25]],
    ], dtype=np.float16),
    np.array([
        [[2., -2., 127.5],  [-127.5, 0., 0.]],
        [[0., 0., 0.],  [0., 0., 0.]],
    ], dtype=np.float16),
]

class TestWavelet(unittest.TestCase):

    def test_generate_round_trip_single_level(self):
        source_image = TEST_IMAGES[0]
        w = Wavelet().prepare_from_image(source_image)
        wavelet_image = w.as_image()
        self.assertTrue(np.array_equal(source_image, wavelet_image))

    def test_generate_round_trip_two_level(self):
        source_image = TEST_IMAGES[1]
        w = Wavelet().prepare_from_image(source_image)
        wavelet_image = w.as_image()
        self.assertTrue(np.array_equal(source_image, wavelet_image))

    def test_hard_threshold(self):
        source_wavelet = TEST_WAVELETS[0]
        expected_wavelet = TEST_WAVELETS[1]
        w = Wavelet()
        w.wavelet = source_wavelet
        w.apply_hard_threshold(1.0)
        observed_wavelet = w.wavelet
        self.assertTrue(np.array_equal(expected_wavelet, observed_wavelet))

    def test_soft_threshold(self):
        source_wavelet = TEST_WAVELETS[0]
        expected_wavelet = TEST_WAVELETS[2]
        w = Wavelet()
        w.wavelet = source_wavelet
        w.apply_soft_threshold(2.0)
        observed_wavelet = w.wavelet
        self.assertTrue(np.array_equal(expected_wavelet, observed_wavelet))

    def test_round_trip_all_files_in_media_folder(self):
        training_image_dir = pathlib.Path("../media")
        media_directory = [item for item in training_image_dir.rglob('*')]
        for path in media_directory:
            source_image = io.imread(path)
            w = Wavelet().prepare_from_image(source_image)
            wavelet_image = w.as_image()
            self.assertTrue(np.array_equal(source_image, wavelet_image))

if __name__ == '__main__':
    unittest.main()
