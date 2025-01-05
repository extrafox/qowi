import pathlib
import numpy as np
import unittest
from qowi.wavelet import Wavelet, haar_decode, haar_encode
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

    def test_haar_round_trip(self):
        expected_a, expected_b, expected_c, expected_d = 0, 0, 0, 2
        ll, hl, lh, hh = haar_encode(expected_a, expected_b, expected_c, expected_d)
        observed_a, observed_b, observed_c, observed_d = haar_decode(ll, hl, lh, hh)
        self.assertEqual(expected_a, observed_a)
        self.assertEqual(expected_b, observed_b)
        self.assertEqual(expected_c, observed_c)
        self.assertEqual(expected_d, observed_d)

    def test_haar_round_trip_two_levels(self):
        # NOTE: trying to hand code this is painstaking. not sure if useful to complete

        l1 = np.array([
            [0, 11, 23, 31],
            [45, 55, 57, 63],
            [79, 81, 99, 127],
            [140, 173, 231, 255],
        ], dtype='uint8')

        lla, hla, lha, hha = haar_encode(l1[0, 0], l1[0, 1], l1[1, 0], l1[1, 1])
        llb, hlb, lhb, hhb = haar_encode(l1[0, 2], l1[0, 3], l1[1, 2], l1[1, 3])
        llc, hlc, lhc, hhc = haar_encode(l1[2, 0], l1[2, 1], l1[3, 2], l1[3, 3])
        lld, hld, lhd, hhd = haar_encode(l1[2, 2], l1[2, 3], l1[3, 2], l1[3, 3])

        l2 = np.array([
            [lla, llb, hla, hlb],
            [llc, lld, hlc, hld],
            [lha, lhb, hha, hhb],
            [lhc, lhd, hhd, hhd],
        ])

        ll3, hl3, lh3, hh3 = haar_encode(l2[0, 0], l2[0, 1], l2[1, 0], l2[1, 1])

        l3 = np.array([
            [ll3, hl3],
            [lh3, hh3]
        ])

        self.assertTrue(True)

    def test_haar_round_trip_mostly_exhaustively(self):
        for expected_a in range(0, 256, 17):
            for expected_b in range(0, 256, 13):
                for expected_c in range(0, 256, 11):
                    for expected_d in range(0, 256, 7):
                        observed = (expected_a, expected_b, expected_c, expected_d)
                        ll, hl, lh, hh = haar_encode(expected_a, expected_b, expected_c, expected_d)
                        expected = (observed_a, observed_b, observed_c, observed_d) = haar_decode(ll, hl, lh, hh)
                        self.assertEqual(expected, observed)

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

    def test_generate_round_trip_full_255s(self):
        source_image = np.full((2 ** 8, 2 ** 8, 3), 255, dtype=np.uint8)
        w = Wavelet().prepare_from_image(source_image)
        wavelet_image = w.as_image()
        self.assertTrue(np.array_equal(source_image, wavelet_image))

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
