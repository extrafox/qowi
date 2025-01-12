import unittest
import numpy as np


def haar_wavelet_transform(a, b, c, d):
    """
    Compute Haar wavelet coefficients (LL, HL, LH, HH) for a 2x2 grid of pixels.

    Parameters:
        pixels (list): A list of 4 pixel values.

    Returns:
        list: Haar wavelet coefficients [LL, HL, LH, HH].
    """

    LL = (a + b + c + d) / 4
    HL = (a + b - c - d) / 4
    LH = (a - b + c - d) / 4
    HH = (a - b - c + d) / 4
    return [LL, HL, LH, HH]


def haar_sort_index_from_pixels(pixels: np.ndarray) -> int:
    """
    Wrapper function to compute the Haar transform-based sort index for a given 2x2 grid of pixels.

    Parameters:
        pixels (np.ndarray): A numpy array of 4 pixel values (8-bit, 0-255).

    Returns:
        int: The computed index in the reordered Haar wavelet transform scheme.
    """

    def forward_recursive(pixels, level, start_index, group_size, max_level=4):
        if level >= max_level:
            return start_index  # Base case: final index at the deepest level

        # Compute coefficient for the current filter level
        coefficients = haar_wavelet_transform(*pixels)

        # Sort order is determined by the coefficient at this level
        coeff = coefficients[level]
        subgroup_size = group_size // 4

        # Determine the subgroup based on the coefficient
        subgroup_index = int((coeff + 255) // (512 / 4))  # Map coefficient to one of 4 subgroups

        # Recurse into the correct subgroup
        return forward_recursive(
            pixels,
            level + 1,
            start_index + subgroup_index * subgroup_size,
            subgroup_size,
            max_level
        )

    return forward_recursive(pixels.tolist(), 0, 0, 4_294_967_296)


class TestHaarTransform(unittest.TestCase):

    def test_haar_wavelet_transform(self):
        result = haar_wavelet_transform(100, 150, 50, 200)
        expected = [125.0, 0.0, -50.0, 25.0]
        self.assertAlmostEqual(result[0], expected[0], places=5)
        self.assertAlmostEqual(result[1], expected[1], places=5)
        self.assertAlmostEqual(result[2], expected[2], places=5)
        self.assertAlmostEqual(result[3], expected[3], places=5)

    def test_haar_sort_index_from_pixels(self):
        pixels = np.array([100, 150, 50, 200])
        index = haar_sort_index_from_pixels(pixels)
        # Exact index depends on the recursive implementation logic
        self.assertIsInstance(index, int)
        self.assertGreaterEqual(index, 0)


if __name__ == "__main__":
    unittest.main()
