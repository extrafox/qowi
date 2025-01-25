import math
import numpy as np
from numpy import ndarray
from qowi.haar_sort_table import HaarSortTable


class HaarSortWavelet:
    def __init__(self, haar_sort_table: HaarSortTable, width=0, height=0, color_depth=0):
        self.width = width
        self.height = height
        self.color_depth = color_depth
        self.haar_sort_table = haar_sort_table
        self.num_levels = 0
        self.wavelet = None
        self._initialize_from_shape(width, height, color_depth)

    def _initialize_from_shape(self, width, height, color_depth):
        """Initialize the wavelet array based on image dimensions and color depth."""
        self.width = width
        self.height = height
        self.color_depth = color_depth

        if width == 0 or height == 0:
            self.num_levels = 0
            self.wavelet = None
        else:
            self.num_levels = max(math.ceil(math.log2(width)), math.ceil(math.log2(height)))
            length = 2 ** self.num_levels
            self.wavelet = np.zeros((length, length, self.color_depth), dtype=np.uint8)

    def _generate_wavelet(self):
        """Generate the wavelet transform levels using vectorized operations."""
        for dest_level in reversed(range(self.num_levels)):
            dest_length = 2 ** dest_level
            source_length = dest_length * 2

            # Extract 2x2 blocks for all channels at once
            a = self.wavelet[0:source_length:2, 0:source_length:2]
            b = self.wavelet[0:source_length:2, 1:source_length:2]
            c = self.wavelet[1:source_length:2, 0:source_length:2]
            d = self.wavelet[1:source_length:2, 1:source_length:2]

            # Prepare grids for all blocks and channels
            grids = np.stack([a, b, c, d], axis=-1)  # Shape: (dest_length, dest_length, color_depth, 4)

            # Vectorized transformation using HaarSortTable
            ll, hl, lh, hh = np.split(
                self.haar_sort_table.pixels_to_wavelets(grids.reshape(-1, 4)), 4, axis=-1
            )

            # Reshape the results back to grid shape
            ll = ll.reshape(dest_length, dest_length, self.color_depth)
            hl = hl.reshape(dest_length, dest_length, self.color_depth)
            lh = lh.reshape(dest_length, dest_length, self.color_depth)
            hh = hh.reshape(dest_length, dest_length, self.color_depth)

            # Store in appropriate sub-bands
            self.wavelet[:dest_length, :dest_length] = ll
            self.wavelet[:dest_length, dest_length:source_length] = hl
            self.wavelet[dest_length:source_length, :dest_length] = lh
            self.wavelet[dest_length:source_length, dest_length:source_length] = hh

    def prepare_from_image(self, image: ndarray):
        """Prepare the wavelet transform from an input image."""
        self._initialize_from_shape(image.shape[0], image.shape[1], image.shape[2])

        # Copy the input image into the top-left corner of the wavelet array
        self.wavelet[:self.width, :self.height] = image

        # Generate the wavelet transform
        self._generate_wavelet()
        return self

    def prepare_from_wavelet(self, wavelet: ndarray):
        """Prepare the object from an existing wavelet array."""
        self._initialize_from_shape(wavelet.shape[0], wavelet.shape[1], wavelet.shape[2])
        self.wavelet = wavelet

    def as_image(self):
        """Reconstruct the image from the wavelet transform using vectorized operations."""
        reconstructed = self.wavelet.copy()

        for source_level in range(self.num_levels):
            source_length = 2 ** source_level
            dest_length = source_length * 2

            # Extract sub-bands for all channels
            ll = reconstructed[:source_length, :source_length]
            hl = reconstructed[:source_length, source_length:dest_length]
            lh = reconstructed[source_length:dest_length, :source_length]
            hh = reconstructed[source_length:dest_length, source_length:dest_length]

            # Prepare Haar components for all blocks and channels
            components = np.stack([ll, hl, lh, hh], axis=-1)  # Shape: (source_length, source_length, color_depth, 4)

            # Vectorized inverse transformation using HaarSortTable
            a, b, c, d = np.split(
                self.haar_sort_table.wavelets_to_pixels(components.reshape(-1, 4)), 4, axis=-1
            )

            # Reshape the results back to grid shape
            a = a.reshape(source_length, source_length, self.color_depth)
            b = b.reshape(source_length, source_length, self.color_depth)
            c = c.reshape(source_length, source_length, self.color_depth)
            d = d.reshape(source_length, source_length, self.color_depth)

            # Reconstruct the larger grid
            reconstructed[:dest_length:2, :dest_length:2] = a
            reconstructed[:dest_length:2, 1:dest_length:2] = b
            reconstructed[1:dest_length:2, :dest_length:2] = c
            reconstructed[1:dest_length:2, 1:dest_length:2] = d

        # Return the reconstructed image trimmed to the original dimensions
        return reconstructed[:self.width, :self.height].astype(np.uint8)
