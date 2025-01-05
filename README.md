The Quite OK Wavelet Image (QOWI) Format
============

The QOWI (pronounced: COW-ee) image compression format is a reference
implementation and a work-in-progress for a format that can scale
from losslessly compressing images and wavelets to lossy compression
at competitive file sizes and image quality. This project was inspired
by, and the name is an homage to, the
[QOI Format](https://qoiformat.org/qoi-specification.pdf).

Current Project Goals
---------------------

* Support lossless compression of images and wavelets on par formats like PNG and QOI
* Support lossy compression with file size and image quality comparable to JPEG or WebP
* Have fun and learn about image and wavelet compression

Project Status
--------------

* CLI for performing QOWI encode and decode operations on images
* Wavelet generation at configurable number of levels
* Losslessly encode (tested on single image) at around 80% of original
* Lossy encoding using wavelet thresholding
* (coming soon) Configurable rounding at successive wavelet approximation levels
* (coming soon) Hard and soft thresholding of the wavelet coefficients
* (coming soon) Optimized traversal of approximations for encoding
* (coming soon) Optimized universal code, e.g. [Golomb](https://en.wikipedia.org/wiki/Golomb_coding) or similar code
* (coming soon) Better performance (within limits of Python)

Usage
-----

    usage: qowi.py [-h] [-t HARD_THRESHOLD] [-s SOFT_THRESHOLD] [-w WAVELET_LEVELS] [-p WAVELET_PRECISION] {encode,decode} source destination
    
    Quite OK Wavelet Image (QOWI) Encoder/Decoder
    
    positional arguments:
      {encode,decode}       Operation to perform: encode or decode
      source                Path to the source file
      destination           Path to the destination file
    
    optional arguments:
      -h, --help            show this help message and exit
      -t HARD_THRESHOLD, --hard-threshold HARD_THRESHOLD
                            Wavelet hard threshold
      -s SOFT_THRESHOLD, --soft-threshold SOFT_THRESHOLD
                            Wavelet soft threshold
      -w WAVELET_LEVELS, --wavelet-levels WAVELET_LEVELS
                            Number of wavelet levels to encode. Defaults to 10
      -p WAVELET_PRECISION, --wavelet-precision WAVELET_PRECISION
                            Precision to round at each wavelet level. Defaults to 0

Examples
--------

    user@host:~/dev/qowi$ ./qowi.py encode media/dog_32x32.bmp dog_32x32.qowi
    |================================================================================| 1023/1023 (100.00%)
    Encoding completed successfully.
    user@host:~/dev/qowi$:~/dev/qowi$ ./qowi.py decode dog_32x32.qowi dog_32x32.png
    |================================================================================| 1023/1023 (100.00%)
    Decoding completed successfully.

Discussion
----------

(COMING SOON)