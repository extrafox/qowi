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

* Support lossless compression of images and wavelets on par with formats like PNG and QOI
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

### Key Characteristics:
- Lossless compression comparable to PNG and QOI.
- Competitive lossy compression with file sizes and quality comparable to JPEG or WebP.
- Flexible implementation with configurable wavelet levels and thresholding.
- Designed to be fun and educational for developers interested in wavelet and image compression.

## Features

- **Lossless Image Compression**: Preserves all details while reducing file size.
- **Lossy Compression**: Uses configurable wavelet thresholding to optimize file size and quality.
- **Wavelet Transformations**: Supports Haar wavelets with multiple levels.
- **Universal Codes**: Efficient coding for variable-length integer representations.
- **Customizable Precision**: Rounding options for successive wavelet approximations.
- **Caching and Optimization**: Includes MFLRU caching for better performance.
- **Cross-platform CLI Tool**: Easily encode and decode images.

---

## Usage

### CLI Commands

Run the `qowi.py` script for encoding and decoding operations.

#### General Syntax:
```bash
usage: qowi.py [-h] [-t HARD_THRESHOLD] [-s SOFT_THRESHOLD] [-w WAVELET_LEVELS] [-p WAVELET_PRECISION] {encode,decode} source destination
```

#### Positional Arguments:
- **{encode,decode}**: Operation to perform.
- **source**: Path to the source file.
- **destination**: Path to the destination file.

#### Optional Arguments:
- **-h, --help**: Show help message and exit.
- **-t, --hard-threshold**: Wavelet hard threshold.
- **-s, --soft-threshold**: Wavelet soft threshold.
- **-w, --wavelet-levels**: Number of wavelet levels to encode (default: 10).
- **-p, --wavelet-precision**: Precision to round at each wavelet level (default: 0).

#### Examples:
1. **Encoding an Image**:
   ```bash
   ./qowi.py encode media/dog_32x32.bmp output.qowi
   ```

2. **Decoding an Image**:
   ```bash
   ./qowi.py decode output.qowi decoded.png
   ```

---

## Contact

For questions or suggestions, feel free to reach out to the project maintainer, me, via GitHub issues.

Table of Contents
-----------------

1. Overview of QOWI Approach
2. A Little Bit About the Author (DISCLAIMERS)
3. Wavelets 
   1. Haar Wavelets
   2. Lossless Haar with Integer Based Coefficients
   3. Handling Differently Sized Source Images
   4. Hard and Soft Thresholding
   5. Rounding Approximations
4. Entropy Coding
   1. Universal Codes
   2. Run Length Coding
   3. Difference Coding
   4. Value Coding
   5. Most Frequent Least Recently Used (MFLRU) Cache
   6. Zig-Zag Coding
5. The Encoding and Decoding Process
   1. Wavelet Traversal
   2. Integer Encoder

Overview of QOWI Approach
-------------------------

The Quite OK Wavelet Image (QOWI) format is inspired by the QOI format. QOI is intended
to only encode 3 or 4 channel 8-bit images losslessly. It does this by
using several operations to compress the original 8-bit pixel values:
INDEX, DIFF, LUMA, RUN and RGB (also RGBA). By looking at the current
pixel position and comparing to the last one, the encoder will see which
coding method results in the shortest code and it will use it.

QOWI follows this model, but with some modifications that are motivated
by the desire to encode wavelets, and specifically, wavelet coefficients.
The considerations for encoding wavelets include, dealing with signed floating
point values, more than 8 bits required to represent any particular
coefficient and less similarity between adjacent codes.

The rest of this document will cover the details of the QOWI approach.

Note that this format is still under active development and experimentation,
so assumptions and implementation details are likely to change over time.

A Little Bit About the Author (DISCLAIMERS)
-------------------------------------------

I am coming to wavelets and compression with no academic training
in the subject. I am a software engineer by training and I like to think
about technical problems. I learned about wavelets years ago and
thought they were interesting, but until I learned about QOI, it
was only a passing interest. But, QOI surprised me. How could it be
that such a simple format could emerge in the 2020s when academics
have been working in this problem domain for decades?

QOI is an approach that aligns well with my software development
experience and it started me thinking. If academics had failed to
find such a simple solution for standard raster image compression, opting instead
for convoluted solutions like PNG, then maybe wavelet compression
could be tackled with a similar approach and comparable results.

We are yet to see if QOWI will live up to expectations, but here we are...

Haar Wavelets
-------------

I'm not going to go into great detail about Wavelets in general
or even Haar wavelets specifically. For that, you can find many
[resources online](https://en.wikipedia.org/wiki/Haar_wavelet). However,
I will talk about how Haar was a good
choice for this project. I would love to hear from wavelet experts
about this approach and where better solutions have already been
found or where this approach could be extended.

In a nutshell, a 2D Haar wavelet construction allows you to take
four pixels in the spatial domain and transform them into the wavelet
domain. The translation is lossless, but it does require more bits
to store the same pixel values in the wavelet domain. [You can
thank Heisenberg](https://en.wikipedia.org/wiki/Uncertainty_principle)
for that, I suppose.

### Basic Construction ###

1. Break your source image up into 2x2 grids

|   a   |   b   |
|-------|-------|
|   c   |   d   |

2. Perform the following transform calculation

~~~python
ll = a + b + c + d
hl = a + b - c - d
lh = a - b + c - d
hh = a - b - c + d
~~~

In the normal construction, you would divide these expressions
by four (making the LL item the average, or approximation). But
for my approach, I leave these as integers so that I don't lose
precision

3. You then group the values by their filter type and repeat the process on the LL grouping

To get back to your original pixels, you just do everything in
reverse, only you are working in the wavelet domain now

1. Regroup your filter coefficients

| LL | HL |
|----|----|
| LH | HH |

2. Perform the reverse arithmetic

~~~python
a = (ll + hl + lh + hh) // 4
b = (ll + hl - lh - hh) // 4
c = (ll - hl + lh - hh) // 4
d = (ll - hl - lh + hh) // 4
~~~~

NOTE: I do the division in this direction, which is opposite of
the standard because I want to stay in the Integer domain.

3. Repeat through all the wavelet levels

This is a pretty hurried explanation of the whole process, but if
you really want to know how to do it, just read my code

Lossless Haar with Multiple Levels
----------------------------------

I alluded to it above, but, since wavelets are typically done using
floating point numbers, they are expected to be lossy since you
lose some information due to the limited size of the floating
point number. With each wavelet level you encode, you need to add
more bits in order to avoid rounding errors.

If you are starting with 8-bit integers in a typical raster image,
then, you will need 10-bits to encode the first wavelet level. To see how this
is the case, you can look at the maximum value for the addition of
four 8-bit integers:

~~~python
LL = (255 + 255 + 255 + 255) = 1020
~~~

The filter coefficients are not typically processed recursively,
but the approximation coefficients are. This means that you
will need to add two additional bits for each wavelet level
in order to store the approximation sum without data loss.

The current implementation uses 64-bit ints to store the
coefficients which means that you will be limited to 28 wavelet levels.
However, in practice, while I don't know how many levels is useful
in other contexts, you probably won't want to go that deep for
practical image compression.

Handling Differently Sized Source Images
----------------------------------------

For the straightforward implementation of Haar, the source image
needs to be square. There are probably better ways to handle this,
but I chose to find the smallest square image that the source
image will fit in, filled with zero (0) values, then I copy the
source image to the top-left corner. I perform the encode and
decode, then crop the original image out.

Normally, I would expect that this is pretty wasteful, and maybe
it is. But, with run-length encoding (described below) this
hasn't turned out to be a major problem. So, for now, I have left
it that way.

Hard and Soft Thresholding
--------------------------

The main way wavelets can result in compression is that they throw
away data that won't have a big effect on the source image. Haar
coefficients represent differences between pairs of pixels. When
these coefficients have a small magnitude, it means that the pixel
pairs are close in value. Throwing this data away has limited effect
on the perception of the image to the human viewer. For more information on
thresholding, try Google (or even better, chatGPT)

Rounding Approximations
-----------------------

In typical floating point wavelets, there is a natural rounding
that occurs in the approximations as you transform to higher and
higher levels. In the case of QOWI, I want to have the option
of performing a lossless encoding, so I need to keep all of the
precision. This is helped by only using integers to store the
coefficients. However, we can dial in the level of precision we
want to have by rounding at each successive wavelet level.

I have no knowledge of research into this type of operation, so I
can't really say how useful it is. I will be experimenting with it 
more, but my expectation is that,
by reducing entropy, RUN and CACHE op codes are more likely to hit.
Another option I will likely try is, not only to round the higher
level values, but to bit shift the coefficients during encode (and
shift back during decode) so that the values will encode better
with the universal code.

Universal Codes
---------------

Universal codes allow you to take any integer and represent it
with a code. These codes are variable length with higher magnitude
integers requiring more bits to encode. Currently, I am using a
code of my own design, though, I later did some research and found
that Elias invented these in 1975. I guess I was a little late for
that one. I tried to use Golomb codes, but my implementation of them
was too slow, so I went back to my version of Elias codes which I
was able to implement with a fast algorithm.

This is a major area that still needs some work in QOWI. The final
compression amount is greatly affected by choice of universal code.
Notably, QOI does not use a universal code and I believe this is
one of the factors that makes QOI so effective.

I will probably move back to Golomb codes which have the advantage
that they can be finely tuned based on empirical measurement of
frequencies of the numbers you want to represent. I just need to
find or code my own fast implementation. You have one? Let me know in
the comments.

Run Length Coding
-----------------

QOWI uses run-length coding to greatly compress consecutive
coefficients that have the same value. This, combined with
universal codes, allows run lengths of any length to be
represented by a single code

Difference Coding
-----------------

Of course, Haar wavelets themselves represent differences between
pairs of pixels in the spatial domain, but, when consecutive
coefficients have a small difference, it can be more efficient to
encode the difference between coefficients instead of the directly
encoding their values.

Value Coding
------------

This is the obvious one and, often, the least efficient. QOI tries
to avoid full value codes, which take 4 bytes in that system, and
it will generally be able to represent pixels with smaller codes.
Given that wavelets require more bits in the best case, compared
to spatial domain pixels, and since QOWI needs to be able to
encode losslessly, universal codes are used for each individual
color channel.

Most Frequent Least Recently Used (MFLRU) Cache
-----------------------------------------------

In QOI, a simple, 6-bit hash is used to store already seen
pixels in the cache, with a total capacity of 62 (not 64 as you
might expect) entries. This has the advantage of guaranteeing small
codes and having low performance overhead, but at the expense of
missing a lot of opportunities for caching.

For wavelet coefficients, this didn't seem like a good option.
At the very least, there is more entropy in adjacent and
consecutive coefficients relative to pixels meaning that
recent cached coefficients are less likely to match. For this
and other reasons, I opted to use a larger cache. Given the
Elias codes I'm currently using, the cache capacity is 65k,
since this is the largest number that will still result in a
code that is shorter than the longest VALUE code.

Universal codes work best for geometric sequences. This means
that smaller integer values should be more frequent and are, 
therefore, encoded to shorter bit sequences. The trade off is
that less frequent values will require more bits than a typical
fixed bit encoding (like uint) would provide

The solution I am using is to store coefficients in the cache in
order from the most frequent to the least frequent. For values
that have been observed the same number of times, the values are
stored in order of the most recent to least recently used.

I call this a Most Frequently Least Recently Used, or MFLRU, cache.
Anecdotally, and not yet backed by data, when I switched from an LRU
to the MFLRU cache, a saw a big bump in CACHE op code uses and a
shortening of the index values stored in the codes.

Zig-Zag Coding
--------------

Since coefficients are naturally represented as signed integers or
floats, you need to store an additional sign bit along with the uint
value before coding using the universal code. QOWI uses zig-zag
codes which append the sign bit to the right side of the uint
magnitude. This has the advantage of taking better advantage of
efficient encoding with universal codes without the waste of
storing the sign separately. The tradeoff is a little more
complexity.

Zig-zag codes are used whenever the encoder needs to store a
signed integer. Unsigned integers can be coded as universal codes
directly without this extra step.

Wavelet Traversal
-----------------

There are likely many traversal strategies to choose from for
serializing the wavelet coefficients. The one I have chosen is
depth-first traversal working from parent to children within a
single filter before moving to other filters.

I expect that parent to child similarities will be higher than
for other relationships between coefficients. Also, zero trees
(trees with only zero values from a parent node) will naturally
be encoded as a single RUN op code using this approach.

Caveat: at the moment, when the wavelet levels don't go to a
single set of parent nodes, the approximations are getting
traversed as if they are parent child relationship. Probably not
a big deal, but it may be possible to get more compression by
optimizing the traversal of the approximation values.

Integer Encoder
---------------

Saving the best for last, the Integer Encoder... QOI works
exclusively with 8-bit uint values as source data. This has
some big advantages that are not available when encoding longer
signed integers. So, the key changes that QOWI makes is to work
in the signed integer domain and to use universal codes to handle
the larger and variable sized input values.

The DELTA and VALUE op codes are interesting because they are
similar, but slightly change the order of operations.

### VALUE ###

1. Takes an integer coefficient as input
2. Converts to zig-zag integer
3. Converts to universal code

### DELTA ###

1. Takes an integer coefficient as input
2. Takes the difference between this integer and the last seen value
3. Converts to zig-zag integer
4. Converts to universal code

By trying these two strategies with every input value (as well as RUN
and CACHE strategies), the encoder can choose the strategy for
a given coefficient that results in the shortest encoded
representation.

Conclusion
----------

Well, if you got this far, you are a champion. This has been a
really fun project for me up to now. I have been more than a
little obsessed by it. I hope it is useful to someone, but, if
not, it has been great for me. I have learned a ton by thinking
through these problems. Cheers!















