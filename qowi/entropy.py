import math
import numpy as np
from bitstring import BitArray, Bits, BitStream

DEFAULT_M = 60 # TODO: figure out how to optimize this number for each data distribution

def golomb_encode_tuple(uint_tuple, m=DEFAULT_M) -> Bits:
    ret = BitArray()
    for uint_value in uint_tuple:
        ret.append(golomb_encode(uint_value, m))
    return ret


def golomb_decode_tuple(bitstream: BitStream, num_to_decode=1, m=DEFAULT_M) -> tuple:
    ret = [None] * num_to_decode
    for i in range(num_to_decode):
        ret[i] = golomb_decode(bitstream, m)
    return tuple(ret)


def golomb_encode(n, m=DEFAULT_M):
    """
    Encodes an integer n using Golomb coding with parameter M and returns a Bits object.

    :param n: The integer to encode (non-negative).
    :param m: The Golomb parameter (positive integer).
    :return: A Bits object containing the Golomb-encoded value.
    """
    if m <= 0:
        raise ValueError("M must be a positive integer")
    if n < 0:
        raise ValueError("n must be a non-negative integer")

    # Quotient part (unary encoding)
    q = n // m
    unary = BitArray(bool=True) * q  # q '1' bits
    unary.append(BitArray(bool=False))  # 1 '0' bit

    # Remainder part (binary encoding)
    r = n % m
    b = (m - 1).bit_length()  # Number of bits to encode the remainder
    threshold = (1 << b) - m

    if r < threshold:  # Smaller remainder fits in b-1 bits
        remainder = BitArray(uint=r, length=b - 1)
    else:
        r += threshold
        remainder = BitArray(uint=r, length=b)

    # Combine unary and remainder parts
    unary.append(remainder)
    return unary


def golomb_decode(bitstream: BitStream, m=DEFAULT_M):
    """
    Decodes a Golomb-encoded value from a BitStream.

    :param bitstream: The BitStream object containing the encoded data.
    :param m: The Golomb parameter (positive integer).
    :return: The decoded integer and the number of bits read.
    """
    if m <= 0:
        raise ValueError("M must be a positive integer")
    if not isinstance(bitstream, BitStream):
        raise TypeError("bit_stream must be an instance of BitStream")

    # Decode the unary part
    q = 0
    while bitstream.read(1).bin == '1':
        q += 1

    # Decode the remainder part
    b = (m - 1).bit_length()  # Number of bits for the remainder
    threshold = (1 << b) - m

    # Read the next `b-1` bits first
    r_bits = bitstream.read(b - 1).bin
    r = int(r_bits, 2)

    if r >= threshold:  # If remainder is above threshold, read one more bit
        r = (r << 1) | int(bitstream.read(1).bin, 2)
        r -= threshold

    # Calculate the decoded value
    n = q * m + r
    return n
