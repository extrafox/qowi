import math
from bitstring import BitArray, Bits, BitStream

DEFAULT_M = 4 # TODO: figure out how to optimize this number for each data distribution

def calculate_order(value: int) -> int:
    return math.floor(math.log2(value + 2))


def simple_encode(uint_value: int) -> Bits:
    if uint_value < 0:
        raise ValueError("Entropy encoding cannot be negative")

    order = calculate_order(uint_value)
    offset = 2 ** order - 2
    delta = uint_value - offset
    if delta < 0:
        raise ValueError("Invalid delta calculation: value={}, offset={}, delta={}".format(uint_value, offset, delta))

    leading_bits = Bits(bin='1' * (order - 1) + '0')
    data_bits = Bits(uint=delta, length=order)
    return leading_bits + data_bits


def simple_decode(bit_stream: BitStream) -> int:
    order = 1
    offset = 0
    leading_ones = 0
    while bit_stream.peek(1).uint == 1:  # Peek without consuming
        leading_ones += 1
        bit_stream.read(1)  # Consume bit
        offset += 2 ** order
        order += 1

    bit_stream.read(1) # Skip the terminating '0' bit

    delta = bit_stream.read(order).uint
    return offset + delta


def simple_encode_tuple(uint_tuple) -> Bits:
    ret = BitArray()
    for uint_value in uint_tuple:
        ret.append(simple_encode(uint_value))
    return ret


def simple_decode_tuple(bit_stream: BitStream, num_to_decode=1) -> tuple:
    ret = [None] * num_to_decode
    for i in range(num_to_decode):
        ret[i] = simple_decode(bit_stream)
    return tuple(ret)


# def golomb_encode_tuple(uint_tuple, m=DEFAULT_M) -> Bits:
#     ret = BitArray()
#     for uint_value in uint_tuple:
#         ret.append(golomb_encode(uint_value, m))
#     return ret
#
#
# def golomb_decode_tuple(bitstream: BitStream, num_to_decode=1, m=DEFAULT_M) -> tuple:
#     ret = [None] * num_to_decode
#     for i in range(num_to_decode):
#         ret[i] = golomb_decode(bitstream, m)
#     return tuple(ret)
#
#
# def golomb_encode(n, m=DEFAULT_M):
#     """
#     Encode an integer n using Golomb coding with divisor m.
#
#     :param n: The integer to encode (non-negative).
#     :param m: The Golomb divisor (positive integer).
#     :return: A Bits object representing the encoded value.
#     """
#     if m <= 0:
#         raise ValueError("m must be a positive integer")
#     if n < 0:
#         raise ValueError("n must be a non-negative integer")
#
#     # Quotient and remainder
#     q = n // m
#     r = n % m
#
#     # Unary part: q repetitions of '1' followed by '0'
#     unary = Bits(bin='1' * q + '0')
#
#     # Binary representation of the remainder
#     b = (m - 1).bit_length()
#     threshold = (1 << b) - m
#
#     if r < threshold:
#         remainder = Bits(uint=r, length=b - 1)
#     else:
#         remainder = Bits(uint=r + threshold, length=b)
#
#     # Combine unary and remainder parts
#     return unary + remainder
#
# def golomb_decode(bitstream, m=DEFAULT_M):
#     """
#     Decode a Golomb-encoded value from a BitStream.
#
#     :param bitstream: The BitStream containing the encoded value.
#     :param m: The Golomb divisor (positive integer).
#     :return: The decoded integer value.
#     """
#     if m <= 0:
#         raise ValueError("m must be a positive integer")
#
#     # Read unary part (count number of '1's before the first '0')
#     q = 0
#     while bitstream.read(1).bin == '1':
#         q += 1
#
#     # Determine the number of bits for the remainder
#     b = (m - 1).bit_length()
#     threshold = (1 << b) - m
#
#     # Read the remainder bits
#     remainder_bits = b - 1
#     remainder = bitstream.read(remainder_bits).uint
#     if remainder >= threshold:
#         remainder = (remainder << 1) | bitstream.read(1).uint
#         remainder -= threshold
#
#     # Decode the value
#     n = q * m + remainder
#     return n
