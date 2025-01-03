import math
import numpy as np
from bitstring import BitArray, Bits, BitStream

def calculate_order(value: int) -> int:
    return math.floor(math.log2(value + 2))


def encode(uint_value: int) -> Bits:
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


def decode(bit_stream: BitStream) -> int:
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


def encode_array(uint_array: np.ndarray) -> Bits:
    if uint_array.dtype.kind != 'u':
        raise TypeError("Encoded array must be of an unsigned int type")

    ret = BitArray()
    for uint_value in uint_array:
        ret.append(encode(uint_value))
    return ret


def decode_array(bit_stream: BitStream, num_to_decode=1) -> np.ndarray:
    ret = np.empty(shape=num_to_decode, dtype=np.uint16)
    for i in range(num_to_decode):
        ret[i] = decode(bit_stream)
    return ret
