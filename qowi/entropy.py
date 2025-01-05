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
