import math
import numpy as np
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
    orders = np.array([calculate_order(v) for v in uint_tuple], dtype=int)
    offsets = (2 ** orders) - 2
    deltas = np.array(uint_tuple) - offsets

    leading_bits = [Bits(bin='1' * (o - 1) + '0') for o in orders]
    data_bits = [Bits(uint=d, length=int(o)) for d, o in zip(deltas, orders)]

    ret = BitArray()
    for lb, db in zip(leading_bits, data_bits):
        ret.append(lb + db)
    return ret


def simple_decode_tuple(bit_stream: BitStream, num_to_decode=1) -> tuple:
    ret = []
    for _ in range(num_to_decode):
        ret.append(simple_decode(bit_stream))
    return tuple(ret)


def simple_encode_ndarray(uint_array: np.ndarray) -> Bits:
    if not np.issubdtype(uint_array.dtype, np.integer):
        raise ValueError("Input array must have an unsigned integer dtype.")

    # Vectorized calculation for encoding
    order = np.floor(np.log2(uint_array + 2)).astype(int)
    offset = (2 ** order) - 2
    delta = uint_array - offset

    leading_bits = [Bits(bin='1' * (o - 1) + '0') for o in order.ravel()]
    data_bits = [Bits(uint=d, length=o) for d, o in zip(delta.ravel(), order.ravel())]

    ret = BitArray()
    for lb, db in zip(leading_bits, data_bits):
        ret.append(lb + db)
    return ret


def simple_decode_ndarray(bit_stream: BitStream, num_to_decode=1, dtype=np.uint32) -> np.ndarray:
    values = []
    for _ in range(num_to_decode):
        values.append(simple_decode(bit_stream))
    return np.array(values, dtype=dtype)
