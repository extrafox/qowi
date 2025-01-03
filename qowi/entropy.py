import math
from bitstring import Bits, BitStream

def calculate_order(value: int) -> int:
    return math.floor(math.log2(value + 2))

def entropy_encode(value: int) -> Bits:
    if value < 0:
        raise ValueError("Entropy encoding cannot be negative")

    order = calculate_order(value)
    offset = 2 ** order - 2
    delta = value - offset
    if delta < 0:
        raise ValueError(f"Invalid delta calculation: value={value}, offset={offset}, delta={delta}")

    leading_bits = Bits(bin='1' * (order - 1) + '0')
    data_bits = Bits(uint=delta, length=order)
    return leading_bits + data_bits

def entropy_decode(bit_stream: BitStream) -> int:
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
