from ApschedulerStandalone.triggers.cron import DEFAULT_VALUES
from bitstring import BitArray, Bits, BitStream

DEFAULT_START_ORDER = 2
DEFAULT_ORDER_INCREMENT = 3

def entropy_encode(value: int, start_order=DEFAULT_START_ORDER, order_increment=DEFAULT_ORDER_INCREMENT) -> BitArray:
    ret = BitArray()
    order = start_order
    min_value = 0
    while True:
        num_values = 2 ** order
        max_value = min_value + num_values - 1

        if min_value <= value <= max_value:
            delta = value - min_value
            ret.append(Bits(uint=0, length=1))
            ret.append(Bits(uint=delta, length=order))
            return ret

        ret.append(Bits(uint=1, length=1))
        min_value = max_value + 1
        order += order_increment

def entropy_decode(bit_stream: BitStream, start_order=DEFAULT_START_ORDER, order_increment=DEFAULT_ORDER_INCREMENT) -> int:
    order = start_order
    min_value = 0
    while True:
        num_values = 2 ** order
        max_value = min_value + num_values - 1

        b = bit_stream.read(1)
        if b.uint == 0:
            value = min_value + bit_stream.read(order).uint
            return value

        min_value = max_value + 1
        order += order_increment
