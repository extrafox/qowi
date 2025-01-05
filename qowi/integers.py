import numpy as np


def integer_to_zigzag(int_value: int) -> int:
    return abs(int_value << 1) + (0 if int_value >= 0 else 1)


def zigzag_to_integer(zigzag_value: int) -> int:
    return (1 if zigzag_value & 1 == 0 else -1) * (zigzag_value >> 1)


def int_tuple_to_zigzag_tuple(integers: tuple) -> tuple:
    return tuple(integer_to_zigzag(a) for a in integers)


def zigzag_tuple_to_int_tuple(zigzag_tuple: tuple) -> tuple:
    return tuple(zigzag_to_integer(a) for a in zigzag_tuple)


def subtract_tuples(first: tuple, second: tuple) -> tuple:
    return tuple(int(a) - int(b) for a, b in zip(first, second))
