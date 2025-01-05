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


# def zigzag_array_to_int16_array(zigzag_array: np.ndarray) -> np.ndarray:
#     if zigzag_array.dtype.kind != 'u':
#         raise TypeError("Unsupported dtype {}. Uint10 array much be of an unsigned integer type".format(zigzag_array.dtype))
#
#     magnitude_array = (zigzag_array >> 1).astype(np.int16)
#     signs = np.where(zigzag_array & 1, -1, 1)
#     return signs * magnitude_array
#
#
# def int_array_to_zigzag_array(int_array: np.ndarray) -> np.ndarray:
#     if int_array.dtype.kind != 'i':
#         raise TypeError("Unsupported dtype {}. Input array much be of an integer type".format(int_array.dtype))
#
#     magnitude_array = np.abs(int_array)
#     sign_bits = (int_array < 0)
#     return ((magnitude_array << 1) + sign_bits).astype(np.uint16)


# def float_to_uint10(float_value: float) -> int:
#     return (int(abs(float_value * 4)) << 1) + (0 if float_value >= 0 else 1)
#
#
# def uint10_to_float(uint10_value: int) -> float:
#     return (1 if uint10_value & 1 == 0 else -1) * (uint10_value >> 1) / 4


# def uint10_array_to_float16_array(uint10_array: np.ndarray) -> np.ndarray:
#     if uint10_array.dtype.kind != 'u':
#         raise TypeError("Unsupported dtype {}. Uint10 array much be of an unsigned integer type".format(uint10_array.dtype))
#
#     magnitude_array = (uint10_array >> 1).astype(np.float16)
#     signs = np.where(uint10_array & 1, -1, 1)
#     return signs * (magnitude_array / 4)


# def float_array_to_uint10_array(float_array: np.ndarray) -> np.ndarray:
#     if float_array.dtype.kind != 'f':
#         raise TypeError("Unsupported dtype {}. Input array much be of a float type".format(float_array.dtype))
#
#     magnitude_array = np.abs(float_array * 4).astype(np.uint16)
#     sign_bits = (float_array < 0).astype(np.uint16)
#     return (magnitude_array << 1) + sign_bits


# def right_bit_shift(uint_array: np.ndarray, bit_shift: int) -> np.ndarray:
#     float_array = uint10_array_to_float16_array(uint_array)
#     magnitude_array = np.abs(float_array * 4).astype(np.uint16)
#     sign_bits = (float_array < 0).astype(np.uint16)
#     shifted_array = magnitude_array >> bit_shift
#     return (shifted_array << 1) + sign_bits
#
#
# def left_bit_shift(uint_array: np.ndarray, bit_shift: int) -> np.ndarray:
#     float_array = uint10_array_to_float16_array(uint_array)
#     magnitude_array = np.abs(float_array * 4).astype(np.uint16)
#     sign_bits = (float_array < 0).astype(np.uint16)
#     shifted_array = magnitude_array << bit_shift
#     return (shifted_array << 1) + sign_bits
