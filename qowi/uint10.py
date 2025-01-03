import numpy as np

# The uint10 format:
# [magnitude: 9 bits][sign: 1 bit]
# sign: 0 is positive, 1 is negative
# int: the magnitude portion is used as-is
# float: the uint10 magnitude portion is 4 times the float magnitude

def from_int(int_value: int) -> int:
    return abs(int_value << 1) + (0 if int_value >= 0 else 1)


def to_int(uint10_value: int) -> int:
    return (1 if uint10_value & 1 == 0 else -1) * (uint10_value >> 1)


def from_float(float_value: float) -> int:
    return (int(abs(float_value * 4)) << 1) + (0 if float_value >= 0 else 1)


def to_float(uint10_value: int) -> float:
    return (1 if uint10_value & 1 == 0 else -1) * (uint10_value >> 1) / 4


def to_float_array(uint10_array: np.ndarray) -> np.ndarray:
    if uint10_array.dtype.kind != 'u':
        raise TypeError("Unsupported dtype {}. Uint10 array much be of an unsigned integer type".format(uint10_array.dtype))

    abs_values = (uint10_array >> 1).astype(np.float16)
    signs = np.where(uint10_array & 1, -1, 1)
    return signs * (abs_values / 4)


def from_float_array(float_array: np.ndarray) -> np.ndarray:
    if float_array.dtype.kind != 'f':
        raise TypeError("Unsupported dtype {}. Input array much be of a float type".format(float_array.dtype))

    abs_arr = np.abs(float_array * 4).astype(np.uint16)
    sign_bits = (float_array < 0).astype(np.uint16)
    return (abs_arr << 1) + sign_bits


def to_int_array(uint10_array: np.ndarray) -> np.ndarray:
    if uint10_array.dtype.kind != 'u':
        raise TypeError("Unsupported dtype {}. Uint10 array much be of an unsigned integer type".format(uint10_array.dtype))

    abs_values = (uint10_array >> 1)
    signs = np.where(uint10_array & 1, -1, 1)
    return signs * abs_values


def from_int_array(int_array: np.ndarray) -> np.ndarray:
    if int_array.dtype.kind != 'i':
        raise TypeError("Unsupported dtype {}. Input array much be of an integer type".format(int_array.dtype))

    abs_array = np.abs(int_array)
    sign_bits = (int_array < 0)
    return ((abs_array << 1) + sign_bits).astype(np.uint16)

