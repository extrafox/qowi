import numpy as np


def integer_to_shifted(integer: int, num_values) -> int:
    return int(integer) - num_values // 2

def shifted_to_integer(shifted_integer: int, num_values) -> int:
    return shifted_integer + num_values // 2

def integer_to_zigzag(int_value: int) -> int:
    return abs(int_value << 1) + (0 if int_value >= 0 else 1)


def zigzag_to_integer(zigzag_value: int) -> int:
    return (1 if zigzag_value & 1 == 0 else -1) * (zigzag_value >> 1)


def int_tuple_to_zigzag_tuple(integers: tuple) -> tuple:
    return tuple(integer_to_zigzag(a) for a in integers)


def zigzag_tuple_to_int_tuple(zigzag_tuple: tuple) -> tuple:
    return tuple(zigzag_to_integer(a) for a in zigzag_tuple)


def int_tuple_to_shifted_tuple(integers: tuple, num_values) -> tuple:
    return tuple(integer_to_shifted(a, num_values) for a in integers)


def shifted_tuple_to_int_tuple(zigzag_tuple: tuple, num_values) -> tuple:
    return tuple(shifted_to_integer(a, num_values) for a in zigzag_tuple)


def subtract_tuples(first: tuple, second: tuple) -> tuple:
    return tuple(int(a) - int(b) for a, b in zip(first, second))


def round_scaled_integer(value, scale, round_to_fraction):
    """
    Round a scaled integer to a specified fraction of its scale.

    Args:
        value (int): The integer to be rounded.
        scale (int): The scale factor (e.g., 4, 16, 64).
        round_to_fraction (float): The fraction of the scale to round to (e.g., 0.25 for 1/4th of scale).

    Returns:
        int: The rounded integer value at the same scale as the input.
    """
    # Convert the rounding fraction to the scaled integer domain
    rounding_increment = int(round_to_fraction * scale)

    if rounding_increment == 0:
        return value

    # Perform rounding in the scaled integer domain
    rounded_value = ((value + rounding_increment // 2) // rounding_increment) * rounding_increment

    return rounded_value

def round_scaled_integer_ndarray(values, scale, round_to_fraction):
    """
    Round a scaled ndarray of integers to a specified fraction of its scale.

    Args:
        values (ndarray): The array of integers to be rounded.
        scale (int): The scale factor (e.g., 4, 16, 64).
        round_to_fraction (float): The fraction of the scale to round to (e.g., 0.25 for 1/4th of scale).

    Returns:
        ndarray: The rounded array of integers at the same scale as the input.
    """
    # Convert the rounding fraction to the scaled integer domain
    rounding_increment = int(round_to_fraction * scale)

    if rounding_increment == 0:
        return values

    # Perform rounding in the scaled integer domain
    rounded_values = ((values + rounding_increment // 2) // rounding_increment) * rounding_increment

    return rounded_values