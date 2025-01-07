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


def rescale(value: int, rescale_digits: int) -> int:
    """
    Rescales an integer that represents a scaled float value by applying a bit shift.
    A positive rescale_digits value shifts to the left (increasing scale), while a
    negative rescale_digits value shifts to the right (reducing scale).

    Parameters:
    value (int): The input integer value to be rescaled.
    rescale_digits (int): The number of bits to shift. Positive for left shift,
                          negative for right shift.

    Returns:
    int: The rescaled integer value.
    """
    if rescale_digits == 0:
        return value

    if rescale_digits > 0:
        # Apply the left bit shift for positive rescale_digits
        return value << rescale_digits

    else:
        shift_amount = 1 << abs(rescale_digits)  # Equivalent to 2 ** abs(rescale_digits)
        remainder = value & (shift_amount - 1)  # Extract the bits being discarded

        # Perform rounding: Add half the shift amount to the value if rounding up
        if value >= 0:
            value += shift_amount // 2
        else:
            value -= shift_amount // 2

        # Apply the right bit shift
        return value >> abs(rescale_digits)

def rescale_ndarray(values: np.ndarray, rescale_digits: int) -> np.ndarray:
    """
    Rescales an ndarray of integers that represent scaled float values by applying a bit shift.
    A positive rescale_digits value shifts to the left (increasing scale), while a
    negative rescale_digits value shifts to the right (reducing scale).

    Parameters:
    values (np.ndarray): The input ndarray of integer values to be rescaled.
    rescale_digits (int): The number of bits to shift. Positive for left shift,
                          negative for right shift.

    Returns:
    np.ndarray: The rescaled ndarray.
    """
    if rescale_digits == 0:
        return values

    if rescale_digits > 0:
        # Apply the left bit shift for positive rescale_digits
        return values << rescale_digits

    else:
        shift_amount = 1 << abs(rescale_digits)  # Equivalent to 2 ** abs(rescale_digits)
        remainder = values & (shift_amount - 1)  # Extract the bits being discarded

        # Perform rounding: Add half the shift amount to the values if rounding up
        adjustment = shift_amount // 2
        values = np.where(values >= 0, values + adjustment, values - adjustment)

        # Apply the right bit shift
        return values >> abs(rescale_digits)


