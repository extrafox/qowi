import numpy as np

def validate_bit_depth(pixel_bit_depth: int):
    if pixel_bit_depth not in {2, 4, 8}:
        raise ValueError("Bit depth must be 2, 4, or 8.")

def get_struct_format(pixel_bit_depth: int) -> str:
    return {2: "B", 4: "H", 8: "I"}[pixel_bit_depth]

def calculate_haar_coefficients(grids: np.ndarray) -> np.ndarray:
    grids = np.array(grids, dtype=np.int16)
    coefficients = np.zeros((grids.shape[0], 4), dtype=np.int16)

    coefficients[:, 0] = np.sum(grids, axis=1)  # LL
    coefficients[:, 1] = grids[:, 0] - grids[:, 1] + grids[:, 2] - grids[:, 3]  # HL
    coefficients[:, 2] = grids[:, 0] + grids[:, 1] - grids[:, 2] - grids[:, 3]  # LH
    coefficients[:, 3] = grids[:, 0] - grids[:, 1] - grids[:, 2] + grids[:, 3]  # HH

    return coefficients

def grid_to_index(grids: np.ndarray, pixel_bit_depth: int) -> int:
    validate_bit_depth(pixel_bit_depth)
    grids = np.array(grids, dtype=np.uint8)
    shifts = np.array([3, 2, 1, 0]) * (pixel_bit_depth // 4)
    return np.sum(grids << shifts, axis=1)

def calculate_haar_keys(coefficients: np.ndarray) -> np.ndarray:
    # Constants for bit shifts
    BIT_SHIFTS = np.array([24, 16, 8, 0], dtype=np.int32)

    # Apply bit shifts and combine into keys
    keys = np.sum((coefficients & 0xFF) << BIT_SHIFTS, axis=1)
    return keys

def index_to_grid(index: int, pixel_bit_depth: int) -> np.ndarray:
    validate_bit_depth(pixel_bit_depth)
    max_value = (1 << pixel_bit_depth) - 1
    shifts = np.array([0, 1, 2, 3]) * pixel_bit_depth
    grids = [(index >> shift) & max_value for shift in shifts]
    return np.array(grids[::-1], dtype=np.uint8)
