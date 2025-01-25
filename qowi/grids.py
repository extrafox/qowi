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

BIT_SHIFTS = np.array([24, 16, 8, 0], dtype=np.int32)
def calculate_haar_keys(coefficients: np.ndarray) -> np.ndarray:
    keys = np.sum((coefficients & 0xFF) << BIT_SHIFTS, axis=1)
    return keys

def grids_to_indices(grids: np.ndarray, pixel_bit_depth: int) -> np.ndarray:
    validate_bit_depth(pixel_bit_depth)

    # Avoid unnecessary wrapping; grids is already an ndarray
    total_bits = pixel_bit_depth * 4  # Since we have 4 values per grid
    shifts = np.array([total_bits - (pixel_bit_depth * (i + 1)) for i in range(4)])

    # No need to wrap grids in np.array, directly use the ndarray
    return np.sum(grids * (1 << shifts), axis=1)

def indices_to_grids(indices: np.ndarray, pixel_bit_depth: int) -> np.ndarray:
    validate_bit_depth(pixel_bit_depth)
    max_value = (1 << pixel_bit_depth) - 1
    shifts = np.array([0, 1, 2, 3]) * pixel_bit_depth

    # Create the grids in a more efficient manner by modifying in-place
    grids = np.zeros((indices.shape[0], 4), dtype=np.uint8)
    for i, index in enumerate(indices):
        grids[i] = [(index >> shift) & max_value for shift in shifts[::-1]]

    return grids
