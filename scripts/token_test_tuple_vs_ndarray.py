import random
import numpy as np
import time

from bitstring import BitStream

import qowi.entropy as entropy

ITERATIONS = 10000

# Generate random test data
test_tuple = tuple(random.randint(0, 1020) for _ in range(3))
test_array = np.array(test_tuple, dtype=np.uint32)

# Measure tuple encoding/decoding
tuple_start = time.time()
for _ in range(ITERATIONS):
    encoded_tuple = entropy.simple_encode_tuple(test_tuple)
    decoded_tuple = entropy.simple_decode_tuple(BitStream(encoded_tuple), num_to_decode=3)
tuple_end = time.time()

# Measure ndarray encoding/decoding
array_start = time.time()
for _ in range(ITERATIONS):
    encoded_array = entropy.simple_encode_ndarray(test_array)
    decoded_array = entropy.simple_decode_ndarray(BitStream(encoded_array), num_to_decode=3, dtype=np.uint32)
array_end = time.time()

print("Tuple encoding/decoding time:", tuple_end - tuple_start)
print("Ndarray encoding/decoding time:", array_end - array_start)
