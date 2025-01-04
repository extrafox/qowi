import numpy as np
from bitstring import BitStream
from qowi.uint10_decoder import Decoder
from qowi.uint10_encoder import UintEncoder
from qowi.wavelet import Wavelet

count = 0
while True: # go until something breaks
	count += 1
	print("Processing image, count: {}".format(count))

	random_image = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
	print("Image with shape: {}".format(random_image.shape))
	np.set_printoptions(threshold=np.prod(random_image.shape), linewidth=200)
	print(repr(random_image))

	hard_threshold = 1
	bit_shift = 2
	carry_over_bits = 1

	print("Encoding with bit shift {}, carry over {} and hard threshold {}...".format(bit_shift, carry_over_bits, hard_threshold))
	w = Wavelet().prepare_from_image(random_image)
	w.apply_hard_threshold(hard_threshold)
	e = UintEncoder(w, bit_shift, carry_over_bits)
	encoded_image = e.encode()
	bit_shift_threshold_size = len(encoded_image)
	original_image_size = random_image.shape[0] * random_image.shape[1] * random_image.shape[2] * 8
	print("Bit shift (bits): {} ({}%)".format(bit_shift_threshold_size, round(bit_shift_threshold_size / original_image_size * 100, 2)))
	print("Decoding...")
	d = Decoder(BitStream(encoded_image))
	decoded_image = d.decode()
	print("Done.")


