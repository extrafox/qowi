import numpy as np
import random
import unittest
from bitstring import BitStream
from qowi.uint10_decoder import Uint10Decoder
from qowi.uint10_encoder import Uint10Encoder
from qowi.wavelet import Wavelet

class TestUint10Decoder(unittest.TestCase):

    def test_instantiation(self):
        bitstream = BitStream()
        d = Uint10Decoder(bitstream, 1024)
        self.assertIsInstance(d, Uint10Decoder)

    def test_round_trip_one_token(self):
        bitstream = BitStream()
        expected_token = (1, 2, 3)

        e = Uint10Encoder(bitstream, 32)
        e.encode_next(expected_token)
        e.finish()

        d = Uint10Decoder(bitstream, 32)
        observed_token = d.decode_next()

        self.assertEqual(expected_token, observed_token)

    def test_round_trip_16_tokens(self):
        bitstream = BitStream()
        e = Uint10Encoder(bitstream, 32)

        expected_token_list = [(218, 111,  28),  (126, 238,  24),  (198,  57, 226),  ( 75, 132,  92),  ( 11,  11,  55),  ( 19,  25, 194),  (112,  87,  28),  (247, 236,  12),  ( 55, 211,   0),  (149,  70, 195),  ( 10,  89, 201),  (146,  78, 195),  (200, 181, 156),  (195,  54, 207),  (231, 228, 173),  ( 53, 249,  21)]
        for i in range(len(expected_token_list)):
            e.encode_next(expected_token_list[i])
        e.finish()

        d = Uint10Decoder(bitstream, 32)
        observed_token_list = []
        for i in range(len(expected_token_list)):
            observed_token = d.decode_next()
            observed_token_list.append(observed_token)

        self.assertEqual(expected_token_list, observed_token_list)

if __name__ == '__main__':
    unittest.main()
