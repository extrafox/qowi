import numpy as np
import qowi.entropy as entropy
import unittest
from bitstring import BitStream

from qowi import integers
from qowi.integer_encoder import IntegerEncoder, OP_CODE_CACHE, OP_CODE_RUN, OP_CODE_VALUE, OP_CODE_DELTA, ZERO_INTEGER


class TestIntegerEncoder(unittest.TestCase):

    def test_instantiation(self):
        bitstream = BitStream()
        e = IntegerEncoder(bitstream, 1024)
        self.assertIsInstance(e, IntegerEncoder)

    def test_encode_and_read_delta_token(self):
        bitstream = BitStream()
        expected_token = (-2, 0, 2)

        e = IntegerEncoder(bitstream, 1024)
        e.encode_next(expected_token)
        e.finish()

        op_code = bitstream.read(2)
        self.assertEqual(OP_CODE_DELTA, op_code)

        delta_zigzag = entropy.decode_tuple(bitstream, 3)
        delta_value = integers.zigzag_tuple_to_int_tuple(delta_zigzag)
        observed_token = integers.subtract_tuples(ZERO_INTEGER, delta_value)

        self.assertEqual(expected_token, observed_token)

    def test_encode_and_read_value_token(self):
        bitstream = BitStream()
        tokens = [(255, 255, 255), (1, 1, 1)]

        e = IntegerEncoder(bitstream, 1024)
        e.encode_next(tokens[0])
        e.encode_next(tokens[1])
        e.finish()

        op_code = bitstream.read(2)
        delta_zigzag = entropy.decode_tuple(bitstream, 3)
        op_code = bitstream.read(2)
        self.assertEqual(OP_CODE_VALUE, op_code)

        value_zigzag = entropy.decode_tuple(bitstream, 3)
        observed_token = integers.zigzag_tuple_to_int_tuple(value_zigzag)
        self.assertEqual(tokens[1], observed_token)


    def test_encode_and_read_run_token(self):
        bitstream = BitStream()

        e = IntegerEncoder(bitstream, 1024)
        e.encode_next(ZERO_INTEGER)
        e.finish()

        op_code = bitstream.read(2)
        self.assertEqual(OP_CODE_RUN, op_code)

        run_value = entropy.decode(bitstream) + 1
        self.assertEqual(1, run_value)


    def test_encode_and_read_cache_token(self):
        bitstream = BitStream()
        tokens = [(255, 255, 255), (0, 0, 0)]

        e = IntegerEncoder(bitstream, 1024)
        e.encode_next(tokens[0])
        e.encode_next(tokens[1])
        e.finish()

        op_code = bitstream.read(2)
        delta_zigzag = entropy.decode_tuple(bitstream, 3)
        op_code = bitstream.read(2)
        self.assertEqual(OP_CODE_CACHE, op_code)

        pos_value = entropy.decode(bitstream)
        self.assertEqual(1, pos_value)


if __name__ == '__main__':
    unittest.main()
