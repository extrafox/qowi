import numpy as np
from bitstring import BitArray, Bits, BitStream
from qowi.entropy import entropy_decode, entropy_encode
from typing import List

INTEGER_ORDER_AND_INCREMENT = (1, 1)
FLOAT_ORDER_AND_INCREMENT = (1, 1)

class Primitive:
    @property
    def uint10(self) -> int:
        raise NotImplementedError()

    @property
    def value(self):
        raise NotImplementedError()

    @property
    def entropy_coded(self) -> Bits:
        raise NotImplementedError()

class PUnsignedInteger(Primitive):
    @classmethod
    def from_uint10(cls, uint10):
        return PUnsignedInteger(uint10)

    def __init__(self, value):
        self._value = value

    @property
    def uint10(self) -> int:
        return self._value

    @property
    def value(self) -> int:
        return self._value

    @property
    def entropy_coded(self) -> Bits:
        return entropy_encode(self.uint10, INTEGER_ORDER_AND_INCREMENT[0], INTEGER_ORDER_AND_INCREMENT[1])

class PFloat(Primitive):
    @classmethod
    def from_uint10(cls, uint10: int):
        sign = -1 if uint10 & 1 == 1 else 1
        magnitude = (uint10 >> 1) / 4
        if magnitude == 0.0 and sign == -1: # a 1 value creates an ambiguity between 0.0 and -0.0
            raise ValueError("A uint10 value of 1 is not allowed")
        return PFloat(magnitude * sign)

    def __init__(self, value):
        self._value = value

    @property
    def uint10(self) -> int:
        sign = 0 if self._value >= 0 else 1
        uint10 = (abs(int(self._value * 4)) << 1)
        if uint10 > 0: # a 1 value creates an ambiguity between 0.0 and -0.0
            uint10 += sign
        return uint10

    @property
    def value(self) -> float:
        return self._value

    @property
    def entropy_coded(self) -> Bits:
        return entropy_encode(self.uint10, FLOAT_ORDER_AND_INCREMENT[0], FLOAT_ORDER_AND_INCREMENT[1])

class PList:
    @classmethod
    def from_token(cls, token: tuple, dtype=PFloat):
        ret = []
        for element in token:
            if dtype == PFloat:
                ret.append(PFloat.from_uint10(element))
            elif dtype == PUnsignedInteger:
                ret.append(PUnsignedInteger.from_uint10(element))
            else:
                raise TypeError("dtype must be a Primitive")
        return PList.from_list(ret)

    @classmethod
    def from_bitstream(cls, bitstream: BitStream, length=3, dtype=PFloat):
        ret = []
        for i in range(length):
            if dtype == PFloat:
                uint10 = entropy_decode(bitstream, FLOAT_ORDER_AND_INCREMENT[0], FLOAT_ORDER_AND_INCREMENT[1])
                ret.append(PFloat.from_uint10(uint10))
            elif dtype == PUnsignedInteger:
                uint10 = entropy_decode(bitstream, INTEGER_ORDER_AND_INCREMENT[0], INTEGER_ORDER_AND_INCREMENT[1])
                ret.append(PUnsignedInteger(uint10))
            else:
                raise TypeError("dtype must be a Primitive")
        return PList.from_list(ret)

    @classmethod
    def from_ndarray(cls, array: np.ndarray):
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be an ndarray")

        if array.dtype.kind == "f":
            dtype = PFloat
        elif array.dtype.kind == "u":
            dtype = PUnsignedInteger
        else:
            raise TypeError("unsupported data type {}".format(l.dtype))

        ret = PList()
        for element in array:
            if dtype == PFloat:
                ret._list.append(PFloat(element))
            elif dtype == PUnsignedInteger:
                ret._list.append(PUnsignedInteger(element))
            else:
                raise TypeError("unsupported data type {}".format(dtype))

        return ret

    @classmethod
    def from_list(cls, array):
        if len(array) == 0:
            return PList()

        dtype = type(array[0])
        ret = PList()
        for element in array:
            if dtype == PFloat:
                ret._list.append(element)
            elif dtype == PUnsignedInteger:
                ret._list.append(element)
            elif dtype == float:
                ret._list.append(PFloat(element))
            elif dtype == int:
                ret._list.append(PUnsignedInteger(element))
            else:
                raise TypeError("unsupported data type {}".format(data_type))

        return ret

    _list: List[Primitive]

    def __init__(self):
        self._list = []

    @property
    def token(self):
        length = len(self._list)
        ret = [0] * length
        for i in range(length):
            ret[i] = self._list[i].uint10
        return tuple(ret)

    @property
    def ndarray(self):
        if type(self._list[0]) == PUnsignedInteger:
            dtype = np.uint16
        else:
            dtype = np.float16

        length = len(self._list)
        ret = np.empty(length, dtype=dtype)
        for i in range(length):
            ret[i] = self._list[i].value
        return ret

    @property
    def entropy_coded(self):
        ret = BitArray()
        for primitive in self._list:
            ret.append(primitive.entropy_coded)
        return ret

    def __getitem__(self, i):
        return self._list[i]

    def __setitem__(self, i, value):
        if not isinstance(value, Primitive):
            raise TypeError("value must be a Primitive")

        self._list[i] = value

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)
