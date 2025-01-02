import numpy as np
from bitstring import BitArray, Bits, BitStream
from typing import List

DEFAULT_START_ORDER = 2
DEFAULT_ORDER_INCREMENT = 3

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
        return entropy_encode(self)

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
        return entropy_encode(self)

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
            ret.append(entropy_decode(bitstream, dtype=dtype))
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

def entropy_encode(primitive: Primitive, start_order=DEFAULT_START_ORDER, order_increment=DEFAULT_ORDER_INCREMENT) -> Bits:
    ret = BitArray()
    value = primitive.uint10
    order = start_order
    min_value = 0
    while True:
        num_values = 2 ** order
        max_value = min_value + num_values - 1

        if min_value <= value <= max_value:
            delta = value - min_value
            ret.append(Bits(uint=0, length=1))
            ret.append(Bits(uint=delta, length=order))
            return ret

        ret.append(Bits(uint=1, length=1))
        min_value = max_value + 1
        order += order_increment

def entropy_decode(bit_stream: BitStream, start_order=DEFAULT_START_ORDER, order_increment=DEFAULT_ORDER_INCREMENT, dtype=PFloat) -> Primitive:
    order = start_order
    min_value = 0
    while True:
        num_values = 2 ** order
        max_value = min_value + num_values - 1

        b = bit_stream.read(1)
        if b.uint == 0:
            value = min_value + bit_stream.read(order).uint
            if dtype == PFloat:
                return PFloat.from_uint10(value)
            elif dtype == PUnsignedInteger:
                return PUnsignedInteger(value)
            else:
                raise TypeError("dtype must be a Primitive")

        min_value = max_value + 1
        order += order_increment
