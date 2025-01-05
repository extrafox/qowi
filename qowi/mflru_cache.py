from sortedcontainers import SortedList

class ValueNode:
    def __init__(self, value):
        self.value = value
        self.observed_count = 0
        self.observation_index = None

    def __lt__(self, other):
        if self.observed_count == other.observed_count:
            return self.observation_index > other.observation_index # a higher index is more recent
        else:
            return self.observed_count > other.observed_count # a higher frequency count gets lower index

    def __repr__(self):
        return f"Value({self.value}, {self.observed_count}, {self.observation_index})"

class MFLRUCache:

    def __init__(self, capacity):
        self._list = SortedList()
        self._value_index = {}
        self._capacity = capacity
        self._observation_index = -1

    def observe(self, value):
        self._observation_index += 1

        if value not in self._value_index:
            node = ValueNode(value)
            self._value_index[value] = node
        else:
            node = self._value_index[value]

        self._list.discard(node)
        node.observed_count += 1
        node.observation_index = self._observation_index
        self._list.add(node)

        if len(self._list) > self._capacity:
            self._list.pop()

    def index(self, value):
        if value not in self._value_index:
            raise IndexError

        try:
            return self._list.index(self._value_index[value])
        except ValueError:
            raise IndexError

    def __getitem__(self, key):
        return self._list[key].value

    def __repr__(self):
        ret = "["
        count = 0
        for node in self._list:
            if count > 0:
                ret += ", "
            ret += f"{node.value}"
            count += 1
        ret += "]"
        return ret