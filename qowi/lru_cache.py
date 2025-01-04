
class Node:
    def __init__(self, last, next, value, index):
        self.last  = last
        self.next = next
        self.value = value
        self.index = index

    def __repr__(self):
        return f'{self.index}: {self.value}'


class LRUCache:

    def __init__(self, capacity):
        self._value_lookup = {}
        self._root = Node(None, None, None, None)
        self._capacity = capacity
        self._last_node = None

    def _prepend_node(self, node):
        if self._root.next is node:
            return

        old_first = self._root.next
        node_old_last = node.last
        node_old_next = node.next

        # update root
        self._root.next = node

        # update node
        node.last = self._root
        node.next = old_first
        node.index = 0

        # update old first node
        if old_first is not None:
            old_first.last = node

        # update node's old neighbors
        if node_old_last is not None:
            if node_old_next is not None:
                node_old_last.next = node_old_next
                node_old_next.last = node_old_last
            else:
                node_old_last.next = None

        if self._last_node is None:
            self._last_node = node
        elif self._last_node == node:
            self._last_node = node_old_last

    def observe(self, value):
        if value not in self._value_lookup:
            node = Node(None, None, value, None)
            self._value_lookup[value] = node
            self._prepend_node(node)
            self._reindex_to(self._last_node)
            self._expire_old_nodes()
        else:
            node = self._value_lookup[value]
            old_next = node.next
            self._prepend_node(node)
            self._reindex_to(old_next)

    def index(self, value):
        if value in self._value_lookup:
            return self._value_lookup[value].index
        else:
            raise IndexError

    def _reindex_to(self, target):
        node = self._root.next
        i = 0
        while True:
            if node is None:
                return

            node.index = i
            if node == target:
                break
            node = node.next
            i += 1

    def _expire_old_nodes(self):
        if self._last_node.index >= self._capacity:
            old_last = self._last_node
            self._last_node = old_last.last
            self._delete_after(self._last_node)

    def _delete_after(self, parent: Node):
        if parent.next is None:
            return

        node = parent.next
        parent.next = None

        while node is not None:
            del self._value_lookup[node.value]
            node = node.next

    def __getitem__(self, key):
        if key >= self._capacity:
            raise IndexError("Index {} is greater than capacity: {}".format(key, self._capacity))

        node = self._root
        i = -1
        while node.next is not None and i < key:
            i += 1
            node = node.next
            if i == key:
                return node.value

        raise IndexError

    def __repr__(self):
        ret = "["
        node = self._root
        i = -1
        while node.next is not None and i < self._capacity - 1:
            i += 1
            node = node.next
            if i > 0:
                ret += ", "
            ret += "{}".format(repr(node.value))
        ret += "]"
        return ret

    def __len__(self):
        return self._last_node.index + 1