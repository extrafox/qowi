
# NOTES:
# 1. using a dictionary index to speed up checking if the value is in the cache
# 2. using a doubly linked list with index to speed up observe() performance
# 3. cleaning up out-of-bounds nodes and their indices opportunistically

class Node:
    def __init__(self, last, next, value):
        self.last  = last
        self.next = next
        self.value = value

class LRUCache:
    def __init__(self, capacity):
        self._index = {}
        self._root = Node(None, None, None)
        self._capacity = capacity

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

    def observe(self, value):
        if value not in self._index:
            node = Node(None, None, value)
            self._index[value] = node
            self._prepend_node(node)
        else:
            node = self._index[value]
            self._prepend_node(node)

    def index(self, value):
        # fail quickly
        if value not in self._index:
            raise ValueError("{} not found".format(value))

        # search for index
        node = self._root
        i = -1
        while node.next is not None and i < self._capacity - 1:
            i += 1
            node = node.next
            if node.value == value:
                return i

        # clean up when the value is out-of-bounds
        if node.next is not None:
            self._delete_after(node)

        raise ValueError("{} not found".format(value))

    def _delete_after(self, parent: Node):
        node = parent.next
        parent.next = None

        while node is not None:
            del self._index[node.value]
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