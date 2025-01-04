import unittest
from qowi.lru_cache import LRUCache

class TestLRUCache(unittest.TestCase):

    def test_capacity_limit(self):
        c = LRUCache(3)
        c.observe(1)
        c.observe(2)
        c.observe(3)
        c.observe(4)
        self.assertEqual(repr(c), "[4, 3, 2]")  # add assertion here

    def test_get_item(self):
        c = LRUCache(3)
        c.observe(1)
        c.observe(2)
        c.observe(3)
        self.assertEqual(c[1], 2)  # add assertion here

    def test_get_item_index_error_quick(self):
        c = LRUCache(3)
        c.observe(1)
        exception_raised = False
        try:
            v = c[2] # this should raise an exception since there are not enough items
        except IndexError:
            exception_raised = True
        self.assertTrue(exception_raised)

    def test_get_item_index_error_slow(self):
        c = LRUCache(3)
        exception_raised = False
        try:
            v = c[3] # this should raise an exception since it is out-of-bounds
        except IndexError:
            exception_raised = True
        self.assertTrue(exception_raised)

    def test_get_item_that_existed_but_is_now_out_of_bounds(self):
        c = LRUCache(3)
        c.observe('A')
        c.observe('B')
        c.observe('C')
        c.observe('D')
        exception_raised = False
        try:
            i = c.index('A') # this should raise an exception since the item is now old
        except IndexError:
            exception_raised = True
        self.assertTrue(exception_raised)

    def test_last_node_and_size(self):
        c = LRUCache(3)
        c.observe('A')
        self.assertEqual(c._last_node.value, c[0])
        self.assertEqual(len(c), 1)
        c.observe('B')
        self.assertEqual(c._last_node.value, c[1])
        self.assertEqual(len(c), 2)
        c.observe('C')
        self.assertEqual(c._last_node.value, c[2])
        self.assertEqual(len(c), 3)
        c.observe('D')
        self.assertEqual(c._last_node.value, c[2])
        self.assertEqual(len(c), 3)
        self.assertEqual(repr(c), "['D', 'C', 'B']")  # add assertion here

if __name__ == '__main__':
    unittest.main()
