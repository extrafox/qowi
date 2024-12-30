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
        c.observe(1)
        c.observe(2)
        c.observe(3)
        c.observe(4)
        exception_raised = False
        try:
            i = c.index(1) # this should raise an exception since the item is now old
        except ValueError:
            exception_raised = True
        self.assertTrue(exception_raised)

if __name__ == '__main__':
    unittest.main()
