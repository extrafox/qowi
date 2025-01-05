import unittest
from qowi.mflru_cache import MFLRUCache

class TestMFLRUCache(unittest.TestCase):

    def test_get_item(self):
        c = MFLRUCache(3)
        c.observe(1)
        c.observe(2)
        c.observe(3)
        self.assertEqual(c[1], 2)  # add assertion here

    def test_get_item_index_error_quick(self):
        c = MFLRUCache(3)
        c.observe(1)
        exception_raised = False
        try:
            v = c[2] # this should raise an exception since there are not enough items
        except IndexError:
            exception_raised = True
        self.assertTrue(exception_raised)

    def test_get_item_index_error_slow(self):
        c = MFLRUCache(3)
        exception_raised = False
        try:
            v = c[3] # this should raise an exception since it is out-of-bounds
        except IndexError:
            exception_raised = True
        self.assertTrue(exception_raised)

    def test_get_item_that_existed_but_is_now_out_of_bounds(self):
        c = MFLRUCache(3)
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

    def test_sort_order(self):
        c = MFLRUCache(3)
        c.observe('A') # count == 1
        c.observe('B') # count == 1, observed more recently
        self.assertTrue(repr(c) == '[B, A]')

        c.observe('A') # count == 2
        self.assertTrue(repr(c) == '[A, B]')

        c.observe('A') # count == 3
        self.assertTrue(repr(c) == '[A, B]')

        c.observe('B') # count == 2
        self.assertTrue(repr(c) == '[A, B]')

    def test_index(self):
        c = MFLRUCache(3)
        c.observe('A')
        c.observe('B')
        self.assertEqual(0, c.index('B'))

    def test_value_not_in_cache(self):
        c = MFLRUCache(3)
        c.observe('A')
        c.observe('B')
        observed_exception_raised = False
        try:
            c.index('C')
        except IndexError:
            observed_exception_raised = True
        self.assertTrue(observed_exception_raised)

if __name__ == '__main__':
    unittest.main()
