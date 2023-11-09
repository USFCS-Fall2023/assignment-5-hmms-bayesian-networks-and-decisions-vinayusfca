import unittest

from HMM import HMM


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    # unittest.main()
    model = HMM()
    model.load('two_english')