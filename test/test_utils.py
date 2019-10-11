import unittest
import lib.utils.math_utils as mu
import lib.utils.array_utils as au


class MathUtilsTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_sigmoid(self):
        self.assertEqual(mu.sigmoid(0), 0.5)
        self.assertLessEqual(mu.sigmoid(100), 1)
        self.assertGreaterEqual(mu.sigmoid(-100), 0)
        self.assertEqual(mu.sigmoid(10 ** 1000), 1)
        self.assertEqual(mu.sigmoid(-1000), 0)


class ArrayUtilsTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_flatten(self):
        self.assertListEqual(au.flatten([1, [2,3], [4]]), [1, 2, 3, 4])
        self.assertListEqual(au.flatten(['a', ['b']]), ['a', 'b'])
        self.assertListEqual(au.flatten([1, [2, [3, 4]], 5]), [1, 2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()