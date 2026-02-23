import unittest

from convolution import _compute_conv  # pyright: ignore[reportPrivateUsage]


class ComputeConvTests(unittest.TestCase):
    def test_conv2d(self):
        result = _compute_conv(227, 2, 11, 4)
        self.assertEqual(result, 56)


if __name__ == "__main__":
    _ = unittest.main()
