import unittest

import torch.nn as tnn

from convolution import _compute_conv  # pyright: ignore[reportPrivateUsage]
from convolution import _compute_shape  # pyright: ignore[reportPrivateUsage]
from convolution import TensorShape


class ComputeConvTests(unittest.TestCase):
    def test_conv(self):
        result = _compute_conv(227, 2, 11, 4)
        self.assertEqual(result, 56)


class ComputShapeTests(unittest.TestCase):
    def test_conv(self):
        shape = TensorShape(32, 32, 3)
        padding = (2, 2)
        kernel_size = (5, 5)
        stride = (3, 3)
        conv = tnn.Conv2d(
            3, 64, padding=padding, kernel_size=kernel_size, stride=stride
        )
        result = _compute_shape(conv, shape)

        checker_height = _compute_conv(
            shape.height, padding[0], kernel_size[0], stride[0]
        )
        checker_width = _compute_conv(
            shape.width, padding[1], kernel_size[1], stride[1]
        )
        checker = TensorShape(checker_height, checker_width, 64)

        self.assertEqual(result, checker)

    def test_relu(self):
        activation = tnn.ReLU()
        shape = TensorShape(32, 32, 3)

        result = _compute_shape(activation, shape)

        self.assertEqual(result, shape)

    def test_dropout(self):
        activation = tnn.Dropout(0.5)
        shape = TensorShape(32, 32, 3)

        result = _compute_shape(activation, shape)

        self.assertEqual(result, shape)

    def test_batch_norm(self):
        activation = tnn.BatchNorm2d(3024)
        shape = TensorShape(32, 32, 3)

        result = _compute_shape(activation, shape)

        self.assertEqual(result, shape)

    def test_max_pool(self):
        shape = TensorShape(32, 32, 3)
        padding = (2, 2)
        kernel_size = (5, 5)
        stride = (3, 3)
        pool = tnn.MaxPool2d(kernel_size[0], stride[0], padding[0])
        result = _compute_shape(pool, shape)

        checker_height = _compute_conv(
            shape.height, padding[0], kernel_size[0], stride[0]
        )
        checker_width = _compute_conv(
            shape.width, padding[1], kernel_size[1], stride[1]
        )
        checker = TensorShape(checker_height, checker_width, 3)

        self.assertEqual(result, checker)

    def test_adaptive_max_pool(self):
        shape = TensorShape(32, 32, 3)
        out_size = (3, 3)
        pool = tnn.AdaptiveMaxPool2d(out_size)
        result = _compute_shape(pool, shape)

        checker = TensorShape(out_size[0], out_size[1], 3)

        self.assertEqual(result, checker)

    def test_adaptive_avg_pool(self):
        shape = TensorShape(32, 32, 3)
        out_size = (3, 3)
        pool = tnn.AdaptiveAvgPool2d(out_size)
        result = _compute_shape(pool, shape)

        checker = TensorShape(out_size[0], out_size[1], 3)

        self.assertEqual(result, checker)


if __name__ == "__main__":
    _ = unittest.main()
