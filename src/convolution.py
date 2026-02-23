from collections.abc import Iterable
from functools import singledispatch
from typing import NamedTuple, Never, Self, overload

import torch.nn as tnn


@overload
def _to_pair(obj: int | tuple[int, int]) -> tuple[int, int]: ...


@overload
def _to_pair(obj: tuple[int | None, int | None]) -> tuple[int | None, int | None]: ...


@overload
def _to_pair(obj: None) -> Never: ...


def _to_pair(obj: int | tuple[int, int] | tuple[int | None, int | None] | None):
    match obj:
        case int():
            return (obj, obj)
        case None:
            raise ValueError("Shape does not exist")
        case _:
            return obj


class TensorShape(NamedTuple):
    height: int
    width: int
    channels: int

    def in_features(self):
        return self.height * self.channels * self.width


@singledispatch
def _compute_shape(module: tnn.Module, previous_shape: TensorShape) -> TensorShape:  # pyright: ignore[reportUnusedParameter]
    raise NotImplementedError(
        f"Cannot compute features map for the module: {type(module).__name__}"
    )


def _compute_conv(origin: int, padding: int, kernel_size: int, stride: int) -> int:
    return int((origin + 2 * padding - (kernel_size - 1) - 1) / stride + 1)


@_compute_shape.register
def _(module: tnn.Conv2d, previous_shape: TensorShape):
    if isinstance(module.padding, str):
        raise NotImplementedError("Padding is string")
    new_height = _compute_conv(
        previous_shape.height,
        module.padding[0],
        module.kernel_size[0],
        module.stride[0],
    )
    new_width = _compute_conv(
        previous_shape.width, module.padding[1], module.kernel_size[1], module.stride[1]
    )
    return TensorShape(new_height, new_width, module.out_channels)


@_compute_shape.register
def _(module: tnn.ReLU | tnn.Dropout | tnn.BatchNorm2d, previous_shape: TensorShape):  # pyright: ignore[reportUnusedParameter]
    return previous_shape


@_compute_shape.register
def _(module: tnn.MaxPool2d, previous_shape: TensorShape):
    padding_height, padding_width = _to_pair(module.padding)
    kernel_height, kernel_width = _to_pair(module.kernel_size)
    stride_height, stride_width = _to_pair(module.stride)

    new_height = _compute_conv(
        previous_shape.height, padding_height, kernel_height, stride_height
    )
    new_width = _compute_conv(
        previous_shape.width, padding_width, kernel_width, stride_width
    )
    return TensorShape(new_height, new_width, previous_shape.channels)


@_compute_shape.register
def _(
    module: tnn.AdaptiveAvgPool2d | tnn.AdaptiveMaxPool2d, previous_shape: TensorShape
):
    new_height, new_width = _to_pair(module.output_size)

    if new_height is None:
        new_height = previous_shape.height

    if new_width is None:
        new_width = previous_shape.width

    return TensorShape(
        new_height,
        new_width,
        previous_shape.channels,
    )


class Convolution:
    def __init__(self, modules: list[tnn.Module], input_shape: TensorShape):
        super().__init__()
        if any(isinstance(module, tnn.Linear) for module in modules):
            raise ValueError("Linear layers cannot be in convolutions")

        self._modules: list[tnn.Module] = modules
        self._out_shape: TensorShape = TensorShape(0, 0, 0)

        head = self._modules[0]

        if isinstance(head, tnn.Conv2d):
            self._modules[0] = tnn.Conv2d(
                input_shape.channels,
                head.out_channels,
                head.kernel_size,  # pyright: ignore[reportArgumentType]
                head.stride,  # pyright: ignore[reportArgumentType]
                head.padding,  # pyright: ignore[reportArgumentType]
            )

        previous_shape = input_shape
        for i in range(len(modules) - 1):
            new_shape = _compute_shape(self._modules[i], previous_shape)

            if isinstance((module := self._modules[i + 1]), tnn.Conv2d):
                self._modules[i + 1] = tnn.Conv2d(
                    new_shape.channels,
                    module.out_channels,
                    module.kernel_size,  # pyright: ignore[reportArgumentType]
                    module.stride,  # pyright: ignore[reportArgumentType]
                    self._modules[i + 1].padding,  # pyright: ignore[reportArgumentType]
                )

            previous_shape = new_shape

        self._out_shape = previous_shape

    def append(self, module: tnn.Module) -> None:
        if isinstance(module, tnn.Linear):
            raise ValueError("Linear layers cannot be in convolutions")

        if isinstance(module, tnn.Conv2d):
            tail = tnn.Conv2d(
                self._out_shape.channels,
                module.out_channels,
                module.kernel_size,  # pyright: ignore[reportArgumentType]
                module.stride,  # pyright: ignore[reportArgumentType]
                module.padding,  # pyright: ignore[reportArgumentType]
            )
        else:
            tail = module

        self._modules.append(tail)
        self._out_shape = _compute_shape(module, self._out_shape)

    @singledispatch
    def extend(self, arg: Iterable[tnn.Module] | Self) -> None:
        if isinstance(arg, "Convolution"):
            for module in arg._modules:
                self.append(module)
        else:
            raise NotImplementedError(
                f"Unsupported argument type: {type(arg).__name__}"
            )

    @extend.register(Iterable)
    def _(self, modules: Iterable[tnn.Module]) -> None:
        if any(isinstance(module, tnn.Linear) for module in modules):
            raise ValueError("Linear layers cannot be in convolutions")

        for module in modules:
            self.append(module)

    @property
    def out_shape(self):
        return self._out_shape

    def sequential(self):
        return tnn.Sequential(*self._modules)
