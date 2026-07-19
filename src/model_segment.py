"""Model registry and segmentation for feature extraction.

This module provides `SupportedModels`, an enum of torchvision
models, and `ModelSegment`, which slices a donor into a reusable
`torch.nn.Module` separating convolutional and classifier layers.
"""

from collections.abc import Iterable
from typing import cast, override

import torch
import torch.nn as tnn
from timm.models import FastVit, VisionTransformer
from torchvision.models.googlenet import InceptionAux as GoogLeNetAux
from torchvision.models.inception import InceptionAux

from tensor_shape import TensorShape, compute_shape


class ViTPatch(tnn.Module):
    def __init__(self, conv_proj: tnn.Conv2d, class_token: tnn.Parameter) -> None:
        super().__init__()
        self.conv_proj = conv_proj
        self.class_token = class_token

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x = self.conv_proj(x)
        x = x.flatten(2).permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1)
        return torch.cat([batch_class_token, x], dim=1)


@compute_shape.register
def _(module: ViTPatch, previous_shape: TensorShape) -> TensorShape:
    patch_shape = compute_shape(module.conv_proj, previous_shape)
    seq_length = patch_shape.height * patch_shape.width + 1
    return TensorShape(1, seq_length, patch_shape.channels)


class ModelSegment(tnn.Module):
    def __init__(
        self,
        modules: list[tnn.Module],
        index: int | slice,
        donor: str,
        class_token: tnn.Parameter | None = None,
    ) -> None:
        super().__init__()

        as_slice = slice(index) if isinstance(index, int) else index
        selected_modules = modules[as_slice]

        if donor.startswith("vit_"):
            if class_token is None:
                raise ValueError(f"class_token is required for '{donor}'")
            selected_modules = [
                ViTPatch(module, class_token)
                if isinstance(module, tnn.Conv2d)
                else module
                for module in selected_modules
            ]

        self._convolution_layers: tnn.Sequential = tnn.Sequential()
        self._classifier_layers: tnn.Sequential = tnn.Sequential()
        self._donor = donor

        for module in selected_modules:
            self.append(module)

    def compute_shape(self, input_shape: TensorShape) -> TensorShape | int:
        result_shape = input_shape
        for module in self.get_modules():
            if isinstance(module, tnn.Linear):
                result_shape = module.out_features
            elif (
                isinstance(module, tnn.Sequential)
                and len(list(module.children())) > 0
                and isinstance(module[-1], tnn.Linear)
            ):
                result_shape = module[-1].out_features
            elif isinstance(module, FastVit):
                result_shape = module.head.fc.out_features
            elif isinstance(module, VisionTransformer):
                result_shape = cast(int, module.head.out_features)
            else:
                result_shape = compute_shape(module, result_shape)

        return result_shape

    def extend(self, modules: Iterable[tnn.Module]) -> None:
        for module in modules:
            self.append(module)

    def append(self, module: tnn.Module) -> None:
        if isinstance(module, (InceptionAux, GoogLeNetAux)):
            return
        if isinstance(module, tnn.Sequential):
            if any(isinstance(submodule, tnn.Linear) for submodule in module):
                self._classifier_layers.append(module)
            else:
                self._convolution_layers.append(module)
        elif isinstance(module, tnn.Linear):
            self._classifier_layers.append(module)
        else:
            self._convolution_layers.append(
                module
            )  # For the future: _DenseBlock & _Transition come here too

    def get_modules(self) -> tnn.Sequential:
        return self._convolution_layers + self._classifier_layers

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self._convolution_layers) == 0:
            x = self._classifier_layers(x)
        elif len(self._classifier_layers) == 0:
            x = self._convolution_layers(x)
        else:
            x = self._convolution_layers(x)
            if self._donor.startswith("densenet"):
                x = torch.nn.functional.relu(x, inplace=True)
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            elif self._donor.startswith("mnasnet"):
                x = x.mean([2, 3])
                return self._classifier_layers(x)
            elif self._donor == "mobilenet_v2":
                x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            elif self._donor.startswith("vit_"):
                x = x[:, 0]
            x = torch.flatten(x, 1)
            x = self._classifier_layers(x)

        return x
