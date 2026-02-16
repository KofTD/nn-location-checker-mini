from typing import Iterable

import torch.nn


class Segment:
    """This class should validate sequence of layers in a network"""

    def __init__(self, modules: Iterable[torch.nn.Module]):
        self._modules: list[torch.nn.Module] = modules

    def append(self, module: torch.nn.Module):
        self._modules.append(module)

    def extend(self, modules: Iterable[torch.nn.Module]):
        self._modules.extend(modules)

    def sequential(self):
        return torch.nn.Sequential(*self._modules)

    @property
    def modules(self):
        return self._modules
