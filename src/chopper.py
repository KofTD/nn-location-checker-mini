import torch.nn
import torchvision.models as models


class Chopper:
    def __init__(self, nn: torch.nn.Module):
        if isinstance(nn, models.AlexNet):
            self._modules = list(nn.children())
        else:
            raise ValueError("Unsupported neural network")

    def head_chop(self, stop: int):
        if stop > len(self._modules):
            raise ValueError("Incorrect stop")
        return self._modules[:stop]

    def chop(self, start: int, stop: int):
        if start > stop or start > len(self._modules):
            raise ValueError("Incorrect start")
        if stop > len(self._modules):
            raise ValueError("Incorrect stop")

        return self._modules[start:stop]

    def tail_chop(self, start: int):
        if start > len(self._modules):
            raise ValueError("Incorrect start")

        return self._modules[start:]
