import time
from datetime import timedelta
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader

from classification_network import ClassificationNetwork


class RunResult(NamedTuple):
    total: int
    true_positives: int
    false_negative: int
    total_time: timedelta


def _run_model(
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    model: ClassificationNetwork,
    device: torch.device,
) -> RunResult:
    n = 0
    tp = 0
    total_time = timedelta()
    with torch.no_grad():
        for images, labels in data_loader:  # pyright: ignore[reportAny]
            images = images.requires_grad_().to(device)  # pyright: ignore[reportAny]
            labels = labels.to(device)  # pyright: ignore[reportAny]
            batch_start_time = timedelta(seconds=time.time())
            outputs = model(images)  # pyright: ignore[reportAny]
            batch_time = timedelta(seconds=time.time()) - batch_start_time
            total_time += batch_time
            _, predicted = torch.max(outputs.data, 1)  # pyright: ignore[reportAny]
            n += labels.size(0)  # pyright: ignore[reportAny]
            tp += (predicted == labels).sum()  # pyright: ignore[reportAny]

    return RunResult(n, tp, n - tp, total_time)


def get_accuracy(
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    model: ClassificationNetwork,
    device: torch.device,
) -> float:
    result = _run_model(data_loader, model, device)
    return result.true_positives / result.total


def get_avg_time_per_image(
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    model: ClassificationNetwork,
    device: torch.device,
) -> timedelta:
    results = _run_model(data_loader, model, device)

    return results.total_time / results.total
