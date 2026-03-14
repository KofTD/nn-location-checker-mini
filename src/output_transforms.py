import time

import torch
from torch.utils.data import DataLoader

from classification_network import ClassificationNetwork


def get_accuracy(
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    model: ClassificationNetwork,
    device: torch.device,
) -> float:
    tp = 0
    n = 0
    with torch.no_grad():
        for images, labels in data_loader:  # pyright: ignore[reportAny]
            images = images.requires_grad_().to(device)  # pyright: ignore[reportAny]
            labels = labels.to(device)  # pyright: ignore[reportAny]
            outputs = model(images)  # pyright: ignore[reportAny]
            _, predicted = torch.max(outputs.data, 1)  # pyright: ignore[reportAny]
            n += labels.size(0)  # pyright: ignore[reportAny]
            tp += (predicted == labels).sum()  # pyright: ignore[reportAny]
    return float(tp / n)


def get_avg_time_per_image(
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    model: ClassificationNetwork,
    device: torch.device,
) -> float:
    total_time = 0.0
    n = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.requires_grad_().to(device)
            labels = labels.to(device)
            batch_start_time = time.time()
            _ = model(images)  # pyright: ignore[reportAny]
            batch_time = time.time() - batch_start_time
            total_time += batch_time
            n += labels.size(0)
    return float(total_time / n)
