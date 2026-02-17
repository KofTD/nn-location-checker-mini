from math import ceil
from pathlib import Path
from random import sample, shuffle

import cv2
import torch
import torchvision

from src.dataset import Dataset, Marker


class Data_loader:
    transform = torchvision.transforms.ToTensor()

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle_required = shuffle

    def __iter__(self):
        self._data_queue: list[tuple[Path, Marker]] = []

        markers_quantity = len(Marker)

        objects_per_category = int(
            ceil(self._batch_size / markers_quantity)
        )  # I hope the batch_size is at least the number of objects

        if self._shuffle_required or len(self._data_queue) == 0:
            self._data_queue = self._form_queue(
                self._dataset, objects_per_category
            )  # It's probably not random enough for our purpose
            shuffle(self._data_queue)

        self._queue_pos = 0

        return self

    @staticmethod
    def _form_queue(
        dataset: Dataset, objects_per_category: int
    ) -> list[tuple[Path, Marker]]:
        data_queue: list[tuple[Path, Marker]] = []

        for start_category, end_category in dataset.category_bounds:
            category_elements: list[tuple[Path, Marker]] = []

            if end_category - start_category < objects_per_category:
                category_paths = dataset.pool[start_category:end_category]
            else:
                category_paths = sample(dataset.pool, objects_per_category)

            data_queue.extend(category_paths)

        return data_queue

    def _load_to_tensor(self, photo: Path):
        image = cv2.imread(photo)
        return self.transform(image)

    def __next__(self):
        if self._queue_pos >= self._batch_size:
            raise StopIteration

        photo, marker = self._data_queue[self._queue_pos]
        self._queue_pos += 1

        return (self._load_to_tensor(photo), torch.tensor([marker.value]))

    @property
    def dataset(self):
        return self._dataset
