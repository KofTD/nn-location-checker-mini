from enum import Enum
from os import PathLike
from pathlib import Path
from typing import overload

import cv2
import cv2.typing as cv2t


class Marker(Enum):
    OTHER = 0
    KREMLIN = 1
    CHKALOV_STAIRCASE = 2
    RUKAVISHNIKOV_ESTATE = 3
    ARHANGELSK_CATHEDRAL = 4
    PECHERSKY_MONASTERY = 5
    CHURCH_OF_THE_NATIVITY = 6
    STATE_BANK = 7
    PALACE_OF_LABOR = 8
    CATHEDRAL_MOSQUE = 9
    ALEXANDER_NEVSKY_CATHEDRAL = 10
    SPASSKY_OLD_FAIR_CATHEDRAL = 11
    FAIR = 12
    DRAMA_THEATER_GORKY = 13
    CHURCH_OF_THE_NATIVITY_WITH_THE_ROYAL_CHAPEL = 14


class Dataset:
    def __init__(
        self,
        images_directory: str | PathLike[str] | Path,
    ):
        self._images_directory: Path = Path(images_directory)

        directories = list(self._images_directory.iterdir())
        directories.sort()

        self._pool: list[tuple[Path, Marker]] = self._load_pool(directories)
        self._pool_idx: int = -1

    def __len__(self):
        return len(self._pool)

    @overload
    def __getitem__(self, idx: int) -> tuple[cv2t.MatLike, Marker]: ...

    @overload
    def __getitem__(self, idx: slice) -> list[tuple[cv2t.MatLike, Marker]]: ...

    def __getitem__(
        self, idx: int | slice
    ) -> tuple[cv2t.MatLike, Marker] | list[tuple[cv2t.MatLike, Marker]]:
        def load_image(image_path: Path):
            image = cv2.imread(str(image_path))

            if image is None:
                raise RuntimeError(f"Could not load an image: {image_path}")

            return image

        if isinstance(idx, int):
            image_path, label = self._pool[idx]

            image = load_image(image_path)

            return image, label
        else:
            pool_slice = self._pool[idx]

            result_slice: list[tuple[cv2.typing.MatLike, Marker]] = []
            for image_path, label in pool_slice:
                image = load_image(image_path)
                result_slice.append((image, label))

            return result_slice

    def __iter__(self):
        self._pool_idx = -1
        return self

    def __next__(self):
        self._pool_idx += 1

        return self[self._pool_idx]

    @staticmethod
    def _load_pool(
        directories: list[Path],
    ) -> list[tuple[Path, Marker]]:
        pool: list[tuple[Path, Marker]] = []

        for directory in directories:
            marker_number = int(directory.name[: directory.name.find("_")])
            marker = Marker(marker_number)

            for photo in directory.iterdir():
                pool.append((photo, marker))

        return pool

    @property
    def pool(self) -> list[tuple[Path, Marker]]:
        return self._pool
