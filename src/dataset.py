from enum import Enum
from os import PathLike
from pathlib import Path


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
        path_to_dataset: str | PathLike[str] | Path,
    ):
        self._dataset_path = Path(path_to_dataset)

        directories = list(self._dataset_path.iterdir())
        directories.sort()

        self._pool, self._category_bounds = self._load_pool(directories)

    @staticmethod
    def _load_pool(directories: list[Path]):
        pool: list[tuple[Path, Marker]] = []
        category_bounds: list[tuple[int, int]] = []

        previous_bound_end = 0

        for directory in directories:
            marker_number = int(directory.name[: directory.name.find("_")])
            marker = Marker(marker_number)

            for photo in directory.iterdir():
                pool.append((photo, marker))

            category_bounds.append((previous_bound_end, len(pool)))
            previous_bound_end = len(pool)
        return (pool, category_bounds)

    @property
    def pool(self):
        return self._pool

    @property
    def category_bounds(self):
        return self._category_bounds
