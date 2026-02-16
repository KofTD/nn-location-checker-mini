from abc import ABC, abstractmethod

from src.segment import Segment


class BaseLoader(ABC):
    """Interface for segment loaders from any markup format"""

    @abstractmethod
    def load(self) -> Segment:
        raise NotImplementedError
