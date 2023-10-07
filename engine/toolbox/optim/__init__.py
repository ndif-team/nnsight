

from abc import abstractmethod
from typing import Any


class Optimization:

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def __call__(self) -> Any:
        pass