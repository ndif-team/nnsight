from __future__ import annotations

from contextlib import AbstractContextManager


class Accumulator(AbstractContextManager):

    def __enter__(self) -> Accumulator:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass