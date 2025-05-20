from typing import Any

import torch
from typing_extensions import Self

from ..._c.py_mount import mount, unmount


class Object(torch.Tensor):

    def save(obj: Any) -> Self:

        Globals.saves.add(id(obj))

        return obj


class Globals:

    stack = 0

    saves = set()

    @staticmethod
    def enter():
        if Globals.stack == 0:
            mount(Object.save, "save")
        Globals.stack += 1

    @staticmethod
    def exit():
        Globals.stack -= 1
        if Globals.stack == 0:
            unmount("save")
