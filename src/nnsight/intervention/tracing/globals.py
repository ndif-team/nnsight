from typing import Any, Callable, Union

import torch
from typing_extensions import Self

from ..._c.py_mount import mount, unmount


class Object(torch.Tensor):

    def save(obj: Any) -> Self:

        Globals.saves.add(id(obj))

        return obj
    
    def __getattr__(self, name: str) -> Self:

        return super().__getattr__(name)
    
    def __getitem__(self, key: Any) -> Self:
        
        return super().__getitem__(key)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Self:
        
        return super().__call__(*args, **kwargs)


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
