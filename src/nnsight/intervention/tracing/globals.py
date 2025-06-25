from typing import Any, Callable, Union

import torch
from typing_extensions import Self
from ... import deprecated
from ..._c.py_mount import mount, unmount


class Object(torch.Tensor):

    def save(self, _=0):
        """
        Save an object to be accessable after the trace context is exited.
        
        Example:
        
        >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
        >>> with model.trace("Hello World"):
        >>>     attn_0 = model.transformer.h[0].attn.output.save()
        >>> print(attn_0)
        """

        Globals.saves.add(id(self))

        return self
    
    @deprecated(message="Use `tracer.stop()` instead.")
    def stop(self, _=0):
        """
        Stop the trace context.
        """
        
        from ..interleaver import EarlyStopException

        raise EarlyStopException()
        
    
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
            mount(Object.stop, "stop")
        Globals.stack += 1

    @staticmethod
    def exit():
        Globals.stack -= 1
        if Globals.stack == 0:
            unmount("save")
            unmount("stop")