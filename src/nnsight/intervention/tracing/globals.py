from typing import Any, Callable, Tuple, Union

import torch
from typing_extensions import Self
from ... import deprecated
from ..._c.py_mount import mount, unmount
from ... import CONFIG


def save(object: Any):

    Globals.saves.add(id(object))

    return object


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

        save(self)

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


class TracingCache:

    def __init__(self):
        self.cache = {}

    def get(self, cache_key: Tuple):
        """
        Check if the given filename and lineno is in the cache.
        """
        return self.cache.get(cache_key, None)

    def add(self, cache_key: Tuple, value: Any):
        """
        Add the given value to the cache.
        """
        self.cache[cache_key] = value


class Globals:

    stack = 0

    saves = set()

    cache = TracingCache()

    @staticmethod
    def enter():
        if CONFIG.APP.PYMOUNT and Globals.stack == 0:
            mount(Object.save, "save")
            mount(Object.stop, "stop")
        Globals.stack += 1

    @staticmethod
    def exit():
        Globals.stack -= 1
        if CONFIG.APP.PYMOUNT and Globals.stack == 0:
            unmount("save")
            unmount("stop")
