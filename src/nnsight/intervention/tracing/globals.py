from typing import Any, Callable, Tuple, Union

import torch
from typing_extensions import Self
from ..._c.py_mount import mount
from ... import CONFIG


def save(object: Any):

    Globals.saves.add(id(object))

    return object


class Object(torch.Tensor):

    def save(self, _=0):
        """
        Save an object to be accessable after the trace context is exited.

        Examples:

        >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
        >>> with model.trace("Hello World"):
        ...     attn_0 = model.transformer.h[0].attn.output.save()
        >>> print(attn_0)
        """

        if Globals.stack == 0:
            raise RuntimeError(
                ".save() called outside of a trace context. "
                "Use .save() only inside a `with model.trace(...)` block."
            )

        save(self)

        return self

    def __getattr__(self, name: str) -> Self:

        return super().__getattr__(name)

    def __getitem__(self, key: Any) -> Self:

        return super().__getitem__(key)

    def __call__(self, *args: Any, **kwargs: Any) -> Self:

        return super().__call__(*args, **kwargs)


class TracingCache:

    def __init__(self):
        self.cache = {}
        self.code_cache = {}

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

    def get_code(self, cache_key):
        """
        Get a cached compiled code object.
        """
        return self.code_cache.get(cache_key, None)

    def add_code(self, cache_key, code_obj):
        """
        Cache a compiled code object.
        """
        self.code_cache[cache_key] = code_obj

    def clear(self):
        """
        Clear all cached source, AST, and code objects.
        """
        self.cache.clear()
        self.code_cache.clear()


class Globals:

    stack = 0

    saves = set()

    cache = TracingCache()

    _mounted = False

    @staticmethod
    def enter():

        if CONFIG.APP.PYMOUNT and not Globals._mounted:
            mount(Object.save, "save")
            Globals._mounted = True
        Globals.stack += 1

    @staticmethod
    def exit():
        Globals.stack -= 1

    @staticmethod
    def clear():
        Globals.saves.clear()
        Globals.cache.clear()
        Globals.stack = 0
