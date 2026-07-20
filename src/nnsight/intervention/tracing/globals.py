from typing import Any, Tuple

import torch
from typing_extensions import Self
from ..._c.py_mount import mount
from ... import CONFIG


_mounted = False


def _ensure_mounted():
    """Mount Object.save / Object.carry as the universal `.save` / `.carry` methods.

    Lazy one-time setup, run at trace setup (``_setup_interleaver``) and from
    ``.save()`` / ``.carry()`` so we only pay the C-level mount cost once.
    """
    global _mounted
    if CONFIG.APP.PYMOUNT and not _mounted:
        mount(Object.save, "save")
        mount(Object.carry, "carry")
        _mounted = True


def save(object: Any):

    Globals.saves.add(id(object))

    return object


def carry(object: Any):
    """Mark ``object`` to be handed to a later trace in the same ``model.session()``
    WITHOUT surfacing it as a saved output.

    The portable counterpart to relying on an inner trace's locals flowing implicitly:
    in-process a non-saved value already crosses to the next trace, but under isolation
    each inner trace runs in a worker that only ships its outputs home, so a non-saved
    value would vanish. ``.carry()`` explicitly registers the value to cross the boundary
    — so the same code is correct in-process AND isolated. Unlike :func:`save`, a carried
    value is dropped at session exit (it is not in ``Globals.saves``), so it never appears
    in the caller's frame; use it for cross-trace handoffs (an activation to patch into a
    later run) that are not themselves results.
    """

    Globals.shared.add(id(object))

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

        save(self)

        return self

    def carry(self, _=0):
        """Hand this value to a later trace in the same ``model.session()`` without saving
        it as an output. See :func:`carry`.

        Examples:

        >>> with model.session():
        ...     with model.trace("clean prompt"):
        ...         act = model.transformer.h[6].output.carry()   # not a result
        ...     with model.trace("corrupt prompt") as tracer:
        ...         tracer.patch(model.transformer.h[6], act)     # transplanted in
        ...         logits = model.lm_head.output.save()
        """

        carry(self)

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
    """Process-wide tracing state.

    Holds these pieces of true global state:
    - ``saves``: set of ``id()`` for objects marked via ``.save()``.
      The root tracer's ``push()`` filters its frame locals against this
      set so only saved values propagate out of the trace.
    - ``shared``: set of ``id()`` for objects marked via ``.carry()`` —
      cross-trace handoffs within a ``model.session()`` that are NOT surfaced
      as outputs. Consulted by the isolated worker's ``end()`` to ship carried
      values across the boundary; dropped at session exit (not in ``saves``).
    - ``cache``: source/AST/code-object memoization across traces.

    Root-vs-inner detection lives on the tracer itself — see
    ``Tracer.push`` — by checking whether the target frame is an
    nnsight-generated frame (i.e., another trace's compiled body).
    """

    saves = set()

    shared = set()

    cache = TracingCache()

    @staticmethod
    def clear():
        Globals.saves.clear()
        Globals.shared.clear()
        Globals.cache.clear()
