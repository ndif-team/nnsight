"""Record module activations during a trace with ``tracer.cache()``.

A cache observes the values flowing past the interleaver and keeps the ones for
the modules you asked for. Unlike reading a single location with ``.output``, a
cache captures *every* selected module across the whole run — every layer, and
(in a generation loop) every step::

    with model.trace(prompt) as tracer:
        cache = tracer.cache()                       # every module's output
    cache["model.transformer.h.0"].output            # by path
    cache.transformer.h[0].output                    # or by navigation

Because the interleaver already funnels every module input/output through
[`handle`][nnsight.intervention.interleaver.Interleaver.handle] (applying
interventions first), a cache is just a post-intervention observer: it needs no
per-module hooks. Navigation and alias/index resolution are delegated to the
model's [`Envoy`][nnsight.intervention.envoy.Envoy] tree, so a [`CacheView`][nnsight.intervention.cache.CacheView]
stays thin and aliases / ``ModuleList`` indexing work for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..util import apply

if TYPE_CHECKING:
    from .envoy import Envoy


@dataclass
class Entry:
    """One visit to a module: its output and/or its inputs.

    A module visited more than once in a run (e.g. a generation loop) produces
    one [`Entry`][nnsight.intervention.cache.Entry] per visit; see [`CacheView`][nnsight.intervention.cache.CacheView].
    """

    output: Any = None
    inputs: tuple[tuple[Any, ...], dict[str, Any]] | None = None

    @property
    def input(self) -> Any:
        """The first input — the first positional argument, else the first keyword."""
        if self.inputs is None:
            return None
        args, kwargs = self.inputs
        return (list(args) + list(kwargs.values()))[0]


class Cache:
    """Records selected modules' activations as the run reaches them.

    Created by [`cache`][nnsight.intervention.tracer.InterleavingTracer.cache] and
    registered on the calling mediator, so [`observe`][nnsight.intervention.cache.Cache.observe] is fed every location
    the run reaches (post-intervention). Values for the selected module paths are
    stored in [`entries`][nnsight.intervention.cache.Cache.entries] — one list of [`Entry`][nnsight.intervention.cache.Entry] per module path, an
    entry appended per visit.

    Attributes:
        model: The root envoy, used to resolve paths / aliases for [`CacheView`][nnsight.intervention.cache.CacheView].
        targets: The module paths to keep, or ``None`` to keep every module.
        entries: Recorded values, ``{module_path: [Entry, ...]}``.
    """

    def __init__(
        self,
        model: Envoy,
        modules: list[Envoy | str] | None = None,
        device: torch.device | None = torch.device("cpu"),
        dtype: torch.dtype | None = None,
        detach: bool = True,
        include_output: bool = True,
        include_inputs: bool = False,
        non_blocking: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.detach = detach
        self.non_blocking = non_blocking
        self.include_output = include_output
        self.include_inputs = include_inputs
        # None => every module; else the exact set of paths to keep.
        self.targets: set[str] | None = (
            None
            if modules is None
            else {m if isinstance(m, str) else m.path for m in modules}
        )
        self.entries: dict[str, list[Entry]] = {}

    def __getstate__(self) -> dict:
        # `model` is a live Envoy used only for CacheView navigation, and it drags
        # in the interleaver's unpicklable instrumented forwards. Drop it so the
        # recorded data (entries) can be serialized — e.g. torch.save-d as a remote
        # trace's result. Access by path string still works without it (see
        # CacheView); tree navigation needs a model re-attached.
        return {**self.__dict__, "model": None}

    def _select(self, location: str) -> "tuple[str, str] | None":
        """``(module_path, slot)`` if ``location`` is one this cache keeps, else None.

        ``location`` is ``"{module_path}.output"`` or ``"{module_path}.input"``.
        Anything else (source ops, skip gates, the run ``result``) has a base that
        isn't a module path, so it's filtered out.
        """
        if location.endswith(".output"):
            path, key = location[: -len(".output")], "output"
            if not self.include_output:
                return None
        elif location.endswith(".input"):
            path, key = location[: -len(".input")], "inputs"
            if not self.include_inputs:
                return None
        else:
            return None

        if self.targets is not None and path not in self.targets:
            return None

        return path, key

    def wants(self, location: str) -> bool:
        """Whether this cache would record ``location``.

        The same question [`observe`][nnsight.intervention.cache.Cache.observe]
        answers by recording, asked without recording — for a caller that has to
        do work *before* the value can be offered, and only wants to do it for
        locations some cache actually keeps (see
        [`TPFragments`][nnsight.modeling.tp.fragments.TPFragments], which
        must run a collective first). Shares `_select` with ``observe`` so the two
        can't drift.
        """
        return self._select(location) is not None

    def observe(self, location: str, value: Any) -> None:
        """Record ``value`` if ``location`` is a selected module's input/output."""
        selected = self._select(location)
        if selected is None:
            return

        path, key = selected
        self._record(path, key, self._transform(value))

    def _transform(self, value: Any) -> Any:
        # Detach and move off the compute device so the cache doesn't pin the
        # graph or GPU memory. Applied to every tensor in the value.
        if self.detach:
            value = apply(value, lambda t: t.detach(), torch.Tensor)
        if self.device is not None or self.dtype is not None:
            value = apply(
                value,
                lambda t: t.to(
                    device=self.device, dtype=self.dtype, non_blocking=self.non_blocking
                ),
                torch.Tensor,
            )
        return value

    def _record(self, path: str, key: str, value: Any) -> None:
        # Pair the input and output of one visit into a single Entry: the input
        # pre-hook fires before the output post-hook, so fill the open slot on the
        # current entry, or start a new entry (a new visit) when it's taken.
        entries = self.entries.setdefault(path, [])
        if entries and getattr(entries[-1], key) is None:
            setattr(entries[-1], key, value)
        else:
            entries.append(Entry(**{key: value}))


class CacheView:
    """Path- and attribute-addressable view over a [`Cache`][nnsight.intervention.cache.Cache]'s entries.

    The object returned by ``tracer.cache()``. Read a module's captured value
    with ``.output`` / ``.inputs`` / ``.input`` after selecting it, either by path
    (``cache["model.transformer.h.0"]``) or by navigating the tree
    (``cache.model.transformer.h[0]`` — or the short ``cache.transformer.h[0]``).
    Navigation is resolved against the model's envoy tree, so renamed modules and
    ``ModuleList`` indices resolve the same way they do on the model
    (``cache.model.transformer.h["second_layer"]`` works when ``1`` is renamed
    ``second_layer``). When a module was visited multiple times (a generation
    loop), ``len(view)`` is the visit count and ``.output`` returns the list.

    The object handed back is a *virtual root* above the model: its child is the
    model itself, reached by the model's name (``cache.model``), mirroring the
    full paths used as keys. ``_envoy`` is ``None`` there and a real envoy at every
    node below.
    """

    def __init__(
        self, cache: Cache, envoy: Envoy | None, path: str | None = None
    ) -> None:
        # Set through the instance dict directly; __getattr__ handles misses.
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_envoy", envoy)
        # An explicit module path, used to address entries when there's no envoy to
        # read `.path` from — e.g. a deserialized cache (its model was dropped),
        # where access is by path string rather than tree navigation.
        object.__setattr__(self, "_path", path)

    def _node_path(self) -> str | None:
        """This node's module path — from the explicit `_path`, else its envoy."""
        if self._path is not None:
            return self._path
        return self._envoy.path if self._envoy is not None else None

    def _entries(self) -> list[Entry]:
        path = self._node_path()
        if path is None:
            raise KeyError("the cache root is not a module")
        if path not in self._cache.entries:
            raise KeyError(f"{path!r} was not cached")
        return self._cache.entries[path]

    def _values(self, attr: str) -> Any:
        # One value per visit; unwrap the single-visit case for convenience.
        values = [getattr(entry, attr) for entry in self._entries()]
        return values[0] if len(values) == 1 else values

    @property
    def output(self) -> Any:
        return self._values("output")

    @property
    def inputs(self) -> Any:
        return self._values("inputs")

    @property
    def input(self) -> Any:
        return self._values("input")

    def keys(self) -> list[str]:
        """The cached module paths at or below this view's node (all, at the root)."""
        prefix = self._node_path()
        if prefix is None:
            return list(self._cache.entries)
        return [
            key
            for key in self._cache.entries
            if key == prefix or key.startswith(prefix + ".")
        ]

    def __getattr__(self, name: str) -> CacheView:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._navigate(self._child(name))

    def __getitem__(self, key: str | int) -> CacheView:
        if isinstance(key, int):
            return self._navigate(self._envoy[key])
        if "." in key:  # absolute dotted path, resolved from the virtual root
            # A path that names a cached module directly is addressable without an
            # envoy (a deserialized cache has no model to navigate); fall back to
            # tree navigation for partial paths / when a model is present.
            if key in self._cache.entries:
                return CacheView(self._cache, None, path=key)
            node: Any = None
            view = CacheView(self._cache, None)
            for part in key.split("."):
                node = view._child(part)
                view = CacheView(self._cache, node)
            return view
        return self._navigate(self._child(key))  # single component, relative

    def __contains__(self, key: str) -> bool:
        try:
            node = self[key]
        except (AttributeError, IndexError, KeyError):
            return False
        path = node._node_path()
        return path is not None and path in self._cache.entries

    def __len__(self) -> int:
        return len(self._entries())

    def __repr__(self) -> str:
        path = self._node_path()
        at = "root" if path is None else repr(path)
        return f"CacheView({at}, cached={self.keys()})"

    def _child(self, name: str) -> Any:
        """Resolve one path component to a child envoy, alias- and index-aware.

        At the virtual root the model's own name descends to the model; elsewhere
        a numeric component indexes a ``ModuleList`` and any other name is an
        attribute lookup on the current envoy (which resolves renames).
        """
        if self._envoy is None:
            model = self._cache.model
            if name == model.path:
                return model
            return getattr(model, name)  # short form: skip the leading model name
        if name.lstrip("-").isdigit():
            return self._envoy[int(name)]
        return getattr(self._envoy, name)

    def _navigate(self, node: Any) -> CacheView:
        from .envoy import Envoy

        if not isinstance(node, Envoy):
            raise AttributeError(f"{node!r} is not a module")
        return CacheView(self._cache, node)
