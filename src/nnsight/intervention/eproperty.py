"""A descriptor for a value the interleaver serves during a trace.

An `eproperty` turns a plain attribute — ``model.h[0].output``,
``tracer.result``, a source op's ``.input`` — into a handoff point in the run: reading
it parks the worker until the model reaches that location and returns the value there;
writing it swaps the worker's value in. The location is ``"{obj.path}.{key}"`` (or
just ``key`` when the host has no ``path``, as for ``tracer.result``).

The decorated stub *is* the **preprocess**: it takes the raw value the interleaver
served and returns what the user reads, so an identity view is just
``def output(self, value): return value``. Two more callbacks refine it:

* [`postprocess`][nnsight.intervention.eproperty.eproperty.postprocess] — runs on a written value before it's swapped in
  (e.g. repack a lone ``input`` back into the ``(args, kwargs)`` the model wants).
* [`transform`][nnsight.intervention.eproperty.eproperty.transform] — the write-back half of `preprocess`. When
  preprocess hands back a reshaped/sliced *view*, the user's in-place edits to it
  are invisible to the model (which still holds the original). A transform maps the
  edited view back to the model's layout; it fires once, after the block is done
  with that read, and the result is spliced in as if swapped. It receives the
  edited view and the raw served value: ``transform(self, view, raw)``.

    class Heads(Envoy):
        @eproperty(key="output")
        def heads(self, value):                     # preprocess: [B,S,H] -> heads
            b, s, h = value.shape
            return value.view(b, s, self.n_heads, h // self.n_heads).transpose(1, 2)

        @heads.transform
        def heads(self, value, raw):                # write the edited heads back
            b, nh, s, hd = value.shape
            return value.transpose(1, 2).reshape(b, s, nh * hd)

    with model.trace(prompt):
        model.attn.heads[:, 5] = 0                  # zero head 5; transform swaps it back

Both callbacks work in the shape of the **location**, which is not always a bare
tensor. ``key="output"`` on a tuple-returning module serves — and must be handed
back — the whole tuple. ``key="input"`` always serves the raw ``(args, kwargs)``
pair, the same value ``.inputs`` gives you, so the example above written against
``c_proj``'s input destructures on the way in and repacks on the way out::

    @eproperty(key="input")
    def heads(self, value):
        (x,), _ = value
        ...

    @heads.transform
    def heads(self, value, raw):
        ...
        return ((flat,), {})

``raw`` is the value as served, before preprocess. A transform that hands
back a *copy* of one element of a container — a block that returns
``(hidden_states, present)``, read as a clone of ``hidden_states`` so a saved
read cannot be mutated later — needs it to rebuild the container around the
edited copy: the other elements are not in the view, and a transform cannot
read the location itself, since it fires on the model side, after the read,
where nothing is parked there::

    @eproperty(key="output")
    def hidden(self, value):
        return value[0].clone()

    @hidden.transform
    def hidden(self, edited, raw):
        return (edited.clone(), *raw[1:])

One failure mode is worth knowing before you meet it: an `eproperty` is a
``property``, and a ``property`` getter that raises `AttributeError` falls through
to ``__getattr__``. A preprocess that raises one — a typo, a wrong unpack — is
swallowed and resurfaces as ``'Heads' object (nor its module) has attribute
'heads'``, blaming the attribute rather than the line inside it.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .interleaver import Interleaver, Mediator


@runtime_checkable
class IEnvoy(Protocol):
    """Interface for objects that host `eproperty` descriptors.

    An eproperty reads and writes its value through the interleaver at a location
    derived from the host, so a host must provide:

    Attributes:
        interleaver: The [`Interleaver`][nnsight.intervention.interleaver.Interleaver]
            managing execution flow (used by [`eproperty.provide`][nnsight.intervention.eproperty.eproperty.provide] to serve a
            value from the model side).
        path: Optional location prefix used to build the eproperty's location
            (``"{path}.{key}"``). May be ``None`` / empty — `eproperty._location`
            then falls back to the key alone. This is how tracer-level eproperties
            such as [`InterleavingTracer.result`][nnsight.intervention.tracer.InterleavingTracer.result] work without a path prefix.

    Notes:
        Hosts with no meaningful path (e.g. tracers) need not declare ``path`` at
        all — `eproperty._location` uses ``getattr(obj, "path", "")``, so a
        missing attribute is treated the same as ``None`` / ``""``. It is declared
        ``Optional[str]`` here only for type clarity.
    """

    interleaver: Interleaver
    path: Optional[str]


class eproperty(property):
    """A served value on an interleaving host (Envoy, SourceEnvoy, tracer, ...).

    Define one by decorating a stub with ``@eproperty`` (or ``@eproperty(key=...)``);
    the stub is the `preprocess`. The host only needs a ``path`` attribute (and
    an ``interleaver`` for [`provide`][nnsight.intervention.eproperty.eproperty.provide]); the value is read/written through
    [`Mediator`][nnsight.intervention.interleaver.Mediator], which raises outside a trace.

    Args:
        key: The location suffix appended to the host's ``path`` (``"{path}.{key}"``).
            Defaults to the stub's name. Several eproperties may share a key to give
            different views of the same location (``input`` and ``inputs``).
        description: A short label; only used to surface the attribute in a repr.
    """

    def __init__(
        self, key: Optional[str] = None, description: Optional[str] = None
    ) -> None:
        property.__init__(self)
        self.name: Optional[str] = None
        self.key: Optional[str] = None
        self.description = description
        #: the decorated stub — maps the served value to what the user reads
        self._preprocess: Optional[Callable] = None
        #: maps a written value to what's swapped in
        self._postprocess: Optional[Callable] = None
        #: maps an edited preprocess view back to the model's layout (write-back)
        self._transform: Optional[Callable] = None
        # Bare `@eproperty` (no parens): `key` is really the decorated function.
        if callable(key):
            self(key)
        else:
            self.key = key

    def __call__(self, preprocess: Callable) -> "eproperty":
        """Register the decorated stub as the preprocess and adopt its name/doc."""
        self.name = preprocess.__name__
        self._preprocess = preprocess
        self.__doc__ = preprocess.__doc__
        if self.key is None:
            self.key = self.name
        return self

    def postprocess(self, func: Callable) -> "eproperty":
        """Register the write-side callback, run on a value before it's swapped in."""
        self._postprocess = func
        return self

    def transform(self, func: Callable) -> "eproperty":
        """Register the write-back for an edited preprocess view (see class doc).

        ``func(self, view, raw)``: the edited view, and the value as served
        before preprocess, for a view that is one element of a container the
        write-back has to rebuild.
        """
        self._transform = func
        return self

    def __str__(self) -> str:
        """The line an envoy's repr shows for this value: ``(name) -> ReturnType: description``.

        The return type is the preprocess stub's return annotation when it has
        one, as written (a ``from __future__ import annotations`` module gives
        the alias name, ``Residual``; a class gives its ``__name__``); an
        unannotated or ``Any`` stub shows no arrow.
        """
        hint = getattr(self._preprocess, "__annotations__", {}).get("return")
        if hint is not None and not isinstance(hint, str):
            hint = getattr(hint, "__name__", repr(hint))
        typed = f" -> {hint}" if hint and hint != "Any" else ""
        return f"({self.name}){typed}: {self.description}"

    def _location(self, obj: IEnvoy) -> str:
        path = getattr(obj, "path", "")
        return f"{path}.{self.key}" if path else self.key

    def __get__(self, obj: Optional[IEnvoy], owner: Any = None) -> Any:
        if obj is None:
            return self
        location = self._location(obj)
        raw = Mediator.value(location)
        value = self._preprocess(obj, raw) if self._preprocess is not None else raw
        if self._transform is not None:
            # Bind the (about-to-be-returned) view into the write-back now, so the
            # user's in-place edits are visible when the mediator fires it after
            # this read — see Mediator.handle. The raw served value rides along:
            # this is the only chance to hand it over, since the transform fires
            # where the location can't be read.
            Mediator.current(location).transform = partial(self._transform, obj, value, raw)
        return value

    def __set__(self, obj: IEnvoy, value: Any) -> None:
        if self._postprocess is not None:
            value = self._postprocess(obj, value)
        Mediator.swap(self._location(obj), value)

    def provide(self, obj: IEnvoy, value: Any) -> Any:
        """Serve this eproperty's value into the run from the model side.

        The counterpart to a worker reading it: hands ``value`` to the interleaver at
        this location so a worker parked there is resumed with it. Used for values
        the model produces outside any module — e.g. a driver feeding
        ``tracer.result``.
        """
        return obj.interleaver.handle(self._location(obj), value)
