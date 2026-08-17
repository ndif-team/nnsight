"""The [`Envoy`][nnsight.intervention.envoy.Envoy] — nnsight's window into a running PyTorch model.

An [`Envoy`][nnsight.intervention.envoy.Envoy] wraps a `torch.nn.Module` and mirrors its submodule
tree, so every module in the model has a matching envoy reachable by the same
attribute path (``model.transformer.h[0].mlp``). Envoys are the objects you
interact with when tracing: they expose each module's live ``input``/``output``
during a forward pass, let you overwrite those values, read gradients, skip
whole modules, and reach individual operations inside a forward via
[`source`][nnsight.intervention.envoy.Envoy.source].

You open a trace with ``with model.trace(x):`` and, inside the block, read or
write envoy attributes as if the forward pass had paused at each module for you.
Capture a value with ``.save()`` to use it after the trace:

.. code-block:: python

    from nnsight.intervention.envoy import Envoy

    model = Envoy(my_module)

    with model.trace(x):
        hidden = model.layer1.output.save()   # captured mid-forward
        model.layer2.output[:] = 0            # overwrite layer2's output in place

    print(hidden.shape)                        # available after the block

Gradients are available the same way. Call ``.backward()`` on a captured value
as a context manager and, inside it, read ``.grad`` on tensors you captured
earlier in the forward — you can edit gradients too:

.. code-block:: python

    with model.trace(x):
        a1 = model.fc1.output
        loss = model.output.sum()
        with loss.backward():
            g = a1.grad.save()     # gradient flowing into fc1's output
            a1.grad = a1.grad * 2  # and it can be edited in place of autograd's

Locations must be read in execution order: asking for an earlier module's output
after a later one has already run raises
[`OutOfOrderError`][nnsight.intervention.interleaver.OutOfOrderError].
"""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterator

import torch

from .. import deprecated
from ..util import apply

from ..tracing.hint import Object
from ..tracing.tracer import Tracer, WithBlockNotFoundError
from .interleaver import (
    EarlyStopException,
    Interleaver,
    Mediator,
)
from .batching import Batcher
from .editing import EditingTracer
from .eproperty import eproperty
from .source import Source
from .tracer import InterleavingTracer
from .util import first_input, replace_first_input


def traceable(method: Callable) -> Callable:
    """Make an Envoy method usable as a trace context.

    ``with envoy.method(...):`` traces the method (runs it interleaved with the
    block's interventions); ``envoy.method(...)`` just calls it. While already
    interleaving, it always just calls the method (we're inside a trace).
    """

    @functools.wraps(method)
    def wrapper(self: Envoy, *args: Any, **kwargs: Any) -> Any:
        fn = method.__get__(self, type(self))
        # Already inside a trace: just run the method — we're mid-interleave.
        if self.interleaver.interleaving:
            return method(self, *args, **kwargs)
        tracer = self.trace(*args, fn=fn, **kwargs)
        try:
            tracer.capture()
        except WithBlockNotFoundError:
            # Called directly (not as a `with` block): run it through interleave
            # so dispatch, device placement, and input prep still happen — same
            # path as trace(trace=False). Only edits apply (interleave adds them).
            return self.interleave(fn, *args, **kwargs)
        return tracer

    return wrapper


def _addindent(text: str, spaces: int) -> str:
    lines = text.split("\n")
    if len(lines) == 1:
        return text
    first = lines.pop(0)
    lines = [(spaces * " ") + line for line in lines]
    return first + "\n" + "\n".join(lines)


class Envoy:
    """Wraps a `torch.nn.Module` to expose and edit its values during a trace.

    One envoy mirrors one module and reads or overwrites that module's live
    ``input``/``output`` (and gradients) as the forward pass runs, driving those
    interventions through a shared [`Interleaver`][nnsight.intervention.interleaver.Interleaver].
    The child envoys mirror the module's submodule tree, so the whole model is
    reachable by attribute path from the root envoy. See the module docstring for
    the mental model.

    Attributes:
        path: The module's dotted location in the tree, e.g. ``"model.transformer.h.0"``.
            Every location the interleaver reads (``{path}.output``, ``{path}.skip``)
            is derived from it.
        interleaver: The [`Interleaver`][nnsight.intervention.interleaver.Interleaver]
            shared across the whole tree; it installs the hooks and routes values.
        _module: The wrapped `torch.nn.Module`.
        _edits: Default interventions registered by [`edit`][nnsight.intervention.envoy.Envoy.edit], replayed on every
            trace (a list of [`Mediator`][nnsight.intervention.interleaver.Mediator]).
        _children: The direct child envoys, in module order.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        path: str = "model",
        interleaver: Interleaver | None = None,
        rename: dict[str, str | list[str]] | None = None,
        envoys: dict | None = None,
    ) -> None:
        self._module = module
        self.path = path
        self.interleaver = interleaver if interleaver is not None else Interleaver()
        # instrument installs the input/output hooks and the source/skip controller
        # (registering this interleaver on the module) — see Interleaver.instrument.
        self.interleaver.instrument(self)

        # Default interventions registered via .edit(), replayed on every trace.
        self._edits: list[Mediator] = []

        # Module-name aliases (see `rename` / `_bind_aliases`). `_rename` is
        # the raw spec, inherited by children; `_aliases` maps each alias bound on
        # *this* envoy to the real path it resolved from (used by `__repr__`).
        self._rename = rename
        self._aliases: dict[str, str] = {}

        # Optional map choosing a custom Envoy subclass per child module (by module
        # type or path suffix); inherited by children so it applies all the way
        # down. See `_resolve_envoy_class`.
        self._envoys = envoys

        self._children: list[Envoy] = []

        for name, child in module.named_children():
            self._add_envoy(name, child)

        # Children exist now, so multi-component alias paths (e.g. "h.0") resolve.
        self._bind_aliases()

    def _wrap_envoy(self, name: str, module: torch.nn.Module) -> Envoy:
        # Mirror a module already on self._module as an envoy child. __dict__.get
        # (not getattr) so this is safe to call from __getattr__.
        existing = self.__dict__.get(name)
        if isinstance(existing, Envoy):
            self._children.remove(existing)
        child_path = f"{self.path}.{name}"
        envoy = self._resolve_envoy_class(module, child_path)(
            module,
            path=child_path,
            interleaver=self.interleaver,
            rename=self._rename,
            envoys=self._envoys,
        )
        self._children.append(envoy)
        # A submodule whose name shadows an Envoy attribute (e.g. BERT's `output`)
        # would otherwise be masked by that attribute — or trip its setter on the
        # object.__setattr__ below. Give the submodule the name and relocate the
        # nnsight attribute to `nns_<name>`.
        if not name.startswith("_") and hasattr(Envoy, name):
            self._mount_overloaded(name, envoy)
        else:
            object.__setattr__(self, name, envoy)
        return envoy

    def _mount_overloaded(self, name: str, envoy: Envoy) -> None:
        # Keep `name` for the submodule and move nnsight's attribute to `nns_name`.
        # The override lives on a per-instance subclass so only this envoy is
        # affected; all its siblings and the shared Envoy class are untouched.
        warnings.warn(
            f"Module '{self.path}' has a submodule named '{name}', which shadows "
            f"Envoy's '{name}'. The submodule keeps '.{name}'; nnsight's '{name}' "
            f"is available as '.nns_{name}' on this module."
        )

        cls = type(self)
        if not cls.__name__.endswith("__Overloaded"):
            cls = type(f"{cls.__name__}__Overloaded", (cls,), {})
            object.__setattr__(self, "__class__", cls)

        original = getattr(Envoy, name)
        setattr(cls, f"nns_{name}", original)

        # A property is a data descriptor (it wins over the instance dict), so a
        # plain stored child would still be masked — override `name` on the
        # subclass to hand back the stored child, keeping the original setter so
        # `envoy.name = value` still writes the intervention (for output/input).
        if isinstance(original, property):
            setattr(
                cls,
                name,
                property(lambda self, _n=name: self.__dict__[_n], original.fset, original.fdel),
            )
        # A method is not a data descriptor, so the stored child already wins;
        # nothing more to override.
        self.__dict__[name] = envoy

    def _bind_aliases(self) -> None:
        """Bind each ``rename`` alias as an attribute pointing at the same Envoy.

        For every ``path -> alias(es)`` entry, resolve ``path`` *relative to this
        envoy*; if it names a descendant envoy, bind each alias as an attribute on
        this envoy pointing at that same descendant object. Because every envoy in
        the tree runs this, a single-component path like ``"mlp"`` binds wherever
        it resolves (each block that has one), while a multi-component path like
        ``"transformer.h.3.mlp"`` binds only on the envoy it resolves from. A
        leading dot is a no-op — path components are matched by name (an empty
        first component is skipped, mirroring `nnsight.util.fetch_attr`).

        Aliases are ordinary attributes referencing the *same* child object (not
        copies, and not added to `_children`), so ``__getattr__`` needs no
        alias branch, iteration doesn't double-count, and re-pointing the tree on
        dispatch (`_update`, in place) keeps them valid with no rebuild.
        """
        if not self._rename:
            return
        for path, aliases in self._rename.items():
            path = path.lstrip(".")
            try:
                target = self.get(path)
            except AttributeError:
                continue
            if not isinstance(target, Envoy):
                continue
            for alias in [aliases] if isinstance(aliases, str) else aliases:
                object.__setattr__(self, alias, target)
                self._aliases[alias] = path

    def _add_envoy(self, name: str, module: torch.nn.Module) -> Envoy:
        # Register a (possibly new) module on self._module, then mirror it.
        self._module.add_module(name, module)
        return self._wrap_envoy(name, module)

    def _resolve_envoy_class(self, module: torch.nn.Module, path: str) -> type["Envoy"]:
        """The [`Envoy`][nnsight.intervention.envoy.Envoy] class to wrap ``module`` (at ``path``) with.

        Consults the `_envoys` map (``None`` -> the base [`Envoy`][nnsight.intervention.envoy.Envoy]): a
        single class wraps every child with it; a dict's keys are either a
        ``torch.nn.Module`` subclass (matched against the module's MRO, tried
        first) or a string dotted path-suffix (``"attn"``, ``"transformer.h"``).
        Falls back to the base [`Envoy`][nnsight.intervention.envoy.Envoy] when nothing matches — so a model can
        give, e.g., its attention modules a subclass exposing a ``.heads`` eproperty.
        """
        mapping = self._envoys
        if mapping is None:
            return Envoy
        if isinstance(mapping, type):
            return mapping
        for cls in type(module).__mro__:
            if cls in mapping:
                return mapping[cls]
        for key, envoy_cls in mapping.items():
            if isinstance(key, str) and self._path_ends_with(path, key):
                return envoy_cls
        return Envoy

    @staticmethod
    def _path_ends_with(path: str, key: str) -> bool:
        """Whether ``path`` ends with dotted ``key`` component-wise (not substring)."""
        parts = path.split(".")
        key_parts = key.removeprefix(".").split(".")
        return len(key_parts) <= len(parts) and parts[-len(key_parts):] == key_parts

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self) -> dict:
        # For serialization: tag the heavy/server-side objects as persistent so
        # they're referenced by id rather than serialized. The server resolves
        # them from its own model (Module:<path>) and interleaver (Interleaver).
        state = self.__dict__.copy()
        state["interleaver"]._persistent_id = "Interleaver"
        state["_module"]._persistent_id = f"Module:{self.path}"
        return state

    def _update(self, module: torch.nn.Module) -> None:
        # Re-point an existing envoy tree at a new module of the same structure
        # (e.g. swapping meta weights for real ones). instrument() removes this
        # path's old hooks before re-adding, and we recurse over children by
        # name (as in __init__), so modules shared across paths line up.
        self._module = module
        # instrument re-installs the hooks and the source/skip controller on the new
        # module (its own forward; the previous module's controller doesn't carry
        # over) — see Interleaver.instrument.
        self.interleaver.instrument(self)
        children = dict(module.named_children())
        for child in self._children:
            name = child.path.rsplit(".", 1)[-1]
            # A child that isn't a submodule of the new module — e.g. a standalone
            # module added to the tree (TransformersModel's `generator`) — has nothing
            # to re-point at, so leave it as-is (it keeps its own module and hooks).
            if name in children:
                child._update(children[name])

    def trace(
        self,
        *args: Any,
        fn: Any = None,
        backend: Any = None,
        tracer_cls: type[InterleavingTracer] | None = None,
        trace: bool = True,
        **kwargs: Any,
    ) -> InterleavingTracer:
        """Open a trace: a ``with`` block that runs the module and lets you read and
        edit its intermediate values.

        Inside the block, read an envoy's ``.output``/``.input`` to capture a value,
        or assign to it to overwrite what the module passes on. Mark a value with
        ``.save()`` to keep it past the block.

        Examples:
            >>> model = TransformersModel("openai-community/gpt2", dispatch=True)
            >>> with model.trace("Hello World"):
            ...     model.transformer.h[0].attn.output[0][:] = 0   # zero the attn output
            ...     output = model.output.save()
            >>> print(output)

        Args:
            *args: Arguments to pass to the tracer
            tracer_cls: Tracer class to construct instead of the default
                [`InterleavingTracer`][nnsight.intervention.tracer.InterleavingTracer] — an
                extension point for a custom tracer.
            trace: If ``False``, bypass tracing — run the module directly on the
                inputs and return its output. A one-shot forward with no
                intervention: input prep, dispatch, and device placement still
                happen (via `interleave`), but no ``with`` block is captured.
            **kwargs: Keyword arguments to pass to the tracer

        Returns:
            An InterleavingTracer for this module, or — when ``trace=False`` —
            the module's output directly.
        """
        if fn is None:
            fn = "__call__"
        if not trace:
            # One-shot forward, no trace body — only registered edits apply
            # (interleave adds them).
            return self.interleave(fn, *args, **kwargs)
        if tracer_cls is None:
            tracer_cls = InterleavingTracer
        return tracer_cls(self, fn, *args, backend=backend, **kwargs)

    def edit(self, *, inplace: bool = False, backend: Any = None) -> EditingTracer:
        """Open an editing tracer: capture interventions and store them as defaults.

        The block is not executed against a live forward; instead the interventions
        it captures are stored and replayed on every later trace of the (edited)
        model. Clear them with [`clear_edits`][nnsight.intervention.envoy.Envoy.clear_edits].

        Examples:
            >>> model = TransformersModel("openai-community/gpt2", dispatch=True)
            >>> # The first layer's attention output will always be zeroed.
            >>> with model.edit() as (tracer, edited_model):
            ...     edited_model.transformer.h[0].attn.output[0][:] = 0

            >>> with model.trace("Hello World"):
            ...     output = model.output.save()          # original model, unedited
            >>> print(output)

            >>> with edited_model.trace("Hello World"):
            ...     edited_output = edited_model.output.save()   # edit applied
            >>> print(edited_output)

        Args:
            inplace: If ``False`` (default), store the edit on a shallow copy and
                leave this envoy clean, so the edited behavior is opt-in through
                the copy. If ``True``, store it on this envoy itself.
            backend: Backend for the underlying trace.

        Returns:
            An [`EditingTracer`][nnsight.intervention.editing.EditingTracer]. Entering it
                binds the tracer — for its ``iter`` API — and, when
                ``inplace=False``, the edited copy as well:
                ``with model.edit() as (tracer, edited):``. With ``inplace=True``
                only the tracer is bound.
        """
        # Trace a block but store it instead of running it: the captured
        # interventions become defaults replayed on every later trace (see
        # EditingTracer). inplace=False edits a shallow copy, leaving self clean.
        return EditingTracer(self, inplace=inplace, backend=backend)

    def clear_edits(self) -> None:
        """
        Clear all edits for this Envoy.
        """
        self._edits = []

    def session(self, backend: Any = None, tracer_cls: type[Tracer] | None = None) -> Tracer:
        """Open a session: a scope enclosing several traces that share values.

        Inside ``with model.session():`` you can open multiple ``with
        model.trace(...)`` blocks and pass values between them — a value read in
        one trace is available in a later trace *without* an explicit ``.save()``,
        because the session (not each individual trace) is the save boundary. Only
        values marked with `nnsight.save` survive past the session itself.
        Ordinary Python — loops, conditionals, building lists — runs natively in
        the session body.

        Examples:
            >>> with model.session():
            ...     with model.trace(x):
            ...         hidden = model.layer1.output      # no .save() needed
            ...     with model.trace(x):
            ...         out = (hidden * 2).save()         # `hidden` flows in
            >>> print(out)

        Args:
            backend: What runs the captured session block. Defaults to running it
                in place (which executes the nested traces as it reaches them).
            tracer_cls: Tracer class to construct instead of the default
                [`Tracer`][nnsight.tracing.tracer.Tracer] — an extension point for a
                custom session tracer.

        Returns:
            A [`Tracer`][nnsight.tracing.tracer.Tracer] acting as the session scope.
        """
        # A plain Tracer already captures the block, execs it as real Python (so
        # nested traces run as reached), and gates saves at its own — outermost —
        # boundary. That is exactly a session; no model state is needed here.
        if tracer_cls is None:
            tracer_cls = Tracer
        return tracer_cls(backend=backend)

    def _shallow_copy(self) -> Envoy:
        # A twin sharing this envoy's module/interleaver/children but with its own
        # _edits list, so editing the copy doesn't touch the original's defaults.
        copy = object.__new__(type(self))
        copy.__dict__.update(self.__dict__)
        copy._edits = list(self._edits)
        return copy

    @eproperty(key="input")
    def inputs(self, value: Any) -> Any:
        """The module's forward inputs as an ``(args, kwargs)`` tuple.

        Read or replace the whole input during a trace::

            with model.trace("Hello World"):
                args, kwargs = model.transformer.h[0].attn.inputs
        """
        return value

    @eproperty
    def input(self, value: Any) -> Object:
        """The module's first forward input (first positional, else first keyword).

        A convenience view over `inputs`; writing it repacks into the full
        ``(args, kwargs)`` the model expects::

            with model.trace("Hello World"):
                hidden_states = model.transformer.h[0].attn.input.save()
        """
        args, kwargs = value
        return first_input(args, kwargs)

    @input.postprocess
    def input(self, value: Any) -> Any:
        args, kwargs = Mediator.value(f"{self.path}.input")
        return replace_first_input(args, kwargs, value)

    @eproperty
    def output(self, value: Any) -> Object:
        """The module's forward output — read or replace it during a trace::

            with model.trace("Hello World"):
                attn = model.transformer.h[0].attn.output[0].save()
        """
        return value

    @property
    @deprecated(
        "model.iter is deprecated and will be removed in a future version. "
        "Use tracer.iter instead."
    )
    def iter(self):
        """Deprecated: use ``tracer.iter``.

        An alias for ``tracer.iter[...]``; the iteration API lives on the tracer.
        """
        from .iterator import Iterations

        return Iterations()

    @deprecated(
        "model.all() is deprecated and will be removed in a future version. "
        "Use tracer.all() instead."
    )
    def all(self):
        """Deprecated: use ``tracer.all()``."""
        from .iterator import Iterations

        # Build the range directly rather than via self.iter, so only the
        # model.all() deprecation warns (not model.iter as well).
        return Iterations()[:]

    def skip(self, replacement: Any) -> None:
        """Skip this module's execution, returning ``replacement`` as its output.

        The module's forward is not run; ``replacement`` is used in its place.

        Examples:
            >>> model = TransformersModel("openai-community/gpt2", dispatch=True)
            >>> with model.trace("Hello World"):
            ...     # Skip the first layer, passing its input through as its output.
            ...     model.transformer.h[0].skip(model.transformer.h[0].input)
            ...     output = model.output.save()
            >>> print(output)

        Args:
            replacement: The value to use as the module's output; must match the
                shape the module would return. Read it from anywhere — including
                this module's own ``.input``, which is offered before the skip gate.

        Returns:
            Nothing; the skip takes effect when the module runs.
        """
        # The skip controller is installed on every module up front (see __init__),
        # so the skip gate is offered whenever the module runs — this works even
        # when ``replacement`` was read from the module's own input before calling
        # skip. (nn.Module.__call__ binds `forward` before running its pre-hooks, so
        # a controller installed only now — after the input read resumes the worker
        # mid-pre-hook — would be too late for the already-bound forward.)
        Mediator.skip(f"{self.path}.skip", replacement)

    @property
    def source(self) -> Source:
        """Get the source code representation of the module.

        Examples:
            >>> model = TransformersModel("openai-community/gpt2", dispatch=True)
            >>> print(model.transformer.h[0].attn.source)   # list the operations
            >>> with model.trace("Hello World"):
            ...     attn = model.transformer.h[0].attn.source.attention_interface_0.output.save()

        Returns:
            A [`Source`][nnsight.intervention.source.Source] exposing operation-level access.
        """
        # Intermediate operations inside this module's forward, addressable as
        # source.{name}_{occurrence} (e.g. source.relu_0.output). A class-level
        # property, so it wins over the __getattr__ module fallthrough. Building
        # it permanently source-instruments the module (inert outside a trace).
        return Source(self)

    def to(self, device: torch.device) -> Envoy:
        """Move the wrapped module to ``device`` (in place).

        Args:
            device: The device to move the module to.

        Returns:
            This envoy, for method chaining.
        """
        self._module.to(device)
        return self

    def cpu(self, *args: Any, **kwargs: Any) -> Envoy:
        """Move the wrapped module to the CPU (in place); return this envoy."""
        self._module.cpu(*args, **kwargs)
        return self

    def cuda(self, *args: Any, **kwargs: Any) -> Envoy:
        """Move the wrapped module to the GPU (in place); return this envoy."""
        self._module.cuda(*args, **kwargs)
        return self

    @property
    def device(self) -> torch.device | None:
        """The device of the module's first parameter, or ``None`` if it has none."""
        try:
            return next(self._module.parameters()).device
        except StopIteration:
            return None

    @property
    def devices(self) -> set[torch.device]:
        """The set of devices the module's parameters live on (empty if it has none)."""
        return {parameter.device for parameter in self._module.parameters()}

    #: The [`Batcher`][nnsight.intervention.batching.Batcher] class to batch with.
    #: Base default is the plain dim-0-stack [`Batcher`][nnsight.intervention.batching.Batcher]; a model whose batch
    #: layout differs (e.g. [`DiffusionBatcher`][nnsight.modeling.diffusion.DiffusionBatcher]
    #: for classifier-free-guidance doubling) overrides it with its subclass.
    _batcher_class: type["Batcher"] = Batcher

    def _batch_size(self, *inputs: Any, **kwargs: Any) -> int:
        """Number of batch rows an invoke's input contributes (0 if it has none).

        Base default: any input is a single row. Models that batch (e.g.
        [`TransformersModel`][nnsight.modeling.transformers.TransformersModel]) override this to
        report the true row count of a prompt / list / tensor.
        """
        return 1 if (inputs or kwargs) else 0

    def _batch(self, invokes: list[tuple], fn: Any) -> tuple:
        """Combine invokes' inputs into one ``(args, kwargs)`` for ``fn``.

        ``invokes`` is a list of ``(inputs, kwargs)``. Base default: pass a single
        invoke straight through; batching two or more requires a model that
        overrides this ([`TransformersModel`][nnsight.modeling.transformers.TransformersModel]).
        """
        if not invokes:
            return tuple(), {}
        if len(invokes) == 1:
            return invokes[0]
        raise NotImplementedError(
            f"{type(self).__name__} does not support batching multiple invokes"
        )

    def interleave(
        self, fn: Any, *args: Any, batcher: "Batcher | None" = None, **kwargs: Any
    ) -> Any:
        """Run ``fn`` interleaved with the interleaver's registered workers.

        This is the low-level driver behind `trace`; you rarely call it
        directly. The workers (edits + per-invoke interventions, with their batch
        groups) are set up on the interleaver first; this runs ``fn(*args, **kwargs)``
        alongside them and clears them afterward.

        Args:
            fn: The callable to run, or a method name resolved against the module.
            *args: Positional inputs for a direct (untraced) call. They're wrapped as
                a single implicit invoke and assembled like a one-invoke trace
                (tokenized/collated), then moved to `device`. Ignored when
                ``batcher`` is given (the inputs come from assembling it).
            batcher: The per-invoke inputs to combine into one call. When given, it's
                assembled into ``(args, kwargs)`` and registered on the interleaver so
                [`handle`][nnsight.intervention.interleaver.Interleaver.handle] can
                narrow each worker to its own rows; any ``kwargs`` passed alongside
                (trace-level params like ``max_new_tokens``) override the assembled
                ones.
            **kwargs: Keyword inputs — part of the direct call's single invoke when no
                ``batcher`` is given, else trace-level params for the assembled call.

        Returns:
            Any: Whatever ``fn`` returned (typically the module's forward output).
        """
        # Add the passed (args, kwargs) as one more input set, then assemble the
        # combined call and register the batcher so handle() can narrow per worker.
        # A direct (untraced) call has no batcher: create one. Either way the add is
        # uniform — a direct call's input contributes a row; a trace's forward params
        # (max_new_tokens, no data rows) fold into the assembled call.
        if batcher is None:
            batcher = self._batcher_class(self, kwargs)
        self.interleaver.batcher = batcher
        batcher.add(*args, **kwargs)
        args, kwargs = batcher.assemble(fn)
        # Move input tensors onto the model's device so the user doesn't have to.
        device = self.device
        if device is not None:
            args, kwargs = apply(
                (args, kwargs), lambda tensor: tensor.to(device), torch.Tensor
            )
        # Resolve a named fn against the live module now (after any dispatch),
        # so it binds to the module actually running rather than an earlier one.
        if isinstance(fn, str):
            fn = getattr(self._module, fn)
        # Registered edits run first — prepend them so an edit's swap lands before a
        # trace intervention reads that location. They act on the whole batch (no
        # group); the interleaver's `prepare` has already added any invoke workers.
        # Assigned rather than spliced in place, so this goes through the setter
        # and the interleaver's index of who is parked where is re-derived with it.
        self.interleaver.mediators = self._edits + self.interleaver.mediators
        result = None
        try:
            with self.interleaver:
                result = fn(*args, **kwargs)
                # The model has produced its return value; serve it to any
                # intervention parked on `tracer.result` while still interleaving.
                self.interleaver.handle("result", result)
            # The model finished: surface any worker still parked on a location
            # the run never reached (an out-of-order request).
            self.interleaver.check_dangling_mediators()
        except EarlyStopException:
            # An intervention stopped before the model began running: the
            # worker raised during start(), so __enter__ never finished and
            # __exit__ did not run to swallow it.
            pass
        finally:
            self.interleaver.cancel()
        return result

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails, so envoy children (set on the
        # instance) and real attributes are already handled. Fall through to the
        # wrapped module. __dict__.get avoids recursing before _module is set.
        module = self.__dict__.get("_module")
        if module is None or not hasattr(module, name):
            raise AttributeError(
                f"{type(self).__name__!r} object (nor its module) has attribute {name!r}"
            )

        value = getattr(module, name)

        # A submodule not yet mirrored as an envoy: wrap it so interventions
        # work on it too (and it's reachable next time as a normal attr).
        if isinstance(value, torch.nn.Module):
            return self._wrap_envoy(name, value)

        return value

    def __call__(self, *args: Any, hook: bool = False, **kwargs: Any) -> Any:
        """Run this module's forward — applying it ad hoc, out of execution order.

        Inside a trace you can feed a module any input to compute with it away from
        its place in the forward pass — e.g. the logit lens, running ``lm_head`` on
        an intermediate hidden state::

            with model.trace(prompt):
                hidden = model.transformer.h[-1].output
                logits = model.lm_head(model.transformer.ln_f(hidden))

        While interleaving, ``module.forward`` is called directly rather than
        ``module(...)``, so PyTorch's hook dispatch is skipped. That both keeps
        this ad-hoc call from re-firing the interleaver's hooks (which would try to
        switch into the very worker greenlet making the call) and leaves the
        module's real place in the forward pass untouched. Outside a trace it is an
        ordinary module call.

        Pass ``hook=True`` to force the full ``module(...)`` call so the module's
        own hooks *do* fire. Use it for a module attached to the tree that isn't
        part of the real forward pass — an adapter, LoRA, or SAE applied in an
        edit — so its internals become observable at ``.submodule.output``::

            model.transformer.h[0].adapter = MyAdapter()
            with model.edit() as (tracer, edited):
                acts = edited.transformer.h[0].output
                edited.transformer.h[0].output[:] = \
                    edited.transformer.h[0].adapter(acts, hook=True)
            with edited.trace(prompt):
                inner = edited.transformer.h[0].adapter.inner.output.save()

        The block above applies the adapter once, at the first time the layer runs.
        To apply it every time — each step of a generation loop — put the
        passthrough under the edit tracer's ``iter``::

            with model.edit(inplace=True) as tracer:
                for _ in tracer.iter[:]:
                    acts = model.transformer.h[0].output
                    model.transformer.h[0].output[:] = \
                        model.transformer.h[0].adapter(acts, hook=True)

        Args:
            *args: Inputs to run the module's forward on.
            hook: If ``False`` (default), run the forward directly while
                interleaving, leaving the module's hooks — and its real place in
                the forward pass — untouched. If ``True``, run the full
                ``module(...)`` so its hooks fire and its submodules become
                addressable; use it for a module attached to the tree rather than
                one the forward pass already runs.
            **kwargs: Keyword inputs to the module's forward.

        Returns:
            The module's output for these inputs.
        """
        if self.interleaver.interleaving and not hook:
            return self._module.forward(*args, **kwargs)
        return self._module(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, torch.nn.Module) and not name.startswith("_"):
            self._add_envoy(name, value)
        else:
            object.__setattr__(self, name, value)

    def __iter__(self) -> Iterator[Envoy]:
        """Iterate over this envoy's direct children.

        Yields each immediate child envoy — e.g. the blocks of a
        `ModuleList`, so ``for layer in model.model.layers:``
        walks the layers. This is *not* recursive; use [`modules`][nnsight.intervention.envoy.Envoy.modules] to walk the
        whole subtree.

        Yields:
            Envoy: Each direct child envoy, in order.

        Example:
            ::

                for layer in model.model.layers:
                    print(layer.path)
        """
        return iter(self._children)

    def __getitem__(self, key: Any) -> Envoy:
        """Index into direct child envoys, e.g. for a `ModuleList`.

        Args:
            key: Any index the underlying child list accepts (an int, or a slice).

        Returns:
            Envoy: The child envoy at ``key`` (e.g. ``model.layers[0]`` for the
            first block of a ``ModuleList``).
        """
        return self._children[key]

    def __len__(self) -> int:
        """The number of entries in the wrapped module (e.g. a ``ModuleList``'s length)."""
        return len(self._module)

    def get(self, path: str) -> Any:
        """Resolve a dotted ``path`` from this envoy, e.g. ``"transformer.h.0.mlp"``.

        A programmatic alternative to attribute access, useful when the path is
        built at runtime. Outside a trace it returns the descendant envoy; inside
        one, a trailing ``.output``/``.input`` resolves through to the live value.

        Examples:
            >>> model = TransformersModel("openai-community/gpt2", dispatch=True)
            >>> module = model.get("transformer.h.0.mlp")
            >>> with model.trace("Hello"):
            ...     value = model.get("transformer.h.0.mlp.output").save()

        Args:
            path: A ``.``-separated attribute path relative to this envoy.

        Returns:
            The resolved child [`Envoy`][nnsight.intervention.envoy.Envoy], or the live value when ``path``
            ends in an intervention attribute during a trace.
        """
        obj: Any = self
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    def modules(
        self,
        include_fn: Callable[[Envoy], bool] | None = None,
        names: bool = False,
    ) -> list[Any]:
        """Flatten this envoy's whole subtree (children first, then self) into a list.

        Args:
            include_fn: Optional predicate on an envoy; only those it returns
                ``True`` for are kept.
            names: If ``True``, yield ``(path, envoy)`` tuples instead of envoys.

        Returns:
            A list of [`Envoy`][nnsight.intervention.envoy.Envoy] (or ``(path, Envoy)`` tuples when ``names``).
        """
        # Flatten the envoy tree (children first, then self), optionally filtered
        # by include_fn and paired with each envoy's path when names=True.
        result: list[Any] = []
        for child in self._children:
            result.extend(child.modules(include_fn=include_fn, names=names))
        if include_fn is None or include_fn(self):
            result.append((self.path, self) if names else self)
        return result

    def named_modules(
        self, include_fn: Callable[[Envoy], bool] | None = None
    ) -> list[tuple[str, Envoy]]:
        """Flatten the subtree into ``(path, envoy)`` tuples: [`modules`][nnsight.intervention.envoy.Envoy.modules] with ``names=True``.

        Args:
            include_fn: Optional predicate on an envoy; only those it returns
                ``True`` for are kept.

        Returns:
            A list of ``(path, Envoy)`` tuples for the included envoys.
        """
        return self.modules(include_fn=include_fn, names=True)

    def _name(self) -> str:
        return self._module._get_name()

    def _repr_modulelist(self) -> str:
        reprs = [repr(child) for child in self._children]

        start_end = [[0, 0]]
        blocks = [reprs[0]]
        for index, child_repr in enumerate(reprs[1:], 1):
            if child_repr == blocks[-1]:
                start_end[-1][1] += 1
                continue
            start_end.append([index, index])
            blocks.append(child_repr)

        lines = []
        for (start, end), block in zip(start_end, blocks):
            if start == end:
                line = f"({start}): {block}"
            else:
                line = f"({start}-{end}): {end - start + 1} x {block}"
            lines.append(_addindent(line, 2))

        return self._name() + "(\n  " + "\n  ".join(lines) + "\n)"

    def __repr__(self) -> str:
        if self._children and isinstance(self._module, torch.nn.ModuleList):
            return self._repr_modulelist()

        extra_lines = []
        extra_repr = self._module.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        # Split aliases bound here into those naming a direct child (decorate its
        # line as "alias/realname") and multi-component mounts (their own lines).
        direct: dict[str, list[str]] = {}
        mounts: list[str] = []
        for alias, path in self._aliases.items():
            if "." in path:
                mounts.append(alias)
            else:
                direct.setdefault(path, []).append(alias)

        child_lines = []
        for child in self._children:
            name = child.path.rsplit(".", 1)[-1]
            label = "/".join([*direct.get(name, []), name])
            child_lines.append(f"({label}): " + _addindent(repr(child), 2))
        for alias in mounts:
            child_lines.append(f"({alias}): " + _addindent(repr(getattr(self, alias)), 2))

        # eproperties given a description surface as their own lines, so special
        # hookable values (e.g. a model's .logits) show up in the tree; the plain
        # .input/.output views carry no description and stay hidden.
        eproperty_lines = []
        seen = set()
        for cls in type(self).__mro__:
            for attr in cls.__dict__.values():
                if (
                    isinstance(attr, eproperty)
                    and attr.description is not None
                    and attr.name not in seen
                ):
                    seen.add(attr.name)
                    eproperty_lines.append(f"({attr.name}): {attr.description}")

        lines = extra_lines + child_lines + eproperty_lines

        main_str = self._name() + "("
        if lines:
            if len(extra_lines) == 1 and not child_lines and not eproperty_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str
