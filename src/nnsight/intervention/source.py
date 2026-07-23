"""Expose, edit, and skip a module's ``forward`` at the operation level.

Module ``input``/``output`` are the only two locations the forward *hooks*
(:mod:`nnsight.intervention.interleaver`) surface. Everything in between — the
individual operations a ``forward`` performs — is invisible to them because it
isn't a submodule with its own hook.

This module makes those intermediates observable, editable, and skippable without
the interleaver knowing anything about source. The interleaver runs on one
primitive: a provider-location string and :meth:`Interleaver.handle`, which
serves a value to interventions and returns it back edited if one wrote to that
location. Inputs/outputs are just the two locations hooks emit; here we add more,
mid-forward:

1. Parse the module's ``forward`` and rewrite every call ``fn(*args, **kwargs)``
   into ``__nnsight_op__("source.{name}_{n}", fn, *args, **kwargs)``. At run time
   :func:`_make_op` brackets the call with :meth:`Interleaver.handle` on its
   ``.input`` (before) and ``.output`` (after) — both readable/replaceable — and a
   ``.skip`` gate that can bypass the call entirely.
2. ``name`` is the called function's dotted path joined with ``_``
   (``self.act(...)`` → ``self_act``, ``torch.relu(...)`` → ``torch_relu``,
   ``dropout(...)`` → ``dropout``); ``n`` is a per-name counter in **execution
   order** (nested calls run inner-first, so the inner call is ``_0``), which is
   the order the interleaver serves values.

Installation is lazy and permanent. The first time a module is sourced or skipped
its ``forward`` is replaced by a :func:`_make_controller` closure that reads a
single per-module :class:`_State` (see :data:`_STATE`): it gates on ``.skip``, then
runs the *body* — the original forward, or the source-instrumented one once
``.source`` is used. The controller is inert outside a trace and stays installed,
so later runs work regardless of request order, and source and skip compose on one
wrapper. Because the ``_State`` is rebound on each access, a module wrapped by
several Envoys/Interleavers reports to whichever touched it most recently.

An :class:`Envoy` exposes operations as ``envoy.source.{name}_{n}``, whose
``.input``/``.inputs``/``.output``/``.skip`` mirror an Envoy's own, one level finer.
"""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
import weakref
from types import CodeType, FunctionType
from typing import TYPE_CHECKING, Any, Callable, Iterator, NamedTuple

import torch

from .eproperty import eproperty
from .interleaver import Mediator
from .util import first_input, replace_first_input

if TYPE_CHECKING:
    from .envoy import Envoy

#: Global name the instrumented forward calls to bracket each operation.
OP = "__nnsight_op__"

#: Attribute holding a module's :class:`_State` once it has been sourced/skipped.
_STATE = "__nnsight__"

#: Sentinel returned by the skip gate when no skip is pending for a location.
_NO_SKIP = object()


class SourceNotAvailable(Exception):
    """A module's ``forward`` can't be source-instrumented (no source, decorated…)."""


class Compiled(NamedTuple):
    """Everything the source machinery needs about one instrumented ``forward``."""

    code: CodeType  #: the instrumented forward's code object
    names: tuple[str, ...]  #: operation labels, in execution order
    lines: dict[str, int]  #: label -> 1-based line within :attr:`source`
    source: str  #: dedented source text of the original forward


class _State:
    """Per-module source/skip state, stored at ``module.__dict__[_STATE]``.

    Created when a module is first sourced or skipped. The controller and op probes
    read it live on every call, so re-wrapping — or wrapping the same module in more
    than one Envoy at once — just registers another interleaver here.

    :attr:`interleavers` maps each interleaver that instrumented this module to the
    path it addresses the module by; :meth:`active` picks the one whose trace is
    currently running (there is at most one, exactly as each of a module's stacked
    input/output hooks fires only while its own interleaver is interleaving).

    Because this state lives on the module (``module.__dict__[_STATE]``), it holds
    neither the module nor its interleavers *strongly* — the module would sit in a
    reference cycle and never be freed by refcounting. The interleavers are held in
    a :class:`weakref.WeakKeyDictionary` (a finished local wrapper's interleaver
    drops out on its own; a server's persistent interleaver stays, so the same
    module serves request after request), and :attr:`body` is the *unbound* forward
    (a plain function taking ``self``) rather than a bound method that would pin it.
    """

    __slots__ = ("interleavers", "body", "sourced")

    def __init__(self, body: Callable) -> None:
        #: interleaver -> the path it addresses this module by (weak keys)
        self.interleavers: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self.body = body  #: unbound forward to run when not skipped
        self.sourced = False  #: whether body is the source-instrumented forward

    def register(self, interleaver: Any, path: str) -> None:
        """Record that ``interleaver`` reaches this module at ``path``."""
        self.interleavers[interleaver] = path

    def active(self) -> "tuple[Any, str] | tuple[None, None]":
        """The (interleaver, path) whose trace is running now, or (None, None)."""
        for interleaver, path in self.interleavers.items():
            if interleaver.interleaving:
                return interleaver, path
        return None, None


#: Instrumented forwards, memoized per original ``forward`` code object. Value is a
#: :class:`Compiled`, or a :class:`SourceNotAvailable` (cached so a forward we
#: can't instrument isn't re-parsed on every access).
_FORWARD_CACHE: dict[CodeType, "Compiled | SourceNotAvailable"] = {}


# ---------------------------------------------------------------------------
# Compiling the instrumented forward
# ---------------------------------------------------------------------------


class _Instrument(ast.NodeTransformer):
    """Rewrite every call into an ``__nnsight_op__(location, fn, *args, **kwargs)``.

    Numbers occurrences in execution order: :meth:`generic_visit` descends into a
    call's arguments (running inner calls first) *before* this node is assigned
    its counter, so ``f(f(x))`` gives the inner ``f`` ``f_0`` and the outer ``f_1``.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.names: list[str] = []
        # label -> 1-based line of the call within the (dedented) forward source,
        # captured here before increment_lineno shifts it to file coordinates.
        self.lines: dict[str, int] = {}

    @staticmethod
    def _op_name(call: ast.Call) -> str:
        """The called function's dotted name, joined with ``_``.

        The whole attribute chain is used, not just the last component:
        ``self.act`` → ``self_act``, ``nn.functional.dropout`` →
        ``nn_functional_dropout``, ``module.attn_dropout`` → ``module_attn_dropout``.
        A bare name stays as-is (``dropout`` → ``dropout``).
        """
        func = call.func
        if isinstance(func, ast.Attribute):
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return "_".join(reversed(parts))
        if isinstance(func, ast.Name):
            return func.id
        return "call"

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Descend first so nested calls are numbered before this (outer) one —
        # matching execution order. The wrapper we return is not re-visited, so
        # our own op insertions never get counted as operations.
        self.generic_visit(node)

        name = self._op_name(node)
        occurrence = self.counts.get(name, 0)
        self.counts[name] = occurrence + 1
        label = f"{name}_{occurrence}"
        self.names.append(label)
        self.lines[label] = node.lineno

        # fn(*args, **kwargs)  ->  __nnsight_op__("source.<label>", fn, *args, **kwargs)
        return ast.Call(
            func=ast.Name(id=OP, ctx=ast.Load()),
            args=[ast.Constant(value=f"source.{label}"), node.func, *node.args],
            keywords=node.keywords,
        )


def _compile_instrumented(func: Callable) -> Compiled:
    """Parse, instrument, and compile a Python ``func`` (a ``forward`` or a drilled-into
    callable), or raise.

    Raises :class:`SourceNotAvailable` for callables we can't safely rewrite:
    non-Python ones (builtins/C functions have no ``__code__``), those with free
    variables, no recoverable source, or a decorator (which is load-bearing and
    would be dropped by extracting the raw code object).
    """
    code_object = getattr(func, "__code__", None)
    if code_object is None:
        raise SourceNotAvailable("callable has no Python source (builtin or C function)")
    if code_object.co_freevars:
        # Recompiled at module level, free variables would become globals and
        # break — methods essentially never close over an enclosing scope, so
        # bail rather than silently misbehave.
        raise SourceNotAvailable("callable closes over free variables")
    try:
        lines, start = inspect.getsourcelines(func)
    except (OSError, TypeError) as error:
        raise SourceNotAvailable("callable source is unavailable") from error

    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source)
    definition = tree.body[0]
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise SourceNotAvailable("callable is not a plain function")
    if definition.decorator_list:
        raise SourceNotAvailable("decorated callable is not supported")

    instrument = _Instrument()
    instrument.visit(tree)
    # Line up line numbers with the original file so tracebacks read right.
    ast.increment_lineno(tree, start - 1)
    ast.fix_missing_locations(tree)

    module = compile(tree, code_object.co_filename, "exec")
    code = next(
        (
            const
            for const in module.co_consts
            if isinstance(const, CodeType) and const.co_name == func.__name__
        ),
        None,
    )
    if code is None:
        raise SourceNotAvailable("could not locate the compiled callable")
    return Compiled(code, tuple(instrument.names), instrument.lines, source)


def _compiled(func: Callable) -> Compiled:
    """Cached :func:`_compile_instrumented`, keyed by ``func``'s code object."""
    key = getattr(func, "__code__", None)
    if key is None:
        # Builtins / C functions have no code object to key on or parse.
        raise SourceNotAvailable("callable has no Python source (builtin or C function)")
    cached = _FORWARD_CACHE.get(key)
    if isinstance(cached, SourceNotAvailable):
        raise cached
    if cached is not None:
        return cached
    try:
        result = _compile_instrumented(func)
    except SourceNotAvailable as error:
        _FORWARD_CACHE[key] = error
        raise
    _FORWARD_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# The installed forward: controller (skip gate + body) and per-op probes
# ---------------------------------------------------------------------------


def _skipped(interleaver: Any, location: str) -> Any:
    """Query the skip gate at ``location``; return the replacement or :data:`_NO_SKIP`."""
    return interleaver.handle(f"{location}.skip", _NO_SKIP)


def _run_op(
    interleaver: Any, base: str, fn: Callable, args: tuple, kwargs: dict
) -> Any:
    """Bracket one operation at location ``base``: input, skip, (recursive) run, output.

    Reports/replaces ``.input``, honors a ``.skip`` gate, and reports/replaces
    ``.output`` — the same three handles a module hook emits, one level finer.
    Between them, if a worker asked to drill into this op (``base`` is a key in
    :attr:`Interleaver.sourced`), the raw ``fn`` is offered over ``{base}.fn`` so
    the worker can hand back a source-instrumented copy (cached in
    :attr:`Interleaver.sourced` for later fires); that copy runs in place of ``fn``,
    making *its* operations addressable under ``{base}.source.*`` — recursively.
    """
    args, kwargs = interleaver.handle(f"{base}.input", (args, kwargs))
    skipped = _skipped(interleaver, base)
    if skipped is not _NO_SKIP:
        # Skip: don't call fn; report the replacement as this op's output too.
        return interleaver.handle(f"{base}.output", skipped)
    if base in interleaver.sourced:
        # First fire: the entry is still None (requested, not built) — serve the
        # live callable to the parked worker, which builds and caches the
        # instrumented version. Later fires: the entry is already built, no worker
        # is parked, so this handle is a no-op and we reuse it.
        interleaver.handle(f"{base}.fn", fn)
        entry = interleaver.sourced.get(base)
        if entry is not None:
            fn = entry[0]
    value = fn(*args, **kwargs)
    return interleaver.handle(f"{base}.output", value)


def _make_op(module: Any) -> Callable:
    """Build the ``__nnsight_op__`` the instrumented forward calls at each call site.

    Anchored to the *module* (reads its live :class:`_State`), so re-wrapping and
    multiple wrappers just work. Outside a trace it calls straight through; inside
    one it brackets the call via :func:`_run_op` under ``{path}.{location}`` for the
    interleaver whose trace is running. Holds the module by weakref: this op lives in
    the instrumented forward's globals, which is the module's own ``body`` — a strong
    reference would cycle.
    """
    module_ref = weakref.ref(module)

    def op(location: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        interleaver, path = module_ref().__dict__[_STATE].active()
        if interleaver is None:
            return fn(*args, **kwargs)
        return _run_op(interleaver, f"{path}.{location}", fn, args, kwargs)

    return op


def _make_nested_op(interleaver: Any, prefix: str) -> Callable:
    """Build the ``__nnsight_op__`` for a source-instrumented *called* function.

    Like :func:`_make_op`, but the location namespace is fixed to ``prefix`` (the
    drilled-into op's path) rather than resolved from a module's live state — the
    callable isn't a module and has no :class:`_State`. ``location`` already carries
    the ``source.`` prefix (added by :class:`_Instrument`), so its operations land
    under ``{prefix}.source.{label}``.
    """

    def op(location: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if not interleaver.interleaving:
            return fn(*args, **kwargs)
        return _run_op(interleaver, f"{prefix}.{location}", fn, args, kwargs)

    return op


def _build_instrumented(fn: Callable, compiled: Compiled, op: Callable) -> Callable:
    """Materialize a source-instrumented copy of ``fn`` that calls ``op`` per call site.

    ``compiled`` is ``fn``'s instrumented code (from :func:`_compiled`); ``op`` is the
    :func:`_make_nested_op` to bind as its ``__nnsight_op__`` global. Bound methods
    are rebuilt from their underlying function and re-bound to the same instance, so
    the recompiled body still receives ``self``.
    """
    if inspect.ismethod(fn):
        target, receiver = fn.__func__, fn.__self__
    else:
        target, receiver = fn, None
    globals_ = {**target.__globals__, OP: op}
    instrumented = FunctionType(
        compiled.code, globals_, target.__name__, target.__defaults__, None
    )
    if target.__kwdefaults__:
        instrumented.__kwdefaults__ = target.__kwdefaults__
    return instrumented.__get__(receiver) if receiver is not None else instrumented


def _make_controller(module: Any) -> Callable:
    """Build the forward installed on an instrumented module: skip gate, then body.

    Reads the module's live :class:`_State` each call, so it composes cleanly:
    ``body`` is the (unbound) original forward or, once sourced, the instrumented
    one, and a ``.skip`` bypasses it entirely (and, if sourced, all its ops).

    Holds the module by weakref and wraps the original forward: the module owns this
    controller (as its ``forward``), so a strong back-reference would cycle; and
    ``functools.wraps`` preserves the forward's signature, which ``generate()``
    introspects to decide whether to pass ``attention_mask``/``position_ids``.
    """
    module_ref = weakref.ref(module)
    original = type(module).forward  # unbound; used only for signature metadata

    @functools.wraps(original)
    def controller(*args: Any, **kwargs: Any) -> Any:
        module = module_ref()
        state = module.__dict__[_STATE]
        interleaver, path = state.active()
        if interleaver is None:
            return state.body(module, *args, **kwargs)
        skipped = _skipped(interleaver, path)
        if skipped is not _NO_SKIP:
            return skipped
        return state.body(module, *args, **kwargs)

    return controller


def _ensure_controller(envoy: "Envoy") -> _State:
    """Install the controller forward on ``envoy``'s module once; (re)bind and return
    its :class:`_State`.

    Installed directly into the module's ``__dict__`` (shadowing the class method
    for ``__call__``) and left there permanently — inert outside a trace. The body
    defaults to the original forward; :func:`install_source` upgrades it.
    """
    module = envoy._module
    state = module.__dict__.get(_STATE)
    if state is None:
        # Store the *unbound* forward (a bound method would pin the module, and the
        # module holds this state — a cycle); the controller supplies the module.
        state = _State(type(module).forward)
        module.__dict__[_STATE] = state
        module.__dict__["forward"] = _make_controller(module)
    # Register this envoy's interleaver (weakly) under its path, so several envoys
    # can share the module and each routes to its own trace.
    state.register(envoy.interleaver, envoy.path)
    return state


def install_source(envoy: "Envoy") -> Compiled:
    """Source-instrument ``envoy``'s module and install the controller.

    Returns the module's :class:`Compiled`. Upgrades the controller's body to the
    instrumented forward (built once per module, from code cached per code object).
    """
    func = type(envoy._module).forward
    compiled = _compiled(func)  # raises SourceNotAvailable if it can't

    state = _ensure_controller(envoy)
    if not state.sourced:
        globals_ = {**func.__globals__, OP: _make_op(envoy._module)}
        forward = FunctionType(
            compiled.code, globals_, func.__name__, func.__defaults__, None
        )
        if func.__kwdefaults__:
            forward.__kwdefaults__ = func.__kwdefaults__
        # Unbound (the controller passes the module); a bound method would pin it.
        state.body = forward
        state.sourced = True
    return compiled


def install_skip(envoy: "Envoy") -> None:
    """Install the controller so ``envoy``'s module forward can be skipped."""
    _ensure_controller(envoy)


# ---------------------------------------------------------------------------
# User-facing views
# ---------------------------------------------------------------------------


class SourceEnvoy:
    """A single operation inside a module's ``forward``, e.g. ``source.torch_relu_0``.

    You never construct one directly; you reach it by indexing a :class:`Source`
    with an operation's ``{callable}_{occurrence}`` name (``envoy.source.torch_relu_0``).
    It is the operation-level analogue of an
    :class:`~nnsight.intervention.envoy.Envoy`: where an Envoy exposes a
    *submodule's* ``.input``/``.output``, a ``SourceEnvoy`` exposes those same
    handles for a *single call* the ``forward`` makes — one level finer.

    Inside a trace, each handle both reads and writes the live value:

    * :attr:`output` — the operation's return value.
    * :attr:`input` — the operation's first argument.
    * :attr:`inputs` — the operation's full ``(args, kwargs)``.
    * :meth:`skip` — bypass the call, substituting a value for its output.
    * :attr:`source` — drill *into* the called function, exposing its own
      operations one level deeper (recursively).

    Reading returns the value; assigning replaces it for the rest of the forward.
    These handles are only meaningful inside a ``with envoy.trace(...):`` block.
    To use a captured value after the trace, call ``.save()`` on it.

    Examples:
        Capture and edit an intermediate operation (for a ``forward`` that runs
        ``h = torch.relu(self.fc1(x))``)::

            with model.trace(x):
                pre = model.layer1.source.torch_relu_0.output.save()  # capture
                model.layer1.source.torch_relu_0.output = pre * 2      # and rescale it

        On a real transformer, reach the activation inside an MLP::

            with model.trace(prompt):
                act = model.transformer.h[0].mlp.source.self_act_0.output.save()

        Drill into a called function to reach an operation inside it::

            with model.trace(prompt):
                attn = model.transformer.h[0].attn.source
                out = attn.attention_interface_0.source.attn_output_transpose_0.output.save()
    """

    def __init__(
        self,
        envoy: "Envoy",
        name: str,
        path: str,
        source: str = "",
        line: int | None = None,
    ) -> None:
        self._envoy = envoy
        self.name = name
        self.path = path
        self._source = source
        self._line = line

    @property
    def source(self) -> "Source":
        """Drill into the called function, exposing *its* operations recursively.

        Returns a :class:`Source` over the function this operation calls, so its
        internal operations become addressable as
        ``...source.{name}.source.{inner}`` — with the same
        ``.input``/``.output``/``.inputs``/``.skip``/``.source`` handles, to any
        depth.

        Only available **inside a trace**: the called function is resolved from the
        live value flowing through the call at run time (a call target is often a
        local variable, e.g. an attention implementation, so it can't be found
        statically). Raises :class:`SourceNotAvailable` if the target has no
        recoverable Python source (a builtin/C function) or is itself a submodule
        (call ``.source`` on that submodule directly instead).

        Examples:
            >>> with model.trace(prompt):
            ...     attn = model.transformer.h[0].attn.source.attention_interface_0
            ...     out = attn.source.attn_output_transpose_0.output.save()
        """
        interleaver = self._envoy.interleaver
        if not interleaver.interleaving:
            raise SourceNotAvailable(
                "recursive `.source` is only available inside a trace"
            )
        if self.path not in interleaver.sourced:
            # Mark requested (None placeholder), then park until the operation fires
            # and the model side hands back the live callable (see _run_op). Build
            # and cache its instrumented copy so later fires this run reuse it.
            interleaver.sourced[self.path] = None
            fn = Mediator.value(f"{self.path}.fn")
            if isinstance(fn, torch.nn.Module):
                raise SourceNotAvailable(
                    f"{self.name!r} calls a submodule; call `.source` on that "
                    f"submodule directly instead of drilling into the call"
                )
            compiled = _compiled(fn)  # raises SourceNotAvailable if it can't
            instrumented = _build_instrumented(
                fn, compiled, _make_nested_op(interleaver, self.path)
            )
            interleaver.sourced[self.path] = (instrumented, compiled)
        return Source(self._envoy, prefix=self.path, compiled=interleaver.sourced[self.path][1])

    @eproperty
    def output(self, value: Any) -> Any:
        """The operation's return value.

        Read it to capture the value the call produced; assign to it to replace
        that value for the remainder of the ``forward`` (downstream operations
        see the replacement). In-place edits work too.

        Examples:
            >>> with model.trace(x):
            ...     h = model.layer1.source.torch_relu_0.output.save()          # capture
            ...     model.layer1.source.torch_relu_0.output = h.clamp(min=0)    # replace
        """
        return value

    @eproperty(key="input")
    def inputs(self, value: Any) -> Any:
        """The operation's arguments as an ``(args, kwargs)`` pair.

        Use this when you need every argument (or the keyword arguments); for the
        common case of a single leading argument, :attr:`input` is more direct.
        Assigning a new ``(args, kwargs)`` pair replaces the arguments the call
        runs with.

        Examples:
            >>> with model.trace(x):
            ...     args, kwargs = model.layer1.source.self_fc2_0.inputs
            ...     model.layer1.source.self_fc2_0.inputs = ((args[0] * 0,), {})
        """
        return value

    @eproperty
    def input(self, value: Any) -> Any:
        """The operation's first argument (first positional, else first keyword).

        A convenience view over :attr:`inputs` for the usual single-argument call.
        Assigning replaces just that first argument, leaving any other arguments
        untouched.

        Examples:
            >>> with model.trace(x):
            ...     model.layer1.source.self_fc2_0.input = torch.zeros(2, 8)
        """
        args, kwargs = value
        return first_input(args, kwargs)

    @input.postprocess
    def input(self, value: Any) -> Any:
        args, kwargs = Mediator.value(f"{self.path}.input")
        return replace_first_input(args, kwargs, value)

    def skip(self, replacement: Any) -> None:
        """Skip this operation, using ``replacement`` as its output.

        The call never runs; ``replacement`` takes the place of its return value
        and flows on to whatever consumed it. Use it to short-circuit expensive
        or unwanted compute, or to splice in a value of your own.

        Reading ``.output`` for a skipped operation returns ``replacement``.

        Args:
            replacement: Value substituted for the operation's return value.

        Examples:
            >>> with model.trace(x):
            ...     # fc1 doesn't run; the forward proceeds as if it returned zeros
            ...     model.layer1.source.self_fc1_0.skip(torch.zeros(2, 8))
        """
        Mediator.skip(f"{self.path}.skip", replacement)

    def __repr__(self) -> str:
        """A window of the ``forward`` source around this operation's call site.

        The call site is flagged with ``-->`` / ``<--`` so you can confirm you
        indexed the operation you meant, with a few surrounding lines for context
        (``....`` marks truncation above or below). Falls back to the operation's
        dotted path when the source text is unavailable.

        Examples:
            >>> print(model.layer1.source.self_fc1_0)
            model.layer1.source.self_fc1_0:

                def forward(self, x):
                --> h = torch.relu(self.fc1(x)) <--
                    return self.fc2(h)
        """
        if not self._source:
            return self.path
        source_lines = self._source.split("\n")
        # ._line is 1-based; -2 lands in the body frame where the highlighted line
        # is index (line_number + 1) into source_lines.
        line_number = self._line - 2
        start = max(0, line_number - 5)
        end = min(len(source_lines) - 1, line_number + 8)

        out = [self.path + ":\n"]
        if start != 0:
            out.append("    ....")
        for i in range(start, end):
            line = source_lines[i]
            if i == line_number + 1:
                out.append(f"    --> {line[4:]} <--")
            else:
                out.append("    " + line)
        if end != len(source_lines) - 1:
            out.append("    ....")
        return "\n".join(out)


class Source:
    """A module's ``forward`` decomposed into its individual operations.

    Reached as ``envoy.source`` (e.g. ``model.layer1.source``). Every call the
    ``forward`` makes becomes an operation named ``{callable}_{occurrence}``,
    where ``callable`` is the called function's dotted path joined with ``_``:
    ``self.fc1(x)`` → ``self_fc1_0``, ``torch.relu(...)`` → ``torch_relu_0``. The
    occurrence counter is per name and runs in execution order (nested calls run
    inner-first, so the inner call gets ``_0``); two ``torch.relu(...)`` calls are
    therefore ``torch_relu_0`` and ``torch_relu_1``. Index in with that name to get
    a :class:`SourceEnvoy`::

        model.layer1.source.torch_relu_0    # -> SourceEnvoy for the relu call

    You rarely need to memorize the names: ``print(model.layer1.source)`` renders
    the whole ``forward`` with each operation labelled at its call site, and
    ``print(model.layer1.source.torch_relu_0)`` zooms in on one. Iterating a
    ``Source`` yields its operations in execution order.

    Source values are only meaningful inside a trace, and ordinary inference is
    unaffected. Requesting an operation on a ``forward`` whose source can't be
    recovered (e.g. a decorated ``forward``) raises :class:`SourceNotAvailable`.

    A ``Source`` also decomposes a *called function* — reached as
    ``some_op.source`` (see :attr:`SourceEnvoy.source`) — the same way, one level
    deeper. In that nested form the operations live under the drilled-into op's
    path rather than the module's ``forward``.

    Examples:
        Inspect, then capture and edit, an intermediate operation::

            print(model.layer1.source)                          # list the operations
            with model.trace(x):
                h = model.layer1.source.torch_relu_0.output.save()  # capture
                model.layer1.source.self_fc2_0.input = h * 0        # edit a later op's input

        Iterate every operation::

            for op in model.layer1.source:
                print(op.name)
    """

    def __init__(
        self,
        envoy: "Envoy",
        prefix: str | None = None,
        compiled: "Compiled | None" = None,
    ) -> None:
        self._envoy = envoy
        if compiled is None:
            # Top level: instrument the module's own forward (inert outside a trace).
            self._compiled = install_source(envoy)  # raises SourceNotAvailable
            self._prefix = f"{envoy.path}.source"
        else:
            # Nested: `compiled` is the drilled-into callable, already parsed by
            # SourceEnvoy.source; its operations hang off the op's path.
            self._compiled = compiled
            self._prefix = f"{prefix}.source"

    @property
    def _names(self) -> tuple[str, ...]:
        return self._compiled.names

    def _node(self, name: str) -> SourceEnvoy:
        """A :class:`SourceEnvoy` for ``name``, carrying source text for its repr."""
        return SourceEnvoy(
            self._envoy,
            name,
            f"{self._prefix}.{name}",
            self._compiled.source,
            self._compiled.lines[name],
        )

    def __getattr__(self, name: str) -> SourceEnvoy:
        """Resolve ``source.<name>`` to its :class:`SourceEnvoy`.

        Raises :class:`AttributeError` for an unknown operation, listing the
        available names so a mistyped or wrong-occurrence label is easy to fix.

        Examples:
            >>> model.layer1.source.self_fc1_0  # -> SourceEnvoy
            >>> model.layer1.source.nope_0      # AttributeError: ... available: self_fc1_0, torch_relu_0, self_fc2_0
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._compiled.names:
            available = ", ".join(self._compiled.names) or "(none)"
            raise AttributeError(
                f"{self._prefix!r} has no operation {name!r}; available: {available}"
            )
        return self._node(name)

    def __iter__(self) -> Iterator[SourceEnvoy]:
        """Iterate the operations in execution order, yielding a :class:`SourceEnvoy` each.

        Examples:
            >>> [op.name for op in model.layer1.source]
            ['self_fc1_0', 'torch_relu_0', 'self_fc2_0']
        """
        for name in self._compiled.names:
            yield self._node(name)

    def __repr__(self) -> str:
        """The whole ``forward`` source with every op labelled at its call site.

        Each source line is shown with the operations that occur on it in a left
        gutter (the ``def`` line is marked ``*``); when several operations share a
        line, the extras appear as ``+`` continuations. This is the map from
        source code to ``{callable}_{occurrence}`` names, so you never have to
        guess an occurrence number.

        Examples:
            >>> print(model.layer1.source)
                              * def forward(self, x):
             self_fc1_0   ->  0     h = torch.relu(self.fc1(x))
             torch_relu_0 ->  +     ...
             self_fc2_0   ->  1     return self.fc2(h)
        """
        names = self._compiled.names
        # ._compiled.lines is 1-based; -2 matches the loop below, which enumerates
        # source_lines[1:] (the body) from 0.
        line_numbers = {name: self._compiled.lines[name] - 2 for name in names}
        max_name_length = max((len(name) for name in names), default=0)

        source_lines = self._compiled.source.split("\n")
        formatted = [" " * (max_name_length + 6) + "* " + source_lines[0]]

        ops_by_line: dict[int, list[str]] = {}
        for name in names:
            ops_by_line.setdefault(line_numbers[name], []).append(name)

        for i, line in enumerate(source_lines[1:]):
            if i in ops_by_line:
                ops = ops_by_line[i]
                formatted.append(f" {ops[0]:{max_name_length}} ->{i:3d} {line}")
                for op in ops[1:]:
                    indent = " " * (len(line) - len(line.lstrip()))
                    formatted.append(f" {op:{max_name_length}} ->  + {indent}...")
            else:
                formatted.append(" " * (max_name_length + 4) + f"{i:3d} {line}")

        return "\n".join(formatted)
