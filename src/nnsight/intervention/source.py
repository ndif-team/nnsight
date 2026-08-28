"""Expose, edit, and skip a module's ``forward`` at the operation level.

A module's ``input``/``output`` are the two locations its controller hands to the
[`Interleaver`][nnsight.intervention.interleaver.Interleaver]. Everything in
between — the individual operations a ``forward`` performs — is invisible to it
because it isn't a submodule with a controller of its own.

This module makes those intermediates observable, editable, and skippable without
the interleaver knowing anything about source. The interleaver runs on one
primitive: a provider-location string and [`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle], which
serves a value to interventions and returns it back edited if one wrote to that
location. Inputs/outputs are just the two locations a controller emits; here we
add more, mid-forward:

1. Parse the module's ``forward`` and rewrite every call ``fn(*args, **kwargs)``
   into ``__nnsight_op__("source.{name}_{n}", fn, *args, **kwargs)``. At run time
   `make_op` brackets the call with [`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle] on its
   ``.input`` (before) and ``.output`` (after) — both readable/replaceable — and a
   ``.skip`` gate that can bypass the call entirely.
2. ``name`` is the called function's dotted path joined with ``_``
   (``self.act(...)`` → ``self_act``, ``torch.relu(...)`` → ``torch_relu``,
   ``dropout(...)`` → ``dropout``); ``n`` is a per-name counter in **execution
   order** (nested calls run inner-first, so the inner call is ``_0``), which is
   the order the interleaver serves values.
3. Rewrite every assignment ``x = value`` into
   ``x = __nnsight_op__("source.x_{n}", __nnsight_bind__, value)`` — the same
   bracket around an identity, so a value that is *not* a call's return (a
   ``q @ k`` product, a running state ``S = S * decay + update`` inside a loop) is
   addressable too. ``n`` is the same per-name counter calls use, so a name that
   is bound and then called (``attention_interface = ...; attention_interface(...)``)
   is ``attention_interface_0`` at the binding and ``attention_interface_1`` at the call.

A decorated ``forward`` is instrumented through its decorators: a wrapper that
calls the function it closes over is peeled and rebuilt around the instrumented
function (`decorator_chain`, `rewrap`); one that doesn't — a dispatcher that
hands the function to a lookup and calls the result — is instrumented as it is,
its closure intact (`compile_source`), so the call that actually runs is
the operation to drill into.

Installation is permanent. When an envoy is built its module's ``forward`` is
replaced by a `make_controller` closure over a single per-module `State` (see
`STATE`): it hands off ``.input``, gates on ``.skip``, runs the *body* — the
original forward, or the source-instrumented one once ``.source`` is used — and
hands off ``.output``. The controller is inert outside a trace, so later runs
work regardless of request order, and source and skip compose on one wrapper. A
module wrapped by several envoys routes to whichever interleaver is running
(`State.active`).

An [`Envoy`][nnsight.intervention.envoy.Envoy] exposes operations as ``envoy.source.{name}_{n}``, whose
``.input``/``.inputs``/``.output``/``.skip`` mirror an Envoy's own, one level finer.
"""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
import weakref
from types import CellType, CodeType, FunctionType
from typing import TYPE_CHECKING, Any, Callable, Iterator, NamedTuple

import torch

from .eproperty import eproperty
from .interleaver import Mediator
from .util import first_input, replace_first_input

if TYPE_CHECKING:
    from .envoy import Envoy

#: Global name the instrumented forward calls to bracket each operation.
OP = "__nnsight_op__"

#: Global name of the identity an instrumented assignment routes its value through.
BIND = "__nnsight_bind__"


def bind(value: Any) -> Any:
    """The callee of an assignment operation: ``x = e`` runs as
    ``x = __nnsight_op__("source.x_n", __nnsight_bind__, e)``, so the bracket
    `run_op` puts around every call serves the assigned value as ``.output``."""
    return value


#: Attribute holding a module's `State` once it has been sourced/skipped.
STATE = "__nnsight__"

#: Sentinel returned by the skip gate when no skip is pending for a location.
NO_SKIP = object()


class SourceNotAvailable(Exception):
    """A module's ``forward`` can't be source-instrumented (no source, decorated…)."""


class Compiled(NamedTuple):
    """Everything the source machinery needs about one instrumented ``forward``."""

    code: CodeType  #: the instrumented forward's code object
    names: tuple[str, ...]  #: operation labels, in execution order
    lines: dict[str, int]  #: label -> 1-based line within `source`
    source: str  #: dedented source text of the original forward


class State:
    """Per-module source/skip state, stored at ``module.__dict__[STATE]``.

    Created when a module is first sourced or skipped. The controller and op probes
    read it live on every call, so re-wrapping — or wrapping the same module in more
    than one Envoy at once — just registers another interleaver here.

    [`routes`][nnsight.intervention.source.State.routes] lists each interleaver that instrumented this module with the
    path it addresses the module by; [`active`][nnsight.intervention.source.State.active] picks the one whose trace is
    currently running (there is at most one).

    Because this state lives on the module (``module.__dict__[STATE]``), it holds
    neither the module nor its interleavers *strongly* — the module would sit in a
    reference cycle and never be freed by refcounting. The interleavers are held by
    weakref (a finished local wrapper's interleaver drops out on its own; a
    server's persistent interleaver stays, so the same module serves request after
    request), and [`body`][nnsight.intervention.source.State.body] is the *unbound* forward
    (a plain function taking ``self``) rather than a bound method that would pin it.
    """

    __slots__ = ("routes", "body", "sourced", "compiled")

    def __init__(self, body: Callable) -> None:
        #: (weakref to interleaver, the path it addresses this module by, and the
        #: three locations the handoff uses -- built once, not per call)
        self.routes: list[tuple[Any, str, tuple[str, str, str]]] = []
        self.body = body  #: unbound forward to run when not skipped
        self.sourced = False  #: whether body is the source-instrumented forward
        self.compiled: Compiled | None = None  #: the instrumented forward's `Compiled`, once sourced

    def register(self, interleaver: Any, path: str) -> None:
        """Record that ``interleaver`` reaches this module at ``path``."""
        self.routes = [r for r in self.routes if r[0]() not in (None, interleaver)]
        locations = (f"{path}.input", f"{path}.skip", f"{path}.output")
        self.routes.append((weakref.ref(interleaver), path, locations))

    def active(self) -> "tuple[Any, str, tuple] | tuple[None, None, tuple]":
        """The (interleaver, path, locations) whose trace is running now and has
        workers, or Nones — the gate every module call and every op passes.

        A plain list walked on every module call, so it has to stay cheap: one
        entry in all but the shared-module case, and a dead weakref costs a test.
        ``busy`` is False for a run with no workers (a vLLM step with no nnsight
        requests in it), where every handoff would be a no-op.
        """
        for ref, path, locations in self.routes:
            interleaver = ref()
            if interleaver is not None and interleaver.interleaving and interleaver.busy:
                return interleaver, path, locations
        return None, None, ()


#: Instrumented forwards, memoized per original ``forward`` code object.
FORWARD_CACHE: dict[CodeType, "Compiled"] = {}


# ---------------------------------------------------------------------------
# Compiling the instrumented forward
# ---------------------------------------------------------------------------


class Instrument(ast.NodeTransformer):
    """Rewrite every call into ``__nnsight_op__(location, fn, *args, **kwargs)`` and
    every assignment into ``target = __nnsight_op__(location, __nnsight_bind__, value)``.

    Numbers occurrences in execution order: a call's arguments (and an
    assignment's value) are visited *before* the node is assigned its counter, so
    ``f(f(x))`` gives the inner ``f`` ``f_0`` and the outer ``f_1``, and
    ``h = relu(x)`` gives ``relu_0`` then ``h_0``. Calls and assignments share
    one counter per name.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.names: list[str] = []
        # label -> 1-based line within the (dedented) source, captured before
        # increment_lineno shifts it to file coordinates.
        self.lines: dict[str, int] = {}

    @staticmethod
    def dotted(expr: ast.expr) -> tuple[list[str], bool]:
        """The attribute chain of ``expr`` and whether it is rooted in a name.

        ``self.a.b`` → ``(['self', 'a', 'b'], True)``; ``x[i].y`` → ``(['x', 'y'], True)``
        (a subscript says where in the object, not what the object is called);
        ``(a @ b).sum`` → ``(['sum'], False)``.
        """
        parts = []
        while True:
            if isinstance(expr, ast.Attribute):
                parts.append(expr.attr)
                expr = expr.value
            elif isinstance(expr, ast.Subscript):
                expr = expr.value
            elif isinstance(expr, ast.Name):
                parts.append(expr.id)
                return parts[::-1], True
            else:
                return parts[::-1], False

    def wrap(self, name: str, fn: ast.expr, args: list, keywords: list, node: ast.AST) -> ast.Call:
        """``__nnsight_op__("source.{name}_{n}", fn, *args, **keywords)`` at ``node``."""
        occurrence = self.counts.get(name, 0)
        self.counts[name] = occurrence + 1
        label = f"{name}_{occurrence}"
        self.names.append(label)
        self.lines[label] = node.lineno
        # Copy the node's source location onto the wrapper so an exception raised
        # inside the instrumented forward points at the real line; a locationless
        # node would take increment_lineno's raw offset instead.
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=OP, ctx=ast.Load()),
                args=[ast.Constant(value=f"source.{label}"), fn, *args],
                keywords=keywords,
            ),
            node,
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name) and node.func.id == "super" and not (node.args or node.keywords):
            # Zero-argument super() reads `__class__` and the first argument off
            # the frame that calls it; from inside __nnsight_op__ there is neither.
            return node
        # Descend first so nested calls are numbered before this (outer) one. The
        # wrapper returned is not re-visited, so it is never counted as an op.
        self.generic_visit(node)
        parts, _ = self.dotted(node.func)
        return self.wrap("_".join(parts) or "call", node.func, node.args, node.keywords, node)

    def bound(self, target: ast.expr, value: ast.expr, node: ast.AST) -> ast.expr:
        """``value`` routed through the identity under the target's name.

        ``a, b = e1, e2`` binds each name its own value, so each element gets its
        own op (the tuple is still built before any name is bound, so ``a, b = b, a``
        still swaps). Any other unpacking, and a target with no name to label
        (``f()[0] = v``), is left as it is.
        """
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            elements = [*target.elts, *value.elts]
            if len(target.elts) == len(value.elts) and not any(isinstance(e, ast.Starred) for e in elements):
                value.elts = [self.bound(t, v, node) for t, v in zip(target.elts, value.elts)]
            return value
        parts, rooted = self.dotted(target)
        if not rooted:
            return value
        return self.wrap("_".join(parts), ast.Name(id=BIND, ctx=ast.Load()), [value], [], node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        # The value runs before the targets (`x[f(i)] = g()` evaluates g, then x,
        # then f), so visit it first to keep the counters in execution order.
        node.value = self.visit(node.value)
        node.targets = [self.visit(target) for target in node.targets]
        if len(node.targets) == 1:  # `a = b = v` binds one value to two names; left alone
            node.value = self.bound(node.targets[0], node.value, node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if node.value is not None:  # the annotation is not evaluated at run time
            node.value = self.bound(node.target, self.visit(node.value), node)
        return node


#: Name of the shell every function is recompiled inside (see `compile_source`).
SHELL = "__nnsight_shell__"


def source_tree(code: CodeType) -> tuple[ast.Module, int, str]:
    """Parse the function ``code`` was compiled from: its module AST, first line, and text.

    From the code object, not the function: given a function, ``inspect`` follows
    ``__wrapped__`` and hands back the decorated function's source instead of the
    wrapper's. Raises [`SourceNotAvailable`][nnsight.intervention.source.SourceNotAvailable] when there is nothing to read.
    """
    try:
        lines, start = inspect.getsourcelines(code)
    except (OSError, TypeError) as error:
        raise SourceNotAvailable("callable source is unavailable") from error
    source = textwrap.dedent("".join(lines))
    return ast.parse(source), start, source


def compile_source(func: Callable) -> Compiled:
    """Parse, instrument, and compile a Python ``func``, or raise.

    The definition is compiled inside a shell function whose parameters are
    ``func``'s free variables: recompiled at module level they would become
    globals and break; as a nested definition they compile as free variables
    again, and `instrument` attaches the original cells. The caller has peeled
    ``func``'s decorators and rebuilds them around the result, so the ``@`` lines
    (which ``getsourcelines`` includes) are dropped rather than doubled.
    """
    code_object = func.__code__
    tree, start, source = source_tree(code_object)
    definition = tree.body[0]
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise SourceNotAvailable("callable is not a plain function")
    definition.decorator_list = []

    rewriter = Instrument()
    rewriter.visit(tree)
    shell = ast.FunctionDef(
        name=SHELL,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in code_object.co_freevars],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[definition, ast.Return(value=ast.Name(id=definition.name, ctx=ast.Load()))],
        decorator_list=[],
    )
    tree.body[0] = ast.copy_location(shell, definition)
    # Line up line numbers with the original file so tracebacks read right.
    ast.increment_lineno(tree, start - 1)
    ast.fix_missing_locations(tree)

    module = compile(tree, code_object.co_filename, "exec")
    shell_code = next(c for c in module.co_consts if isinstance(c, CodeType) and c.co_name == SHELL)
    # By the code's own name, not `func.__name__`: functools.wraps renames the
    # wrapper after the function it wraps, but its `def` line does not change.
    code = next(c for c in shell_code.co_consts if isinstance(c, CodeType) and c.co_name == code_object.co_name)
    return Compiled(code, tuple(rewriter.names), rewriter.lines, source)


def compiled(func: Callable) -> Compiled:
    """Cached `compile_source`, keyed by ``func``'s code object."""
    key = getattr(func, "__code__", None)
    if key is None:
        raise SourceNotAvailable("callable has no Python source (builtin or C function)")
    if key not in FORWARD_CACHE:
        FORWARD_CACHE[key] = compile_source(func)
    return FORWARD_CACHE[key]


def peel_index(wrapper: Callable) -> int | None:
    """Index of the closure cell holding the function ``wrapper`` decorates, or ``None``.

    A decorator's wrapper keeps the function it decorates in a closure cell and
    calls it, so the cell is found from the wrapper's own source: the free names
    it calls directly (``fn(*args, **kwargs)``) that hold a Python function.
    Exactly one is the decorated function. None means the wrapper doesn't call
    what it closes over — a dispatcher that hands the function to a lookup and
    calls the result (transformers' experts wrapper, which runs a fused kernel
    instead of the eager loop it wraps) — and several is ambiguous; either way
    the wrapper is instrumented as it is, so the call that actually runs is what
    shows up. Matching by ``__wrapped__`` would peel the dispatcher too.
    """
    code = getattr(wrapper, "__code__", None)
    if code is None or not code.co_freevars:
        return None
    try:
        tree, _, _ = source_tree(code)
    except SourceNotAvailable:
        return None
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    candidates = [
        index
        for index, (name, cell) in enumerate(zip(code.co_freevars, wrapper.__closure__))
        if name in called and isinstance(getattr(cell, "cell_contents", None), FunctionType)
    ]
    return candidates[0] if len(candidates) == 1 else None


def decorator_chain(func: Callable) -> tuple[Callable, list[tuple[Callable, int]]]:
    """Peel ``func``'s decorators: the innermost function and the ``(wrapper, cell)``
    chain, outermost first, that `rewrap` rebuilds around its replacement."""
    chain: list[tuple[Callable, int]] = []
    seen = {func}
    while (index := peel_index(func)) is not None:
        chain.append((func, index))
        func = func.__closure__[index].cell_contents
        if func in seen:  # a closure that calls itself
            break
        seen.add(func)
    return func, chain


def rewrap(chain: list[tuple[Callable, int]], innermost: Callable) -> Callable:
    """Rebuild ``chain``'s decorators around ``innermost``, inside out.

    Each wrapper is rebuilt with a fresh closure rather than having its cell
    assigned: the wrapper is the *class*'s attribute, shared by every instance in
    the process, so mutating its cell in place would redirect models nobody is
    tracing.
    """
    for wrapper, index in reversed(chain):
        cells = list(wrapper.__closure__)
        cells[index] = CellType(innermost)
        innermost = function_like(wrapper, wrapper.__code__, tuple(cells))
    return innermost


def function_like(fn: Callable, code: CodeType, closure: tuple | None, **globals_: Any) -> Callable:
    """A new function with ``fn``'s globals, defaults and names but ``code`` and ``closure``."""
    new = FunctionType(code, {**fn.__globals__, **globals_}, fn.__name__, fn.__defaults__, closure)
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__qualname__ = fn.__qualname__
    return new


def instrument(fn: Callable, op: Callable) -> tuple[Callable, Compiled]:
    """A source-instrumented replacement for ``fn``, calling ``op`` per operation,
    and its `Compiled` — or raise [`SourceNotAvailable`][nnsight.intervention.source.SourceNotAvailable].

    Peels ``fn``'s decorators, instruments the function they wrap, and rebuilds
    them around it so their behaviour still runs. The instrumented copy shares
    the function's closure cells, matched by name (the shell can order them
    differently), so a wrapper keeps reaching what it closed over. A bound method
    is rebuilt from its function and re-bound to the same instance.
    """
    receiver = fn.__self__ if inspect.ismethod(fn) else None
    inner, chain = decorator_chain(fn.__func__ if receiver is not None else fn)
    result = compiled(inner)
    cells = dict(zip(inner.__code__.co_freevars, inner.__closure__ or ()))
    closure = tuple(cells[name] for name in result.code.co_freevars) or None
    built = rewrap(chain, function_like(inner, result.code, closure, **{OP: op, BIND: bind}))
    return (built.__get__(receiver) if receiver is not None else built), result


# ---------------------------------------------------------------------------
# The installed forward: controller (skip gate + body) and per-op probes
# ---------------------------------------------------------------------------


def run_op(interleaver: Any, base: str, fn: Callable, args: tuple, kwargs: dict) -> Any:
    """Bracket one operation at location ``base``: input, skip, (recursive) run, output.

    Reports/replaces ``.input``, honors a ``.skip`` gate, and reports/replaces
    ``.output`` — the same three handles a module's controller emits, one level finer.
    Between them, if a worker asked to drill into this op (``base`` is a key in
    [`Interleaver.sourced`][nnsight.intervention.interleaver.Interleaver.sourced]), the raw ``fn`` is offered over ``{base}.fn`` so
    the worker can hand back a source-instrumented copy (cached in
    [`Interleaver.sourced`][nnsight.intervention.interleaver.Interleaver.sourced] for later fires); that copy runs in place of ``fn``,
    making *its* operations addressable under ``{base}.source.*`` — recursively.
    """
    args, kwargs = interleaver.handle(f"{base}.input", (args, kwargs))
    skipped = interleaver.handle(f"{base}.skip", NO_SKIP)
    if skipped is not NO_SKIP:
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


def make_op(locate: Callable[[], tuple]) -> Callable:
    """Build the ``__nnsight_op__`` an instrumented function calls at each operation.

    ``locate`` answers "which trace is running, and under what path?" — for a
    module's forward, its live `State` (so re-wrapping and multiple wrappers just
    work); for a drilled-into callable, the interleaver and op path it was drilled
    from. With no trace running the op calls straight through; inside one it
    brackets the call via `run_op` under ``{path}.{location}``.
    """

    def op(location: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        interleaver, path = locate()
        if interleaver is None:
            return fn(*args, **kwargs)
        return run_op(interleaver, f"{path}.{location}", fn, args, kwargs)

    return op


def run_body(state: "State", module: Any, args: tuple, kwargs: dict) -> Any:
    """Run the module's body, honouring accelerate's device-alignment hook.

    ``accelerate.add_hook_to_module`` installs alignment by replacing
    ``module.forward`` (instance ``__dict__``) and keeping the real forward on
    ``module._old_forward``. We install our controller into that same slot, so its
    wrapper is gone and ``pre_forward``/``post_forward`` would never run -- which
    silently breaks any model sharded across devices, because the inter-module
    tensor moves are exactly what those do. ``_hf_hook`` stays attached either way,
    so the omission is invisible.

    Bracketing the body here restores it, and works for both the original forward
    and the source-instrumented one.
    """
    # An instance attribute, so read it off `__dict__`: `getattr` on a module
    # without one goes through `nn.Module.__getattr__` and raises, on every call.
    hook = module.__dict__.get("_hf_hook")
    if hook is None:
        return state.body(module, *args, **kwargs)
    args, kwargs = hook.pre_forward(module, *args, **kwargs)
    output = state.body(module, *args, **kwargs)
    return hook.post_forward(module, output)


def make_controller(module: Any, state: "State") -> Callable:
    """Build the forward installed on an instrumented module: the module's handoff.

    This is where a module's ``.input``, ``.skip`` and ``.output`` reach the
    interleaver -- the same three handles [`run_op`][nnsight.intervention.source.run_op] emits for an operation,
    one level up. Being the forward rather than a hook keeps the module on
    PyTorch's fast call path, and it runs inside the module's own hooks, so a
    runtime that keeps collectives in them sees the pre-collective value here --
    which is what its [`Fragments`][nnsight.intervention.fragments.Fragments] describes.
    ``body`` is the (unbound) original forward or, once sourced, the instrumented
    one, and a ``.skip`` bypasses it entirely (and, if sourced, all its ops).

    Holds the module by weakref: the module owns this controller (as its
    ``forward``), so a strong back-reference would cycle. ``functools.wraps``
    preserves the forward's signature, which ``generate()`` introspects to decide
    whether to pass ``attention_mask``/``position_ids``.
    """
    module_ref = weakref.ref(module)
    original = type(module).forward  # unbound; used only for signature metadata

    @functools.wraps(original)
    def controller(*args: Any, **kwargs: Any) -> Any:
        module = module_ref()
        interleaver, _, locations = state.active()
        if interleaver is None:
            return run_body(state, module, args, kwargs)
        handle = interleaver.handle
        args, kwargs = handle(locations[0], (args, kwargs))
        output = handle(locations[1], NO_SKIP)  # the skip gate
        if output is NO_SKIP:
            output = run_body(state, module, args, kwargs)
        return handle(locations[2], output)

    return controller


def install_controller(envoy: "Envoy") -> State:
    """Install the controller forward on ``envoy``'s module once; (re)bind and return
    its `State`.

    Installed directly into the module's ``__dict__`` (shadowing the class method
    for ``__call__``) and left there permanently — inert outside a trace. The body
    defaults to the original forward; [`install_source`][nnsight.intervention.source.install_source] upgrades it.
    """
    module = envoy._module
    state = module.__dict__.get(STATE)
    if state is None:
        # Store the *unbound* forward (a bound method would pin the module, and the
        # module holds this state — a cycle); the controller supplies the module.
        state = State(type(module).forward)
        module.__dict__[STATE] = state
        module.__dict__["forward"] = make_controller(module, state)
    # Register this envoy's interleaver (weakly) under its path, so several envoys
    # can share the module and each routes to its own trace.
    state.register(envoy.interleaver, envoy.path)
    return state


def install_source(envoy: "Envoy") -> Compiled:
    """Source-instrument ``envoy``'s module and install the controller.

    Returns the module's [`Compiled`][nnsight.intervention.source.Compiled]. Upgrades the controller's body to the
    instrumented forward (built once per module, from code cached per code object).
    """
    state = install_controller(envoy)
    if not state.sourced:
        # The class's forward: unbound (the controller passes the module), and a
        # bound method would pin it.
        state.body, state.compiled = instrument(
            type(envoy._module).forward, make_op(lambda: state.active()[:2])
        )
        state.sourced = True
    return state.compiled


# ---------------------------------------------------------------------------
# User-facing views
# ---------------------------------------------------------------------------


class SourceEnvoy:
    """A single operation inside a module's ``forward``, e.g. ``source.torch_relu_0``.

    You never construct one directly; you reach it by indexing a [`Source`][nnsight.intervention.source.Source]
    with an operation's ``{callable}_{occurrence}`` name (``envoy.source.torch_relu_0``).
    It is the operation-level analogue of an
    [`Envoy`][nnsight.intervention.envoy.Envoy]: where an Envoy exposes a
    *submodule's* ``.input``/``.output``, a ``SourceEnvoy`` exposes those same
    handles for a *single call* the ``forward`` makes — one level finer.

    Inside a trace, each handle both reads and writes the live value:

    * `output` — the operation's return value.
    * `input` — the operation's first argument.
    * `inputs` — the operation's full ``(args, kwargs)``.
    * `skip` — bypass the call, substituting a value for its output.
    * `source` — drill *into* the called function, exposing its own
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
                out = attn.attention_interface_1.source.attn_output_transpose_0.output.save()
    """

    def __init__(
        self,
        envoy: "Envoy",
        name: str,
        path: str,
        source: str,
        line: int,
    ) -> None:
        self.envoy = envoy
        self.name = name
        self.path = path
        self.text = source
        self.line = line

    @property
    def source(self) -> "Source":
        """Drill into the called function, exposing *its* operations recursively.

        Returns a [`Source`][nnsight.intervention.source.Source] over the function this operation calls, so its
        internal operations become addressable as
        ``...source.{name}.source.{inner}`` — with the same
        ``.input``/``.output``/``.inputs``/``.skip``/``.source`` handles, to any
        depth.

        Only available **inside a trace**: the called function is resolved from the
        live value flowing through the call at run time (a call target is often a
        local variable, e.g. an attention implementation, so it can't be found
        statically). Raises [`SourceNotAvailable`][nnsight.intervention.source.SourceNotAvailable] if the target has no
        recoverable Python source (a builtin/C function) or is itself a submodule
        (call ``.source`` on that submodule directly instead).

        Examples:
            >>> with model.trace(prompt):
            ...     attn = model.transformer.h[0].attn.source.attention_interface_1
            ...     out = attn.source.attn_output_transpose_0.output.save()
        """
        interleaver = self.envoy.interleaver
        if not interleaver.interleaving:
            raise SourceNotAvailable(
                "recursive `.source` is only available inside a trace"
            )
        if self.path not in interleaver.sourced:
            # Mark requested (None placeholder), then park until the operation fires
            # and the model side hands back the live callable (see run_op). Build
            # and cache its instrumented copy so later fires this run reuse it.
            interleaver.sourced[self.path] = None
            fn = Mediator.value(f"{self.path}.fn")
            if isinstance(fn, torch.nn.Module):
                raise SourceNotAvailable(
                    f"{self.name!r} calls a submodule; call `.source` on that "
                    f"submodule directly instead of drilling into the call"
                )
            if fn is bind:
                raise SourceNotAvailable(
                    f"{self.name!r} is an assignment, not a call; there is no "
                    f"function to drill into"
                )
            interleaver.sourced[self.path] = instrument(  # raises SourceNotAvailable if it can't
                fn, make_op(lambda: (interleaver, self.path) if interleaver.interleaving else (None, None))
            )
        return Source(self.envoy, prefix=self.path, compiled=interleaver.sourced[self.path][1])

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
        common case of a single leading argument, `input` is more direct.
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

        A convenience view over `inputs` for the usual single-argument call.
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
        source_lines = self.text.split("\n")
        # .line is 1-based; -2 lands in the body frame where the highlighted line
        # is index (line_number + 1) into source_lines.
        line_number = self.line - 2
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
    therefore ``torch_relu_0`` and ``torch_relu_1``. Every assignment is an
    operation too, named ``{target}_{occurrence}`` — ``h = q @ k`` gives
    ``h_0``, whose ``.output`` is the assigned value — so a value that is not
    a call's return (a matmul, a running state inside a loop) is reachable by the
    name the forward gives it. Index in with that name to get a
    [`SourceEnvoy`][nnsight.intervention.source.SourceEnvoy]::

        model.layer1.source.torch_relu_0    # -> SourceEnvoy for the relu call
        model.layer1.source.h_0        # -> SourceEnvoy for `h = ...`

    You rarely need to memorize the names: ``print(model.layer1.source)`` renders
    the whole ``forward`` with each operation labelled at its call site, and
    ``print(model.layer1.source.torch_relu_0)`` zooms in on one. Iterating a
    ``Source`` yields its operations in execution order.

    Source values are only meaningful inside a trace, and ordinary inference is
    unaffected. Requesting an operation on a ``forward`` whose source can't be
    recovered (e.g. a decorated ``forward``) raises [`SourceNotAvailable`][nnsight.intervention.source.SourceNotAvailable].

    A ``Source`` also decomposes a *called function* — reached as
    ``some_op.source`` (see [`SourceEnvoy.source`][nnsight.intervention.source.SourceEnvoy.source]) — the same way, one level
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

    def __init__(self, envoy: "Envoy", prefix: str, compiled: "Compiled") -> None:
        # `compiled` is the instrumented forward (a module's, from
        # `install_source`, or a drilled-into callable's); its operations hang off
        # `prefix`.
        self.envoy = envoy
        self.compiled = compiled
        self.prefix = f"{prefix}.source"

    @property
    def names(self) -> tuple[str, ...]:
        return self.compiled.names

    def node(self, name: str) -> SourceEnvoy:
        """A [`SourceEnvoy`][nnsight.intervention.source.SourceEnvoy] for ``name``, carrying source text for its repr."""
        return SourceEnvoy(
            self.envoy,
            name,
            f"{self.prefix}.{name}",
            self.compiled.source,
            self.compiled.lines[name],
        )

    def __getattr__(self, name: str) -> SourceEnvoy:
        """Resolve ``source.<name>`` to its [`SourceEnvoy`][nnsight.intervention.source.SourceEnvoy].

        Raises `AttributeError` for an unknown operation, listing the
        available names so a mistyped or wrong-occurrence label is easy to fix.

        Examples:
            >>> model.layer1.source.self_fc1_0  # -> SourceEnvoy
            >>> model.layer1.source.nope_0      # AttributeError: ... available: self_fc1_0, torch_relu_0, self_fc2_0
        """
        if name.startswith("__"):
            # Dunder probes (pickle, copy, IPython) are not operations; a private
            # helper's op (`_grouped_linear_0`) is.
            raise AttributeError(name)
        if name not in self.compiled.names:
            available = ", ".join(self.compiled.names) or "(none)"
            raise AttributeError(
                f"{self.prefix!r} has no operation {name!r}; available: {available}"
            )
        return self.node(name)

    def __iter__(self) -> Iterator[SourceEnvoy]:
        """Iterate the operations in execution order, yielding a [`SourceEnvoy`][nnsight.intervention.source.SourceEnvoy] each.

        Examples:
            >>> [op.name for op in model.layer1.source]
            ['self_fc1_0', 'torch_relu_0', 'self_fc2_0']
        """
        for name in self.compiled.names:
            yield self.node(name)

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
        names = self.compiled.names
        # .compiled.lines is 1-based; -2 matches the loop below, which enumerates
        # source_lines[1:] (the body) from 0.
        line_numbers = {name: self.compiled.lines[name] - 2 for name in names}
        max_name_length = max((len(name) for name in names), default=0)

        source_lines = self.compiled.source.split("\n")
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
