"""Source-level operation tracing.

Architecture
============

Each nnsight-wrapped module has one *global* :class:`SourceAccessor` that
is lazily built on the first ``.source`` access (by any Envoy / Interleaver
/ Mediator). The SourceAccessor is cached on the module itself as
``module.__source_accessor__`` so it survives any later replacement of
``module.forward`` (e.g. ``torch.compile``, accelerate hot-swap). It owns:

- The injected version of the module's forward — its AST has been
  rewritten so every call site is wrapped by a ``wrap(fn, name=...)``
  lookup that consults the per-operation hook state.
- A dict ``{op_name: OperationAccessor}`` — one entry per call site.

Each :class:`OperationAccessor` is also global per (module, op): it owns
the hook lists (``pre_hooks``, ``post_hooks``, ``fn_hooks``,
``fn_replacement``) and, for recursive source tracing, a nested
:class:`SourceAccessor` for the operation's own fn.

Per-Envoy wrappers — :class:`SourceEnvoy` and :class:`OperationEnvoy` —
sit on top of the accessors and provide the user-facing API (eproperties
for ``.input``/``.output``, pretty-printed ``.source``). Multiple Envoys
or Interleavers wrapping the same module share the same underlying
accessors; only the per-Envoy wrappers are duplicated.

Forward routing
---------------

``nnsight_forward`` (installed by :meth:`Interleaver.wrap_module`)
checks ``module.__source_accessor__`` on each call:

- If a SourceAccessor exists and ``.hooked`` is True (any
  OperationAccessor under it has any active hook), it invokes
  ``source_accessor(module, *args, **kwargs)`` which runs the injected
  forward.
- Otherwise it calls the unwrapped original via
  ``module.__nnsight_forward__`` — zero overhead for modules that
  nobody is source-tracing.

Lifetimes
---------

- ``SourceAccessor`` and ``OperationAccessor`` live as long as the module's
  ``nnsight_forward`` wrapper does (effectively the lifetime of the model).
- ``SourceEnvoy`` / ``OperationEnvoy`` live as long as their owning Envoy.
- Hooks on an OperationAccessor are one-shot and self-remove when they
  fire; they are also tracked on ``mediator.hooks`` so session cleanup
  drains them.
- ``fn_replacement`` is one-shot too: cleared after :func:`wrap_operation`
  runs once. Re-accessing ``.source`` on an OperationEnvoy reinstalls it.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from builtins import compile, exec
from collections import defaultdict
from functools import partial, wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import astor
import torch

from .hooks import (
    hooked_operation_input,
    hooked_operation_output,
)
from .interleaver import IEnvoy, Interleaver

if sys.version_info >= (3, 9):
    _ast_to_source = ast.unparse
else:
    _ast_to_source = astor.to_source


# ---------------------------------------------------------------------------
# AST-based injection (formerly inject.py)
# ---------------------------------------------------------------------------


class FunctionCallWrapper(ast.NodeTransformer):

    def __init__(self, name: str):
        self.name_index = defaultdict(int)
        self.line_numbers = {}
        self.name = name
        # Cache the name prefix to avoid repeated string operations
        self._name_prefix = f"{name}."
        # Track the line number of the first function definition encountered.
        # Only wrap calls that occur after this line (inside the function body)
        self._function_start_line = None

    def get_name(self, node: ast.Call):
        """Extract and index function name from a Call node."""
        func = node.func
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            func_name = "_".join(reversed(parts))
        else:
            func_name = "unknown"

        index = self.name_index[func_name]
        self.name_index[func_name] = index + 1
        return f"{func_name}_{index}"

    def visit_Call(self, node):
        if (
            self._function_start_line is not None
            and node.lineno <= self._function_start_line
        ):
            return self.generic_visit(node)

        self.generic_visit(node)
        func_name = self.get_name(node)
        self.line_numbers[func_name] = node.lineno - 2

        wrapped_name = f"{self._name_prefix}{func_name}"

        return ast.Call(
            func=ast.Call(
                func=ast.Name(id="wrap", ctx=ast.Load()),
                args=[node.func],
                keywords=[
                    ast.keyword(arg="name", value=ast.Constant(value=wrapped_name))
                ],
            ),
            args=node.args,
            keywords=node.keywords,
        )

    def visit_FunctionDef(self, node):
        if self._function_start_line is None:
            self._function_start_line = node.lineno
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if self._function_start_line is None:
            self._function_start_line = node.lineno
        return self.generic_visit(node)


def convert(fn: Callable, wrap: Callable, name: str):
    """Rewrite ``fn``'s AST so every call site is wrapped by ``wrap(fn, name=...)``.

    Returns the source string, a ``{op_name: line_number}`` map, and the
    compiled & executed wrapped function.
    """

    source = textwrap.dedent(inspect.getsource(fn))

    module_globals = inspect.getmodule(fn).__dict__

    tree = ast.parse(source)

    # Strip decorators — they're irrelevant to operation tracing and can fail
    # when re-executed outside the original class context (e.g. transformers'
    # @auto_docstring tries cls.__mro__ which fails on a bare function).
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []

    transformer = FunctionCallWrapper(name)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    local_namespace = {"wrap": wrap}

    global_namespace = globals().copy()
    global_namespace.update(module_globals)
    global_namespace["wrap"] = wrap

    filename = "<nnsight>"

    if isinstance(tree, ast.Module):
        code_obj = compile(tree, filename, "exec")
    else:
        code_obj = compile(_ast_to_source(tree), filename, "exec")

    exec(code_obj, global_namespace, local_namespace)

    fn = local_namespace[fn.__name__]

    return source, transformer.line_numbers, fn


# ---------------------------------------------------------------------------
# Operation wrapper — hook dispatcher built per-call-site
# ---------------------------------------------------------------------------


def wrap_operation(
    fn: Callable,
    name: str,
    bound_obj: Optional[Any] = None,
    op_accessor: Optional["OperationAccessor"] = None,
) -> Callable:
    """Wrap ``fn`` so it processes the hook lists on ``op_accessor``.

    Installed by :meth:`SourceAccessor.wrap` only when the OperationAccessor
    has at least one active hook (otherwise the call site uses ``fn`` directly).
    Hook lists are read live at call time so hooks registered after wrapper
    creation are still seen.

    ``fn_replacement`` is one-shot: cleared after the operation completes so
    subsequent forward passes fall back to the original ``fn`` unless a new
    replacement is installed (typically by re-accessing ``.source`` on the
    matching :class:`OperationEnvoy`).
    """

    @wraps(fn)
    def inner(*args, **kwargs):

        actual_fn = (
            op_accessor.fn_replacement
            if op_accessor.fn_replacement is not None
            else fn
        )

        # Consume the one-shot replacement *now*, before invoking actual_fn.
        # Clearing only at the end would race with the worker thread, which
        # may set ``fn_replacement`` for the next forward pass while
        # ``actual_fn`` is still running here (the worker resumes from
        # mid-wrap_op after a hook delivers a value). An end-of-call clear
        # would silently overwrite that next-step setup.
        op_accessor.fn_replacement = None

        for hook in list(op_accessor.fn_hooks):
            actual_fn = hook(actual_fn)

        for hook in list(op_accessor.pre_hooks):
            result = hook((args, kwargs))
            if result is not None:
                args, kwargs = result

        if not inspect.ismethod(actual_fn) and bound_obj is not None:
            value = actual_fn(bound_obj, *args, **kwargs)
        else:
            value = actual_fn(*args, **kwargs)

        for hook in list(op_accessor.post_hooks):
            result = hook(value)
            if result is not None:
                value = result

        return value

    return inner


# ---------------------------------------------------------------------------
# Global accessors — per-module / per-operation
# ---------------------------------------------------------------------------


class OperationAccessor:
    """Global hook state for a single call site inside a module's forward.

    Exactly one instance per (module, op) pair, owned by the parent
    :class:`SourceAccessor`. Hook registrations from any Envoy /
    Interleaver / Mediator that touches this operation land on the *same*
    accessor — by design, so multiple consumers can coexist over the
    module's lifetime.

    Hook lists (read live by :func:`wrap_operation` at call time):

    - ``pre_hooks`` — appended by :func:`operation_input_hook`. Each is
      called with ``(args, kwargs)``; non-``None`` return replaces them.
    - ``post_hooks`` — appended by :func:`operation_output_hook`. Each is
      called with the return value; non-``None`` return replaces it.
    - ``fn_hooks`` — appended by :func:`operation_fn_hook` for recursive
      source tracing. Each receives the current fn and returns a
      (possibly replaced) fn.
    - ``fn_replacement`` — a one-shot fn replacement installed by
      :attr:`OperationEnvoy.source`. When set, :func:`wrap_operation` uses
      it in place of the original fn for one call, then clears it.

    All input/output/fn hooks are one-shot and self-remove when they
    fire. The ``hooked`` property is True if any list is non-empty;
    :class:`SourceAccessor.wrap` checks it to take the zero-overhead
    fast path for unhooked sites.
    """

    def __init__(self, name: str, source: str, line_number: int):
        """Initialize an OperationAccessor.

        Args:
            name: Fully-qualified path of the operation
                (e.g. ``"model.transformer.h.0.attn.split_1"``).
            source: Source code of the enclosing module's forward
                (used by ``__str__`` for pretty-printing).
            line_number: Line number of the operation in ``source``.
        """
        self.path = name
        self.source_code = source
        self.line_number = line_number

        self.pre_hooks: List[Callable] = []
        self.post_hooks: List[Callable] = []
        self.fn_hooks: List[Callable] = []
        self.fn_replacement: Optional[Callable] = None

        # Nested SourceAccessor for recursive source tracing on this op's fn.
        # Built on first access in :attr:`OperationEnvoy.source` and reused
        # for the lifetime of the parent module (across Envoys / sessions).
        self._source_accessor: Optional["SourceAccessor"] = None

    @property
    def hooked(self) -> bool:
        """True if the op has any active hook or a pending fn replacement."""
        return bool(
            self.pre_hooks
            or self.post_hooks
            or self.fn_hooks
            or self.fn_replacement is not None
        )

    def __str__(self):
        source_lines = self.source_code.split("\n")
        start_idx = max(0, self.line_number - 5)
        end_idx = min(len(source_lines) - 1, self.line_number + 8)

        highlighted_lines = [self.path + ":\n"]

        if start_idx != 0:
            highlighted_lines.append("    ....")

        for i in range(start_idx, end_idx):
            line = source_lines[i]
            if i == self.line_number + 1:
                highlighted_lines.append(f"    --> {line[4:]} <--")
            else:
                highlighted_lines.append("    " + line)

        if end_idx != len(source_lines) - 1:
            highlighted_lines.append("    ....")

        return "\n".join(highlighted_lines)


class SourceAccessor:
    """Global injected-forward + operation accessor map for one fn.

    Built once on first ``.source`` access for a module (or for an
    operation's fn, in the recursive case) and cached. Subsequent
    accesses — even from different Envoys / Interleavers — reuse the
    same accessor.

    The injected forward is **not** written onto the module. Instead,
    ``nnsight_forward`` (installed by :meth:`Interleaver.wrap_module`)
    branches on ``module.__source_accessor__``: if present, it calls the
    accessor; otherwise it calls the original ``__nnsight_forward__``
    directly. This keeps the non-source path zero-overhead.
    """

    def __init__(self, fn: Callable, path: str):
        """Inject ``fn`` and build an OperationAccessor for each call site.

        Args:
            fn: Unwrapped original function whose AST should be rewritten.
                For modules this is found by :func:`resolve_true_forward`;
                for recursive source it is the op's own fn (delivered via
                :func:`operation_fn_hook`).
            path: Dotted prefix for operation names (e.g.
                ``"model.transformer.h.0.attn"``).
        """
        self.path = path
        source, line_numbers, injected = convert(fn, self.wrap, path)

        self.source = source
        self.line_numbers = line_numbers
        self._forward = injected

        self.operations: Dict[str, OperationAccessor] = {}
        for op_short_name, line in line_numbers.items():
            full_name = f"{path}.{op_short_name}" if path else op_short_name
            self.operations[full_name] = OperationAccessor(
                full_name, source, line
            )

    def wrap(self, fn: Callable, **kwargs) -> Callable:
        """Per-call-site dispatcher baked into the injected forward.

        Fast path: return ``fn`` unchanged when the op's accessor has no
        hooks. Lazy path: build a wrapper via :func:`wrap_operation` that
        runs the hook lists at call time.
        """
        name = kwargs["name"]

        op = self.operations.get(name)
        if op is None or not op.hooked:
            return fn

        bound_obj = (
            fn.__self__
            if inspect.ismethod(fn) and getattr(fn, "__name__", None) != "forward"
            else None
        )
        return wrap_operation(fn, name=name, bound_obj=bound_obj, op_accessor=op)

    @property
    def hooked(self) -> bool:
        """True if any OperationAccessor under this SourceAccessor is hooked.

        Provided for introspection; not load-bearing on the forward routing
        path. ``nnsight_forward`` routes through the SourceAccessor whenever
        it exists (regardless of ``hooked``), since per-op hooks may register
        mid-forward and the injected ``wrap`` closure already has a per-op
        fast path for unhooked sites.
        """
        return any(op.hooked for op in self.operations.values())

    def rebind(self, fn: Callable) -> None:
        """Re-inject against ``fn`` while preserving OperationAccessor state.

        Called by :meth:`Envoy._update` when a module is replaced (typically
        on dispatch — meta-tensor module swapped for the loaded one). The new
        ``fn`` is expected to share the source code of the old (same class),
        so operation names line up; their hook lists, ``fn_replacement``, and
        nested SourceAccessors are kept intact, so any pre-existing
        OperationEnvoy / SourceEnvoy references remain valid.
        """
        source, line_numbers, injected = convert(fn, self.wrap, self.path)
        self.source = source
        self.line_numbers = line_numbers
        self._forward = injected

        for op_short_name, line in line_numbers.items():
            full_name = (
                f"{self.path}.{op_short_name}" if self.path else op_short_name
            )
            existing = self.operations.get(full_name)
            if existing is None:
                self.operations[full_name] = OperationAccessor(
                    full_name, source, line
                )
            else:
                existing.line_number = line
                existing.source_code = source

    def __call__(self, *args, **kwargs):
        """Invoke the injected forward.

        Called by ``nnsight_forward`` when ``hooked`` is True. The first
        positional argument should be the module (``self`` of the original
        forward method), since the injected fn is unbound.
        """
        return self._forward(*args, **kwargs)

    def __iter__(self):
        """Yield each operation's short name, deduplicated by line number."""
        seen_lines = set()
        for name, line_number in self.line_numbers.items():
            if line_number in seen_lines:
                continue
            seen_lines.add(line_number)
            yield name

    def __str__(self):
        """Pretty-print the source with operation names at their call sites."""
        max_name_length = (
            max(len(name) for name in self.line_numbers.keys())
            if self.line_numbers
            else 0
        )

        source_lines = self.source.split("\n")
        formatted_lines = [
            " " * (max_name_length + 6) + "* " + source_lines[0]
        ]

        operations_by_line: Dict[int, List[str]] = {}
        for name, line_number in self.line_numbers.items():
            operations_by_line.setdefault(line_number, []).append(name)

        for i, line in enumerate(source_lines[1:]):
            line_number = i

            if line_number in operations_by_line:
                operations = operations_by_line[line_number]

                first_op = operations[0]
                line_prefix = f" {first_op:{max_name_length}} ->{line_number:3d} "
                formatted_lines.append(f"{line_prefix}{line}")

                if len(operations) > 1:
                    for op in operations[1:]:
                        continuation_prefix = f" {op:{max_name_length}} ->  + "
                        formatted_lines.append(
                            f"{continuation_prefix}{' ' * (len(line) - len(line.lstrip()))}..."
                        )
            else:
                line_prefix = " " * (max_name_length + 4) + f"{line_number:3d} "
                formatted_lines.append(f"{line_prefix}{line}")

        return "\n".join(formatted_lines)


def resolve_true_forward(module: torch.nn.Module) -> Callable:
    """Find the unwrapped fn whose AST should be injected.

    A module's ``forward`` may have been wrapped by accelerate
    (``module.forward = partial(new_forward, module)``, which calls
    ``module._old_forward(*args, **kwargs)``) or by nnsight
    (``module.forward = nnsight_forward``, which calls
    ``module.__nnsight_forward__(module, *args, **kwargs)``). In both
    cases the *true* forward — the user's actual compute — lives one
    level deeper.

    - Accelerate: ``module._old_forward`` (often a bound method).
    - nnsight: ``module.__nnsight_forward__`` (set by :meth:`Interleaver.wrap_module`).
    - Plain module: ``type(module).forward``.

    Returns an unbound function suitable for re-execution with the module
    as the first positional argument (which is how the injected fn is
    called by :class:`SourceAccessor.__call__`).
    """
    if hasattr(module, "_old_forward"):
        fn = module._old_forward
        if hasattr(fn, "__func__"):
            return fn.__func__
        if isinstance(fn, partial):
            return fn.func
        return fn
    if hasattr(module, "__nnsight_forward__"):
        return module.__nnsight_forward__
    return type(module).forward


def get_or_create_source_accessor(module: torch.nn.Module) -> SourceAccessor:
    """Return the module's :class:`SourceAccessor`, building it on first access.

    The accessor is cached on ``module.__source_accessor__`` directly so it
    survives any replacement of ``module.forward`` (e.g. by ``torch.compile``,
    accelerate's hot-swap, or other wrappers that re-bind forward after
    nnsight has wrapped the module). Subsequent calls — even from different
    Envoys / Interleavers / Mediators — return the same instance.
    """
    accessor = getattr(module, "__source_accessor__", None)
    if accessor is None:
        fn = resolve_true_forward(module)
        path = getattr(module, "__path__", "") or ""
        accessor = SourceAccessor(fn, path)
        module.__source_accessor__ = accessor
    return accessor


# ---------------------------------------------------------------------------
# Per-Envoy user-facing wrappers
# ---------------------------------------------------------------------------


class OperationEnvoy:
    """Per-Envoy proxy for a single call site.

    Implements :class:`IEnvoy` so it can back ``eproperty`` descriptors
    for ``.output``, ``.input``, ``.inputs``, and ``.source`` (recursive
    source tracing). All hook state lives on the underlying
    :class:`OperationAccessor`; this class is a thin per-Envoy view that
    routes hook registration to the shared accessor.
    """

    def __init__(
        self,
        accessor: OperationAccessor,
        interleaver: Optional[Interleaver] = None,
    ):
        """Initialize an OperationEnvoy.

        Args:
            accessor: The shared :class:`OperationAccessor` that owns this
                operation's hook state. Multiple OperationEnvoys (from
                different parent Envoys) may share the same accessor.
            interleaver: The :class:`Interleaver` whose ``current`` mediator
                should receive the registered hook callbacks.
        """
        self.accessor = accessor
        self.path = accessor.path
        self.interleaver = interleaver
        self._source: Optional["SourceEnvoy"] = None

    def __str__(self):
        return str(self.accessor)

    @hooked_operation_output()
    def output(self) -> Any:
        """Get the output of this operation.

        Examples:
            >>> with model.trace("Hello World"):
            ...     attn = model.transformer.h[0].attn.source.attention_interface_0.output.save()
        """

    @hooked_operation_input()
    def inputs(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Get the inputs to this operation as ``(args, kwargs)``."""

    @hooked_operation_input()
    def input(self) -> Any:
        """Get the first positional (or first keyword) input to this operation."""

    @input.preprocess
    def input(self, value):
        return [*value[0], *value[1].values()][0]

    @input.postprocess
    def input(self, value):
        inputs = self.inputs
        return (value, *inputs[0][1:]), inputs[1]

    @property
    def source(self) -> "SourceEnvoy":
        """Get the source of this operation's fn for recursive source tracing.

        On first access for the *underlying op* (any Envoy), builds a nested
        :class:`SourceAccessor` on the :class:`OperationAccessor` and uses
        the fn-hook + swap dance to substitute the injected fn into the
        currently-running operation. The injected fn is also installed as
        ``fn_replacement`` so the operation wrapper picks it up.

        Because ``fn_replacement`` is one-shot (see :func:`wrap_operation`),
        subsequent ``.source`` accesses re-install it from the cached
        nested accessor.
        """
        # Local import to avoid circular dependency at module import time.
        from .hooks import operation_fn_hook

        accessor = self.accessor
        mediator = self.interleaver.current

        if accessor._source_accessor is None:
            # First-ever recursive .source access for this operation.
            # Use the fn-hook to intercept the currently-bound fn.
            operation_fn_hook(mediator, accessor)
            fn = mediator.request(f"{self.path}.fn")

            if isinstance(fn, torch.nn.Module):
                msg = (
                    f"Don't call .source on a module ({getattr(fn, '__path__', '')}) "
                    f"from within another .source. Call it directly with: "
                    f"{getattr(fn, '__path__', '')}.source"
                )
                raise ValueError(msg)

            nested = SourceAccessor(fn, self.path)
            accessor._source_accessor = nested

            # Substitute the injected fn into the currently-running op
            # (handled by :meth:`Mediator.handle_swap_event` and the
            # fn-hook's mediator.handle call).
            mediator.swap(f"{self.path}.fn", nested._forward)
            accessor.fn_replacement = nested._forward
        else:
            nested = accessor._source_accessor
            # `fn_replacement` is one-shot — reinstall directly from the
            # cached nested accessor. The op's wrap_operation will pick it
            # up on the next call within this forward pass.
            if accessor.fn_replacement is None:
                accessor.fn_replacement = nested._forward

        if self._source is None:
            self._source = SourceEnvoy(nested, self.interleaver)

        return self._source


class SourceEnvoy:
    """Per-Envoy user-facing wrapper around a :class:`SourceAccessor`.

    Provides named attribute access to :class:`OperationEnvoy` instances
    (one per call site) and pretty-prints by delegating to the accessor.
    Multiple SourceEnvoys may wrap the same accessor — they share hook
    state via the underlying :class:`OperationAccessor`s.
    """

    def __init__(
        self,
        accessor: SourceAccessor,
        interleaver: Optional[Interleaver] = None,
    ):
        """Initialize a SourceEnvoy.

        Args:
            accessor: The shared :class:`SourceAccessor` for the underlying
                module. Several SourceEnvoys may wrap the same accessor.
            interleaver: The :class:`Interleaver` whose ``current`` mediator
                will receive operation-level hook callbacks.
        """
        self.accessor = accessor
        self.path = accessor.path
        self.interleaver = interleaver

        self.operations: List[OperationEnvoy] = []
        self._operations_by_name: Dict[str, OperationEnvoy] = {}

        for short_name in accessor.line_numbers.keys():
            full_name = (
                f"{accessor.path}.{short_name}" if accessor.path else short_name
            )
            op_accessor = accessor.operations[full_name]
            op_envoy = OperationEnvoy(op_accessor, interleaver=interleaver)
            setattr(self, short_name, op_envoy)
            self.operations.append(op_envoy)
            self._operations_by_name[full_name] = op_envoy

    def _get_operation(self, name: str) -> Optional[OperationEnvoy]:
        """Look up an OperationEnvoy by its fully qualified path."""
        return self._operations_by_name.get(name)

    def __str__(self):
        return str(self.accessor)

    def __iter__(self):
        return iter(self.accessor)

    def __getattribute__(self, name: str) -> Union[OperationEnvoy, Any]:
        return object.__getattribute__(self, name)
