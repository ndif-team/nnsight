"""Source-based serialization for remote execution.

Standard cloudpickle serializes code as *bytecode*, which is tied to the exact
Python version — a payload pickled on 3.10 can fail to load on 3.11. This module
serializes code by its **source** instead, so it reconstructs on any Python that
can parse the syntax (client and server need not match).

Two mechanisms layer on top of cloudpickle:

* [`code_reduce`][nnsight.intervention.serialization.code_reduce] — reduces a block of source plus its scope to a filtered
  ``(source, globals, locals)`` tuple (only the names the source references, and
  the code as text). ``RequestModel.serialize`` ships the traced block as one of
  these tuples, and the function reducer runs a dynamic ``def``'s source/globals
  through it too; `_load_function` compiles the source back to a code
  object on the far side.
* Persistent ids: given any envoy of a model (``tree=``), the pickler replaces
  objects that already exist on the receiver with string persistent IDs, and the
  receiver resolves each ID from its ``persistent_objects`` map (built by
  ``Remotable._remoteable_persistent_objects``). Persistent IDs cover:
  - the model's interleaver and attachments (such as ``Tokenizer`` and ``Pipeline``),
    matched by object identity;
  - each ``torch.nn.Module`` in the tree as ``Module:<path>``, matched by identity;
  - each stateless [`Envoy`][nnsight.intervention.envoy.Envoy] in the tree as
    ``Envoy:<path>``.
  An envoy that carries client-side state at or below it (an `.edit()`, a
  ``rename=`` alias, or a caller-supplied envoy class) is pickled by value while
  its stateless descendants still go as ``Envoy:<path>``. Envoys belonging to any
  other model tree raise ``pickle.PicklingError``.
"""

from __future__ import annotations

import ast
import inspect
import io
import linecache
import pickle
import textwrap
import types
import weakref
from typing import Any, Callable, Optional

import cloudpickle
from cloudpickle.cloudpickle import _get_cell_contents

from .envoy import Envoy

DEFAULT_PROTOCOL = 4


def _referenced_names(source: str) -> set[str]:
    """Names the source needs from its enclosing scope, from its AST.

    Two filters, both there to keep unrelated objects out of the payload — what
    lands here gets pickled and shipped, so a wrong name is at best dead weight
    and at worst an unpicklable object or one the server mis-resolves.

    **Only ``ast.Name`` nodes**, so an attribute is not mistaken for a variable.
    The bytecode's ``co_names`` cannot make that distinction — ``llm.model.layers``
    puts ``model`` and ``layers`` in the same table as a genuine variable load —
    and treating those as names ships whatever global happens to share the
    spelling. ``model``, ``output``, ``input`` and ``config`` are all both common
    attributes and common variable names; when one such global was another model,
    its envoys claimed a conflicting ``Module:<path>`` id and the server failed to
    resolve it against the deployed checkpoint.

    **Only names the block reads** — ``Load`` (and ``Del``, which needs the
    binding to exist), plus an augmented-assignment target, which reads before it
    writes. A name the block merely *binds* (``with model.trace(...) as tracer:``,
    a ``for`` target, a plain assignment) is a local of the block: it is about to
    be overwritten, so capturing the enclosing scope's same-named object is
    pointless. It is also actively harmful, because that object need not be
    picklable — a stale ``tracer`` left over from an earlier cell holds a dead
    frame, and pickling it dies in ``SerializedFrame`` on ``None.f_code``.

    ``ast.walk`` covers nested scopes (comprehensions, lambdas, inner ``def``\\ s)
    in one pass, which is what the previous recursion into ``co_consts`` was for.
    """
    import ast

    tree = ast.parse(source)
    names = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
    }
    names.update(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Load, ast.Del))
    )
    return names


def _lambda_source(func: types.FunctionType) -> str:
    """Recover just a lambda's expression source from its surrounding line.

    ``inspect.getsource`` on a lambda returns the whole line it sits on (e.g.
    ``f = lambda x: x`` or ``f, g = lambda: 1, lambda: 2``), sometimes with several
    lambdas. Parse it and unparse the ``ast.Lambda`` that matches this code object:
    the one whose source span encloses the code's own bytecode positions (and, for
    a nested lambda, the outermost such — that *is* ``func``).
    """
    lines, start = inspect.getsourcelines(func)
    source = "".join(lines)

    # Parse, tracking how far parsing shifted positions from their file
    # coordinates, so AST spans can be compared to the code object's positions.
    col_shift = 0  # columns removed by dedent
    line_base = start - 1  # file line of AST line 1 is `start`
    tree = None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        dedented = textwrap.dedent(source)
        first = next(line for line in source.splitlines() if line.strip())
        col_shift = len(first) - len(first.lstrip())
        try:
            tree = ast.parse(dedented)
        except SyntaxError:
            try:
                # e.g. "return lambda: x" in isolation — wrap it so it parses.
                tree = ast.parse("def __w__():\n" + textwrap.indent(dedented, " "))
                col_shift -= 1  # the one space of indent we added
                line_base -= 1  # the "def __w__():" line we prepended
            except SyntaxError:
                tree = None
    if tree is None:
        # The lambda sits on a continued line that won't parse on its own; parse
        # the whole file so every position is already in file coordinates.
        import linecache

        col_shift, line_base = 0, 0
        tree = ast.parse("".join(linecache.getlines(func.__code__.co_filename)))

    lambdas = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
    if len(lambdas) <= 1:
        return ast.unparse(lambdas[-1]) if lambdas else source.strip()

    # Narrow to lambdas whose parameter names match this code object's. This alone
    # distinguishes a nested lambda from its parent (``lambda x: lambda y: ...``)
    # and a lambda from a lambda default (``lambda x=lambda: 1: ...``).
    params = tuple(func.__code__.co_varnames[: func.__code__.co_argcount])

    def ast_params(node: ast.Lambda) -> tuple[str, ...]:
        return tuple(a.arg for a in node.args.posonlyargs + node.args.args)

    candidates = [n for n in lambdas if ast_params(n) == params] or lambdas
    if len(candidates) == 1:
        return ast.unparse(candidates[0])

    # Same signature on one line (``f, g = lambda x: x, lambda x: -x``): break the
    # tie by source span. Code positions are (line, end_line, col, end_col) in file
    # coordinates; keep only real, non-empty spans (a zero-width RESUME at column 0
    # points at no source and would fall outside every lambda).
    #
    # co_positions is 3.11+, and columns have no pre-3.11 substitute (3.10's dis
    # carries line numbers only), so on 3.10 there is nothing to tie-break with:
    # every candidate "encloses" an empty position list and the narrowest one wins.
    # That still picks the inner lambda of a nested pair; two same-signature lambdas
    # on one line are a coin flip there.
    co_positions = getattr(func.__code__, "co_positions", None)
    positions = (
        [
            (line, end_line, col, end_col)
            for line, end_line, col, end_col in co_positions()
            if None not in (line, end_line, col, end_col)
            and (end_line > line or end_col > col)
        ]
        if co_positions is not None
        else []
    )

    def encloses(node: ast.Lambda) -> bool:
        # The node's file span must contain every one of the code's positions.
        n_line, n_end_line = node.lineno + line_base, node.end_lineno + line_base
        n_col = node.col_offset + col_shift
        n_end_col = node.end_col_offset + col_shift
        for line, end_line, col, end_col in positions:
            if line < n_line or end_line > n_end_line:
                return False
            if line == n_line and col < n_col:
                return False
            if end_line == n_end_line and end_col > n_end_col:
                return False
        return True

    def width(node: ast.Lambda) -> tuple[int, int]:
        return (node.end_lineno - node.lineno, node.end_col_offset - node.col_offset)

    enclosing = [n for n in candidates if encloses(n)]
    if enclosing:
        return ast.unparse(min(enclosing, key=width))
    return ast.unparse(candidates[-1])


def _function_referenced_names(source: str) -> set[str]:
    """Names a ``def``'s body needs from outside it, by Python's own scoping rules.

    A function is not a block: a name assigned *anywhere* in its body is local to
    it, and reading that name before the assignment is an ``UnboundLocalError``,
    never a lookup in the enclosing scope. So the read-anywhere rule
    `_referenced_names` uses for a traced block is wrong here — it captures the
    enclosing scope's same-named object for a name the function only ever binds.
    That is not merely wasteful: a helper doing ``with open(path) as f:`` would
    drag in whatever ``f`` a notebook cell left lying around, and a closed file
    can't be pickled at all ("Cannot pickle closed files").

    ``symtable`` answers this exactly, including ``global`` / ``nonlocal``
    declarations, so it is used rather than re-deriving the rules from the AST.
    Descendant tables are included because a nested ``def`` or comprehension can
    reference a module global too; their *free* variables resolve through the
    enclosing function's closure and are deliberately not collected here.
    """
    import symtable

    def globals_of(table: "symtable.SymbolTable") -> set[str]:
        names = {
            symbol.get_name() for symbol in table.get_symbols() if symbol.is_global()
        }
        for child in table.get_children():
            names |= globals_of(child)
        return names

    top = symtable.symtable(source, "<sourced>", "exec")
    # The source is one `def` (or `lambda`) at module level, so exactly one child
    # table is a function. Select it by type rather than by counting children:
    # under PEP 649 (3.14+) every `def` also gets an `__annotate__` table of type
    # "annotation", and PEP 695 generics add a "type_parameters" table, so a
    # single `def` yields two or more children. Counting them sent every `def` to
    # the block rule instead -- silently reinstating the capture this function
    # exists to prevent. Anything other than exactly one function table, fall
    # back rather than guess.
    functions = [
        child for child in top.get_children() if child.get_type() == "function"
    ]
    if len(functions) != 1:
        return _referenced_names(source)
    return globals_of(functions[0])


def code_reduce(source: str, globals: dict, locals: dict, function: bool = False) -> tuple:
    """Reduce source + its scope to a picklable, filtered tuple.

    Returns ``(source, used_globals, used_locals)`` — the code as text (so it
    recompiles on any Python version) plus only the globals/locals the source
    references, so the whole enclosing scope isn't shipped. Shared by both the
    traced-block payload and the dynamic-function reducer; `_load_function`
    (or the eventual block executor) compiles the source back into a code object.

    Names are looked up by subscript rather than by iterating ``.items()`` so a
    [`Scope`][nnsight.tracing.util.Scope] globals resolves through its fallback
    chain: a nested block (a ``tracer.invoke(...)`` body) runs with a Scope as its
    globals, whose iteration yields only the block's own names, leaving a module
    global it references (e.g. ``torch``) out of the payload.

    ``function`` says ``source`` is a single ``def`` / ``lambda`` rather than a
    traced block, which changes what counts as "referenced" — see
    ``_function_referenced_names`` (private: no reference page publishes it).
    """
    names = _function_referenced_names(source) if function else _referenced_names(source)
    used_globals, used_locals = {}, {}
    for name in names:
        if name in locals:
            used_locals[name] = locals[name]
            continue
        try:
            used_globals[name] = globals[name]
        except KeyError:
            # A builtin or a bare attribute name — resolved at run time, not shipped.
            pass
    return (source, used_globals, used_locals)


def reduce_block(node: Any, globals: dict, locals: dict) -> tuple:
    """Reduce a captured ``with``-block to a picklable ``(source, globals, locals)``.

    Unparses the block *body* to source — padded with leading blank lines so the
    statements keep their original line numbers (a remote traceback then points at
    the right line) — and [`code_reduce`][nnsight.intervention.serialization.code_reduce]\\ s it against its scope. Shared by the
    request payload (the traced block) and edit-mediator serialization.
    """
    import ast

    offset = "\n" * (node.body[0].lineno - 1)
    source = offset + ast.unparse(ast.Module(body=node.body, type_ignores=[]))
    return code_reduce(source, globals, locals)


def _load_function(
    source: str,
    name: str,
    defaults: Optional[tuple],
    kwdefaults: Optional[dict],
    annotations: Optional[dict],
    origin_filename: str,
    origin_line: int,
) -> types.FunctionType:
    """Rebuild a dynamic function from a [`code_reduce`][nnsight.intervention.serialization.code_reduce] tuple's source.

    Compiles the source and lifts out the function's code object rather than
    executing the ``def``, so defaults/annotations aren't re-evaluated against
    incomplete globals. The function starts with empty globals; `_fill`
    populates them after pickle has memoized it (so recursive references resolve).

    The code is padded so its line numbers match the original file, and compiled
    under a ``[<name>:<def-line>] <path>`` filename that names the true origin and
    is registered in linecache, so a remote traceback shows the right file, line,
    and source. The def-line keeps the key unique when two distinct functions
    share a name and file (e.g. same-named nested helpers). (Not under the nnsight
    package, so clean_traceback keeps these frames as the user's own code.)
    """
    # Blank-pad so the function's statements keep their original line numbers.
    padded = "\n" * (origin_line - 1) + source
    filename = f"[{name}:{origin_line}] {origin_filename}"

    module_code = compile(padded, filename, "exec")
    func_code = None
    for const in module_code.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == name:
            func_code = const  # keep the last match (outermost) for nested defs
    if func_code is None:
        raise ValueError(f"Could not find function {name!r} in compiled source")

    linecache.cache[filename] = (
        len(padded),
        None,
        padded.splitlines(keepends=True),
        filename,
    )

    func = types.FunctionType(
        func_code, {"__builtins__": __builtins__}, name, defaults, None
    )
    if kwdefaults:
        func.__kwdefaults__ = kwdefaults
    if annotations:
        func.__annotations__ = annotations
    return func


def _fill(func: types.FunctionType, globals: dict) -> None:
    """Populate a reconstructed function's globals after it's been memoized.

    Deferring this (rather than passing globals to `_load_function`) is
    what lets a function's own name — present in its globals for recursion —
    resolve to the already-memoized function instead of recursing during pickling.
    """
    func.__globals__.update(globals)


# Built-in primitive and container types that never have a persistent ID.
# `persistent_id` is called for every pickled object, so checking this set
# first avoids dict and isinstance lookups on the most common objects.
_ATOMS = frozenset(
    {type(None), bool, int, float, complex, str, bytes, list, tuple, dict, set, frozenset}
)


class _EnvoyName:
    """Wrapper around an envoy's ``path`` that `CustomCloudPickler.persistent_id` emits as ``Envoy:<path>``.

    Used by `CustomCloudPickler.reducer_override` when reducing ``self._tree``:
    passing the `Envoy` itself as a reduce argument would re-enter
    `reducer_override` recursively, whereas `_EnvoyName(path)` resolves to the
    receiver's envoy via ``Envoy:<path>`` before pickle calls `__setstate__`.
    """

    def __init__(self, path: str) -> None:
        self.path = path


def _named_envoy(envoy: Envoy) -> Envoy:
    """Identity reconstructor for ``_tree`` so pickle runs ``__setstate__`` on the resolved envoy."""
    return envoy


class CustomCloudPickler(cloudpickle.CloudPickler):
    """Pickler that serializes dynamic ``def`` functions by source, not bytecode.

    ``tree`` is the [`Envoy`][nnsight.intervention.envoy.Envoy] that the trace or
    session runs against, or ``None``. When ``tree`` is provided, modules,
    stateless envoys, the interleaver, and model attachments (such as the
    tokenizer and pipeline) are emitted as persistent IDs. Pickling any `Envoy`
    when ``tree`` is ``None`` (or when the envoy belongs to a different model)
    raises `pickle.PicklingError`.
    """

    def __init__(self, file: Any, protocol: int = DEFAULT_PROTOCOL, tree: Any = None) -> None:
        super().__init__(file, protocol=protocol)
        # `tree` can be a child envoy when a trace is started on a submodule
        # (`layer.trace(...)`); walk up to `.root` to collect model-level
        # attachments such as the tokenizer and pipeline.
        root = tree.root if tree is not None else None
        self._tree = tree
        self._interleaver = root.interleaver if root is not None else None
        # Map `id(obj) -> persistent_id` for model attachments (`Interleaver`,
        # `Tokenizer`, `Pipeline`) so variables captured before the block
        # (`tok = model.tokenizer`) still serialize as persistent IDs.
        self._ids = (
            {id(obj): name for name, obj in root._persistent_attachments().items()}
            if root is not None
            else {}
        )
        # Per-pickler memo mapping `id(envoy)` to whether its subtree carries state.
        self._stateful: dict[int, bool] = {}
        # Fast-path flag: True when no registered envoy in the tree carries state,
        # allowing `_goes_by_name` to skip the recursive `_state_below` walk.
        self._clean = self._interleaver is not None and not any(
            envoy._carries_state() for envoy in self._interleaver.envoys.values()
        )

    def _dynamic_function_reduce(self, func: types.FunctionType) -> tuple:
        code = func.__code__
        # Serialize by source when we can recover it — a lambda's expression, or
        # a def's block. Functions with no source (C funcs, dataclass methods,
        # exec'd/REPL) fall back to cloudpickle's bytecode reducer.
        try:
            if code.co_name == "<lambda>":
                source = _lambda_source(func)
            else:
                source = textwrap.dedent(inspect.getsource(func))
            # Where the source begins in the original file, so the far side can
            # restore matching line numbers (getsourcelines counts decorators).
            origin_line = inspect.getsourcelines(func)[1]
        except (OSError, TypeError):
            return super()._dynamic_function_reduce(func)

        # Recompiled at module level a function's free variables become globals,
        # so pass the closure as the "locals" for code_reduce to filter, then
        # fold it into the globals the reconstructed function runs against.
        closure = {}
        if func.__closure__ is not None:
            for name, cell in zip(code.co_freevars, func.__closure__):
                try:
                    closure[name] = _get_cell_contents(cell)
                except ValueError:
                    pass  # empty cell (not yet bound); nothing to capture

        source, used_globals, used_closure = code_reduce(
            source, func.__globals__, closure, function=True
        )
        # Globals go in the state (applied after memoization) so a self-reference
        # resolves via the memo instead of recursing during pickling.
        globals = {**used_globals, **used_closure}
        args = (
            source,
            func.__name__,
            func.__defaults__,
            func.__kwdefaults__,
            func.__annotations__,
            code.co_filename,
            origin_line,
        )
        return (_load_function, args, globals, None, None, _fill)

    def _goes_by_name(self, envoy: Envoy) -> bool:
        if self._clean and self._interleaver.envoys.get(id(envoy._module)) is envoy:
            return True
        return not envoy._state_below(self._stateful)

    def reducer_override(self, obj: Any) -> Any:
        # A plain persistent ID returns the resolved object from `persistent_load`
        # without calling `__setstate__`. The envoy running the payload (`_tree`)
        # needs `__setstate__({"interleaver": ...})` to run on the receiver for two
        # reasons: `Meta.__setstate__` marks `dispatched = True`, and receivers that
        # create a per-request interleaver (such as NDIF's sandbox `IPCInterleaver`)
        # bind that interleaver onto `_tree`.
        if obj is self._tree and self._goes_by_name(obj):
            return _named_envoy, (_EnvoyName(obj.path),), {"interleaver": self._interleaver}
        return super().reducer_override(obj)

    def persistent_id(self, obj: Any) -> Optional[Any]:
        # Called for every pickled object, so return early for common built-in types.
        if type(obj) in _ATOMS:
            return None
        pid = self._ids.get(id(obj))
        if pid is not None:
            return pid
        if type(obj) is _EnvoyName:
            return f"Envoy:{obj.path}"
        if self._interleaver is not None:
            # If `obj` is a `torch.nn.Module` wrapped by this tree, emit `Module:<path>`.
            envoy = self._interleaver.envoys.get(id(obj))
            if envoy is not None and envoy._module is obj:
                return f"Module:{envoy.path}"
            # Envoys belonging to this tree share `self._interleaver`. Stateless
            # envoys go as `Envoy:<path>`; stateful envoys return `None` so pickle
            # serializes their `__getstate__` dict and checks each child in turn.
            if isinstance(obj, Envoy) and obj.interleaver is self._interleaver:
                # Return `None` for `_tree` so pickle invokes `reducer_override`.
                if obj is self._tree:
                    return None
                return f"Envoy:{obj.path}" if self._goes_by_name(obj) else None
        if isinstance(obj, Envoy):
            if self._interleaver is None:
                raise pickle.PicklingError(
                    f"Cannot write envoy {obj.path!r} with no model to name it from; "
                    "pass `tree=<the model>` to `dumps`"
                )
            raise pickle.PicklingError(
                f"Envoy {obj.path!r} belongs to a different model than the payload "
                "is written against; the receiver cannot resolve it"
            )
        return None


class UnknownPersistentIdError(pickle.UnpicklingError):
    """A persistent id was encountered with no entry in persistent_objects."""


def _resolve(persistent_objects: dict, pid: Any) -> Any:
    if pid not in persistent_objects:
        raise UnknownPersistentIdError(
            f"{pid!r}: the receiver has no object under this id; a different model, "
            "a module added after the map was built, or an older nnsight on the receiver"
        )
    return persistent_objects[pid]


class _PayloadCopies:
    """Per-unpickler cache of lazy [`Envoy`][nnsight.intervention.envoy.Envoy] copies, keyed by ``id(origin)``.

    A receiver reuses one model tree across requests. Resolving ``Envoy:<path>``
    to a per-payload copy isolates any in-place edits or attribute changes made
    by the payload from the receiver's shared tree. Each copy starts unmaterialized
    (holding ``_origin`` and ``_copies``) and populates its child and parent copies
    on first access via ``Envoy._materialize``. ``self.copies`` is a
    `weakref.WeakValueDictionary` so ``copy -> _copies -> copies[id] -> copy``
    does not form a strong reference cycle keeping ``_origin`` alive after the
    request finishes.
    """

    def __init__(self, persistent_objects: dict) -> None:
        self.objects = persistent_objects
        self.copies: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def __call__(self, origin: Envoy) -> Envoy:
        copy = self.copies.get(id(origin))
        if copy is None:
            copy = object.__new__(type(origin))
            state = copy.__dict__
            state.update(
                (name, dict(value) if type(value) is dict else value)
                for name, value in origin.__dict__.items()
                if not isinstance(value, Envoy) and name not in ("_parent", "_children", "_child_map")
            )
            state["_module"] = _resolve(self.objects, f"Module:{origin.path}")
            state["_edits"] = []
            state["_origin"] = origin
            state["_copies"] = self
            self.copies[id(origin)] = copy
        return copy

    def parent_link(self, parent: Optional[Envoy]) -> Optional[Callable[[], Envoy]]:
        """Return the ``_parent`` callable for a copy whose origin's parent is ``parent``."""
        if parent is None:
            return None
        above = self.copies.get(id(parent))
        if above is not None:
            return weakref.ref(above)
        # When a child envoy was captured directly without its parent in the
        # payload, nothing else holds the parent copy alive; return a strong
        # closure `lambda: above` so `.parent` and `.root` do not expire immediately.
        above = self(parent)
        return lambda: above


class CustomCloudUnpickler(pickle.Unpickler):
    """Unpickler that resolves persistent ids from a ``persistent_objects`` map.

    Each ``Envoy:<path>`` persistent ID resolves to a lazy per-payload copy of
    the receiver's envoy via `_PayloadCopies`, isolating payload mutations from
    the receiver's shared envoy tree. Repeated references to the same envoy
    within one payload resolve to the same copy while it remains alive.
    """

    def __init__(self, file: io.BufferedIOBase, persistent_objects: Optional[dict]):
        super().__init__(file)
        self.persistent_objects = persistent_objects or {}
        self._copies = _PayloadCopies(self.persistent_objects)

    def persistent_load(self, pid: Any) -> Any:
        obj = _resolve(self.persistent_objects, pid)
        if isinstance(obj, Envoy):
            return self._copies(obj)
        return obj


def dump(obj: Any, file: Any, protocol: int = DEFAULT_PROTOCOL, tree: Any = None) -> None:
    """Serialize ``obj`` to an open binary file (``pickle.dump`` semantics).

    ``tree`` is any envoy of the model whose objects travel as names (see
    [`CustomCloudPickler`][nnsight.intervention.serialization.CustomCloudPickler]).
    """
    CustomCloudPickler(file, protocol=protocol, tree=tree).dump(obj)


def dumps(obj: Any, protocol: int = DEFAULT_PROTOCOL, tree: Any = None) -> bytes:
    """Serialize ``obj`` to bytes (``pickle.dumps`` semantics); ``tree`` as in `dump`."""
    buffer = io.BytesIO()
    dump(obj, buffer, protocol=protocol, tree=tree)
    return buffer.getvalue()


def load(
    file: Any,
    persistent_objects: Optional[dict] = None,
    unpickler: type = CustomCloudUnpickler,
) -> Any:
    """Deserialize from an open binary file (``pickle.load`` semantics).

    ``unpickler`` selects the unpickler class (``unpickler(file,
    persistent_objects)``) so a caller can resolve persistent ids differently —
    e.g. a server that rebuilds the model out-of-process passes its own.
    """
    return unpickler(file, persistent_objects).load()


def loads(
    data: bytes,
    persistent_objects: Optional[dict] = None,
    unpickler: type = CustomCloudUnpickler,
) -> Any:
    """Deserialize from bytes (``pickle.loads`` semantics)."""
    return load(io.BytesIO(data), persistent_objects, unpickler)
