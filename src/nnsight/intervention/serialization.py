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
* Persistent ids — objects carrying a ``_persistent_id`` in their ``__dict__``
  are written as that id and resolved on the other side from a
  ``persistent_objects`` map (keeps the model out of the payload).
"""

from __future__ import annotations

import ast
import inspect
import io
import linecache
import pickle
import textwrap
import types
from typing import Any, Optional

import cloudpickle
from cloudpickle.cloudpickle import _get_cell_contents

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
    [`_function_referenced_names`][nnsight.intervention.serialization._function_referenced_names].
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


class CustomCloudPickler(cloudpickle.CloudPickler):
    """Pickler that serializes dynamic ``def`` functions by source, not bytecode."""

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

    def persistent_id(self, obj: Any) -> Optional[Any]:
        # Reference (don't serialize) anything tagged with a _persistent_id.
        try:
            return obj.__dict__["_persistent_id"]
        except (AttributeError, KeyError, TypeError):
            return None


class UnknownPersistentIdError(pickle.UnpicklingError):
    """A persistent id was encountered with no entry in persistent_objects."""


class CustomCloudUnpickler(pickle.Unpickler):
    """Unpickler that resolves persistent ids from a ``persistent_objects`` map."""

    def __init__(self, file: io.BufferedIOBase, persistent_objects: Optional[dict]):
        super().__init__(file)
        self.persistent_objects = persistent_objects or {}

    def persistent_load(self, pid: Any) -> Any:
        if pid in self.persistent_objects:
            return self.persistent_objects[pid]
        raise UnknownPersistentIdError(pid)


def dump(obj: Any, file: Any, protocol: int = DEFAULT_PROTOCOL) -> None:
    """Serialize ``obj`` to an open binary file (``pickle.dump`` semantics)."""
    CustomCloudPickler(file, protocol=protocol).dump(obj)


def dumps(obj: Any, protocol: int = DEFAULT_PROTOCOL) -> bytes:
    """Serialize ``obj`` to bytes (``pickle.dumps`` semantics)."""
    buffer = io.BytesIO()
    dump(obj, buffer, protocol=protocol)
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
