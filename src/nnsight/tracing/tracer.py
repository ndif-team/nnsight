"""Turn a ``with`` block into code we run on our own terms.

The core trick: when the user writes

    with Tracer():
        ...their code...

we don't want the block's body to run normally. Instead we want to capture it,
hand it to a backend, and execute it in a controlled environment (for nnsight,
that means interleaved with a model's forward pass). [`Tracer`][nnsight.tracing.tracer.Tracer] pulls this
off with a small dance across the context-manager protocol:

1. **Capture** ([`Tracer.capture`][nnsight.tracing.tracer.Tracer.capture]) — from the frame that ran the ``with``,
   read the surrounding source, parse it, and find the ``with`` node at that
   line. Compile the block's *body* into a standalone code object. Results are
   memoized per site (see [`nnsight.tracing.globals`][nnsight.tracing.globals]).
2. **Skip** (`Tracer.__enter__` → [`skip_context`][nnsight.tracing.tracer.skip_context]) — install a trace
   function that raises [`ExitTracingException`][nnsight.tracing.tracer.ExitTracingException] the instant the body is
   about to run, so the body never executes inline.
3. **Execute** (`Tracer.__exit__`) — swallow that exception and instead run
   the captured code through the backend.

A block that contains only ``pass`` / a docstring has nothing to intercept, so
step 2 is skipped and it just runs (harmlessly) before the backend is invoked.
"""

from __future__ import annotations

import ast
import builtins
import linecache
import sys
import threading
from types import CodeType, FrameType, TracebackType
from typing import Any

from .backend import Backend
from .globals import BLOCKS, SOURCES
from .util import (
    Scope,
    SerializedFrame,
    clean_traceback,
    push,
)


class ExitTracingException(Exception):
    """Control-flow signal: bail out of the ``with`` body before it runs inline.

    Raised by the installed trace function the moment execution reaches the
    block, and caught in `Tracer.__exit__`, which then runs the captured
    code through the backend instead. Not an error — never surfaces to the user.
    """


class WithBlockNotFoundError(Exception):
    """The tracer call isn't used as a ``with`` block, so there's nothing to trace."""


# The statements a block can't start with (see `skip_context`). `except*` parses
# to its own node from 3.11; on 3.10 `ast.Try` is the only shape there is.
TRY_STATEMENTS = tuple(
    node for node in (ast.Try, getattr(ast, "TryStar", None)) if node is not None
)


def skip_context(tracer: Tracer) -> None:
    """Arm the trace function that skips the block body so the backend runs it.

    Sets a per-frame trace hook on the user's frame that raises
    [`ExitTracingException`][nnsight.tracing.tracer.ExitTracingException] as soon as the body is about to execute. The
    global ``sys.settrace`` is set to a no-op so Python actually delivers
    per-frame trace events (it won't call ``f_trace`` otherwise).

    Python delivers those events per *line*, so the body has to start on a line of
    its own for the hook to reach it first. A body written on the ``with`` line
    would run where it stands — and then again through the backend — so refuse it
    rather than execute it twice.

    Which line the body starts on is the hook's cue to fire, rather than the next
    line to come along: a ``with`` header written over several lines reaches its
    own closing line before the body, and raising there would skip the block
    before the ``as`` target is bound.

    Where the raise lands has to be somewhere the ``with`` can catch it, which
    rules out a body whose first statement is a ``try`` — refuse that shape too.
    """
    node = tracer.node
    body = node.body[0].lineno
    if body == node.lineno:
        raise ValueError(
            "The body of a traced `with` must start on its own line; nnsight runs "
            "the body itself, and can only intercept it at the start of a line."
        )
    if isinstance(node.body[0], TRY_STATEMENTS):
        # CPython carries the `try` keyword's line on an instruction that no
        # exception-table entry covers, so a raise delivered there unwinds the
        # whole frame — past this `with`'s `__exit__` and past any handler around
        # it, uncatchable — rather than reaching the backend. There is no earlier
        # line event to fire on once the header is done, so name the shape and
        # the way out instead of letting the trace disappear.
        raise ValueError(
            "A traced `with` block cannot start with `try:`; nnsight intercepts "
            "the body at its first line, and a `try` there is the one statement "
            "Python gives it no way back out of. Put any statement above the "
            "`try`, or move the `try` outside the block."
        )

    def skip(frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_lineno < body:
            return skip  # still in the header; stay armed
        raise ExitTracingException()

    frame = sys._getframe(2)
    sys.settrace(lambda *args, **kwargs: None)
    frame.f_trace = skip


def _bracket_depth(line: str) -> int:
    """Net open brackets on a line — how a multi-line header stays collected.

    A naive count (blind to brackets inside strings or comments), which is safe
    here: an over-count only keeps collecting lines, which parses fine, and an
    under-count that truncates the block fails the parse and falls back."""
    return (
        line.count("(") + line.count("[") + line.count("{")
        - line.count(")") - line.count("]") - line.count("}")
    )


def _strip_indent(line: str, indent: int) -> str:
    """Remove up to ``indent`` leading spaces, keeping the rest (blank lines and
    their newlines included), so a sliced block dedents to column 0."""
    i = 0
    while i < indent and i < len(line) and line[i] == " ":
        i += 1
    return line[i:]


def skippable(node: ast.With | ast.AsyncWith) -> bool:
    """Whether the block has real code worth intercepting.

    Returns ``False`` for a body made up entirely of ``pass`` statements and bare
    constant expressions (e.g. a docstring or ``...``) — nothing to run through
    the backend, so we let it execute normally instead of installing the skip
    hook. Returns ``True`` as soon as any other statement is found.
    """
    for statement in node.body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        return True
    return False


# --- Saving: choosing which values a trace returns -------------------------
#
# By default nothing a trace computes leaves it; ``save`` (exposed as
# ``nnsight.save``) marks a value to survive past the *outermost* trace. Inner
# (nested) traces push all their locals up to the enclosing block, so values flow
# freely; only the outermost trace — the boundary back into ordinary user code —
# filters to just the saved values ([`push_result`][nnsight.tracing.tracer.push_result]). Depth is tracked around
# the backend call in `Tracer.__exit__`. All state is per-thread, so
# concurrent traces never share saved sets or nesting depth.

_local = threading.local()


def _saves() -> set:
    """This thread's set of saved object ids."""
    saves = getattr(_local, "saves", None)
    if saves is None:
        saves = _local.saves = set()
    return saves


def mark(value: Any) -> Any:
    """Add ``value`` to this thread's saved set (by identity), unconditionally.

    The mechanism behind `save`, without its in-a-trace guard — for internal
    callers that mark values to return *outside* a running trace (e.g. a remote
    backend recording the values a finished request sent back).
    """
    _saves().add(id(value))
    return value


def save(value: Any, _: Any = None) -> Any:
    """Mark ``value`` to be returned to the user after the outermost trace.

    Marks the concrete object by identity and returns it unchanged, so the idiom
    is to save the value you bind: ``h = model.layer1.output.save()``. A value
    built from a saved one is not itself saved — ``(x.save() * 2)`` returns ``x``,
    not the product; write ``(x * 2).save()`` instead.

    Only meaningful inside a trace — a saved value is what the ``with
    model.trace(...):`` block hands back — so calling it with no trace running
    (e.g. ``x = nnsight.save([])`` *before* the block) is an error, not a silent
    no-op whose mark is cleared before anything reads it.

    Args:
        value: The object to mark. Returned unchanged so it can be bound in the
            same expression that saves it.

    Returns:
        ``value`` itself.

    Raises:
        ValueError: If called outside a ``with model.trace(...):`` block.

    Examples:

        >>> with model.trace("The Eiffel Tower is in"):
        ...     hidden = model.transformer.h[0].output.save()
        >>> hidden.shape  # readable after the block
        torch.Size([1, 7, 768])

        The function form ``nnsight.save(x)`` is equivalent to ``x.save()``.
    """
    if getattr(_local, "depth", 0) == 0:
        raise ValueError(
            "save() was called outside a trace. `.save()` / nnsight.save(x) marks a "
            "value to return from the enclosing `with model.trace(...):` block, so "
            "it only works inside one — move the save into the trace block."
        )
    return mark(value)


def inc() -> None:
    """Enter a trace scope (increment this thread's nesting depth).

    Called around the point a trace actually runs its body — the backend call in
    `Tracer.__exit__`. A nested trace's whole ``with`` runs inside that, so
    it sees a depth greater than 1 and is treated as inner.
    """
    _local.depth = getattr(_local, "depth", 0) + 1


def dec() -> None:
    """Leave a trace scope, clearing the saved set once the outermost one exits."""
    depth = getattr(_local, "depth", 1) - 1
    _local.depth = depth
    if depth == 0:
        _saves().clear()


def push_result(frame: FrameType, variables: dict) -> None:
    """Write a trace body's ``variables`` back into ``frame``.

    An inner trace pushes everything, so values flow up to the enclosing block.
    The outermost trace — depth 1 here, since every nested trace has already left
    — pushes only the values marked with `save`.
    """
    if getattr(_local, "depth", 0) == 1:
        saves = _saves()
        variables = {k: v for k, v in variables.items() if id(v) in saves}
    push(frame, variables)


class Tracer:
    """Context manager that captures a ``with`` block and runs it via a backend.

    See the module docstring for the capture → skip → execute flow. Subclasses
    override the small steps ([`parse`][nnsight.tracing.tracer.Tracer.parse], [`build`][nnsight.tracing.tracer.Tracer.build], [`compile`][nnsight.tracing.tracer.Tracer.compile],
    `execute`) or supply a different [`Backend`][nnsight.tracing.backend.Backend]
    to change what "run the block" means.
    """

    class Info:
        """Everything needed to execute a captured block: the live ``frame`` it
        came from (for its globals/locals and traceback filename) and the
        compiled ``code`` of the block body."""

        def __init__(self, frame: FrameType, code: CodeType) -> None:
            self.frame = frame
            self.code = code

        def __getstate__(self) -> dict:
            # The real frame is unpicklable; swap in a stand-in that keeps the
            # code metadata (filename/lineno/name for tracebacks) but no live
            # scope. The code object and captured globals/locals aren't shipped
            # here — they're rebuilt from the source-reduced interventions
            # payload in RequestModel.deserialize.
            return {"frame": SerializedFrame.of(self.frame)}

    def __init__(self, backend: Backend | None = None) -> None:
        """Args:
        backend: What actually runs the captured block in `__exit__`.
            Defaults to a plain [`Backend`][nnsight.tracing.backend.Backend], which
            just executes it in place.
        """
        self.info: Tracer.Info | None = None
        self.node: ast.With | ast.AsyncWith | None = None
        self.backend = backend if backend is not None else Backend()

    def __getstate__(self) -> dict:
        """Serialize the tracer, minus what can't or shouldn't travel.

        Drops the backend (transport-side, may hold credentials), the AST node
        (only needed locally), and the batcher (per-run state, rebuilt by execute
        server-side); [`Info`][nnsight.tracing.tracer.Tracer.Info] handles its own frame swap.
        """
        state = self.__dict__.copy()
        state.pop("backend", None)
        state.pop("node", None)
        state.pop("batcher", None)
        return state

    def __setstate__(self, state: dict) -> None:
        # __getstate__ dropped the AST node (needed only where the block is
        # captured); a tracer run after deserialization — a remote request executed
        # server-side — has no block to point at. execute() still reads self.node,
        # so restore it as None: the mediators it builds run and are never
        # re-serialized, which is exactly the node=None case.
        self.__dict__.update(state)
        self.node = None

    def capture(self) -> None:
        """Capture the ``with`` block at the call site into [`info`][nnsight.tracing.tracer.Tracer.info].

        Looks two frames up (past `__enter__`) to find the user's frame,
        reads and parses its source, locates the ``with`` node at that line, and
        compiles the block body. The parsed node and compiled code are memoized
        per site in [`BLOCKS`][nnsight.tracing.globals.BLOCKS], so re-entering the
        same block is cheap.

        Idempotent: safe to call early (to sniff whether a ``with`` block exists)
        and again on ``__enter__``.

        Raises:
            WithBlockNotFoundError: if the call site isn't a ``with`` statement.
        """
        if self.info is not None:
            return
        frame = sys._getframe(2)
        code = frame.f_code

        # A trace site is fully identified by where it sits in the code, and
        # parsing the whole source + compiling the block is deterministic for
        # that site. Memoize (node, compiled) so re-entering the same ``with``
        # (e.g. a loop opening many tracers) parses and compiles only once — both
        # (None, None) for a site that isn't a ``with`` block, so that verdict is
        # cached too and the traceable fallback doesn't re-parse every call.
        key = (code.co_filename, frame.f_lineno, code.co_name, code.co_firstlineno)

        if key not in BLOCKS:
            source = self.source(code.co_filename)
            node = self.parse(source, frame.f_lineno)
            compiled = None if node is None else self.compile(self.build(node), frame)
            BLOCKS[key] = (node, compiled)

        node, compiled = BLOCKS[key]
        if node is None:
            raise WithBlockNotFoundError()
        self.node = node
        self.info = Tracer.Info(frame, compiled)

    @staticmethod
    def source(filename: str) -> str:
        """Return the source text for ``filename``, reading it at most once.

        Cached in [`SOURCES`][nnsight.tracing.globals.SOURCES] and never
        re-validated, so a file edited mid-run is still traced as it was first
        seen (we deliberately skip ``linecache.checkcache``). Handles files,
        IPython/Jupyter cells, and ``python -c`` programs; see the inline notes.
        """
        if filename in SOURCES:
            return SOURCES[filename]

        # Regular files and IPython/Jupyter cells alike come from linecache.
        # IPython registers each cell there under the frame's filename (with no
        # mtime, so it's never purged) — the exact source it compiled, so line
        # numbers line up with frame.f_lineno regardless of any input
        # transformations (magics, ``!shell``, ``?`` help).
        source = "".join(linecache.getlines(filename))

        # `python -c "<code>"` has no file on disk — its code (and any function
        # defined in it) compiles under the filename "<string>", so recover the
        # literal program from the interpreter args. Gated on the exact
        # "<string>" filename so we don't hijack other source-less contexts under
        # a -c launch (a differently-named exec, "<stdin>", "<console>", ...);
        # dynamic code should register itself in linecache.
        if not source and filename == "<string>" and "-c" in sys.orig_argv:
            source = sys.orig_argv[sys.orig_argv.index("-c") + 1]
            if not source.endswith("\n"):
                source += "\n"

        SOURCES[filename] = source
        return source

    def __enter__(self) -> Tracer:
        """Capture the block and, if it has real code, arm the skip hook."""
        self.capture()
        if skippable(self.node):
            skip_context(self)
        return self

    def parse(self, source: str, lineno: int) -> ast.With | ast.AsyncWith | None:
        """Find the ``with``/``async with`` node that starts on ``lineno``.

        Returns the matching AST node, or ``None`` if there's no ``with``
        statement there (the call site isn't a trace block).

        Parsing the whole file is O(its AST) and dominates a cold capture, so try
        slicing just the block out first (`_parse_block`); parse the whole
        file only if the slice can't be isolated cleanly.
        """
        node = self._parse_block(source, lineno)
        if node is not None:
            return node
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.With, ast.AsyncWith)) and node.lineno == lineno:
                return node
        return None

    def _parse_block(
        self, source: str, lineno: int
    ) -> ast.With | ast.AsyncWith | None:
        """Parse only the ``with`` block at ``lineno``, or ``None`` to fall back.

        Slices from ``lineno`` — the block's (possibly multi-line) header and its
        body, bounded by indentation and open brackets — dedents it to column 0 so
        it stands alone, parses that, and shifts the line numbers back to the file
        so tracebacks and the skip hook still point at the real source.

        Getting the bound wrong is safe: collecting *too much* just parses a few
        trailing statements past the block (``body[0]`` is still the ``with``);
        collecting too little makes the slice unparseable, which returns ``None``
        and lets [`parse`][nnsight.tracing.tracer.Tracer.parse] fall back to the whole file. It never returns a wrong
        node.
        """
        lines = source.splitlines(keepends=True)
        if not 1 <= lineno <= len(lines):
            return None
        first = lines[lineno - 1]
        indent = len(first) - len(first.lstrip())
        collected = [first]
        depth = _bracket_depth(first)
        for line in lines[lineno:]:
            stripped = line.lstrip()
            # A non-blank line dedented back to the header's column, with no bracket
            # still open, is the next statement — the block ended on the line before.
            if depth <= 0 and stripped and len(line) - len(stripped) <= indent:
                break
            collected.append(line)
            depth += _bracket_depth(line)

        chunk = "".join(_strip_indent(line, indent) for line in collected)
        try:
            tree = ast.parse(chunk)
        except SyntaxError:
            return None
        node = tree.body[0] if tree.body else None
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            return None
        ast.increment_lineno(node, lineno - 1)
        return node

    def build(self, node: ast.With | ast.AsyncWith) -> ast.Module:
        """Wrap the block's *body* (not the ``with`` line) in a compilable module.

        ``fix_missing_locations`` backfills line/column info so the synthesized
        module can be compiled.
        """
        module = ast.Module(body=node.body, type_ignores=[])
        ast.fix_missing_locations(module)
        return module

    def compile(self, module: ast.Module, frame: FrameType) -> CodeType:
        """Compile the body module into a code object.

        Compiled under the original frame's filename, and its ``co_name`` is set
        to the frame's so tracebacks read as if the body ran where it was written.
        """
        code = builtins.compile(module, frame.f_code.co_filename, "exec")
        return code.replace(co_name=frame.f_code.co_name)

    def execute(self, code: CodeType) -> None:
        """Run the captured body and write its results back to the user's frame.

        Executes the block against a copy of the caller's locals, then
        [`push_result`][nnsight.tracing.tracer.push_result] writes results back — a nested
        trace passes all its locals up to the enclosing block, while the outermost
        trace returns only the values marked with `nnsight.save`.

        Subclasses override this to change *how* the block runs (interleaved with a
        forward pass, a backward pass, ...), ending with the same
        [`push_result`][nnsight.tracing.tracer.push_result]; the trace-scope depth it reads
        is managed by `__exit__`.
        """
        frame = self.info.frame
        scope = Scope(dict(frame.f_locals), {}, frame.f_globals)
        exec(code, scope)
        push_result(frame, scope)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> bool:
        """Run the captured block through the backend, or propagate a real error.

        Tears down the trace hook, then — for the expected
        [`ExitTracingException`][nnsight.tracing.tracer.ExitTracingException] (body was skipped) or a clean exit (nothing
        to skip) — invokes the backend. An exception from the backend is
        re-raised with a [`clean_traceback`][nnsight.tracing.util.clean_traceback] so nnsight
        internals are dropped, leaving the user's own frames (across any files).
        Any *other* exception from the body propagates normally (returns ``False``).
        """
        sys.settrace(None)
        if exc_type is ExitTracingException or exc_type is None:

            inc()
            try:
                self.backend(self)
            except BaseException as exception:
                exception.__context__ = None
                raise exception.with_traceback(
                    clean_traceback(exception.__traceback__)
                )
            finally:
                dec()
                # Drop the tracer's reference to the caller's frame. Info.frame
                # holds the live frame the block ran in, whose locals bind this
                # tracer and every saved value/model — a reference cycle. Clearing
                # it lets refcounting free the whole trace scope as soon as the
                # caller returns, with no cyclic GC pass.
                if self.info is not None:
                    self.info.frame = None
            return True
        return False
