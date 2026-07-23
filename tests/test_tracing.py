import ast
import linecache
import os
import runpy
import subprocess
import sys
import textwrap
import traceback
import types

import pytest

import nnsight
from nnsight.tracing.backend import Backend
from nnsight.tracing.tracer import ExitTracingException, Tracer, save


class _RecordingBackend(Backend):
    """Runs the block but first snapshots the tracer's info (the frame is released
    at trace exit, so this is where to observe what was captured)."""

    def __init__(self):
        self.info = None

    def __call__(self, tracer):
        self.info = tracer.info
        self.frame = tracer.info.frame
        tracer.execute(tracer.info.code)


class TestParseBlock:
    """Capture slices the block out at its line rather than parsing the whole file.
    The slice has to land on the right node with the file's line numbers, and fall
    back cleanly when it can't be isolated — trailing code, nesting, or a header
    over several lines must all still come out right."""

    def _with(self, source, lineno):
        node = Tracer()._parse_block(source, lineno)
        assert isinstance(node, (ast.With, ast.AsyncWith))
        return node

    def test_indented_block_with_trailing_code(self):
        # Inside a function, code before and after: the slice stops at the block
        # and doesn't choke on the dedented lines after it.
        src = "def f():\n    a = 1\n    with ctx:\n        x = 1\n        y = 2\n    a = 3\n"
        node = self._with(src, 3)
        assert node.lineno == 3  # line numbers are the file's, not the slice's
        assert [s.lineno for s in node.body] == [4, 5]

    def test_multiline_header(self):
        src = "def f():\n    with ctx(\n        1,\n    ) as t:\n        x = 1\n    after = 2\n"
        node = self._with(src, 2)
        assert node.lineno == 2
        assert node.body[0].lineno == 5

    def test_nested_with_picks_the_outer(self):
        src = "with a:\n    with b:\n        x = 1\n"
        outer = self._with(src, 1)
        assert outer.lineno == 1
        inner = self._with(src, 2)
        assert inner.lineno == 2

    def test_non_with_line_returns_none(self):
        # No with at that line — parse() falls back to the whole-file scan.
        src = "def f():\n    a = 1\n    b = 2\n"
        assert Tracer()._parse_block(src, 2) is None

    def test_block_in_a_method_with_a_sibling_after(self):
        # A method whose class has another method after it: the truncation stops at
        # the block, so the sibling never enters the slice (dedenting it in would
        # make a broken block). The whole-file fallback would also be correct, but
        # this confirms the slice itself handles it.
        src = (
            "class C:\n"
            "    def f(self):\n"
            "        with ctx:\n"
            "            x = 1\n"
            "    def g(self):\n"
            "        return 2\n"
        )
        node = self._with(src, 3)
        assert node.lineno == 3
        assert node.body[0].lineno == 4

    def test_block_in_a_function_returning_after(self):
        # The everyday shape: a trace inside a function that returns after it.
        src = "def f(x):\n    with ctx:\n        y = g(x)\n    return y\n"
        node = self._with(src, 2)
        assert node.lineno == 2
        assert node.body[0].lineno == 3

    def test_deeply_indented_block(self):
        # Dedenting has to bring an arbitrarily-indented block to column 0.
        src = (
            "if a:\n"
            "    if b:\n"
            "        for c in d:\n"
            "            with ctx:\n"
            "                x = 1\n"
        )
        node = self._with(src, 4)
        assert node.lineno == 4
        assert node.body[0].lineno == 5

    def test_parse_falls_back_to_whole_file(self, monkeypatch):
        # When the slice can't isolate the block, parse() still finds it by parsing
        # the whole file — force the slice to give up and check the node still comes.
        src = "x = 0\n\ndef f():\n    with ctx:\n        y = 1\n"
        tracer = Tracer()
        monkeypatch.setattr(tracer, "_parse_block", lambda *a, **k: None)
        node = tracer.parse(src, 4)
        assert isinstance(node, ast.With) and node.lineno == 4
        assert node.body[0].lineno == 5

    def test_matches_whole_file_parse(self):
        # The slice and the old whole-file walk agree on the node.
        src = "x = 0\n\ndef f():\n    with ctx:\n        y = 1\n"
        sliced = self._with(src, 4)
        tree = ast.parse(src)
        whole = next(
            n for n in ast.walk(tree)
            if isinstance(n, (ast.With, ast.AsyncWith)) and n.lineno == 4
        )
        assert sliced.lineno == whole.lineno
        assert [s.lineno for s in sliced.body] == [s.lineno for s in whole.body]


class TestSkip:
    def test_original_skipped_recompiled_runs(self):
        ran = []
        with Tracer() as tracer:
            ran.append(1)
        assert ran == [1]
        assert tracer.info is not None

    def test_assignments_pushed_to_parent(self):
        with Tracer():
            computed = save(2 + 3)
            label = save("done")
        assert computed == 5
        assert label == "done"

    def test_body_on_the_with_line_is_refused(self):
        # The body is skipped via a per-line trace hook, so a body sharing the
        # `with` line would run where it stands and again through the backend.
        ran = []
        with pytest.raises(ValueError, match="own line"):
            with Tracer(): ran.append(1)  # noqa: E701
        assert ran == []

    def test_header_spanning_lines_binds_its_target(self):
        # A `with` header written over several lines reaches its own closing line
        # before the body — the skip has to wait for the body, or the block runs
        # with the `as` target never bound.
        ran = []

        class Bound(Tracer):
            pass

        with Bound(
            backend=None,
        ) as tracer:
            ran.append(tracer is not None)
        assert ran == [True]

    def test_pass_on_the_with_line_is_allowed(self):
        # Nothing to skip, so nothing runs twice.
        with Tracer(): pass  # noqa: E701

    def test_multiple_blocks_each_run(self):
        ran = []
        with Tracer():
            ran.append("a")
        with Tracer():
            ran.append("b")
        assert ran == ["a", "b"]

    def test_body_raise_propagates_from_exit(self):
        after = []
        with pytest.raises(RuntimeError) as ctx:
            with Tracer():
                raise RuntimeError("boom")
        assert str(ctx.value) == "boom"
        assert after == []

    def test_pass_body_does_not_leak(self):
        after = []
        with Tracer():
            pass
        after.append("reached")
        assert after == ["reached"]

    def test_ellipsis_body_does_not_leak(self):
        after = []
        with Tracer():
            ...
        after.append("reached")
        assert after == ["reached"]

    def test_noop_prefix_still_runs_real(self):
        ran = []
        with Tracer():
            pass
            ran.append("real")
        assert ran == ["real"]

    def test_read_then_reassign_escapes(self):
        base = 10
        with Tracer():
            total = save(base + 5)
            base = save(99)
        assert total == 15
        assert base == 99

    def test_traceback_points_to_original_source(self):
        try:
            with Tracer():
                value = 1
                raise ValueError("boom")
        except ValueError as error:
            summary = traceback.extract_tb(error.__traceback__)[-1]
            assert summary.filename.endswith("test_tracing.py")
            assert summary.line == 'raise ValueError("boom")'
            assert summary.name == sys._getframe().f_code.co_name
            assert error.__context__ is None
        else:
            pytest.fail("expected ValueError to propagate")


class TestInfo:
    def test_holds_frame_and_code(self):
        backend = _RecordingBackend()
        with Tracer(backend=backend) as tracer:
            value = 1
        # The frame/code are captured and available while the block runs...
        assert backend.frame is sys._getframe()
        assert isinstance(backend.info.code, types.CodeType)
        assert backend.info.code.co_name == sys._getframe().f_code.co_name
        # ...but the frame is released at exit so it can't pin the trace scope.
        assert tracer.info.frame is None


def _tb_files(tb):
    files = []
    while tb is not None:
        files.append(tb.tb_frame.f_code.co_filename)
        tb = tb.tb_next
    return files


def _error_from(filename):
    """A ValueError whose traceback runs through code compiled under filename."""
    code = compile("def boom():\n raise ValueError('x')\nboom()", filename, "exec")
    try:
        exec(code, {})
    except ValueError as error:
        return error


class TestTracebackUtils:
    def test_filter_traceback_keeps_matching_frames(self):
        from nnsight.tracing.util import filter_traceback

        error = _error_from("/u/blk.py")
        tb = filter_traceback(
            error.__traceback__, lambda frame: frame.f_code.co_filename == "/u/blk.py"
        )
        files = _tb_files(tb)
        assert files  # the block frames survived
        assert set(files) == {"/u/blk.py"}  # this test file's frame was dropped

    def test_filter_traceback_empty_is_none(self):
        from nnsight.tracing.util import filter_traceback

        error = _error_from("/u/blk.py")
        assert filter_traceback(error.__traceback__, lambda frame: False) is None

    def _clean_through_internal(self):
        # Route an error through a real nnsight internal frame (Tracer.execute).
        tracer = Tracer()
        tracer.info = Tracer.Info(sys._getframe(1), None)
        try:
            tracer.execute(compile("raise ValueError('x')", __file__, "exec"))
        except ValueError as error:
            from nnsight.tracing.util import clean_traceback

            return clean_traceback(error.__traceback__)

    def test_clean_traceback_drops_nnsight_frames(self, monkeypatch):
        # With DEBUG off, the nnsight internal frame is gone but the user's remains.
        monkeypatch.setattr(nnsight.CONFIG.APP, "DEBUG", False)
        files = _tb_files(self._clean_through_internal())
        assert files
        assert not any(f.endswith("tracer.py") for f in files)
        assert any(f.endswith("test_tracing.py") for f in files)

    def test_clean_traceback_debug_keeps_all_frames(self, monkeypatch):
        # With DEBUG on, nothing is stripped — the full stack (nnsight internals
        # included) is shown.
        monkeypatch.setattr(nnsight.CONFIG.APP, "DEBUG", True)
        assert any(f.endswith("tracer.py") for f in _tb_files(self._clean_through_internal()))

    def test_clean_traceback_keeps_all_user_files(self, monkeypatch):
        from nnsight.tracing.util import clean_traceback

        # Frames from any non-nnsight file are the user's own code and survive —
        # we hide framework internals, not "everything but one file".
        monkeypatch.setattr(nnsight.CONFIG.APP, "DEBUG", False)
        error = _error_from("/some/other/user_module.py")
        assert "/some/other/user_module.py" in _tb_files(
            clean_traceback(error.__traceback__)
        )

    def test_verbose_flag_enables_debug(self, monkeypatch):
        # `-v` / `--verbose` on the command line turns on debug mode at load.
        from nnsight.schema.config import Config

        for argv, expected in ([("prog", "-v"), True], [("prog", "--verbose"), True], [("prog",), False]):
            monkeypatch.setattr(sys, "argv", list(argv))
            config = Config()
            config._from_cli()
            assert config.APP.DEBUG is expected


@pytest.fixture
def tracer():
    return Tracer()


def body(node):
    return "\n".join(ast.unparse(statement) for statement in node.body)


class TestParse:
    def test_single_block(self, tracer):
        source = textwrap.dedent(
            """\
            with A():
                x = 1
                y = 2
            """
        )
        assert body(tracer.parse(source, 1)) == "x = 1\ny = 2"

    def test_multiple_blocks_select_by_lineno(self, tracer):
        source = textwrap.dedent(
            """\
            with A():
                a = 1
            with B():
                b = 2
            """
        )
        assert body(tracer.parse(source, 1)) == "a = 1"
        assert body(tracer.parse(source, 3)) == "b = 2"

    def test_multiline_header(self, tracer):
        source = textwrap.dedent(
            """\
            with A(
                1,
                2,
            ):
                z = 9
            """
        )
        assert body(tracer.parse(source, 1)) == "z = 9"

    def test_multiple_items_one_line(self, tracer):
        source = textwrap.dedent(
            """\
            with A(), B() as b:
                v = b
            """
        )
        assert body(tracer.parse(source, 1)) == "v = b"

    def test_nested_blocks(self, tracer):
        source = textwrap.dedent(
            """\
            with A():
                with B():
                    inner = 1
                outer = 2
            """
        )
        inner = tracer.parse(source, 2)
        assert isinstance(inner, ast.With)
        assert body(inner) == "inner = 1"
        outer = tracer.parse(source, 1)
        assert isinstance(outer.body[0], ast.With)
        assert "outer = 2" in body(outer)

    def test_async_with(self, tracer):
        source = textwrap.dedent(
            """\
            async def f():
                async with A():
                    q = 1
            """
        )
        node = tracer.parse(source, 2)
        assert isinstance(node, ast.AsyncWith)
        assert body(node) == "q = 1"

    def test_block_inside_function(self, tracer):
        source = textwrap.dedent(
            """\
            def f():
                with A():
                    r = 1
                    s = 2
            """
        )
        assert body(tracer.parse(source, 2)) == "r = 1\ns = 2"


class TestBuild:
    def test_builds_module_from_body(self, tracer):
        node = tracer.parse("with A():\n    a = 1\n    b = 2\n", 1)
        module = tracer.build(node)
        assert ast.unparse(module) == "a = 1\nb = 2"


class TestCompile:
    def build(self, tracer, body):
        node = tracer.parse(f"with A():\n    {body}\n", 1)
        return tracer.build(node)

    def test_returns_code_object(self, tracer):
        code = tracer.compile(self.build(tracer, "x = 1 + 1"), sys._getframe())
        assert isinstance(code, types.CodeType)
        variables = {}
        exec(code, {}, variables)
        assert variables["x"] == 2

    def test_executes_raise(self, tracer):
        code = tracer.compile(self.build(tracer, "raise ValueError('ran')"), sys._getframe())
        with pytest.raises(ValueError) as ctx:
            exec(code, {}, {})
        assert str(ctx.value) == "ran"

    def test_read_then_reassign(self, tracer):
        node = tracer.parse("with A():\n    y = base + 5\n    base = 99\n", 1)
        code = tracer.compile(tracer.build(node), sys._getframe())
        variables = {"base": 10}
        exec(code, {}, variables)
        assert variables["y"] == 15
        assert variables["base"] == 99

    def test_renames_code_to_caller(self, tracer):
        code = tracer.compile(self.build(tracer, "x = 1"), sys._getframe())
        assert code.co_name == sys._getframe().f_code.co_name


class TestNesting:
    def test_block_inside_function(self):
        def fn():
            ran = []
            with Tracer() as tracer:
                ran.append(1)
            return ran, tracer.info

        ran, info = fn()
        assert ran == [1]
        assert isinstance(info.code, types.CodeType)

    def test_block_inside_nested_function(self):
        def outer():
            def inner():
                ran = []
                with Tracer() as tracer:
                    ran.append("deep")
                return ran, tracer.info

            return inner()

        ran, info = outer()
        assert ran == ["deep"]
        assert isinstance(info.code, types.CodeType)

    def test_info_captures_frame(self):
        sentinel = object()
        backend = _RecordingBackend()

        def fn():
            local_value = sentinel
            with Tracer(backend=backend):
                pass

        fn()
        # The captured frame exposed fn's locals and globals to the backend.
        assert backend.frame.f_locals["local_value"] is sentinel
        assert "Tracer" in backend.frame.f_globals


_SRC = os.path.join(os.path.dirname(__file__), "..", "src")


class TestBlockCache:
    def test_repeated_site_reuses_compiled_code(self):
        codes = []
        for _ in range(3):
            with Tracer() as tracer:
                value = 1
            codes.append(tracer.info.code)
        # Every iteration is the same source line -> one shared compiled block.
        assert codes[0] is codes[1] is codes[2]

    def test_repeated_site_parses_once(self, monkeypatch):
        from nnsight.tracing import globals as g

        g.BLOCKS.clear()
        calls = []
        original = Tracer.parse
        monkeypatch.setattr(
            Tracer,
            "parse",
            lambda self, source, lineno: calls.append(1) or original(self, source, lineno),
        )
        for _ in range(5):
            with Tracer():
                x = 1
        assert len(calls) == 1  # parsed on the first entry, cached after

    def test_not_a_with_block_verdict_is_cached(self, monkeypatch):
        from nnsight.tracing import globals as g
        from nnsight.tracing.tracer import WithBlockNotFoundError

        def capture_here():
            # capture()'s frame is this function's caller — the fixed line below.
            Tracer().capture()

        g.BLOCKS.clear()
        calls = []
        original = Tracer.parse
        monkeypatch.setattr(
            Tracer,
            "parse",
            lambda self, source, lineno: calls.append(1) or original(self, source, lineno),
        )
        # The call site isn't a ``with`` block, so capture() raises; the negative
        # verdict is cached so repeated calls at the same site don't re-parse.
        for _ in range(4):
            with pytest.raises(WithBlockNotFoundError):
                capture_here()
        assert len(calls) == 1


class TestSource:
    def test_reads_and_caches_file_source(self):
        from nnsight.tracing.globals import SOURCES

        with Tracer():
            token = save("cached-marker")
        assert token in SOURCES[__file__]

    def test_source_not_repulled_after_edit(self, tmp_path):
        from nnsight.tracing.globals import SOURCES

        module = tmp_path / "block_mod.py"
        original = (
            "from nnsight.tracing.tracer import Tracer\n"
            "from nnsight.tracing.tracer import save\n"
            "with Tracer():\n"
            "    marker = save('ORIGINAL')\n"
        )
        module.write_text(original)
        first = runpy.run_path(str(module))
        assert first["marker"] == "ORIGINAL"

        # Edit the file on disk (and clear linecache, as a reused interpreter
        # might). The tracer must keep tracing the source as first seen.
        module.write_text(original.replace("ORIGINAL", "EDITED"))
        linecache.checkcache(str(module))

        second = runpy.run_path(str(module))
        assert second["marker"] == "ORIGINAL"  # edit was not pulled

        SOURCES.pop(str(module), None)

    def test_dash_c_source_recovered_from_argv(self):
        code = (
            "from nnsight.tracing.tracer import Tracer\n"
            "from nnsight.tracing.tracer import save\n"
            "with Tracer():\n"
            "    value = save(6 * 7)\n"
            "print(value)\n"
        )
        env = dict(os.environ, PYTHONPATH=_SRC)
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "42"

    def test_notebook_cell_source_from_linecache(self):
        pytest.importorskip("IPython")
        from IPython.core.interactiveshell import InteractiveShell

        shell = InteractiveShell.instance()
        cell = (
            "from nnsight.tracing.tracer import Tracer\n"
            "from nnsight.tracing.tracer import save\n"
            "with Tracer():\n"
            "    result = save(6 * 7)\n"
        )
        outcome = shell.run_cell(cell, store_history=True)
        assert outcome.error_in_exec is None
        assert shell.user_ns["result"] == 42

    def test_dash_c_trace_inside_function(self):
        # A function defined in -c code also compiles under "<string>".
        code = (
            "from nnsight.tracing.tracer import Tracer\n"
            "from nnsight.tracing.tracer import save\n"
            "def f():\n"
            "    with Tracer():\n"
            "        v = save(123)\n"
            "    return v\n"
            "print(f())\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=_SRC),
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "123"

    def test_dash_c_does_not_hijack_unrelated_dynamic_exec(self):
        # Under a -c launch, an unrelated exec of a differently-named block must
        # NOT resolve to the -c program's source. Both blocks put a `with` on the
        # same line, so a hijack would silently run the wrong body; the guard
        # instead surfaces WithBlockNotFoundError.
        outer = (
            "from nnsight.tracing.tracer import Tracer\n"  # line 1
            "bucket = []\n"                                  # line 2
            "with Tracer():\n"                               # line 3
            "    bucket.append('OUTER')\n"                   # line 4
            "inner = 'x = 1\\ny = 2\\nwith Tracer():\\n    bucket.append(\"INNER\")'\n"
            "exec(compile(inner, '<gen>', 'exec'), globals())\n"
            "print(bucket)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", outer],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=_SRC),
        )
        assert out.returncode != 0
        assert "WithBlockNotFoundError" in out.stderr
        assert "OUTER" not in out.stdout  # never silently mis-traced

    def test_dynamic_exec_traceable_via_linecache(self):
        # The supported way to trace dynamically-exec'd code: give it a name and
        # register the source in linecache.
        name = "<generated-block>"
        src = (
            "from nnsight.tracing.tracer import Tracer\n"
            "from nnsight.tracing.tracer import save\n"
            "with Tracer():\n"
            "    produced = save(7 * 3)\n"
        )
        linecache.cache[name] = (len(src), None, src.splitlines(keepends=True), name)
        try:
            namespace = {}
            exec(compile(src, name, "exec"), namespace)
            assert namespace["produced"] == 21
        finally:
            linecache.cache.pop(name, None)
            from nnsight.tracing.globals import SOURCES

            SOURCES.pop(name, None)

    def test_notebook_trace_in_function_from_earlier_cell(self):
        pytest.importorskip("IPython")
        from IPython.core.interactiveshell import InteractiveShell

        shell = InteractiveShell.instance()
        # Define the traced function in one cell...
        shell.run_cell(
            "from nnsight.tracing.tracer import Tracer\n"
            "from nnsight.tracing.tracer import save\n"
            "def run():\n"
            "    with Tracer():\n"
            "        inner = save(2 * 21)\n"
            "    return inner\n",
            store_history=True,
        )
        # ...and call it from a later cell. Source must come from the function's
        # own (earlier) cell via linecache, not the current cell's input.
        outcome = shell.run_cell("out = run()", store_history=True)
        assert outcome.error_in_exec is None
        assert shell.user_ns["out"] == 42
