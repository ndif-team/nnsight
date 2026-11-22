"""Source-based serialization round-trips (``dumps``/``loads``) and the
``remote="local"`` dry-run of the remote path.
"""

import asyncio
import gc
import linecache
import sys
import textwrap

import pickle
import pickletools
import weakref

import nnsight
import pytest
import torch

from envoy_classes import Doubled, WrappedModel, script_class
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import eproperty
from nnsight.intervention.serialization import (
    CustomCloudUnpickler,
    UnknownPersistentIdError,
    code_reduce,
    dumps,
    loads,
)
from nnsight.ndif import get_local_env
from nnsight.schema.request import RequestModel
from nnsight.tracing.backend import Backend


# Module-level mutually recursive functions (reference each other via globals).
def is_even(n):
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n):
    if n == 0:
        return False
    return is_even(n - 1)


# A local (test-module) helper used to prove ship-by-value under remote="local".
def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


# Telling apart two same-signature lambdas written on one line needs the column
# information in `code.co_positions()`, which is 3.11+. Elsewhere (different
# signatures, different lines) the source parse alone picks the right one.
same_line_lambdas = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="same-line, same-signature lambdas need code.co_positions (3.11+)",
)


class TestScopeFiltering:
    """Only real variable references travel with a block, not attribute names."""

    def test_attribute_name_shadowing_a_global_is_not_shipped(self):
        # `llm.model` is an attribute; a global that happens to be spelled
        # `model` must not be dragged into the payload. co_names cannot tell the
        # two apart, so this used to ship an unrelated model -- whose envoys then
        # claimed a conflicting Module:<path> id on the server.
        source = "h = llm.model.layers[-1].output.save()"
        globals_ = {"llm": "the-traced-model", "model": "an-unrelated-model"}
        _, used_globals, _ = code_reduce(source, globals_, {})
        assert used_globals == {"llm": "the-traced-model"}

    def test_common_attribute_names_do_not_leak(self):
        source = "x = llm.output.save(); y = llm.config.tokenizer"
        globals_ = {
            "llm": "the-traced-model",
            "output": "unrelated",
            "config": "unrelated",
            "tokenizer": "unrelated",
        }
        _, used_globals, _ = code_reduce(source, globals_, {})
        assert used_globals == {"llm": "the-traced-model"}

    def test_names_used_in_nested_scopes_still_ship(self):
        # Comprehensions and lambdas have their own code objects; names they
        # reference must still be collected.
        source = "vals = [torch.relu(t) for t in xs]\nf = lambda z: helper(z)"
        globals_ = {"torch": "torch-mod", "helper": "helper-fn", "xs": [1]}
        _, used_globals, _ = code_reduce(source, globals_, {})
        assert used_globals == {"torch": "torch-mod", "helper": "helper-fn", "xs": [1]}

    def test_names_the_block_only_binds_are_not_shipped(self):
        # `tracer` is a local of the block -- the enclosing scope's same-named
        # object is about to be shadowed, so shipping it is pointless. It is also
        # unsafe: a stale Tracer from an earlier cell holds a dead frame and
        # cannot be pickled at all.
        source = "with model.trace(prompt) as tracer:\n    h = model.layer1.output"
        globals_ = {"model": "the-model", "prompt": "hi", "tracer": "a-stale-tracer"}
        _, used_globals, _ = code_reduce(source, globals_, {})
        assert used_globals == {"model": "the-model", "prompt": "hi"}

    def test_loop_and_assignment_targets_are_not_shipped(self):
        # `count` is only ever bound; `i` is bound and then read, so it stays --
        # the filter is on reads, not on whether the name is also a local.
        source = "for i in items:\n    count = 1\n    use(i)"
        globals_ = {"items": [1, 2], "use": "fn", "i": "stale", "count": "stale"}
        _, used_globals, _ = code_reduce(source, globals_, {})
        assert used_globals == {"items": [1, 2], "use": "fn", "i": "stale"}

    def test_augmented_assignment_target_still_ships(self):
        # `total += x` reads `total` before writing it, even though the AST marks
        # the target as a Store.
        source = "total += x"
        globals_ = {"total": 1, "x": 2}
        _, used_globals, _ = code_reduce(source, globals_, {})
        assert used_globals == {"total": 1, "x": 2}

    def test_deleted_name_still_ships(self):
        _, used_globals, _ = code_reduce("del junk", {"junk": "present"}, {})
        assert used_globals == {"junk": "present"}

    def test_function_locals_are_not_shipped(self):
        # A function is not a block: a name assigned anywhere in the body is
        # local to it, so the enclosing scope's same-named object is never what
        # the body reads. Capturing it dragged a notebook's leftover closed file
        # handle into the payload, where pickling failed outright.
        source = (
            "def load(path):\n"
            "    with open(path) as f:\n"
            "        return json.load(f)\n"
        )
        globals_ = {"json": "json-mod", "open": "builtin", "f": "a-closed-file"}
        _, used_globals, _ = code_reduce(source, globals_, {}, function=True)
        assert "f" not in used_globals
        assert used_globals["json"] == "json-mod"

    def test_function_globals_still_ship(self):
        source = (
            "def score(xs):\n"
            "    return [WEIGHT * torch.relu(x) for x in xs]\n"
        )
        globals_ = {"WEIGHT": 2, "torch": "torch-mod", "xs": "stale", "x": "stale"}
        _, used_globals, _ = code_reduce(source, globals_, {}, function=True)
        assert used_globals == {"WEIGHT": 2, "torch": "torch-mod"}

    def test_function_global_declaration_still_ships(self):
        # `global` makes the assignment target a module global after all.
        source = "def bump():\n    global counter\n    counter += 1\n"
        _, used_globals, _ = code_reduce(source, {"counter": 0}, {}, function=True)
        assert used_globals == {"counter": 0}

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param("def f(x):\n    return g(x)\n", id="plain"),
            pytest.param("def f(x: int) -> int:\n    return g(x)\n", id="annotated"),
            pytest.param("def f(x=1):\n    return g(x)\n", id="default"),
            pytest.param("f = lambda x: g(x)\n", id="lambda"),
        ],
    )
    def test_scope_filter_survives_extra_symtable_children(self, source):
        """The function's own symtable child must be found by type, not by count.

        A single ``def`` does not yield a single child table on every Python.
        Under PEP 649 (3.14+) it also gets an ``__annotate__`` table -- with or
        without annotations present -- and PEP 695 generics add a
        ``type_parameters`` table. Selecting the child by counting meant every
        ``def`` fell through to the block rule on 3.14, so the whole filter was
        silently inert there while the 3.12 CI runner stayed green.

        ``x`` is the parameter, so it is local no matter what: if it ships, the
        filter was bypassed.
        """
        _, used_globals, _ = code_reduce(
            source, {"g": "g-mod", "x": "stale"}, {}, function=True
        )

        assert used_globals == {"g": "g-mod"}

    def test_function_local_shadowing_an_unpicklable_global(self, tmp_path):
        """End-to-end: the payload must not capture an unpicklable shadowed name.

        This is the failure the filter was written for. A closed file cannot be
        pickled at all, so capturing the enclosing scope's ``f`` does not merely
        bloat the payload -- serializing it raises.
        """
        handle = open(tmp_path / "left-over.txt", "w")
        handle.close()

        source = (
            "def load(path):\n"
            "    with open(path) as f:\n"
            "        return f.read()\n"
        )

        _, used_globals, _ = code_reduce(source, {"f": handle}, {}, function=True)

        assert "f" not in used_globals
        # Nothing left that pickle would choke on.
        dumps(used_globals)


class TestLambda:
    def test_simple_lambda(self):
        assert loads(dumps(lambda x: x * 2))(5) == 10

    def test_lambda_with_closure(self):
        multiplier = 3
        assert loads(dumps(lambda x: x * multiplier))(5) == 15

    def test_lambda_with_multiple_args(self):
        assert loads(dumps(lambda x, y, z: x + y + z))(1, 2, 3) == 6

    def test_lambda_with_default_args(self):
        f = loads(dumps(lambda x, y=10: x + y))
        assert f(5) == 15 and f(5, 20) == 25
        # Dict default with colons exercises depth tracking in the source parse.
        g = loads(dumps(lambda x={"a": 1, "b": 2}: x["a"] + x["b"]))
        assert g() == 3 and g({"a": 10, "b": 20}) == 30

    def test_nested_lambda(self):
        assert loads(dumps(lambda x: lambda y: x + y))(5)(3) == 8

    def test_lambda_as_default_value(self):
        f = loads(dumps(lambda x=lambda: 1: x()))
        assert f() == 1 and f(lambda: 42) == 42

    @same_line_lambdas
    def test_multiple_lambdas_same_line(self):
        f, g = lambda x: x * 2, lambda x: x + 1
        assert loads(dumps(f))(5) == 10
        assert loads(dumps(g))(5) == 6

    def test_multiline_lambda_with_neighbors(self):
        # fmt: off
        a, b, c, d = lambda x: x + 1, lambda y: (
            y * 2 +
            y * 3
        ), (lambda z:
            z - 1), lambda w: w * 2
        # fmt: on
        assert loads(dumps(a))(10) == 11
        assert loads(dumps(b))(10) == 50
        assert loads(dumps(c))(10) == 9
        assert loads(dumps(d))(10) == 20

    @same_line_lambdas
    def test_lambda_in_list(self):
        funcs = loads(dumps([lambda x: x + 1, lambda x: x * 2, lambda x: x ** 2]))
        assert funcs[0](5) == 6 and funcs[1](5) == 10 and funcs[2](5) == 25

    @same_line_lambdas
    def test_lambda_in_dict(self):
        ops = loads(dumps({"add": lambda x, y: x + y, "mul": lambda x, y: x * y}))
        assert ops["add"](10, 3) == 13 and ops["mul"](10, 3) == 30


class TestFunctions:
    def test_closure_with_captured_variable(self):
        multiplier = 3.0

        def scale_by_closure(x):
            return x * multiplier

        data = dumps(scale_by_closure)
        assert b"def scale_by_closure" in data
        assert torch.allclose(
            loads(data)(torch.tensor([1.0, 2.0, 3.0])), torch.tensor([3.0, 6.0, 9.0])
        )

    def test_nested_closure(self):
        outer_val = 10

        def outer_func(x):
            inner_val = 5

            def inner_func(y):
                return y + outer_val + inner_val

            return inner_func(x)

        assert loads(dumps(outer_func))(1) == 16

    def test_returned_inner_function(self):
        def make_adder(n):
            def adder(x):
                return x + n

            return adder

        assert loads(dumps(make_adder(5)))(10) == 15

    def test_triple_nested_closure(self):
        def level1(a):
            def level2(b):
                def level3(c):
                    return a + b + c

                return level3

            return level2

        assert loads(dumps(level1(1)(2)))(3) == 6

    def test_mutable_closure_state(self):
        def make_counter():
            count = [0]

            def increment():
                count[0] += 1
                return count[0]

            return increment

        counter = make_counter()
        counter()
        counter()
        restored = loads(dumps(counter))
        assert restored() == 3 and restored() == 4

    def test_lambda_default_argument(self):
        def func_with_lambda_default(processor=lambda x: x * 2):
            return processor(5)

        data = dumps(func_with_lambda_default)
        assert b"lambda" in data
        restored = loads(data)
        assert restored() == 10 and restored(lambda x: x + 1) == 6

    def test_complex_default_arguments(self):
        def func_with_defaults(a, b=10, c="hello", d=None, e=[1, 2, 3], f={"k": "v"}):
            return (a, b, c, d, e, f)

        r = loads(dumps(func_with_defaults))(1)
        assert r == (1, 10, "hello", None, [1, 2, 3], {"k": "v"})

    def test_comprehension_variable_shadowing(self):
        x = 42

        def uses_external_and_comprehension():
            external_value = x
            squares = [x * x for x in range(5)]
            return external_value, squares

        external, squares = loads(dumps(uses_external_and_comprehension))()
        assert external == 42 and squares == [0, 1, 4, 9, 16]

    def test_lambda_parameters_not_external(self):
        def func_with_lambda():
            fn = lambda x: x * 2
            return fn(5)

        assert loads(dumps(func_with_lambda))() == 10

    def test_lambda_with_external_and_param(self):
        multiplier = 3

        def func_with_mixed():
            fn = lambda x: x * multiplier
            return fn(5)

        assert loads(dumps(func_with_mixed))() == 15

    def test_constant_overriding_builtin(self):
        len = 42  # noqa: F841 - shadows a builtin on purpose

        def uses_overridden_len():
            return len

        assert loads(dumps(uses_overridden_len))() == 42

    def test_nested_function_shadowing(self):
        multiplier = 3

        def outer_uses_multiplier():
            base = multiplier * 10

            def inner(multiplier):
                return multiplier * 2

            return base, inner(5)

        base, inner_result = loads(dumps(outer_uses_multiplier))()
        assert base == 30 and inner_result == 10

    def test_import_inside_function(self):
        def func_with_import():
            import json

            return json.dumps([1, 2, 3])

        assert loads(dumps(func_with_import))() == "[1, 2, 3]"

    def test_from_import_inside_function(self):
        def func_with_from_import():
            from collections import Counter

            return Counter([1, 1, 2, 3]).most_common(1)

        assert loads(dumps(func_with_from_import))() == [(1, 2)]

    def test_async_function(self):
        async def async_double(x):
            await asyncio.sleep(0)
            return x * 2

        data = dumps(async_double)
        assert b"async def" in data
        restored = loads(data)
        assert asyncio.iscoroutinefunction(restored)
        assert asyncio.run(restored(5)) == 10

    def test_generator_function(self):
        def generate_numbers(n):
            for i in range(n):
                yield i * 2

        data = dumps(generate_numbers)
        assert b"yield" in data
        assert list(loads(data)(5)) == [0, 2, 4, 6, 8]

    def test_deeply_nested_closure_state(self):
        config = {"a": {"b": {"c": {"value": 42, "tensor": torch.tensor([1.0])}}}}

        def access_deep_config():
            return config["a"]["b"]["c"]["value"]

        assert loads(dumps(access_deep_config))() == 42

    def test_args_kwargs(self):
        def variadic_func(*args, **kwargs):
            return sum(args) + sum(kwargs.values())

        assert loads(dumps(variadic_func))(1, 2, 3, a=4, b=5) == 15

    def test_mixed_positional_keyword_args(self):
        def mixed_args(a, b, *args, c=10, **kwargs):
            return a + b + sum(args) + c + sum(kwargs.values())

        assert loads(dumps(mixed_args))(1, 2, 3, 4, c=5, d=6) == 21

    def test_function_with_annotations(self):
        def annotated_func(x: int, y: float = 1.0) -> float:
            return x * y

        restored = loads(dumps(annotated_func))
        assert restored(5, 2.0) == 10.0
        assert restored.__annotations__ == {"x": int, "y": float, "return": float}

    def test_function_with_docstring(self):
        def documented_func(x):
            """This function doubles its input."""
            return x * 2

        restored = loads(dumps(documented_func))
        assert restored.__doc__ is not None and "doubles" in restored.__doc__

    def test_function_referencing_other_function(self):
        def helper(x):
            return x + 1

        def main_func(x):
            return helper(x) * 2

        assert loads(dumps(main_func))(5) == 12

    def test_shared_closure(self):
        shared_value = 10

        def func1():
            return shared_value * 2

        def func2():
            return shared_value + 5

        assert loads(dumps(func1))() == 20
        assert loads(dumps(func2))() == 15

    def test_error_preserves_context(self):
        def error_func(x):
            y = x + 1
            z = y / 0
            return z

        with pytest.raises(ZeroDivisionError):
            loads(dumps(error_func))(5)

    def test_empty_function(self):
        def empty_func():
            pass

        assert loads(dumps(empty_func))() is None


class TestRecursion:
    def test_module_level_recursion(self):
        def factorial(n):
            return 1 if n <= 1 else n * factorial(n - 1)

        assert loads(dumps(factorial))(5) == 120

    def test_local_recursive_function(self):
        def outer():
            def factorial(n):
                return 1 if n <= 1 else n * factorial(n - 1)

            return factorial

        restored = loads(dumps(outer()))
        assert restored(5) == 120 and restored(10) == 3628800

    def test_local_mutual_recursion(self):
        def local_is_even(n):
            return True if n == 0 else local_is_odd(n - 1)

        def local_is_odd(n):
            return False if n == 0 else local_is_even(n - 1)

        restored_even, restored_odd = loads(dumps((local_is_even, local_is_odd)))
        assert restored_even(4) is True and restored_even(5) is False
        assert restored_odd(4) is False and restored_odd(5) is True

    def test_module_mutual_recursion_single(self):
        restored = loads(dumps(is_even))
        assert restored(4) is True and restored(5) is False

    def test_module_mutual_recursion_together(self):
        restored_even, restored_odd = loads(dumps((is_even, is_odd)))
        assert restored_even(4) is True and restored_odd(5) is True


class TestLocalEnv:
    def test_no_stdlib_modules_marked_as_local(self):
        # Regression for https://github.com/ndif-team/nnsight/issues/619: a
        # platform-dependent path check once marked stdlib modules "local".
        packages = get_local_env().get("packages", {})
        local_stdlib = {
            name
            for name in packages
            if packages[name] == "local" and name in sys.stdlib_module_names
        }
        assert local_stdlib == set()


class TestLinecache:
    """A helper serialized on one machine must be introspectable where its file
    is absent — its source is re-registered in ``linecache`` on load."""

    def test_deserialized_helper_source_registered(self, tmp_path):
        filename = str(tmp_path / "missing_helper.py")
        source = textwrap.dedent(
            """\
            def helper(x):
                y = x + 1
                return y * 2
            """
        )
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        namespace = {}
        exec(compile(source, filename, "exec"), namespace)
        helper = namespace["helper"]

        data = dumps(helper)
        linecache.cache.pop(filename, None)  # simulate the file being absent

        restored = loads(data)
        assert restored(5) == 12
        assert restored.__code__.co_firstlineno == helper.__code__.co_firstlineno
        # The source is registered under a labelled filename ("[helper:1] <path>")
        # so several functions from one file don't clobber each other; the original
        # path is retained as a suffix and the source is retrievable even though the
        # file is gone.
        import inspect

        assert filename in restored.__code__.co_filename
        assert "y * 2" in inspect.getsource(restored)

    def test_two_functions_same_file_no_collision(self, tmp_path):
        filename = str(tmp_path / "shared_helpers.py")
        source = textwrap.dedent(
            """\
            def helper_a(x):
                return x + 1

            def helper_b(x):
                return x * 10
            """
        )
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        namespace = {}
        exec(compile(source, filename, "exec"), namespace)
        a, b = namespace["helper_a"], namespace["helper_b"]
        assert a.__code__.co_firstlineno != b.__code__.co_firstlineno

        data_a, data_b = dumps(a), dumps(b)
        linecache.cache.pop(filename, None)

        ra, rb = loads(data_a), loads(data_b)
        assert ra(5) == 6 and rb(5) == 50
        assert ra.__code__.co_firstlineno == a.__code__.co_firstlineno
        assert rb.__code__.co_firstlineno == b.__code__.co_firstlineno


class TestLocalSimulation:
    """``remote="local"`` serializes the trace, deserializes it with local
    modules hidden (mimicking the server), and runs it in-process."""

    def test_basic_trace_round_trips(self, gpt2):
        with gpt2.trace("The Eiffel Tower is in", remote="local"):
            hidden = gpt2.transformer.h[0].output.save()
        assert isinstance(hidden, torch.Tensor) and hidden.ndim == 3

    def test_intervention_round_trips(self, gpt2):
        with gpt2.trace("The Eiffel Tower is in", remote="local"):
            gpt2.transformer.h[0].output[:] = 0
            zeroed = gpt2.transformer.h[0].output.save()
        assert torch.all(zeroed == 0)

    def test_local_module_function_ships_by_value(self, gpt2):
        # `normalize` lives in this (local, non-installed) test module; the local
        # backend hides it during deserialize, so it must have shipped by value.
        with gpt2.trace("The Eiffel Tower is in", remote="local"):
            h = gpt2.transformer.h[0].output
            normed = normalize(h).save()
        norms = normed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_generate_round_trips(self, gpt2):
        with gpt2.generate("The Eiffel Tower is in", max_new_tokens=3, remote="local"):
            out = gpt2.transformer.h[0].output.save()
        assert isinstance(out, torch.Tensor)


@pytest.fixture(scope="module")
def server_gpt2():
    """A second gpt2, the receiver's: a name in a payload resolves to its tree, so a
    test can tell the receiver's envoy from the sender's."""
    from nnsight.modeling.transformers import TransformersModel

    return TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)


@pytest.fixture(scope="module")
def block_renamed_gpt2():
    """A gpt2 with an alias bound on every block (`self_attn`), and nothing on the
    root: the root reaches the alias by attribute access, so it goes by value for
    the blocks' sake."""
    from nnsight.modeling.transformers import TransformersModel

    return TransformersModel(
        "openai-community/gpt2",
        task="text-generation",
        dispatch=True,
        rename={"attn": "self_attn"},
    )


@pytest.fixture(scope="module")
def classed_gpt2():
    from nnsight.modeling.transformers import TransformersModel

    return TransformersModel(
        "openai-community/gpt2",
        task="text-generation",
        dispatch=True,
        envoys={"mlp": Doubled},
    )


@pytest.fixture(scope="module")
def renamed_gpt2():
    """A gpt2 with an alias bound on the root (`layers`) and one on `transformer`
    (`blocks`): both go by value, and everything below `transformer` by reference."""
    from nnsight.modeling.transformers import TransformersModel

    return TransformersModel(
        "openai-community/gpt2",
        task="text-generation",
        dispatch=True,
        rename={"transformer.h": "layers", "h": "blocks"},
    )


class _ServerRoundTrip(Backend):
    """Serialize a trace and run the *deserialized* tracer, as an NDIF server does.

    ``remote="local"`` runs the original in-process tracer (which still carries its
    AST node) so results push back into the caller's frame; a real server runs the
    tracer rebuilt from the payload, whose node was dropped in transit. This backend
    exercises that server path so the node fallback is covered where it matters.
    ``model`` is the server's: the payload's names resolve on its tree.
    """

    def __init__(self, model):
        self.model = model
        self.ran = False
        self.node_after = "unset"

    def __call__(self, tracer):
        blob = self.blob = RequestModel.serialize(tracer, compress=False)
        persistent = self.model._remoteable_persistent_objects()
        server_tracer = RequestModel.deserialize(blob, persistent, compress=False)
        self.node_after = server_tracer.node
        server_tracer.execute(server_tracer.info.code)
        # What the block saved, as the server's frame holds it after the run.
        self.saved = dict(server_tracer.info.frame.f_locals)
        self.ran = True


PROMPT = "The Eiffel Tower is in"


def _matches(a, b):
    # CPU forward passes have differed by 1.75e-3 with no serialization involved
    # (cause not determined), and a wrong value differs by far more.
    return torch.allclose(a.detach().float(), b.detach().float(), atol=1e-2)


def _names_in(blob):
    """The persistent ids a payload writes."""
    names, last = set(), None
    for opcode, argument, _ in pickletools.genops(blob):
        if isinstance(argument, str):
            last = argument
        elif opcode.name == "BINPERSID":
            names.add(last)
    return names


def _copy_of(copy, envoy):
    """``copy`` is the payload's copy of the receiver's ``envoy``: same class, same
    path, the receiver's module, distinct object."""
    return (
        copy is not envoy
        and copy.path == envoy.path
        and type(copy) is type(envoy)
        and copy._module is envoy._module
    )


class TestServerExecution:
    """The deserialized tracer runs server-side even though serialization drops its
    AST node (needed only where the block is captured)."""

    @torch.no_grad()
    def test_deserialized_tracer_executes(self, gpt2):
        backend = _ServerRoundTrip(gpt2)
        with gpt2.trace("The Eiffel Tower is in the city of", backend=backend):
            gpt2.transformer.h[0].output.save()
        assert backend.ran
        assert backend.node_after is None  # dropped in transit, falls back to None

    @torch.no_grad()
    def test_deserialized_generate_executes(self, gpt2):
        backend = _ServerRoundTrip(gpt2)
        with gpt2.generate(
            "The Eiffel Tower is in", max_new_tokens=3, backend=backend
        ):
            gpt2.transformer.h[0].output.save()
        assert backend.ran

    def test_push_into_a_serialized_frame(self):
        # A deserialized tracer's "frame" is a SerializedFrame, not a real one, so
        # push must not hand it to PyFrame_LocalsToFast: that dereferences it as a
        # PyFrameObject (SIGSEGV on 3.10, silent UB on 3.11/3.12). The plain dict
        # update is the whole write-back here.
        from nnsight.tracing.util import SerializedFrame, push

        frame = SerializedFrame("<block>", 1, "blk")
        push(frame, {"a": 1})
        assert frame.f_locals == {"a": 1}


class TestParentLinks:
    """The weak parent link is not pickled; it is rebuilt on the receiving side."""

    @staticmethod
    def _assert_parents(root):
        for node in root.modules():
            for child in node._children:
                assert child.parent is node, child.path

    def test_a_by_value_envoy_relinks_the_children_that_came_with_it(self, renamed_gpt2, server_gpt2):
        # The root and `transformer` go by value (each has an alias bound on it);
        # `h` below them is the receiver's own and keeps the receiver's parent.
        copy = loads(dumps(renamed_gpt2, tree=renamed_gpt2), server_gpt2._remoteable_persistent_objects())
        assert copy is not server_gpt2 and copy.parent is None
        assert copy.transformer is not server_gpt2.transformer
        assert copy.transformer.parent is copy
        assert _copy_of(copy.transformer.h, server_gpt2.transformer.h)
        assert copy.transformer.h.parent is copy.transformer
        assert copy.transformer.h[2].mlp.root is copy

    def test_a_shipped_subtree_is_its_own_root(self, renamed_gpt2, server_gpt2):
        # Shipped without its parent, the top envoy has none; below it, the receiver's own.
        transformer = loads(
            dumps(renamed_gpt2.transformer, tree=renamed_gpt2),
            server_gpt2._remoteable_persistent_objects(),
        )
        assert transformer is not server_gpt2.transformer
        assert transformer.parent is None and transformer.root is transformer
        assert transformer.h.parent is transformer

    @torch.no_grad()
    def test_a_deserialized_trace_reads_through_parent(self, gpt2):
        # The server runs the block against the envoys rebuilt from the payload.
        backend = _ServerRoundTrip(gpt2)
        with gpt2.trace("The Eiffel Tower is in", backend=backend):
            gpt2.transformer.h[0].attn.parent.mlp.output.save()
        assert backend.ran

    @torch.no_grad()
    def test_remote_local_trace_reads_through_parent(self, gpt2):
        with gpt2.trace("The Eiffel Tower is in", remote="local"):
            block = gpt2.transformer.h[0].attn.parent
            hidden = block.mlp.output.save()
        with gpt2.trace("The Eiffel Tower is in"):
            expected = gpt2.transformer.h[0].mlp.output.save()
        assert torch.equal(hidden, expected)


def _seed_id_cache():
    # to_model_key canonicalizes the repo id via the Hub; pre-seed the cache so the
    # key builds without a network round-trip.
    from nnsight.modeling.huggingface import _ID_CACHE

    _ID_CACHE["openai-community/gpt2"] = "openai-community/gpt2"


class TestModelKey:
    """A remote key names the class the server knows a model by. The deprecated
    aliases (LanguageModel, VisionLanguageModel) share the base TransformersModel's
    key rather than minting their own, so a model deployed as one is reachable when
    a client wraps it as the other."""

    def test_transformers_key_names_its_class(self, gpt2):
        _seed_id_cache()
        assert gpt2.to_model_key().split(":", 1)[0] == (
            "nnsight.modeling.transformers.TransformersModel"
        )

    def test_language_model_shares_the_transformers_key(self, gpt2):
        _seed_id_cache()
        from nnsight import LanguageModel

        lm = LanguageModel("openai-community/gpt2")
        assert lm.to_model_key() == gpt2.to_model_key()

    def test_vision_language_model_inherits_the_redirect(self):
        # VLM defines no redirect of its own; it shares LanguageModel's, so both
        # resolve their key class to TransformersModel (checked without building a
        # VLM, which needs a multimodal checkpoint).
        from nnsight import LanguageModel, VisionLanguageModel

        assert (
            VisionLanguageModel._remoteable_class is LanguageModel._remoteable_class
        )

    def test_key_reconstructs_to_transformers_model(self):
        from nnsight.modeling.transformers import TransformersModel
        from nnsight.util import from_import_path

        import_path = "nnsight.modeling.transformers.TransformersModel"
        assert from_import_path(import_path) is TransformersModel

    def test_key_carries_the_resolved_task(self, gpt2):
        # The task is part of the remote identity: the server rebuilds the
        # pipeline from the key rather than re-inferring one from the Hub, and
        # two tasks over one checkpoint are two deployments. Always resolved —
        # gpt2 was built without a task, and the key still names one.
        import json

        _seed_id_cache()
        suffix = json.loads(gpt2.to_model_key().split(":", 1)[1])
        assert suffix["task"] == "text-generation"

    def test_a_different_task_mints_a_different_key(self, gpt2):
        _seed_id_cache()
        from nnsight.modeling.transformers import TransformersModel

        other = TransformersModel("openai-community/gpt2", task="feature-extraction")
        assert other.to_model_key() != gpt2.to_model_key()

    def test_key_is_stable_across_rebuild(self, gpt2):
        # mint -> from_model_key -> re-mint must be a fixed point: the server
        # rebuilds from the key and anything it re-mints (status, logs) must
        # name the same deployment.
        _seed_id_cache()
        from nnsight.modeling.huggingface import HuggingFaceModel

        key = gpt2.to_model_key()
        assert HuggingFaceModel.from_model_key(key).to_model_key() == key

    def test_alias_task_mints_the_canonical_key(self):
        # "sentiment-analysis" IS "text-classification" (transformers'
        # TASK_ALIASES); two spellings of one pipeline must not become two
        # deployments.
        _seed_id_cache()
        import json

        from nnsight.modeling.transformers import TransformersModel

        aliased = TransformersModel(
            "openai-community/gpt2", task="sentiment-analysis"
        )
        suffix = json.loads(aliased.to_model_key().split(":", 1)[1])
        assert suffix["task"] == "text-classification"


class TestEnvoysByReference:
    """A payload names the model's envoys by path; the receiver resolves them to its
    own tree. State the receiver's tree lacks (an edit, a rename alias) still goes
    by value, and an envoy of any other tree is refused."""

    def test_a_trace_payload_carries_envoys_by_reference(self, gpt2, server_gpt2):
        with server_gpt2.trace(PROMPT):
            want = server_gpt2.transformer.h[0].output.save()
        backend = _ServerRoundTrip(server_gpt2)
        with gpt2.trace(PROMPT, backend=backend):
            out = gpt2.transformer.h[0].output.save()
        assert _matches(backend.saved["out"], want)
        # The tree by value is about 24 KB.
        assert len(backend.blob) < 2000, len(backend.blob)
        # The block loads `gpt2`, the root; `.transformer.h[0]` is attribute access
        # at run time on the receiver's own tree.
        assert b"Envoy:model" in backend.blob
        assert b"Module:" not in backend.blob

    def test_a_referenced_envoy_is_a_copy_of_the_receivers(self, gpt2, server_gpt2):
        objects = server_gpt2._remoteable_persistent_objects()
        block = loads(dumps(gpt2.transformer.h[2], tree=gpt2), objects)
        assert _copy_of(block, server_gpt2.transformer.h[2])

    def test_an_envoy_with_no_tree_to_name_it_is_refused(self, gpt2):
        with pytest.raises(pickle.PicklingError, match="no model to name it from"):
            dumps(gpt2.transformer.h[2])

    def test_an_envoy_of_another_tree_is_refused(self, gpt2, server_gpt2):
        other = Envoy(torch.nn.Sequential(torch.nn.Linear(2, 2)), path="other")
        with pytest.raises(pickle.PicklingError, match="'other' belongs to a different model"):
            dumps(other, tree=gpt2)
        # Captured by a block too: the receiver resolves `gpt2` to its own tree and
        # cannot resolve `other`, so the trace is refused when sent.
        with pytest.raises(pickle.PicklingError, match="'other' belongs to a different model"):
            with gpt2.trace("The Eiffel Tower is in", backend=_ServerRoundTrip(server_gpt2)):
                where = other.path
                gpt2.transformer.h[0].output.save()

    def test_a_captured_module_and_tokenizer_resolve_to_the_receivers(self, gpt2, server_gpt2):
        objects = server_gpt2._remoteable_persistent_objects()
        attn = gpt2.transformer.h[0].attn._module
        blob = dumps(attn, tree=gpt2)
        assert len(blob) < 2000, len(blob)
        assert loads(blob, objects) is server_gpt2.transformer.h[0].attn._module
        assert loads(dumps(gpt2.tokenizer, tree=gpt2), objects) is server_gpt2.tokenizer
        # In a block: a tokenizer captured above the trace travels as its name.
        backend = _ServerRoundTrip(server_gpt2)
        tok = gpt2.tokenizer
        with gpt2.trace(PROMPT, backend=backend):
            n = len(tok(PROMPT)["input_ids"])
            out = gpt2.transformer.h[0].output.save()
        assert backend.saved["out"].shape[1] == len(server_gpt2.tokenizer(PROMPT)["input_ids"])
        assert b"Tokenizer" in backend.blob and len(backend.blob) < 3000, len(backend.blob)

    def test_a_rename_alias_travels_with_the_envoy_it_is_bound_on(self, renamed_gpt2, server_gpt2):
        # `layers` is bound on the root and `blocks` on `transformer`: those two go
        # by value, and a block below them is its path.
        assert b"Envoy:model.transformer.h.0" in dumps(renamed_gpt2.transformer.h[0], tree=renamed_gpt2)
        blob = dumps(renamed_gpt2, tree=renamed_gpt2)
        copy = loads(blob, server_gpt2._remoteable_persistent_objects())
        assert copy is not server_gpt2
        assert _copy_of(copy.layers, server_gpt2.transformer.h)
        assert copy.transformer.blocks is copy.layers
        # The block's source names the alias, and the receiver's tree was built
        # without it: the alias arrives with the envoy it is bound on.
        with server_gpt2.trace(PROMPT):
            want = server_gpt2.transformer.h[0].output.save()
        backend = _ServerRoundTrip(server_gpt2)
        with renamed_gpt2.trace(PROMPT, backend=backend):
            out = renamed_gpt2.layers[0].output.save()
        assert _matches(backend.saved["out"], want)

    def test_an_alias_bound_below_the_root_takes_the_root_by_value(self, block_renamed_gpt2, server_gpt2):
        # `self_attn` is bound on each block. The block's source reaches it from
        # the root by attribute access, so the root, `transformer`, `h` and each
        # block go by value; a block's children are still their paths.
        client = block_renamed_gpt2
        copy = loads(dumps(client, tree=client), server_gpt2._remoteable_persistent_objects())
        assert copy is not server_gpt2
        assert _copy_of(copy.transformer.h[0].self_attn, server_gpt2.transformer.h[0].attn)
        assert _copy_of(copy.transformer.h[0].mlp, server_gpt2.transformer.h[0].mlp)
        assert _copy_of(copy.transformer.ln_f, server_gpt2.transformer.ln_f)
        with server_gpt2.trace(PROMPT):
            want = server_gpt2.transformer.h[0].attn.output[0].save()
        backend = _ServerRoundTrip(server_gpt2)
        with client.trace(PROMPT, backend=backend):
            out = client.transformer.h[0].self_attn.output[0].save()
        assert _matches(backend.saved["out"], want)

    def test_a_class_a_caller_chose_travels_with_the_envoy(self, classed_gpt2, server_gpt2):
        client = classed_gpt2
        assert client.transformer.h[0].mlp._carries_state()
        assert not client.transformer.h[0]._carries_state() and not client._carries_state()
        copy = loads(dumps(client, tree=client), server_gpt2._remoteable_persistent_objects())
        assert type(copy.transformer.h[0].mlp) is Doubled
        assert _copy_of(copy.transformer.h[0].attn, server_gpt2.transformer.h[0].attn)
        backend = _ServerRoundTrip(server_gpt2)
        with client.trace("The Eiffel Tower is in", backend=backend):
            plain = client.transformer.h[0].mlp.output.save()
            doubled = client.transformer.h[0].mlp.doubled.save()
        assert backend.ran
        assert torch.equal(backend.saved["doubled"], backend.saved["plain"] * 2)

    def test_a_class_defined_in_a_script_travels_by_source(self):
        # A notebook's `envoys=` class is one the receiver cannot import: it is
        # rebuilt from its source, eproperties included.
        received = loads(dumps(script_class()))
        prop = received.__dict__["tripled"]
        assert isinstance(prop, eproperty) and prop.key == "output" and prop.name == "tripled"
        assert prop._preprocess(None, 2) == 6

    def test_a_runtime_default_class_map_is_not_state(self, server_gpt2):
        from nnsight.intervention.envoy import RuntimeEnvoys
        from nnsight.modeling.transformers import TransformersModel

        model = TransformersModel(
            "openai-community/gpt2",
            task="text-generation",
            dispatch=True,
            envoys=RuntimeEnvoys({"mlp": Doubled}),
        )
        assert type(model.transformer.h[0].mlp) is Doubled
        assert not model.transformer.h[0].mlp._carries_state()
        assert _names_in(dumps(model, tree=model)) == {"Envoy:model", "Interleaver"}

    def test_a_trace_on_a_block_still_names_the_tokenizer(self, gpt2, server_gpt2):
        hidden = torch.randn(1, 1, 768)
        with server_gpt2.transformer.h[3].trace(hidden):
            want = server_gpt2.transformer.h[3].mlp.output.save()
        backend = _ServerRoundTrip(server_gpt2)
        tok = gpt2.tokenizer
        with gpt2.transformer.h[3].trace(hidden, backend=backend):
            n = len(tok(PROMPT)["input_ids"])
            out = gpt2.transformer.h[3].mlp.output.save()
        assert _matches(backend.saved["out"], want)
        # The input tensor is 3 KB of it; the tokenizer by value would be 2.9 MB.
        assert b"Tokenizer" in backend.blob and len(backend.blob) < 6000, len(backend.blob)

    def test_an_edited_root_goes_by_value_with_its_children_by_reference(self, gpt2, server_gpt2):
        with gpt2.edit() as (_, edited):
            edited.transformer.h[0].output[:] = 0
        copy = loads(dumps(edited, tree=gpt2), server_gpt2._remoteable_persistent_objects())
        assert copy is not server_gpt2 and copy is not edited
        assert len(copy._edits) == 1
        # The children are the receiver's own, copied for this payload.
        assert _copy_of(copy.transformer, server_gpt2.transformer)
        assert copy.transformer.parent is copy
        with edited.trace(PROMPT):
            want = edited.lm_head.output.save()
        with gpt2.trace(PROMPT):
            clean = gpt2.lm_head.output.save()
        assert not _matches(want, clean)
        backend = _ServerRoundTrip(server_gpt2)
        with edited.trace(PROMPT, backend=backend):
            out = edited.lm_head.output.save()
        assert _matches(backend.saved["out"], want)

    def test_a_child_edit_goes_by_value_with_its_children_by_reference(self, gpt2, server_gpt2):
        block = gpt2.transformer.h[1]
        with block.edit(inplace=True):
            block.mlp.output.save()
        try:
            copy = loads(dumps(block, tree=gpt2), server_gpt2._remoteable_persistent_objects())
            assert copy is not server_gpt2.transformer.h[1] and len(copy._edits) == 1
            assert _copy_of(copy.mlp, server_gpt2.transformer.h[1].mlp)
            # The root reaches the edited block by attribute access, so it goes
            # by value too; the blocks beside it are still their paths.
            root = loads(dumps(gpt2, tree=gpt2), server_gpt2._remoteable_persistent_objects())
            assert root is not server_gpt2
            assert _copy_of(root.transformer.h[0], server_gpt2.transformer.h[0])
            assert len(root.transformer.h[1]._edits) == 1
        finally:
            block.clear_edits()

    def test_a_new_attribute_on_an_envoy_is_refused(self, gpt2):
        # A receiver that wraps the same model holds no such attribute, so the
        # write is refused where it is made.
        with pytest.raises(AttributeError, match="takes no new attributes"):
            gpt2.transformer.h[0].alpha = 0.5
        assert "alpha" not in gpt2.transformer.h[0].__dict__

    def test_a_shared_tree_refuses_a_block_write(self, gpt2):
        interleaver = gpt2.interleaver
        interleaver.shared = True
        try:
            with pytest.raises(AttributeError, match="shared by every request"):
                with gpt2.trace("The Eiffel Tower is in"):
                    gpt2.transformer.sae = torch.nn.Linear(2, 2)
            assert "sae" not in gpt2.transformer.__dict__
            assert "sae" not in gpt2.transformer._module._modules
            # An intervention is a write a descriptor takes, and still passes.
            with gpt2.trace("The Eiffel Tower is in"):
                block = gpt2.transformer.h[0]
                block.output = block.output * 0
                out = gpt2.transformer.h[0].output.save()
            assert not out.any()
        finally:
            interleaver.shared = False

    def test_a_referenced_envoy_is_a_copy_holding_the_receivers_module_entry(self, gpt2, server_gpt2):
        # The copy's module is what the receiver's map holds under `Module:<path>`:
        # the entry a server may have wrapped to guard its model.
        objects = server_gpt2._remoteable_persistent_objects()
        guard = torch.nn.Identity()
        objects["Module:model.transformer.h.2"] = guard
        root = loads(dumps(gpt2, tree=gpt2), objects)
        assert root.transformer.h[2]._module is guard
        assert root.transformer.h[3]._module is server_gpt2.transformer.h[3]._module
        # Reached by any spelling, one copy per payload.
        assert root.transformer.h[2] is root.transformer.h._child_map["2"]
        assert root.transformer.h[2] in root.transformer.h._children
        assert root.transformer.h[2].parent is root.transformer.h

    def test_what_a_block_does_to_a_copy_stays_with_its_payload(self, gpt2, server_gpt2):
        objects = server_gpt2._remoteable_persistent_objects()
        root = loads(dumps(gpt2, tree=gpt2), objects)
        with root.edit(inplace=True):
            root.transformer.h[0].output[0][:] = 0
        assert len(root._edits) == 1 and server_gpt2._edits == []
        root.clear_edits()
        assert root._edits == [] and server_gpt2._edits == []
        tokenizer = server_gpt2.tokenizer
        root.tokenizer = None
        assert server_gpt2.tokenizer is tokenizer
        # Another payload gets its own copies.
        again = loads(dumps(gpt2, tree=gpt2), objects)
        assert again is not root and again.transformer is not root.transformer

    def test_replacing_a_child_on_a_copy_replaces_its_entry(self, gpt2):
        # Mounting reaches the receiver's module, so this receiver is its own.
        sender, receiver = Envoy(_Stack()), Envoy(_Stack())
        root = loads(dumps(sender, tree=sender), _objects(receiver))
        before = len(list(root.decoder_blocks[0]))
        root.decoder_blocks[0].mixer = torch.nn.Identity()
        assert len(list(root.decoder_blocks[0])) == before
        assert root.decoder_blocks[0]._child_map["mixer"] is root.decoder_blocks[0].mixer

    def test_a_session_carries_its_model_by_reference(self, gpt2, server_gpt2):
        with server_gpt2.trace(PROMPT):
            want = server_gpt2.transformer.h[0].output.save()
        backend = _ServerRoundTrip(server_gpt2)
        with gpt2.session(backend=backend):
            with gpt2.trace(PROMPT):
                hidden = gpt2.transformer.h[0].output
            with gpt2.trace(PROMPT):
                out = (hidden[0] * 2).save()
        assert _matches(backend.saved["out"], want[0] * 2)
        assert b"Envoy:model" in backend.blob and b"Module:" not in backend.blob

    def test_a_trace_on_a_block_keeps_its_parent_and_root(self, gpt2, server_gpt2):
        backend = _ServerRoundTrip(server_gpt2)
        with gpt2.transformer.h[3].trace(torch.randn(1, 1, 768), backend=backend):
            block = gpt2.transformer.h[3]
            linked = nnsight.save(block.parent is gpt2.transformer.h)
            parent = nnsight.save(block.parent.path)
            root = nnsight.save(block.root.path)
            out = block.mlp.output.save()
        assert backend.saved["linked"] is True
        assert backend.saved["parent"] == "model.transformer.h"
        assert backend.saved["root"] == "model"


class _Stage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = torch.nn.Linear(4, 4)
        self.gate = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.gate(self.mixer(x))


class _Stack(torch.nn.Module):
    """Module names no transformer checkpoint uses, for a tree built on the spot."""

    def __init__(self):
        super().__init__()
        self.decoder_blocks = torch.nn.ModuleList([_Stage(), _Stage()])
        self.output_projection = torch.nn.Linear(4, 2)

    def forward(self, x):
        for block in self.decoder_blocks:
            x = block(x)
        return self.output_projection(x)


class _Loop(torch.nn.Module):
    """A child module holding its parent as a registered submodule."""

    def __init__(self):
        super().__init__()
        self.inner = _Stage()
        self.inner.owner = self

    def forward(self, x):
        return self.inner(x)


def _objects(root):
    table = {"Interleaver": root.interleaver}
    for node in root.modules():
        table[f"Module:{node.path}"] = node._module
        table[f"Envoy:{node.path}"] = node
    return table


class TestPayloadCopies:
    """What a receiver's copy of an envoy reaches, keeps and frees, and which roots
    go by value."""

    def test_a_wrapper_subclass_travels_with_its_members(self, server_gpt2):
        client = WrappedModel("openai-community/gpt2", task="text-generation", dispatch=True)
        with server_gpt2.trace(PROMPT):
            first = server_gpt2.transformer.h[0].output.save()
            last = server_gpt2.transformer.h[-1].output.save()
            logits = server_gpt2.lm_head.output.save()
        backend = _ServerRoundTrip(server_gpt2)
        with client.trace(PROMPT, backend=backend):
            by_attribute = client.stack[0].output.save()
            by_method = client.last_block().output.save()
            doubled = client.doubled_logits.save()
        assert _matches(backend.saved["by_attribute"], first)
        assert _matches(backend.saved["by_method"], last)
        assert _matches(backend.saved["doubled"], logits * 2)
        # The root by value with its children by path; the whole tree is about 24 KB.
        assert len(backend.blob) < 6000, len(backend.blob)

    def test_a_stock_root_class_goes_by_path(self):
        from nnsight import LanguageModel
        from nnsight.modeling.transformers import TransformersModel

        for model in (
            LanguageModel("openai-community/gpt2"),
            TransformersModel("openai-community/gpt2", task="text-generation"),
            Envoy(_Stack()),
        ):
            assert _names_in(dumps(model, tree=model)) == {"Envoy:model", "Interleaver"}

    def test_a_stock_root_is_set_up_by_a_receiver_that_resolves_names_directly(self):
        # NDIF's sandbox runner resolves `Envoy:model` to its undispatched meta model
        # with no copy, and `Interleaver` to one made for the request.
        from nnsight.modeling.transformers import TransformersModel

        request_interleaver = object()

        class Runner(CustomCloudUnpickler):
            def persistent_load(self, pid):
                if pid == "Interleaver":
                    return request_interleaver
                return self.persistent_objects[pid]

        sender = TransformersModel("openai-community/gpt2", task="text-generation")
        meta_model = TransformersModel("openai-community/gpt2", task="text-generation")
        received = loads(
            dumps(sender, tree=sender), meta_model._remoteable_persistent_objects(), Runner
        )
        assert received is meta_model
        assert received.interleaver is request_interleaver
        assert received.dispatched

    def test_a_by_value_root_loads_during_a_run_on_a_shared_tree(self, renamed_gpt2, server_gpt2):
        # A vLLM worker loads a request inside the run, on the tree every request
        # shares, and rebuilding a by-value model writes `dispatched` on it.
        blob = dumps(renamed_gpt2, tree=renamed_gpt2)
        objects = server_gpt2._remoteable_persistent_objects()
        interleaver = server_gpt2.interleaver
        interleaver.shared = True
        try:
            with interleaver:
                copy = loads(blob, objects)
        finally:
            interleaver.shared = False
        assert copy.dispatched and _copy_of(copy.layers, server_gpt2.transformer.h)

    def test_a_child_edit_naming_the_root_round_trips(self):
        # Pickle reaches the child first, then the root its edit names, and the root
        # is rebuilt while the child is still empty.
        sender, receiver = Envoy(_Stack()), Envoy(_Stack())
        stage = sender.decoder_blocks[0]
        root = sender
        with stage.edit(inplace=True):
            root.decoder_blocks[0].mixer.output.save()
        copy = loads(dumps(stage, tree=sender), _objects(receiver))
        assert len(copy._edits) == 1

    def test_a_payload_is_freed_without_a_gc_pass(self):
        sender, receiver = Envoy(_Stack()), Envoy(_Stack())
        table = _objects(receiver)
        tensor = torch.zeros(4)
        gc.collect()
        gc.disable()
        try:
            copy, carried = loads(dumps((sender, tensor), tree=sender), table)
            child = copy.decoder_blocks[0].mixer
            refs = weakref.ref(copy), weakref.ref(child), weakref.ref(carried)
            del copy, child, carried
            assert all(ref() is None for ref in refs)
        finally:
            gc.enable()

    def test_a_child_captured_above_the_block_reaches_its_parent_and_root(self, gpt2, server_gpt2):
        with server_gpt2.trace(PROMPT):
            want = server_gpt2.lm_head.output.save()
        layer = gpt2.transformer.h[3]
        backend = _ServerRoundTrip(server_gpt2)
        with gpt2.trace(PROMPT, backend=backend):
            out = layer.root.lm_head.output.save()
        assert _matches(backend.saved["out"], want)
        # Nothing else holds the parent's copy, here or in a GC-free process.
        gc.disable()
        try:
            copy = loads(dumps(layer, tree=gpt2), server_gpt2._remoteable_persistent_objects())
            parent = weakref.ref(copy.parent)
            assert parent() is not None
            assert _copy_of(copy.parent, server_gpt2.transformer.h)
            assert _copy_of(copy.root, server_gpt2)
        finally:
            gc.enable()

    def test_parent_and_root_reach_the_by_value_envoy_above(self, renamed_gpt2, server_gpt2):
        copy = loads(dumps(renamed_gpt2, tree=renamed_gpt2), server_gpt2._remoteable_persistent_objects())
        assert copy.transformer.ln_f.parent is copy.transformer
        assert copy.transformer.ln_f.root is copy
        assert copy.transformer.ln_f.root.layers is copy.transformer.h

    def test_a_receivers_edit_stays_out_of_a_payload(self, gpt2, server_gpt2):
        with gpt2.trace(PROMPT):
            clean = gpt2.lm_head.output.save()
        with server_gpt2.edit(inplace=True):
            server_gpt2.transformer.h[0].output[:] = 0
        try:
            backend = _ServerRoundTrip(server_gpt2)
            with gpt2.trace(PROMPT, backend=backend):
                out = gpt2.lm_head.output.save()
            assert _matches(backend.saved["out"], clean)
        finally:
            server_gpt2.clear_edits()

    def test_an_edit_copy_made_before_dispatch_names_the_current_objects(self, server_gpt2):
        from nnsight.modeling.transformers import TransformersModel

        client = TransformersModel("openai-community/gpt2", task="text-generation")
        with client.edit() as (_, edited):
            edited.transformer.h[0].output[:] = 0
        before = len(dumps(edited, tree=client))
        client.dispatch()
        blob = dumps(edited, tree=client)
        assert len(blob) == before
        copy = loads(blob, server_gpt2._remoteable_persistent_objects())
        assert copy._module is server_gpt2._module and len(copy._edits) == 1

    def test_a_copy_keeps_its_own_dicts(self, gpt2, server_gpt2):
        objects = server_gpt2._remoteable_persistent_objects()
        copy = loads(dumps(gpt2, tree=gpt2), objects)
        copy.kwargs["planted"] = True
        copy._aliases["planted"] = "x.y"
        again = loads(dumps(gpt2, tree=gpt2), objects)
        assert "planted" not in again.kwargs and "planted" not in again._aliases
        assert "planted" not in server_gpt2.kwargs and "planted" not in server_gpt2._aliases

    def test_base_model_on_a_copy_stays_with_the_payload(self, gpt2, server_gpt2):
        before = dict(server_gpt2._aliases)
        copy = loads(dumps(gpt2, tree=gpt2), server_gpt2._remoteable_persistent_objects())
        assert _copy_of(copy.base_model, server_gpt2.transformer)
        assert server_gpt2._aliases == before

    def test_a_module_missing_from_the_map_raises_the_id_error(self):
        sender, receiver = Envoy(_Stack()), Envoy(_Stack())
        table = _objects(receiver)
        del table["Module:model.output_projection"]
        # Loading links the root's direct children, which needs their modules.
        with pytest.raises(UnknownPersistentIdError, match="Module:model.output_projection"):
            loads(dumps(sender, tree=sender), table)

    def test_a_module_cycle_ends_the_walk(self):
        sender = Envoy(_Loop())
        with sender.inner.edit(inplace=True):
            sender.inner.mixer.output.save()
        assert "Envoy:model" not in _names_in(dumps(sender, tree=sender))

    def test_a_tensor_parallel_class_map_is_not_state(self):
        from nnsight.modeling.tp.envoys import TPEnvoy, tp_envoys

        sender = Envoy(_Stack(), envoys=tp_envoys())
        assert type(sender.output_projection) is TPEnvoy
        assert _names_in(dumps(sender, tree=sender)) == {"Envoy:model", "Interleaver"}
