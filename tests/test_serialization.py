"""Source-based serialization round-trips (``dumps``/``loads``) and the
``remote="local"`` dry-run of the remote path.
"""

import asyncio
import linecache
import sys
import textwrap

import pytest
import torch

from nnsight.intervention.serialization import code_reduce, dumps, loads
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
            hidden = gpt2.transformer.h[0].output[0].save()
        assert isinstance(hidden, torch.Tensor) and hidden.ndim == 2

    def test_intervention_round_trips(self, gpt2):
        with gpt2.trace("The Eiffel Tower is in", remote="local"):
            gpt2.transformer.h[0].output[0][:] = 0
            zeroed = gpt2.transformer.h[0].output[0].save()
        assert torch.all(zeroed == 0)

    def test_local_module_function_ships_by_value(self, gpt2):
        # `normalize` lives in this (local, non-installed) test module; the local
        # backend hides it during deserialize, so it must have shipped by value.
        with gpt2.trace("The Eiffel Tower is in", remote="local"):
            h = gpt2.transformer.h[0].output[0]
            normed = normalize(h).save()
        norms = normed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_generate_round_trips(self, gpt2):
        with gpt2.generate("The Eiffel Tower is in", max_new_tokens=3, remote="local"):
            out = gpt2.transformer.h[0].output[0].save()
        assert isinstance(out, torch.Tensor)


class _ServerRoundTrip(Backend):
    """Serialize a trace and run the *deserialized* tracer, as an NDIF server does.

    ``remote="local"`` runs the original in-process tracer (which still carries its
    AST node) so results push back into the caller's frame; a real server runs the
    tracer rebuilt from the payload, whose node was dropped in transit. This backend
    exercises that server path so the node fallback is covered where it matters.
    """

    def __init__(self, model):
        self.model = model
        self.ran = False
        self.node_after = "unset"

    def __call__(self, tracer):
        blob = RequestModel.serialize(tracer, compress=False)
        persistent = self.model._remoteable_persistent_objects()
        server_tracer = RequestModel.deserialize(blob, persistent, compress=False)
        self.node_after = server_tracer.node
        server_tracer.execute(server_tracer.info.code)
        self.ran = True


class TestServerExecution:
    """The deserialized tracer runs server-side even though serialization drops its
    AST node (needed only where the block is captured)."""

    @torch.no_grad()
    def test_deserialized_tracer_executes(self, gpt2):
        backend = _ServerRoundTrip(gpt2)
        with gpt2.trace("The Eiffel Tower is in the city of", backend=backend):
            gpt2.transformer.h[0].output[0].save()
        assert backend.ran
        assert backend.node_after is None  # dropped in transit, falls back to None

    @torch.no_grad()
    def test_deserialized_generate_executes(self, gpt2):
        backend = _ServerRoundTrip(gpt2)
        with gpt2.generate(
            "The Eiffel Tower is in", max_new_tokens=3, backend=backend
        ):
            gpt2.transformer.h[0].output[0].save()
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
