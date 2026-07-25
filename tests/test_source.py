
import pytest
import torch
import nnsight
import torch.nn as nn

from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import OutOfOrderError
from nnsight.intervention.source import _STATE, SourceNotAvailable


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


class Repeated(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        a = torch.relu(self.fc1(x))
        b = torch.relu(self.fc2(a))
        return b


class Nested(nn.Module):
    def forward(self, x):
        return torch.relu(torch.relu(x))


class Decorated(nn.Module):
    @torch.no_grad()
    def forward(self, x):
        return x + 1


@pytest.fixture
def x():
    return torch.randn(2, 8)


class TestListing:
    def test_lists_operations_in_execution_order(self):
        envoy = Envoy(MLP())
        # fc1 runs, then relu wrapping it, then fc2.
        assert list(envoy.source._names) == ["self_fc1_0", "torch_relu_0", "self_fc2_0"]

    def test_repr_lists_operations(self):
        envoy = Envoy(MLP())
        text = repr(envoy.source)
        for name in ("self_fc1_0", "torch_relu_0", "self_fc2_0"):
            assert name in text

    def test_repr_shows_annotated_source(self):
        # The overview prints the forward with ops labelled in a gutter, the def
        # line marked with '*', and same-line ops as '+' continuations.
        text = repr(Envoy(MLP()).source)
        lines = text.splitlines()
        assert "* def forward" in text
        assert "return self.fc2(h)" in text
        # fc1 labels its line; relu (same line) continues with a '+'.
        assert any("self_fc1_0" in line and "->" in line for line in lines)
        assert any("torch_relu_0" in line and "+" in line for line in lines)

    def test_source_node_repr_marks_call_site(self):
        # Zooming in marks the operation's line with '-->' / '<--' and names it.
        text = repr(Envoy(MLP()).source.self_fc1_0)
        assert text.startswith("model.source.self_fc1_0:")
        marked = [line for line in text.splitlines() if "-->" in line]
        assert len(marked) == 1
        assert "self.fc1(x)" in marked[0] and marked[0].endswith("<--")

    def test_repeated_op_gets_distinct_occurrences(self):
        envoy = Envoy(Repeated())
        assert "torch_relu_0" in envoy.source._names
        assert "torch_relu_1" in envoy.source._names

    def test_unknown_operation_raises_with_available(self):
        envoy = Envoy(MLP())
        with pytest.raises(AttributeError) as exc:
            envoy.source.nope_0
        assert "torch_relu_0" in str(exc.value)

    def test_decorated_forward_rejected(self):
        envoy = Envoy(Decorated())
        with pytest.raises(SourceNotAvailable):
            envoy.source


class TestCapture:
    def test_captures_intermediate_value(self, x):
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            captured["relu"] = envoy.source.torch_relu_0.output
        assert torch.allclose(captured["relu"], torch.relu(model.fc1(x)))

    def test_captures_call_to_submodule(self, x):
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            captured["fc1"] = envoy.source.self_fc1_0.output
        assert torch.allclose(captured["fc1"], model.fc1(x))

    def test_repeated_occurrence_captures_second(self, x):
        model = Repeated()
        envoy = Envoy(model)
        expected = torch.relu(model.fc2(torch.relu(model.fc1(x))))
        captured = {}
        with envoy.trace(x):
            captured["relu1"] = envoy.source.torch_relu_1.output
        assert torch.allclose(captured["relu1"], expected)

    def test_nested_numbering_is_execution_order(self, x):
        model = Nested()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            # inner relu == torch_relu_0, so it equals a single relu of the input.
            captured["inner"] = envoy.source.torch_relu_0.output
        assert torch.allclose(captured["inner"], torch.relu(x))

    def test_source_and_module_output_coexist(self, x):
        # Requested in execution order: relu (mid-forward) before fc2's output.
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            captured["relu"] = envoy.source.torch_relu_0.output
            captured["fc2_out"] = envoy.fc2.output
        assert torch.allclose(captured["relu"], torch.relu(model.fc1(x)))
        assert torch.allclose(captured["fc2_out"], model(x))

    def test_out_of_order_source_request_raises(self, x):
        envoy = Envoy(MLP())
        with pytest.raises(OutOfOrderError):
            with envoy.trace(x):
                later = envoy.source.self_fc2_0.output  # fc2 runs last...
                earlier = envoy.source.self_fc1_0.output  # ...so requesting fc1 now is late


class TestInputs:
    def test_op_input_is_first_arg(self, x):
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            # fc2 receives the relu output as its single positional argument.
            captured["fc2_in"] = envoy.source.self_fc2_0.input
        assert torch.allclose(captured["fc2_in"], torch.relu(model.fc1(x)))

    def test_op_inputs_is_args_kwargs(self, x):
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            captured["fc2_in"] = envoy.source.self_fc2_0.inputs
        args, kwargs = captured["fc2_in"]
        assert torch.allclose(args[0], torch.relu(model.fc1(x)))
        assert kwargs == {}


class TestEditing:
    def test_set_op_output(self, x):
        model = MLP()
        envoy = Envoy(model)
        replacement = torch.full((2, 8), 5.0)
        captured = {}
        with envoy.trace(x):
            envoy.source.torch_relu_0.output = replacement
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.fc2(replacement))

    def test_edit_op_output_from_its_value(self, x):
        model = MLP()
        envoy = Envoy(model)
        ref_relu = torch.relu(model.fc1(x))
        captured = {}
        with envoy.trace(x):
            envoy.source.torch_relu_0.output = envoy.source.torch_relu_0.output + 1
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.fc2(ref_relu + 1))

    def test_inplace_edit_op_output(self, x):
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            envoy.source.torch_relu_0.output[:] = 0
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.fc2(torch.zeros(2, 8)))

    def test_set_op_input(self, x):
        model = MLP()
        envoy = Envoy(model)
        zeros = torch.zeros(2, 8)
        captured = {}
        with envoy.trace(x):
            envoy.source.self_fc2_0.input = zeros
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.fc2(zeros))

    def test_set_op_inputs(self, x):
        model = MLP()
        envoy = Envoy(model)
        zeros = torch.zeros(2, 8)
        captured = {}
        with envoy.trace(x):
            envoy.source.self_fc2_0.inputs = ((zeros,), {})
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.fc2(zeros))


class NestedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = MLP()

    def forward(self, x):
        return self.inner(x)


class TestSkip:
    def test_module_skip_with_constant(self, x):
        model = MLP()
        envoy = Envoy(model)
        v = torch.full((2, 8), 3.0)
        captured = {}
        with envoy.trace(x):
            envoy.fc1.skip(v)  # fc1 doesn't run; its output is v
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.fc2(torch.relu(v)))

    def test_module_skip_with_own_input_first_trace(self, x):
        # The residual-passthrough idiom. Works on the FIRST trace because `.skip`
        # is a property: accessing it installs the controller before `.input` is
        # read, so the forward is intercepted in time.
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            envoy.fc1.skip(envoy.fc1.input)  # fc1 passes its input through
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.fc2(torch.relu(x)))

    def test_module_skip_actually_avoids_compute(self, x):
        # A forward that would raise if run — skip must mean it never executes.
        class Boom(nn.Module):
            def forward(self, x):
                raise AssertionError("forward should not run when skipped")

        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.boom = Boom()

            def forward(self, x):
                return self.boom(x)

        envoy = Envoy(Wrap())
        captured = {}
        with envoy.trace(x):
            envoy.boom.skip(x)
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], x)

    def test_op_skip(self, x):
        model = MLP()
        envoy = Envoy(model)
        v = torch.full((2, 8), 2.0)
        captured = {}
        with envoy.trace(x):
            envoy.source.self_fc1_0.skip(v)  # the fc1 call returns v instead of running
            captured["out"] = envoy.output
        assert torch.allclose(captured["out"], model.fc2(torch.relu(v)))

    def test_op_skip_reports_output(self, x):
        model = MLP()
        envoy = Envoy(model)
        v = torch.full((2, 8), 2.0)
        captured = {}
        with envoy.trace(x):
            envoy.source.self_fc1_0.skip(v)
            captured["fc1_out"] = envoy.source.self_fc1_0.output
        assert torch.allclose(captured["fc1_out"], v)

    def test_skipping_sourced_module_drops_its_ops(self, x):
        # Skipping a whole module means its body never runs, so its source ops
        # never fire — reading one is out of order, like any never-produced value.
        envoy = Envoy(NestedMLP())
        with pytest.raises(OutOfOrderError):
            with envoy.trace(x):
                envoy.inner.skip(torch.zeros(2, 8))
                envoy.inner.source.torch_relu_0.output


class TestInstall:
    def test_forward_installed_permanently(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.trace(x):
            envoy.source.torch_relu_0.output
        # The instrumented forward stays installed (permanent, module-cached).
        assert "forward" in model.__dict__
        assert model.__dict__[_STATE].sourced is True

    def test_untraced_inference_matches_reference(self, x):
        model = MLP()
        envoy = Envoy(model)
        reference = model.fc2(torch.relu(model.fc1(x)))
        # Access .source (installs the instrumented forward) — must not perturb a
        # normal, untraced forward, before or after an actual source trace.
        envoy.source
        assert torch.allclose(model(x), reference)
        with envoy.trace(x):
            envoy.source.torch_relu_0.output
        assert torch.allclose(model(x), reference)

    def test_multiple_envoys_share_module(self, x):
        # Two independent Envoys (each its own Interleaver) over the SAME module:
        # source works through both, rebinding to whichever traces.
        model = MLP()
        e1 = Envoy(model)
        e2 = Envoy(model)
        expected = torch.relu(model.fc1(x))

        with e1.trace(x):
            a = nnsight.save(e1.source.torch_relu_0.output)
        assert torch.allclose(a, expected)

        with e2.trace(x):
            b = nnsight.save(e2.source.torch_relu_0.output)
        assert torch.allclose(b, expected)

        # Back to e1 again — rebinds fine, and edits route to the right run.
        with e1.trace(x):
            e1.source.torch_relu_0.output = torch.zeros(2, 8)
            out = nnsight.save(e1.output)
        assert torch.allclose(out, model.fc2(torch.zeros(2, 8)))

    def test_prior_source_access_makes_ordering_robust(self, x):
        # Once installed (here, before the trace), a source value is observable
        # even after a child-output request that would otherwise start the model.
        model = MLP()
        envoy = Envoy(model)
        envoy.source  # install now
        captured = {}
        with envoy.trace(x):
            captured["fc1_out"] = envoy.fc1.output
            captured["relu"] = envoy.source.torch_relu_0.output
        assert torch.allclose(captured["fc1_out"], model.fc1(x))
        assert torch.allclose(captured["relu"], torch.relu(model.fc1(x)))


# ---------------------------------------------------------------------------
# Recursive / nested source: drilling into a called function
# ---------------------------------------------------------------------------

# Module-level so inspect.getsource can recover them when instrumented.


def relu_double(x):
    a = torch.relu(x)
    b = torch.add(a, a)
    return b


def inner_neg(x):
    return torch.relu(x)


def outer_neg(x):
    y = torch.neg(x)
    return inner_neg(y)


class Calls(nn.Module):
    """forward calls a plain module-level helper."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        h = self.fc(x)
        return relu_double(h)


class Deep(nn.Module):
    """forward -> outer_neg -> inner_neg -> relu (three levels)."""

    def forward(self, x):
        return outer_neg(x)


class Methoded(nn.Module):
    """forward calls a bound method of the module itself."""

    def compute(self, x):
        return torch.relu(x)

    def forward(self, x):
        return self.compute(x)


class TestRecursive:
    def test_capture_nested_output(self, x):
        model = Calls()
        envoy = Envoy(model)
        with envoy.trace(x):
            inner = nnsight.save(envoy.source.relu_double_0.source.torch_relu_0.output)
            outer = nnsight.save(envoy.source.relu_double_0.output)
        assert torch.allclose(inner, torch.relu(model.fc(x)))
        assert torch.allclose(outer, inner + inner)

    def test_edit_nested_output_propagates(self, x):
        model = Calls()
        envoy = Envoy(model)
        with envoy.trace(x):
            envoy.source.relu_double_0.source.torch_relu_0.output = torch.zeros(2, 8)
            out = nnsight.save(envoy.output)
        # relu -> 0, so add(0,0) -> 0, and forward returns that.
        assert torch.allclose(out, torch.zeros(2, 8))

    def test_skip_nested_op(self, x):
        model = Calls()
        envoy = Envoy(model)
        sentinel = torch.ones(2, 8) * 7
        with envoy.trace(x):
            envoy.source.relu_double_0.source.torch_add_0.skip(sentinel)
            out = nnsight.save(envoy.source.relu_double_0.output)
        assert torch.allclose(out, sentinel)

    def test_nested_input_and_inputs(self, x):
        model = Calls()
        envoy = Envoy(model)
        with envoy.trace(x):
            first = nnsight.save(envoy.source.relu_double_0.source.torch_add_0.input)
            args, kwargs = envoy.source.relu_double_0.source.torch_add_0.inputs
            both = nnsight.save(args[0])
        # add(a, a): first arg is relu(fc(x)).
        assert torch.allclose(first, torch.relu(model.fc(x)))
        assert torch.allclose(both, first)

    def test_bound_method_recursion(self, x):
        model = Methoded()
        envoy = Envoy(model)
        with envoy.trace(x):
            r = nnsight.save(envoy.source.self_compute_0.source.torch_relu_0.output)
        assert torch.allclose(r, torch.relu(x))

    def test_depth_three(self, x):
        model = Deep()
        envoy = Envoy(model)
        with envoy.trace(x):
            deep = nnsight.save(
                envoy.source.outer_neg_0.source.inner_neg_0.source.torch_relu_0.output
            )
        assert torch.allclose(deep, torch.relu(torch.neg(x)))

    def test_nested_names_and_repr(self, x):
        model = Calls()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            nested = envoy.source.relu_double_0.source
            captured["names"] = [op.name for op in nested]
            captured["repr"] = repr(nested)
        assert captured["names"] == ["torch_relu_0", "torch_add_0"]
        assert "torch_relu_0" in captured["repr"] and "torch_add_0" in captured["repr"]

    def test_unknown_nested_op_raises(self, x):
        model = Calls()
        envoy = Envoy(model)
        with pytest.raises(AttributeError, match="available: torch_relu_0, torch_add_0"):
            with envoy.trace(x):
                envoy.source.relu_double_0.source.nope_0.output

    def test_builtin_target_raises(self, x):
        # torch_relu_0 inside relu_double calls the builtin torch.relu — no source.
        model = Calls()
        envoy = Envoy(model)
        with pytest.raises(SourceNotAvailable):
            with envoy.trace(x):
                envoy.source.relu_double_0.source.torch_relu_0.source

    def test_submodule_target_raises(self, x):
        # self_fc_0 calls a submodule; drilling in should redirect to its own .source.
        model = Calls()
        envoy = Envoy(model)
        with pytest.raises(SourceNotAvailable, match="submodule"):
            with envoy.trace(x):
                envoy.source.self_fc_0.source

    def test_outside_trace_raises(self):
        model = Calls()
        envoy = Envoy(model)
        with pytest.raises(SourceNotAvailable, match="inside a trace"):
            envoy.source.relu_double_0.source


@pytest.fixture(scope="module")
def gpt2():
    from nnsight.modeling.transformers import TransformersModel

    return TransformersModel("gpt2", task="text-generation", dispatch=True)


class TestRecursiveIntegration:
    def test_capture_nested_in_transformer(self, gpt2):
        attn = gpt2.transformer.h[0].attn
        captured = {}
        with gpt2.trace("The Eiffel Tower is in"):
            captured["names"] = attn.source.attention_interface_0.source._compiled.names
            out = nnsight.save(
                attn.source.attention_interface_0.source.attn_output_transpose_0.output
            )
        assert "attn_output_transpose_0" in captured["names"]
        assert out.shape[0] == 1  # batch

    def test_nested_source_across_generate_steps(self, gpt2):
        attn = gpt2.transformer.h[0].attn
        saved = []
        with gpt2.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
            for _ in tracer.iter[:3]:
                saved.append(
                    nnsight.save(
                        attn.source.attention_interface_0.source.attn_output_transpose_0.output
                    )
                )
        assert len(saved) == 3
        # First step sees the full prompt; later (KV-cached) steps see one token.
        assert saved[0].shape[1] > 1
        assert saved[1].shape[1] == 1
        assert saved[2].shape[1] == 1


def _raises_boom(x):
    raise ValueError("boom in instrumented forward")


class RaisesInForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        h = self.fc(x)
        h = _raises_boom(h)  # the call that raises
        return h


class TestExceptionLineNumbers:
    """An exception inside a `.source`-instrumented forward keeps the real line."""

    def test_instrumented_forward_exception_reports_real_line(self):
        import inspect
        import traceback

        # The real file line of `h = _raises_boom(h)` in RaisesInForward.forward.
        src_lines, start = inspect.getsourcelines(RaisesInForward.forward)
        expected_line = start + next(
            i for i, line in enumerate(src_lines) if "_raises_boom(h)" in line
        )

        model = Envoy(RaisesInForward())
        _ = model.source  # install the AST instrumentation on forward

        try:
            with model.trace(torch.randn(2, 8)):
                nnsight.save(model.output)
        except ValueError as error:
            frames = [
                frame
                for frame in traceback.extract_tb(error.__traceback__)
                if frame.name == "forward" and frame.filename == __file__
            ]
            assert frames, "no forward frame from this file in the traceback"
            # The instrumented forward's frame points at the real raising line,
            # not a drifted one (regression: it reported the raw lineno offset).
            assert frames[-1].lineno == expected_line, (
                frames[-1].lineno,
                expected_line,
            )
            assert frames[-1].line and "_raises_boom(h)" in frames[-1].line
        else:
            pytest.fail("expected ValueError from the instrumented forward")
