import threading

import pytest
import torch
import nnsight
import torch.nn as nn

from nnsight.intervention.envoy import Envoy


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def x():
    return torch.randn(2, 8)


class TestSave:
    def test_saved_value_returns(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.trace(x):
            h = nnsight.save(envoy.fc1.output)
        assert torch.allclose(h, model.fc1(x))

    def test_save_returns_the_same_object(self, x):
        envoy = Envoy(MLP())
        with envoy.trace(x):
            out = envoy.fc1.output
            same = nnsight.save(out)
            # save returns its argument unchanged
            assert same is out

    def test_save_outside_trace_raises(self):
        # A save with no trace active can't return anywhere — its mark is cleared
        # before anything reads it — so it errors instead of silently no-op'ing.
        with pytest.raises(ValueError, match="outside a trace"):
            nnsight.save([])

    def test_unsaved_value_does_not_escape(self, x):
        envoy = Envoy(MLP())
        with envoy.trace(x):
            h = nnsight.save(envoy.fc1.output)
            tmp = envoy.fc2.output  # not saved
        assert isinstance(h, torch.Tensor)
        with pytest.raises(UnboundLocalError):
            tmp

    def test_unsaved_value_does_not_reach_the_frame_at_all(self, x):
        # Stronger than the UnboundLocalError above, which only proves the name
        # never became a fast local: an unsaved value must not be in the frame's
        # locals mapping either. Blocks write to their own shared store, and push
        # is the only thing that writes to the frame.
        envoy = Envoy(MLP())
        with envoy.trace(x):
            h = nnsight.save(envoy.fc1.output)
            tmp = envoy.fc2.output  # not saved
        assert isinstance(h, torch.Tensor)
        assert "tmp" not in locals()

    def test_dict_capture_still_works_without_save(self, x):
        # Mutating an outer object is a side effect, independent of save/push.
        model = MLP()
        envoy = Envoy(model)
        captured = {}
        with envoy.trace(x):
            captured["h"] = envoy.fc1.output
        assert torch.allclose(captured["h"], model.fc1(x))


class TestNested:
    def test_saved_in_inner_reaches_user(self, x):
        # A value saved in the inner (backward) trace reaches the user: inner
        # pushes everything up, the outermost trace keeps only saved values.
        model = MLP()
        envoy = Envoy(model)
        with envoy.trace(x):
            a1 = envoy.fc1.output
            loss = envoy.output.sum()
            with loss.backward():
                g = nnsight.save(a1.grad.clone())
        assert isinstance(g, torch.Tensor)

    def test_inner_value_flows_out_without_save_but_does_not_reach_user(self, x):
        # An unsaved value from the inner trace is usable in the outer block (no
        # save needed inner->outer), but does not itself reach the user; only the
        # value the outer block saves crosses back out.
        model = MLP()
        envoy = Envoy(model)
        with envoy.trace(x):
            a1 = envoy.fc1.output
            loss = envoy.output.sum()
            with loss.backward():
                g = a1.grad.clone()      # NOT saved
            z = nnsight.save(g.sum())    # uses inner g, saves z
        assert isinstance(z, torch.Tensor)
        with pytest.raises(UnboundLocalError):
            g


save_mounted = hasattr(object(), "save")


@pytest.mark.skipif(not save_mounted, reason="C .save() mount not built/enabled")
class TestSaveMethod:
    def test_method_returns_same_object_in_trace(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.trace(x):
            out = envoy.fc1.output
            assert out.save() is out  # save(obj) returns obj unchanged

    def test_method_outside_trace_raises(self):
        # .save() marks a value to return from a trace, so it errors with none
        # active — rather than silently marking into a set nothing will read.
        with pytest.raises(ValueError, match="outside a trace"):
            [1, 2, 3].save()

    def test_method_escapes_a_trace(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.trace(x):
            h = envoy.fc1.output.save()
        assert torch.allclose(h, model.fc1(x))

    def test_method_matches_function_form(self, x):
        model = MLP()
        envoy = Envoy(model)
        with envoy.trace(x):
            m = envoy.fc1.output.save()          # method form
            f = nnsight.save(envoy.fc2.output)   # function form
        assert torch.allclose(m, model.fc1(x))
        assert torch.allclose(f, model(x))


class TestThreadSafety:
    def test_concurrent_traces_keep_separate_saves(self):
        # Save state is per-thread: many traces running at once, each saving a
        # distinct value, must not clobber one another's saved set.
        results = {}
        errors = []

        def run(i):
            try:
                envoy = Envoy(MLP())
                with envoy.trace(torch.randn(2, 8)):
                    v = nnsight.save(torch.full((2, 8), float(i)))
                results[i] = v
            except Exception as error:  # noqa: BLE001
                errors.append((i, error))

        threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i in range(8):
            assert torch.allclose(results[i], torch.full((2, 8), float(i)))
