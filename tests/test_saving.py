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


class TestInjectResults:
    """A local run whose backend doesn't inject keeps the outermost block's saves
    on `backend.results` instead of the block's variables; inner traces and
    `tracer.iter` loops still pass their values on inside the block."""

    def test_saves_kept_on_the_backend_not_the_variables(self, x):
        from nnsight.tracing.backend import Backend

        model = MLP()
        envoy = Envoy(model)
        backend = Backend(inject_results=False)
        with envoy.trace(x, backend=backend):
            h = nnsight.save(envoy.fc1.output)
        assert "h" not in locals()
        assert set(backend.results) == {"h"}
        assert torch.allclose(backend.results["h"], model.fc1(x))

    def test_injecting_backend_keeps_nothing(self, x):
        from nnsight.tracing.backend import Backend

        envoy = Envoy(MLP())
        backend = Backend()
        with envoy.trace(x, backend=backend):
            h = nnsight.save(envoy.fc1.output)
        assert h.shape == (2, 8)
        assert backend.results is None

    def test_config_sets_the_default_for_every_trace(self, x, monkeypatch):
        """A library can switch injection off once, for traces it doesn't
        build a backend for; an explicit backend still decides for itself."""
        from nnsight.schema.config import CONFIG
        from nnsight.tracing.backend import Backend

        envoy = Envoy(MLP())
        monkeypatch.setattr(CONFIG.APP, "INJECT_RESULTS", False)
        with envoy.trace(x) as tracer:
            out = nnsight.save(envoy.output)
        assert "out" not in locals()
        assert set(tracer.backend.results) == {"out"}

        with envoy.trace(x, backend=Backend(inject_results=True)):
            injected = nnsight.save(envoy.output)
        assert injected.shape == (2, 8)

    def test_session_inner_traces_still_feed_the_session(self, x):
        from nnsight.tracing.backend import Backend

        model = MLP()
        envoy = Envoy(model)
        backend = Backend(inject_results=False)
        with envoy.session(backend=backend):
            with envoy.trace(x):
                hidden = envoy.fc1.output
            total = nnsight.save(hidden.sum())
        assert "total" not in locals()
        assert torch.allclose(backend.results["total"], model.fc1(x).sum())

    def test_a_failed_run_does_not_divert_the_next_one(self, x):
        from nnsight.tracing.backend import Backend

        envoy = Envoy(MLP())
        with pytest.raises(ValueError, match="boom"):
            with envoy.trace(x, backend=Backend(inject_results=False)):
                nnsight.save(envoy.fc1.output)
                raise ValueError("boom")
        with envoy.trace(x):
            h = nnsight.save(envoy.fc1.output)
        assert h.shape == (2, 8)

    @torch.no_grad()
    def test_generation_with_an_iter_loop(self, gpt2):
        """A name bound in the loop reaches the code after it; only the block's
        saves go to `results`."""
        from nnsight.tracing.backend import Backend

        backend = Backend(inject_results=False)
        with gpt2.generate(
            "Hello", max_new_tokens=3, do_sample=False, backend=backend
        ) as tracer:
            for _ in tracer.iter[:3]:
                last = gpt2.transformer.h[0].output[0]
            last_rows = nnsight.save(last.shape[0])
        assert "last_rows" not in locals()
        assert backend.results == {"last_rows": 1}  # a cached decode step

    @torch.no_grad()
    def test_every_invoke_contributes_its_saves(self, gpt2):
        from nnsight.tracing.backend import Backend

        backend = Backend(inject_results=False)
        with gpt2.trace(backend=backend) as tracer:
            with tracer.invoke("Hello"):
                first = nnsight.save(gpt2.lm_head.output)
            with tracer.invoke("Goodbye world"):
                second = nnsight.save(gpt2.lm_head.output)
        assert set(backend.results) == {"first", "second"}
        assert backend.results["first"].shape[0] == 1
        assert backend.results["second"].shape[0] == 1
