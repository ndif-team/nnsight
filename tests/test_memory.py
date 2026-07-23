"""Trace teardown leaves no reference cycles: once the scope that ran a trace is
gone, its wrapper, tracer, and unsaved values are freed by refcounting alone — no
cyclic GC pass needed (the tracer drops its hold on the caller's frame at exit).

The cyclic collector is disabled per test, so a surviving object means a real
reference cycle, not deferred collection. Each trace runs in a helper whose frame
is then released — the realistic lifetime, and free of the CPython quirk where
``del`` on a still-live frame's fast local leaves a lingering f_locals snapshot.
"""

import gc
import weakref

import pytest
import torch
import torch.nn as nn

import nnsight
from nnsight.intervention.envoy import Envoy


class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):
        return self.b(self.a(x))


def _x():
    return torch.randn(2, 8)


@pytest.fixture(autouse=True)
def _no_cyclic_gc():
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class TestModelCleanup:
    def test_module_freed_after_trace(self):
        def run():
            model = Envoy(TwoLayer())
            with model.trace(_x()) as tracer:
                nnsight.save(tracer.result)
            return weakref.ref(model._module)

        assert run()() is None

    def test_module_freed_without_trace(self):
        def run():
            model = Envoy(TwoLayer())
            return weakref.ref(model._module)

        assert run()() is None

    def test_interleaver_freed_with_wrapper(self):
        def run():
            model = Envoy(TwoLayer())
            return weakref.ref(model.interleaver)

        assert run()() is None

    def test_many_traces_no_accumulation(self):
        def run():
            model = TwoLayer()
            refs = []
            for _ in range(10):
                w = Envoy(model)
                refs.append(weakref.ref(w))
                with w.trace(_x()) as tracer:
                    nnsight.save(tracer.result)
            return refs

        assert all(r() is None for r in run())


class TestSavedObjectCleanup:
    def test_tracer_freed_but_saved_value_survives(self):
        holder = {}

        def run():
            model = Envoy(TwoLayer())
            with model.trace(_x()) as tracer:
                out = model.a.output.save()
            holder["out"] = out
            return weakref.ref(tracer)

        tracer_ref = run()
        # Trace scope gone -> tracer freed; the saved value we kept is still alive.
        assert tracer_ref() is None
        assert isinstance(holder["out"], torch.Tensor)


class TestTracerCleanup:
    def test_tracer_freed_after_trace(self):
        def run():
            model = Envoy(TwoLayer())
            with model.trace(_x()) as tracer:
                nnsight.save(tracer.result)
            return weakref.ref(tracer)

        assert run()() is None

    def test_frame_released_after_trace(self):
        # The tracer drops its hold on the caller's frame at trace end.
        model = Envoy(TwoLayer())
        with model.trace(_x()) as tracer:
            nnsight.save(tracer.result)
        assert tracer.info.frame is None


class TestExceptionCleanup:
    """An error mid-trace still tears everything down (the frame clear and the
    interleaver's mediator cleanup both run in `finally`)."""

    def test_module_freed_when_trace_body_raises(self):
        def run():
            model = Envoy(TwoLayer())
            ref = weakref.ref(model._module)
            try:
                with model.trace(_x()):
                    raise ValueError("boom")
            except ValueError:
                pass  # the exception (and its traceback) is cleared here
            return ref

        assert run()() is None

    def test_tracer_freed_when_trace_body_raises(self):
        def run():
            with pytest.raises(ValueError):
                model = Envoy(TwoLayer())
                with model.trace(_x()) as tracer:
                    raise ValueError("boom")
            return weakref.ref(tracer)

        assert run()() is None
