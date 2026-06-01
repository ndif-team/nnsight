"""Tests for C4: enforce the serve-path invariants that previously lived
only in comments / docstrings.

Two invariants are promoted to asserted / locked behavior:

1. ``MetaMixin.dispatch()`` is idempotent and thread-safe. Two concurrent
   callers must not both run ``_load`` / ``_update`` and leave the Envoy
   tree in whichever order the last ``_update`` happened to finish.

2. ``Interleaver.initialize()`` refuses to reset state while an
   interleaving session is active. The prior bg-vs-handler race
   (fixed by commit 0bd5b80 — handlers now skip
   ``_init_shared_interleaver()``) is caught in code rather than
   relying on every future handler remembering to skip it.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch


class TestDispatchIdempotency:
    """MetaMixin.dispatch() must be safe under concurrent callers."""

    def test_dispatch_returns_early_when_already_dispatched(self):
        """Fast-path: already dispatched → no ``_load`` call."""
        from nnsight import LanguageModel

        m = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
        assert m.dispatched

        load_calls = {"n": 0}
        original_load = m._load

        def tracking_load(*a, **k):
            load_calls["n"] += 1
            return original_load(*a, **k)

        m._load = tracking_load
        m.dispatch()
        m.dispatch()
        m.dispatch()

        assert load_calls["n"] == 0, (
            f"dispatch() on an already-dispatched model should be a no-op; "
            f"_load was called {load_calls['n']} times"
        )

    def test_concurrent_dispatch_loads_once(self):
        """Two threads racing into ``dispatch()`` on an undispatched
        model must collectively call ``_load`` exactly once.

        Uses a fresh meta-loaded model (``dispatch=False``) and patches
        ``_load`` to introduce a small sleep so the race window is real.
        """
        from nnsight import LanguageModel

        m = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=False)
        assert not m.dispatched

        load_calls = {"n": 0}
        original_load = m._load

        def slow_load(*a, **k):
            load_calls["n"] += 1
            # Hold long enough that a second thread can enter
            # ``dispatch()`` before this finishes.
            time.sleep(0.1)
            return original_load(*a, **k)

        m._load = slow_load

        errors = []

        def runner():
            try:
                m.dispatch()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=runner) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors, f"concurrent dispatch raised: {errors}"
        assert m.dispatched, "model should be dispatched after the threads join"
        assert load_calls["n"] == 1, (
            f"expected exactly one _load call, got {load_calls['n']} — "
            f"the idempotency lock didn't hold under concurrent callers"
        )

    def test_dispatch_lock_survives_pickling(self):
        """The ``_dispatch_lock`` is re-created on unpickle (locks aren't
        picklable). Verifying the attribute exists post-unpickle so
        ``dispatch()`` keeps working.
        """
        import pickle

        from nnsight import LanguageModel

        m = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
        # LanguageModel itself isn't fully picklable in isolation for
        # all cases, but ``MetaMixin.__setstate__`` is exercised any
        # time a state dict is round-tripped. Simulate by calling
        # __setstate__ directly with a minimal state snapshot.
        state = m.__getstate__() if hasattr(m, "__getstate__") else m.__dict__.copy()
        # Drop the lock from the "serialized" state — mirrors pickle's
        # refusal to serialize threading.Lock.
        state.pop("_dispatch_lock", None)

        m.__setstate__(state)
        assert hasattr(m, "_dispatch_lock")
        # And dispatch still works.
        m.dispatch()
        assert m.dispatched


class TestInitializeInvariant:
    """Interleaver.initialize() must refuse to reset state mid-interleaving."""

    def test_initialize_during_interleaving_raises(self):
        """Pre-fix: a handler that forgot to skip
        ``_init_shared_interleaver()`` would call
        ``Interleaver.initialize()`` while the bg vLLM worker was mid-
        ``Mediator.start()``, silently racing on
        ``interleaver.current``. Post-fix: fails loudly.
        """
        from nnsight.intervention.interleaver import Interleaver

        interleaver = Interleaver()

        # Simulate active interleaving (the state the bg worker thread
        # sees when ``Mediator.start()`` is running). We don't need a
        # real forward pass for this — the invariant is expressed via
        # the ``interleaving`` flag.
        interleaver._interleaving = True

        with pytest.raises(RuntimeError, match="initialize.*interleaving"):
            interleaver.initialize(mediators=[], tracer=None)

    def test_initialize_works_when_not_interleaving(self):
        """Sanity check — the invariant is "while interleaving only",
        not "always". Initialize outside an active session still works.
        """
        from nnsight.intervention.interleaver import Interleaver

        interleaver = Interleaver()
        assert not interleaver.interleaving

        # Does not raise — normal initialization path.
        interleaver.initialize(mediators=[], tracer=None)
        assert interleaver.mediators == []
