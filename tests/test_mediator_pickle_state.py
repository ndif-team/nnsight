"""Mediator state must survive pickle round-trips.

Background
----------
``Mediator.__init__`` initializes ``self._trace_saves: Optional[set] = None``.
The attribute was historically per-server-trace state set by
``NNsightRequestHelper._register_mediator``; since the
``_saves_var`` contextvar was dropped (saves are now scoped via
per-frame iteration over the process-wide ``Globals.saves`` set),
``_trace_saves`` is a vestigial back-compat field. It's kept so
unpickled mediators don't ``AttributeError`` on framework code that
hasn't been audited yet.

A pickle round-trip bypasses ``__init__``, so ``__setstate__`` is the
only chance to set the attribute on a deserialized mediator. Without
the default, unpickled mediators arrive without the attribute, and
any access — even an ``is not None`` check — raises ``AttributeError``.

Fix: ``__setstate__`` defaults ``_trace_saves`` to ``None`` — matching
the constructor and the local-trace semantic. These tests pin the
contract.
"""

from __future__ import annotations

import pickle
from collections import defaultdict


def _build_minimal_state() -> dict:
    """Mimic what ``Mediator.__getstate__`` returns. Tests against
    the actual ``__setstate__`` contract — if the contract changes,
    this dict must be updated alongside it.
    """
    return {
        "name": "test-mediator",
        "info": None,
        "batch_group": None,
        "intervention": lambda: None,
        "all_stop": None,
        "iteration_tracker": defaultdict(int),
    }


def test_unpickled_mediator_has_trace_saves_attribute():
    """Direct ``__setstate__`` call: a freshly-restored mediator
    must expose ``_trace_saves`` so ``Interleaver.__enter__`` can
    read it without AttributeError. Default ``None`` matches the
    local-trace semantic.
    """
    from nnsight.intervention.interleaver import Mediator

    m = Mediator.__new__(Mediator)
    m.__setstate__(_build_minimal_state())

    assert hasattr(m, "_trace_saves"), (
        "_trace_saves missing from unpickled mediator — "
        "Interleaver.__enter__ would AttributeError"
    )
    assert m._trace_saves is None, (
        f"_trace_saves default after unpickle must be None "
        f"(matching __init__); got {m._trace_saves!r}"
    )


def test_full_pickle_roundtrip_preserves_trace_saves_default():
    """End-to-end pickle round-trip via cloudpickle (the format
    used by RequestModel.serialize / deserialize in the real
    server path). The deserialized mediator's ``_trace_saves``
    must default to None.
    """
    import cloudpickle

    from nnsight.intervention.interleaver import Mediator

    # Construct a Mediator the way the server path does — through
    # __init__ — then pickle/unpickle it.
    def _intervention():
        return None

    # Mediator.__init__ requires (intervention, info). info is
    # the Tracer.Info object; tests don't need a real one because
    # __getstate__ just stores it as-is and __setstate__ assigns
    # it back. Use None.
    m = Mediator(intervention=_intervention, info=None, name="test")

    # Sanity: pre-pickle the attribute exists and is None.
    assert m._trace_saves is None

    blob = cloudpickle.dumps(m)
    m2 = cloudpickle.loads(blob)

    assert hasattr(m2, "_trace_saves"), (
        "_trace_saves missing after cloudpickle round-trip"
    )
    assert m2._trace_saves is None


def test_setstate_does_not_assume_trace_saves_in_state_dict():
    """``__getstate__`` does NOT include ``_trace_saves`` in its
    returned dict (the server-side trace-saves set isn't
    round-trip-safe; the server reconstructs it post-deserialize).
    ``__setstate__`` must therefore not require it as a key —
    just default it. This test pins that behavior so a future
    change that adds ``_trace_saves`` to ``__getstate__`` doesn't
    silently make ``__setstate__`` start expecting it.
    """
    from nnsight.intervention.interleaver import Mediator

    state = _build_minimal_state()
    assert "_trace_saves" not in state, (
        "this test's premise — that __getstate__ doesn't include "
        "_trace_saves — is no longer true; update the test"
    )

    m = Mediator.__new__(Mediator)
    m.__setstate__(state)
    # No KeyError; default applied.
    assert m._trace_saves is None


def test_unpickled_mediator_passes_interleaver_enter_check():
    """The actual line 484 read: ``if mediator._trace_saves is not None``.
    For a freshly-unpickled mediator, this expression must evaluate
    cleanly to False (not raise AttributeError). This pins the
    contract Interleaver.__enter__ depends on.
    """
    from nnsight.intervention.interleaver import Mediator

    m = Mediator.__new__(Mediator)
    m.__setstate__(_build_minimal_state())

    # The actual code in Interleaver.__enter__:
    if m._trace_saves is not None:
        # Pre-fix: this branch was unreachable because the attribute
        # access raised AttributeError before the comparison.
        assert False, "should not reach this branch with default None"
    # Falls through cleanly. Test passes.
