"""Tests for I1: vLLM serve must apply a finite mediator_timeout.

Background
----------
``Interleaver.mediator_timeout`` defaults to ``None`` (wait forever).
For local ``model.trace()`` this is correct: a debugger break inside
an intervention should not be killed by an outer timeout. For a
multi-tenant continuous-batching server, ``None`` is unsafe — a
single user submitting a trace with ``time.sleep(99999)`` would wedge
the shared forward thread and starve every concurrent request.

The HF vanilla CLI sets ``mediator_timeout=30.0`` on the
``VanillaBatchServer``; the vLLM CLI previously did not set it at
all. These tests lock down the fix:

1. The CLI parser exposes ``--mediator-timeout`` and the default is
   finite.
2. ``_apply_server_config`` actually writes the value through to
   ``model.interleaver.mediator_timeout``.

End-to-end "a hung intervention surfaces TimeoutError within N
seconds" coverage already exists for the HF vanilla path in
``test_hf_serve.py::test_hung_intervention_times_out``. The
underlying timeout mechanism is shared (same ``Interleaver`` field,
same ``Mediator.event_queue.wait(timeout=...)`` call site). We
therefore don't replicate that here — the value-flow tests below
are sufficient to catch the I1 regression class ("CLI doesn't set
the field" or "default goes back to None").
"""

from __future__ import annotations

from types import SimpleNamespace


def test_default_mediator_timeout_is_finite():
    """The fix must keep a finite default — going back to ``None`` would
    silently re-introduce I1 even with the new flag in place.
    """
    from nnsight.modeling.vllm.serve.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["some/model"])
    assert isinstance(args.mediator_timeout, float)
    assert args.mediator_timeout > 0, (
        f"default mediator_timeout must be > 0; got {args.mediator_timeout!r} "
        f"(I1 regression: server would wait forever on a hung intervention)"
    )


def test_mediator_timeout_flag_overrides_default():
    """``--mediator-timeout 12.5`` must round-trip through argparse."""
    from nnsight.modeling.vllm.serve.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["some/model", "--mediator-timeout", "12.5"])
    assert args.mediator_timeout == 12.5


def test_apply_server_config_writes_to_interleaver():
    """The CLI helper must reach the field that the FastAPI handler
    reads. If a future refactor moves ``mediator_timeout`` off
    ``_interleaver`` (e.g. onto a per-step context), this test fails
    and the operator-visible behavior never silently drifts.
    """
    from nnsight.modeling.vllm.serve.cli import _apply_server_config

    fake_interleaver = SimpleNamespace(mediator_timeout=None)
    fake_model = SimpleNamespace(interleaver=fake_interleaver)

    _apply_server_config(fake_model, mediator_timeout=42.0)

    assert fake_model.interleaver.mediator_timeout == 42.0


def test_apply_server_config_overwrites_existing_value():
    """A second call with a different value must take effect (idempotency
    is not the goal here — last writer wins, matching the assignment
    in main()). Catches a regression where someone wraps the assignment
    in ``if model.interleaver.mediator_timeout is None``.
    """
    from nnsight.modeling.vllm.serve.cli import _apply_server_config

    fake_interleaver = SimpleNamespace(mediator_timeout=99.0)
    fake_model = SimpleNamespace(interleaver=fake_interleaver)

    _apply_server_config(fake_model, mediator_timeout=5.0)

    assert fake_model.interleaver.mediator_timeout == 5.0
