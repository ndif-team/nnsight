"""Error delivery across the PP machinery, two gloo ranks, no vLLM.

Wire level: a value the producer cannot serialize must come back as an error
reply the consumer raises, never a hang or a desynced recv.

Interleaver level: a failed pull must land on the worker greenlet at the line
that forced the value, where user code can catch it; an uncaught failure must
unwind only that mediator (``defer_exceptions`` on a shared engine) or
propagate out of ``serve_pulls`` (``defer_exceptions`` off). The upstream
in-place serve raises at the force line directly. Failures are induced by
finalizing the producer while a pull is parked: ``clear_buffer`` error-replies
every parked pull.
"""

import os
import time
import threading

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _init(rank, world, rdv):
    dist.init_process_group(
        "gloo", init_method=f"file://{rdv}", rank=rank, world_size=world
    )


def _start_listener(rank):
    from nnsight.modeling.vllm.pp_listener import PPListener

    buffer: dict = {}
    condition = threading.Condition()
    listener = PPListener(
        buffer, condition, dist.group.WORLD, rank, torch.device("cpu")
    )
    listener.start()
    return listener, buffer, condition


def _drain_and_stop(listener, rank):
    # A pending gloo recv at interpreter teardown SIGABRTs; wake the peer's
    # blocked recv with a dummy request. Production relies on hard worker exit.
    from nnsight.modeling.vllm.pp_listener import REQUEST_MSG_BYTES, TAG_REQUEST

    dist.barrier()
    listener._stop_event.set()
    dist.send(
        torch.zeros(REQUEST_MSG_BYTES, dtype=torch.uint8),
        group_dst=1 - rank, tag=TAG_REQUEST,
    )
    listener._thread.join(timeout=5)
    dist.barrier()


def _wait_for_parked(listener, timeout=10.0):
    """Producer side: block until a consumer's pull is parked here."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with listener._condition:
            if listener._parked:
                return
        time.sleep(0.05)
    raise AssertionError("no pull parked within the wait window")


# ---------------------------------------------------------------------------
# Wire level
# ---------------------------------------------------------------------------


def _run_wire_overflow(rank, world, rdv):
    _init(rank, world, rdv)
    listener, buffer, condition = _start_listener(rank)

    if rank == 1:
        # 16 rank-1 tensors need 2 + 16*2 = 34 shape-header slots, over the
        # 32-slot header; the producer must error-reply, not desync.
        with condition:
            buffer[("model.h.0.output.i0", None)] = tuple(
                torch.zeros(2) for _ in range(16)
            )
    dist.barrier()
    if rank == 0:
        pull = listener.begin_pull(1, "model.h.0.output.i0")
        try:
            pull.complete(timeout=10.0)
            raise AssertionError("oversized value should have error-replied")
        except RuntimeError as error:
            assert "shape-header slots" in str(error), error

    _drain_and_stop(listener, rank)


def test_unserializable_value_error_replies(tmp_path=None):
    rdv = (
        str(tmp_path / "rdv") if tmp_path is not None
        else "/tmp/nnsight_pp_wire_overflow_rdv"
    )
    if os.path.exists(rdv):
        os.remove(rdv)
    mp.spawn(_run_wire_overflow, args=(2, rdv), nprocs=2, join=True)


def _run_wire_mixed_dtype(rank, world, rdv):
    _init(rank, world, rdv)
    listener, buffer, condition = _start_listener(rank)

    if rank == 1:
        with condition:
            buffer[("model.h.0.output.i0", None)] = (
                torch.zeros(2, dtype=torch.float32),
                torch.tensor([5, 6], dtype=torch.int64),
            )
    dist.barrier()
    if rank == 0:
        pull = listener.begin_pull(1, "model.h.0.output.i0")
        try:
            value = pull.complete(timeout=10.0)
        except RuntimeError:
            value = None  # the error reply is the intended behavior
        assert value is None or value[1].dtype == torch.int64, (
            f"mixed-dtype tuple came back silently reinterpreted: "
            f"{[t.dtype for t in value]}"
        )

    _drain_and_stop(listener, rank)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a mixed-dtype tuple is neither error-replied nor faithfully "
        "delivered: torch.cat promotes instead of raising (the same-dtype "
        "validation _serve_reply's comment claims does not happen), the shape "
        "header carries only the first tensor's dtype, and the consumer "
        "rebuilds every element in it — the int64 element arrives as float32."
    ),
)
def test_mixed_dtype_tuple_is_not_silently_reinterpreted(tmp_path=None):
    rdv = (
        str(tmp_path / "rdv") if tmp_path is not None
        else "/tmp/nnsight_pp_wire_mixed_rdv"
    )
    if os.path.exists(rdv):
        os.remove(rdv)
    mp.spawn(_run_wire_mixed_dtype, args=(2, rdv), nprocs=2, join=True)


# ---------------------------------------------------------------------------
# Interleaver level
# ---------------------------------------------------------------------------


def _make_mediator(interleaver, block):
    from nnsight.intervention.interleaver import Mediator

    mediator = Mediator(
        compile(block, "<pp-error-test>", "exec"), {"Mediator": Mediator}, {}
    )
    interleaver.mediators.append(mediator)
    return mediator


def _run_interleaver_errors(rank, world, rdv):
    _init(rank, world, rdv)
    from nnsight.modeling.vllm.pp import PPModuleMap
    from nnsight.modeling.vllm.pp_interleaver import PPInterleaver

    listener, buffer, condition = _start_listener(rank)
    module_map = PPModuleMap(world)
    # One module per scenario: the pull buffer persists across scenarios, so
    # reusing a provider would serve a later doomed pull from a stale entry.
    module_map.set_derived_owners(
        {"h.0": 0, "h.1": 1, "h.2": 1, "h.3": 1, "h.4": 0}
    )
    interleaver = PPInterleaver(module_map, listener, rank)

    # --- a failed downstream pull is thrown at the force line; the worker
    # --- catches it, pulls again, and the drain loop serves the retry
    if rank == 0:
        mediator = _make_mediator(
            interleaver,
            """
try:
    a = Mediator.value("model.h.1.output")
    s = (a + 1).sum().item()
except RuntimeError:
    retry = Mediator.value("model.h.1.output")
    s = (retry * 10).sum().item()
""",
        )
        with interleaver:
            pass  # worker starts, forces the remote value, parks on the pull
        interleaver.serve_pulls(block=True)
        assert not mediator.alive, "worker should have finished"
        assert mediator.exception is None, mediator.exception
        # The first pull error-replied; the retry was served the published
        # value: full((3,), 4.0) * 10 summed.
        assert mediator.lcls["s"] == 120.0, dict(mediator.lcls)
    else:
        _wait_for_parked(listener)
        listener.clear_buffer()  # error-replies the parked first pull
        key = ("model.h.1.output.i0", None)
        with condition:
            buffer[key] = torch.full((3,), 4.0)
        listener.dispatch_parked(key, buffer[key])
    dist.barrier()

    # --- an uncaught pull failure unwinds only its own mediator when the
    # --- interleaver defers exceptions (one request dies, peers finish)
    if rank == 0:
        interleaver.defer_exceptions = True
        doomed = _make_mediator(
            interleaver,
            """
x = Mediator.value("model.h.2.output")
y = (x - 1).sum().item()
""",
        )
        healthy = _make_mediator(
            interleaver,
            """
b = Mediator.value("model.h.0.output")
d = (b.sum() + 1).item()
""",
        )
        with interleaver:
            interleaver.handle("model.h.0.output", torch.arange(3, dtype=torch.float32))
        assert not healthy.alive and healthy.lcls["d"] == 4.0, dict(healthy.lcls)
        interleaver.serve_pulls(block=True)
        assert doomed.exception is not None
        assert "never produced" in str(doomed.exception), doomed.exception
        assert healthy.exception is None
    else:
        _wait_for_parked(listener)
        listener.clear_buffer()
    dist.barrier()

    # --- with defer_exceptions off, the same failure propagates out of
    # --- serve_pulls after being recorded
    if rank == 0:
        interleaver.defer_exceptions = False
        doomed = _make_mediator(
            interleaver,
            """
x = Mediator.value("model.h.3.output")
y = (x - 1).sum().item()
""",
        )
        with interleaver:
            pass
        try:
            interleaver.serve_pulls(block=True)
            raise AssertionError("serve_pulls should have re-raised")
        except RuntimeError as error:
            assert "never produced" in str(error), error
        assert doomed.exception is not None
    else:
        _wait_for_parked(listener)
        listener.clear_buffer()
    dist.barrier()

    # --- an upstream in-place serve raises at the force line, on the worker,
    # --- where user code can catch it (no park, no serve point involved)
    if rank == 1:
        mediator = _make_mediator(
            interleaver,
            """
b = Mediator.value("model.h.1.output")
try:
    a = Mediator.value("model.h.4.output")
    s = (a + b).sum().item()
except RuntimeError as error:
    assert "never produced" in str(error)
    s = -2.0
""",
        )
        with interleaver:
            # Serving the local read resumes the worker, whose upstream force
            # then blocks in place until the peer error-replies.
            interleaver.handle("model.h.1.output", torch.ones(3))
        assert not mediator.alive
        assert mediator.exception is None, mediator.exception
        assert mediator.lcls["s"] == -2.0, dict(mediator.lcls)
    else:
        _wait_for_parked(listener)
        listener.clear_buffer()
    dist.barrier()

    _drain_and_stop(listener, rank)


def test_pull_failures_land_on_the_worker(tmp_path=None):
    rdv = (
        str(tmp_path / "rdv") if tmp_path is not None
        else "/tmp/nnsight_pp_interleaver_errors_rdv"
    )
    if os.path.exists(rdv):
        os.remove(rdv)
    mp.spawn(_run_interleaver_errors, args=(2, rdv), nprocs=2, join=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x", "-p", "no:cacheprovider"]))
