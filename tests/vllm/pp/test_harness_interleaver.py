"""The PP interleaver and listener across two gloo ranks, without vLLM.

Every test here spawns two ranks (`_support.run_two_ranks`), each opening its
half of the harness (`_support.Stage`): a listener over the world group and,
for the interleaver-level tests, a PPInterleaver whose ownership map the test
sets. Both ranks run the SAME intervention block, as under real PP: reads of a
local module park and are served by that rank's ``handle``; reads of the peer's
module are answered with lazies and completed through the pull protocol.

Wire level: buffered and parked serves, container structures and mixed
dtypes, error replies for values the producer cannot pickle, the scoped clear,
the drain barrier.
Interleaver level: cross-stage read and write, the within-stage fragments
gather, publish scoping, collect-time serving, and error delivery onto the
worker greenlet.
"""

import collections
import threading
import time

import torch
import torch.distributed as dist

from _support import Stage, run_two_ranks

FRAGMENTED = "model.h.0.output"

# A namedtuple value crosses the wire as its own type (pickled by reference;
# both ranks import this module).
Row = collections.namedtuple("Row", "ids score")


# ---------------------------------------------------------------------------
# Wire level
# ---------------------------------------------------------------------------


def _listener_protocol(rank, world, rdv):
    stage = Stage(rank, world, rdv)
    listener, buffer, condition = stage.listener, stage.buffer, stage.condition

    if rank == 1:
        # Producer: values up front, one late, one non-tensor, one that
        # cannot be pickled.
        with condition:
            buffer[("model.h.0.output.i0", "req-a")] = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            buffer[("model.h.0.output.i0", None)] = (torch.ones(2, 2), torch.full((2, 2), 5.0))
            buffer[("model.samples.i0", "req-a")] = torch.tensor([7, 8, 9], dtype=torch.int32)
            buffer[("model.h.1.inputs.i0", "req-a")] = {"not": "a tensor"}
            buffer[("model.h.3.output.i0", "req-a")] = threading.Lock()

        def publish_late():
            time.sleep(0.5)
            stage.publish(("model.h.2.output.i0", "req-a"), torch.full((2, 3), 2.5))

        threading.Thread(target=publish_late).start()
        listener.drain_barrier()
        # A scoped clear of a finished request error-replies a pull parked for a
        # value that request never produced.
        time.sleep(0.5)
        listener.clear_buffer(req_ids=["req-a"])
        time.sleep(1.0)
    else:
        # Consumer: several split-phase pulls issued before any wait.
        p1 = listener.begin_pull(1, "model.h.0.output.i0", "req-a")
        p2 = listener.begin_pull(1, "model.h.0.output.i0", None)
        p3 = listener.begin_pull(1, "model.samples.i0", "req-a")
        p4 = listener.begin_pull(1, "model.h.2.output.i0", "req-a")  # parked at the producer
        v1 = p1.complete()
        assert v1.shape == (3, 4) and v1[2, 3] == 11, v1
        v2 = p2.complete()
        assert isinstance(v2, tuple) and torch.all(v2[1] == 5.0), v2
        v3 = p3.complete()
        assert v3.dtype == torch.int32 and v3.tolist() == [7, 8, 9], v3
        v4 = p4.complete()
        assert v4.shape == (2, 3) and torch.all(v4 == 2.5), v4
        # A value without tensors travels whole inside the reply's metadata.
        p5 = listener.begin_pull(1, "model.h.1.inputs.i0", "req-a")
        assert p5.complete() == {"not": "a tensor"}
        p6 = listener.begin_pull(1, "model.h.3.output.i0", "req-a")
        try:
            p6.complete()
            raise AssertionError("an unpicklable value should have error-replied")
        except RuntimeError as error:
            assert "model.h.3" in str(error) and "cannot pickle" in str(error), error
        listener.drain_barrier()
        p7 = listener.begin_pull(1, "model.h.9.output.i0", "req-a")
        try:
            p7.complete(timeout=5.0)
            raise AssertionError("an abandoned pull should have raised")
        except RuntimeError as error:
            assert "never produced" in str(error), error

    stage.close()


def test_listener_protocol():
    run_two_ranks(_listener_protocol)


def _wire_structures(rank, world, rdv):
    stage = Stage(rank, world, rdv)
    values = {
        "single": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "one_tuple": (torch.arange(6, dtype=torch.float32).reshape(2, 3),),
        "named": Row(torch.tensor([1, 2, 3]), 0.5),
        # A module's ``.inputs``: ``((args), {kwargs})`` with int64 positions
        # beside bf16 hidden states and a ``None`` residual.
        "inputs": (
            (torch.arange(4, dtype=torch.int64), torch.full((4, 2), 1.5, dtype=torch.bfloat16), None),
            {"flag": torch.tensor(True)},
        ),
        "many": tuple(torch.full((i + 1,), float(i)) for i in range(64)),
        "empty": torch.empty(0, 4, dtype=torch.float16),
        "scalar": torch.tensor(2.5, dtype=torch.float64),
    }
    if rank == 1:
        with stage.condition:
            for name, value in values.items():
                stage.buffer[(f"model.{name}.output.i0", None)] = value
    dist.barrier()
    if rank == 0:
        pulls = {name: stage.listener.begin_pull(1, f"model.{name}.output.i0") for name in values}
        got = {name: pull.complete(timeout=10.0) for name, pull in pulls.items()}
        assert isinstance(got["single"], torch.Tensor) and torch.equal(got["single"], values["single"])
        one = got["one_tuple"]
        assert type(one) is tuple and len(one) == 1 and torch.equal(one[0], values["single"]), one
        assert type(got["named"]) is Row and got["named"].score == 0.5, got["named"]
        assert got["named"].ids.tolist() == [1, 2, 3]
        (positions, hidden, residual), kwargs = got["inputs"]
        assert positions.dtype == torch.int64 and positions.tolist() == [0, 1, 2, 3]
        assert hidden.dtype == torch.bfloat16 and hidden.shape == (4, 2) and torch.all(hidden == 1.5)
        assert residual is None and kwargs["flag"].dtype == torch.bool and bool(kwargs["flag"])
        assert len(got["many"]) == 64
        assert all(t.shape == (i + 1,) and torch.all(t == i) for i, t in enumerate(got["many"]))
        assert got["empty"].shape == (0, 4) and got["empty"].dtype == torch.float16
        assert got["scalar"].shape == () and got["scalar"].dtype == torch.float64
        assert got["scalar"].item() == 2.5
    stage.close()


def test_reply_preserves_structure_and_dtypes():
    run_two_ranks(_wire_structures)


# ---------------------------------------------------------------------------
# Interleaver level: reads, writes, fragments
# ---------------------------------------------------------------------------


def _crossstage(rank, world, rdv):
    """Rank 0 owns h.0, rank 1 owns h.1; the block reads both and combines them.
    On each rank one read is local and one remote; the combine forces the
    remote lazy (after both reads, keeping forward order) and parks on a pull
    the serve point completes from the peer's published buffer."""
    stage = Stage(rank, world, rdv, {"h.0": 0, "h.1": 1})
    mediator = stage.mediator(
        """
a = Mediator.value("model.h.0.output")
b = Mediator.value("model.h.1.output")
c = (a * 2 + b).sum().item()
"""
    )
    provider, value = {
        0: ("model.h.0.output", torch.arange(4, dtype=torch.float32)),
        1: ("model.h.1.output", torch.full((4,), 5.0)),
    }[rank]
    with stage.interleaver:
        stage.interleaver.handle(provider, value)
    stage.interleaver.serve_pulls()

    assert not mediator.alive, "the block runs to completion on both ranks"
    # sum([0,2,4,6] + [5,5,5,5]) = 32 on BOTH ranks: the pulled value equals the peer's local one.
    assert mediator.lcls["c"] == 32.0, dict(mediator.lcls)
    if rank == 1:
        # The published buffer serves a repeat pull of the same visit.
        pulled = stage.listener.begin_pull(0, "model.h.0.output.i0").complete()
        assert torch.equal(pulled, torch.arange(4, dtype=torch.float32)), pulled
    stage.listener.drain_barrier()
    stage.close()


def test_cross_stage_read_matches_the_owning_rank():
    run_two_ranks(_crossstage)


def _fragments(rank, world, rdv):
    """A fake Fragments marks rank 0's h.0 output as one piece of a whole
    (whole = piece * 2, undone by / 2: an all-reduce over two equal partials,
    with no collective). The local worker reads the whole, the peer's pull
    receives the whole, and the model's forward gets the re-split piece."""
    from nnsight.intervention.fragments import Fragments

    class FakeFragments(Fragments):
        enabled = True

        def fragmented(self, location):
            return location == FRAGMENTED

        def whole(self, location, value):
            return value * 2, lambda edited: edited / 2

    stage = Stage(rank, world, rdv, {"h.0": 0, "h.1": 1}, fragments=FakeFragments())
    mediator = stage.mediator(
        """
a = Mediator.value("model.h.0.output")
b = Mediator.value("model.h.1.output")
c = (a + b).sum().item()
"""
    )
    provider, piece = {
        0: (FRAGMENTED, torch.arange(4, dtype=torch.float32)),
        1: ("model.h.1.output", torch.full((4,), 5.0)),
    }[rank]
    with stage.interleaver:
        returned = stage.interleaver.handle(provider, piece.clone())
    stage.interleaver.serve_pulls()

    assert not mediator.alive
    # sum([0,2,4,6] + [5,5,5,5]) = 32 on BOTH ranks.
    assert mediator.lcls["c"] == 32.0, dict(mediator.lcls)
    if rank == 0:
        assert torch.equal(returned, piece), returned  # the model continues from the piece
    if rank == 1:
        pulled = stage.listener.begin_pull(0, f"{FRAGMENTED}.i0").complete()
        assert torch.equal(pulled, torch.tensor([0.0, 2.0, 4.0, 6.0])), pulled  # the buffer holds the whole
    stage.listener.drain_barrier()
    stage.close()


def test_fragments_gather_before_the_publish():
    run_two_ranks(_fragments)


def _publish_scope(rank, world, rdv):
    """One worker reads one module's output once, while the interleaver handles
    four modules' inputs and outputs for three steps: the buffer holds exactly
    the one entry a worker waited on."""
    modules, steps = 4, 3
    stage = Stage(rank, world, rdv, {f"h.{m}": 0 for m in range(modules)} | {"peer": 1})
    stage.mediator('x = Mediator.value("model.h.0.output")\n')
    if rank == 0:
        value = torch.arange(4.0)
        with stage.interleaver:
            for _ in range(steps):
                for m in range(modules):
                    stage.interleaver.handle(f"model.h.{m}.input", ((value,), {}))
                    stage.interleaver.handle(f"model.h.{m}.output", value)
        assert list(stage.buffer) == [("model.h.0.output.i0", None)], list(stage.buffer)
        assert torch.equal(stage.buffer[("model.h.0.output.i0", None)], value)
    stage.listener.drain_barrier()
    stage.close()


def test_publish_covers_only_waited_locations():
    run_two_ranks(_publish_scope)


# ---------------------------------------------------------------------------
# Interleaver level: collect-time serving
# ---------------------------------------------------------------------------

_READ_H1 = """
a = Mediator.value("model.h.1.output")
result = float(a.sum())
"""


def _scoped_serve(rank, world, rdv):
    """serve_pulls(only=...) resumes the named requests' workers; the others
    keep their parks for their own step serves."""
    stage = Stage(rank, world, rdv, {"h.0": 0, "h.1": 1})
    mediators = {req: stage.mediator(_READ_H1, req_id=req) for req in ("req-A", "req-B")}
    with stage.interleaver:
        if rank == 0:
            stage.interleaver.handle("model.h.0.output", torch.zeros(4))
        else:
            stage.interleaver.handle("model.h.1.output", torch.arange(4.0))
    if rank == 0:
        stage.interleaver.serve_pulls(block=True, only={"req-A"})
        assert not mediators["req-A"].alive and mediators["req-A"].lcls["result"] == 6.0
        assert mediators["req-B"].alive and mediators["req-B"].pending is not None
        stage.interleaver.serve_pulls(block=True)
        assert not mediators["req-B"].alive and mediators["req-B"].lcls["result"] == 6.0
    stage.listener.drain_barrier()
    stage.close()


def test_serve_pulls_scoped_to_requests():
    run_two_ranks(_scoped_serve)


def _finalize_by_round(rank, world, rdv):
    """Collect serves a finished request's workers explicitly (a later step's
    scheduling drops them from the per-step list) under the produced-round
    gate: a pull for a produced round completes, a pull for a round the
    pipeline never made stays parked, and a released worker leaves no pull
    records."""
    stage = Stage(rank, world, rdv, {"h.0": 0, "h.1": 1})
    producible = stage.mediator(_READ_H1, req_id="req-P")
    # No completed rounds and no publisher: a pull the pipeline cannot produce.
    lookahead = stage.mediator(_READ_H1, req_id="req-L", register=(rank == 0))
    stage.interleaver.rounds.update({"req-P": 1, "req-L": 0})
    with stage.interleaver:
        if rank == 0:
            stage.interleaver.handle("model.h.0.output", torch.zeros(4))
        else:
            stage.interleaver.handle("model.h.1.output", torch.arange(4.0))
    if rank == 0:
        stage.interleaver.mediators.clear()
        t0 = time.time()
        stage.interleaver.serve_pulls(block=True, drain=False, mediators=[producible, lookahead])
        elapsed = time.time() - t0
        assert elapsed < 10, f"finalize serve took {elapsed:.1f}s"
        assert not producible.alive and producible.lcls["result"] == 6.0
        assert lookahead.alive and lookahead.pending is not None
        stage.interleaver.discard_pulls(lookahead)
        assert not any(key[0] == id(lookahead) for key in stage.interleaver._pulls)
    stage.listener.drain_barrier()
    stage.close()


def test_finalize_serves_finished_workers_by_round():
    run_two_ranks(_finalize_by_round)


# ---------------------------------------------------------------------------
# Interleaver level: errors land on the worker
# ---------------------------------------------------------------------------


def _wait_for_parked(listener, timeout=10.0):
    """Producer side: block until a consumer's pull is parked here."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with listener._condition:
            if listener._parked:
                return
        time.sleep(0.05)
    raise AssertionError("no pull parked within the wait window")


def _interleaver_errors(rank, world, rdv):
    """A failed pull is thrown at the force line, where user code can catch it;
    an uncaught failure unwinds only its own mediator under defer_exceptions
    and propagates out of serve_pulls without it; an upstream in-place serve
    raises at the force line directly. Failures are induced by finalizing the
    producer while a pull is parked: clear_buffer error-replies every parked
    pull. One module per scenario, since the buffer persists across them."""
    stage = Stage(rank, world, rdv, {"h.0": 0, "h.1": 1, "h.2": 1, "h.3": 1, "h.4": 0})
    interleaver, listener = stage.interleaver, stage.listener

    # A caught failure: the worker pulls again and the drain loop serves the retry.
    if rank == 0:
        mediator = stage.mediator(
            """
try:
    a = Mediator.value("model.h.1.output")
    s = (a + 1).sum().item()
except RuntimeError:
    retry = Mediator.value("model.h.1.output")
    s = (retry * 10).sum().item()
"""
        )
        with interleaver:
            pass  # the worker starts, forces the remote value, parks on the pull
        interleaver.serve_pulls(block=True)
        assert not mediator.alive and mediator.exception is None, mediator.exception
        assert mediator.lcls["s"] == 120.0, dict(mediator.lcls)  # full((3,), 4.0) * 10, summed
    else:
        _wait_for_parked(listener)
        listener.clear_buffer()  # error-replies the parked first pull
        stage.publish(("model.h.1.output.i0", None), torch.full((3,), 4.0))
    dist.barrier()

    # Deferred: one request dies, its peer finishes.
    if rank == 0:
        interleaver.defer_exceptions = True
        doomed = stage.mediator('x = Mediator.value("model.h.2.output")\ny = (x - 1).sum().item()\n')
        healthy = stage.mediator('b = Mediator.value("model.h.0.output")\nd = (b.sum() + 1).item()\n')
        with interleaver:
            interleaver.handle("model.h.0.output", torch.arange(3, dtype=torch.float32))
        assert not healthy.alive and healthy.lcls["d"] == 4.0, dict(healthy.lcls)
        interleaver.serve_pulls(block=True)
        assert doomed.exception is not None and "never produced" in str(doomed.exception), doomed.exception
        assert healthy.exception is None
    else:
        _wait_for_parked(listener)
        listener.clear_buffer()
    dist.barrier()

    # Not deferred: the failure propagates out of serve_pulls after being recorded.
    if rank == 0:
        interleaver.defer_exceptions = False
        doomed = stage.mediator('x = Mediator.value("model.h.3.output")\ny = (x - 1).sum().item()\n')
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

    # Upstream in-place: raised at the force line, on the worker.
    if rank == 1:
        mediator = stage.mediator(
            """
b = Mediator.value("model.h.1.output")
try:
    a = Mediator.value("model.h.4.output")
    s = (a + b).sum().item()
except RuntimeError as error:
    assert "never produced" in str(error)
    s = -2.0
"""
        )
        with interleaver:
            interleaver.handle("model.h.1.output", torch.ones(3))
        assert not mediator.alive and mediator.exception is None, mediator.exception
        assert mediator.lcls["s"] == -2.0, dict(mediator.lcls)
    else:
        _wait_for_parked(listener)
        listener.clear_buffer()
    dist.barrier()

    stage.close()


def test_pull_failures_land_on_the_worker():
    run_two_ranks(_interleaver_errors)
