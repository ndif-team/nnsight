"""Two gloo ranks over the real link and interleaver, no vLLM engine.

Each test spawns two ranks that run ``_run_<name>(rank, world, rdv)``: the
wire (values, requests, failures) and then the interleaver on a block both
ranks run, where one rank's local read is the other's remote one.
"""

import pytest
import torch
import torch.distributed as dist

from _support import Stage, run_two_ranks


def _sync():
    dist.barrier()


# ---------------------------------------------------------------------------
# The wire
# ---------------------------------------------------------------------------


def _run_value(rank, world, rdv):
    from nnsight.modeling.vllm.pp_transport import VALUE

    stage = Stage(rank, world, rdv)
    value = (torch.arange(6, dtype=torch.int64).reshape(2, 3), None, {"h": torch.full((4,), 1.5, dtype=torch.bfloat16)})
    if rank == 0:
        stage.link.publish(VALUE, "r", 0, "model.a.output", value)
        stage.link.publish(VALUE, "r", 0, "model.a.output", (torch.zeros(1),))
        stage.link.publish(VALUE, "r", 1, "model.a.output", torch.ones(2))
    else:
        first = stage.link.take("r", 0, "model.a.output")
        assert torch.equal(first[0], value[0]) and first[1] is None and torch.equal(first[2]["h"], value[2]["h"])
        second = stage.link.take("r", 0, "model.a.output")
        assert torch.equal(second[0], torch.zeros(1))
        other = stage.link.take("r", 1, "model.a.output")
        assert torch.equal(other, torch.ones(2))
        assert not stage.link.has("r", 0, "model.a.output")
    stage.close()


def test_values_arrive_per_worker_in_order():
    run_two_ranks(_run_value)


def _run_request(rank, world, rdv):
    from nnsight.modeling.vllm.pp_transport import RemoteError

    def resolve(provider):
        if provider == "model.b.param.weight":
            return torch.full((3,), 7.0)
        if provider == "model.b.state":
            return {"weight": torch.ones(2), "bias": torch.zeros(2)}
        raise AttributeError(f"no {provider}")

    stage = Stage(rank, world, rdv, resolver=resolve if rank == 1 else None)
    if rank == 0:
        assert torch.equal(stage.link.request(1, "model.b.param.weight"), torch.full((3,), 7.0))
        state = stage.link.request(1, "model.b.state")
        assert torch.equal(state["weight"], torch.ones(2)) and torch.equal(state["bias"], torch.zeros(2))
        with pytest.raises(RemoteError, match="no model.b.param.nope"):
            stage.link.request(1, "model.b.param.nope")
    stage.close()


def test_a_parameter_request_is_answered_from_the_owner():
    run_two_ranks(_run_request)


def _run_failure(rank, world, rdv):
    from nnsight.modeling.vllm.pp_transport import RemoteError

    stage = Stage(rank, world, rdv)
    if rank == 0:
        stage.link.fail("r", 0, "model.a.output", "ran past it")
        stage.link.fail("s", 0, None, "the block failed")
    else:
        with pytest.raises(RemoteError, match="ran past it"):
            stage.link.take("r", 0, "model.a.output")
        with pytest.raises(RemoteError, match="the block failed"):
            stage.link.take("s", 0, "model.anything.output")
        with pytest.raises(RemoteError, match="nothing came"):
            stage.link.take("t", 0, "model.a.output", timeout=0.2)
    stage.close()


def test_a_failure_ends_the_wait_for_that_worker():
    run_two_ranks(_run_failure)


# ---------------------------------------------------------------------------
# The interleaver: one block, both ranks
# ---------------------------------------------------------------------------

OWNERS = {"a": 0, "b": 1}
A, B = "model.a.output", "model.b.output"


def _run_read_both(rank, world, rdv):
    """Rank 0 holds ``a``, rank 1 holds ``b``. The block reads ``a`` then ``b``:
    on rank 1 ``a`` is taken in place at start; on rank 0 ``b`` parks and is
    taken once rank 0 is past the round."""
    stage = Stage(rank, world, rdv, owners=OWNERS)
    block = "a = Mediator.value('model.a.output'); b = Mediator.value('model.b.output'); total = float(a.sum() + b.sum())"
    if rank == 0:
        mediator = stage.mediator(block)
        assert mediator.pending.provider == A
        stage.fire(A, torch.ones(3))
        # Served a, pushed it, and parked on b, which the later stage owns.
        assert mediator.pending.provider == B
        stage.interleaver.rounds["r"] = 1
        stage.interleaver.serve([mediator])
        assert not mediator.alive and mediator.lcls["total"] == 3 + 6
    else:
        mediator = stage.mediator(block)
        # Taken a in place at start; parked on its own b.
        assert mediator.pending.provider == B
        stage.fire(B, torch.full((3,), 2.0))
        assert not mediator.alive and mediator.lcls["total"] == 3 + 6
    stage.close()


def test_a_block_reading_both_stages_finishes_on_both():
    run_two_ranks(_run_read_both)


def _run_write_absorbed(rank, world, rdv):
    """A swap of the other stage's module is absorbed; the owner applies it."""
    stage = Stage(rank, world, rdv, owners=OWNERS)
    block = "Mediator.swap('model.a.output', torch.zeros(3)); b = Mediator.value('model.b.output'); got = float(b.sum())"
    if rank == 0:
        mediator = stage.mediator(block)
        assert mediator.pending.provider == A
        assert torch.equal(stage.fire(A, torch.ones(3)), torch.zeros(3))
        assert mediator.pending.provider == B
        stage.interleaver.rounds["r"] = 1
        stage.interleaver.serve([mediator])
        assert mediator.lcls["got"] == 5.0
    else:
        mediator = stage.mediator(block)
        # The swap was absorbed at start; the block is at b.
        assert mediator.pending.provider == B
        stage.fire(B, torch.full((3,), 5.0 / 3))
        assert mediator.lcls["got"] == pytest.approx(5.0)
    stage.close()


def test_a_write_to_the_other_stage_is_absorbed():
    run_two_ranks(_run_write_absorbed)


def _run_stale(rank, world, rdv):
    """Rank 0's block reads b (later stage) then a (its own, already passed):
    rank 0 stays parked on a, as one GPU would, and tells rank 1, whose block
    is waiting on a in place, that a will not come."""
    stage = Stage(rank, world, rdv, owners=OWNERS)
    block = "b = Mediator.value('model.b.output'); a = Mediator.value('model.a.output'); total = float(a.sum())"
    if rank == 0:
        mediator = stage.mediator(block)
        stage.fire(A, torch.ones(3))  # nobody parked here; the visit passes
        stage.interleaver.rounds["r"] = 1
        stage.interleaver.serve([mediator])
        assert mediator.alive and mediator.pending.provider == A
    else:
        mediator = stage.mediator(block)
        assert mediator.pending.provider == B
        stage.interleaver.defer_exceptions = True
        stage.fire(B, torch.ones(3))
        assert mediator.exception is not None and "ran past" in str(mediator.exception)
    stage.close()


def test_a_read_the_owner_ran_past_fails_the_waiting_peer():
    run_two_ranks(_run_stale)


def _run_rounds(rank, world, rdv):
    """A per-step read of the later stage's value across three rounds: each
    round's value is taken at the next round's start, the last at collect."""
    stage = Stage(rank, world, rdv, owners=OWNERS)
    # Each read pinned to its step, as ``tracer.iter`` pins it.
    block = (
        "m = Mediator.current('step'); vals = []\n"
        "for step in range(3):\n"
        "    m.iteration = step\n"
        "    vals.append(float(Mediator.value('model.b.output').sum()))\n"
    )
    if rank == 0:
        mediator = stage.mediator(block)
        for step in range(3):
            stage.interleaver.serve([mediator])
            assert mediator.alive and mediator.pending.provider == B
            stage.interleaver.rounds["r"] = step + 1
        stage.interleaver.serve([mediator], final=True)
        assert not mediator.alive and mediator.lcls["vals"] == [0.0, 3.0, 6.0]
    else:
        mediator = stage.mediator(block)
        for step in range(3):
            stage.fire(B, torch.full((3,), float(step)))
        assert not mediator.alive and mediator.lcls["vals"] == [0.0, 3.0, 6.0]
    stage.close()


def test_a_per_step_read_follows_the_rounds():
    run_two_ranks(_run_rounds)


class _Kept(list):
    """A list whose ``save`` is its own, so the block can say ``.save()`` on
    a rank with no trace open."""

    def save(self):
        return self


class _Reads:
    """``.output`` reads a location through the mediator, as an envoy does."""

    def __init__(self, provider):
        self.provider = provider

    @property
    def output(self):
        from nnsight.intervention.interleaver import Mediator

        return Mediator.value(self.provider)


def _run_save_only(rank, world, rdv):
    """The block appends ``a`` and ``b`` to a saved container and does nothing
    else with them. Neither rank waits for the other's value or sends its
    own: each binds a placeholder for the remote one and the real tensor for
    its own, in the same positions."""
    from nnsight.modeling.vllm.pp_deferred import Deferred

    stage = Stage(rank, world, rdv, owners=OWNERS)
    block = "kept = Kept().save()\nkept.append(model['a'].output[0])\nkept.append(model['b'].output[0])\n"
    stage_globals = {"Kept": _Kept, "model": {"a": _Reads(A), "b": _Reads(B)}}
    mediator = stage.mediator(block, start=False)
    mediator.glbls.update(stage_globals)
    mediator.lcls.update(stage_globals)
    mediator.start(stage.interleaver)
    stage.interleaver.started(mediator)
    stage.interleaver.mediators.append(mediator)
    stage.interleaver.reindex()
    if rank == 0:
        assert mediator.pending.provider == A
        stage.fire(A, (torch.ones(3),))
        # b, which the later stage owns, was answered with a placeholder at once.
        assert not mediator.alive
        kept = mediator.lcls["kept"]
        assert torch.equal(kept[0], torch.ones(3)) and isinstance(kept[1], Deferred) and kept[1].provider == B
    else:
        # a was answered with a placeholder at start; b is this rank's own.
        assert mediator.pending.provider == B
        stage.fire(B, (torch.full((3,), 2.0),))
        assert not mediator.alive
        kept = mediator.lcls["kept"]
        assert isinstance(kept[0], Deferred) and kept[0].provider == A and torch.equal(kept[1], torch.full((3,), 2.0))
    _sync()
    # Nothing crossed the wire for either read.
    assert not stage.link.has("r", 0, A) and not stage.link.has("r", 0, B)
    stage.close()


def test_a_save_only_read_binds_a_placeholder_and_sends_nothing():
    run_two_ranks(_run_save_only)
