"""Two gloo ranks over the real reply wire and interleaver, no vLLM engine.

Each test spawns two ranks that run ``_run_<name>(rank, world, rdv)``: the
reply wire for parameters, then the interleaver on a block both ranks run,
where one rank's local read is the other's remote one. A step's entries move
between the ranks with ``Stage.exchange``, the way the engine's own
transfers carry them: forward from the earlier rank, backward from the later.
"""

import pytest
import torch
import torch.distributed as dist

from _support import Stage, run_two_ranks


def _sync():
    dist.barrier()


# ---------------------------------------------------------------------------
# The reply wire
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The interleaver: one block, both ranks
# ---------------------------------------------------------------------------

OWNERS = {"a": 0, "b": 1}
A, B = "model.a.output", "model.b.output"


def _run_read_both(rank, world, rdv):
    """Rank 0 holds ``a``, rank 1 holds ``b``. The block reads ``a`` then ``b``:
    on rank 1 ``a`` is taken in place at start, from the step's payload; on
    rank 0 ``b`` parks and is taken once rank 1's payload for the step is back."""
    stage = Stage(rank, world, rdv, owners=OWNERS)
    block = "a = Mediator.value('model.a.output'); b = Mediator.value('model.b.output'); total = float(a.sum() + b.sum())"
    if rank == 0:
        mediator = stage.mediator(block)
        assert mediator.pending.provider == A
        stage.fire(A, torch.ones(3))
        # Served a, and parked on b, which the later stage owns.
        assert mediator.pending.provider == B
        stage.exchange(src=0)
        stage.interleaver.rounds["r"] = 1
        stage.exchange(src=1, reqs=("r",))
        stage.interleaver.serve([mediator])
        assert not mediator.alive and mediator.lcls["total"] == 3 + 6
    else:
        stage.exchange(src=0)
        mediator = stage.mediator(block)
        # Taken a in place at start; parked on its own b.
        assert mediator.pending.provider == B
        stage.fire(B, torch.full((3,), 2.0))
        assert not mediator.alive and mediator.lcls["total"] == 3 + 6
        stage.exchange(src=1, reqs=("r",))
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
        stage.exchange(src=0)
        stage.interleaver.rounds["r"] = 1
        stage.exchange(src=1, reqs=("r",))
        stage.interleaver.serve([mediator])
        assert mediator.lcls["got"] == 5.0
    else:
        stage.exchange(src=0)
        mediator = stage.mediator(block)
        # The swap was absorbed at start; the block is at b.
        assert mediator.pending.provider == B
        stage.fire(B, torch.full((3,), 5.0 / 3))
        assert mediator.lcls["got"] == pytest.approx(5.0)
        stage.exchange(src=1, reqs=("r",))
    stage.close()


def test_a_write_to_the_other_stage_is_absorbed():
    run_two_ranks(_run_write_absorbed)


def _run_not_served(rank, world, rdv):
    """Rank 0's block reads b (later stage) then a (its own, already passed):
    rank 0 stays parked on a, as one GPU would. Rank 1's block waits on a in
    place; the step's payload from rank 0 has no a for it, so it fails there
    and then, with no waiting."""
    stage = Stage(rank, world, rdv, owners=OWNERS)
    block = "b = Mediator.value('model.b.output'); a = Mediator.value('model.a.output'); total = float(a.sum())"
    if rank == 0:
        mediator = stage.mediator(block)
        stage.fire(A, torch.ones(3))  # nobody parked here; the visit passes
        stage.exchange(src=0)
        stage.interleaver.rounds["r"] = 1
        stage.exchange(src=1, reqs=("r",))
        stage.interleaver.serve([mediator])
        assert mediator.alive and mediator.pending.provider == A
    else:
        stage.exchange(src=0)
        stage.interleaver.defer_exceptions = True
        mediator = stage.mediator(block)
        assert mediator.pending.provider == B
        stage.fire(B, torch.ones(3))
        assert mediator.exception is not None and "served nothing for it" in str(mediator.exception)
        stage.exchange(src=1, reqs=("r",))
    stage.close()


def test_a_read_the_owner_ran_past_fails_the_waiting_peer_at_once():
    run_two_ranks(_run_not_served)


def _run_failed_block(rank, world, rdv):
    """Rank 0's copy of the block fails before serving anything. Rank 1's
    copy, waiting on a in place, is told so by the step's payload."""
    stage = Stage(rank, world, rdv, owners=OWNERS)
    block = "a = Mediator.value('model.a.output'); total = float(a.sum())"
    if rank == 0:
        stage.interleaver.defer_exceptions = True
        mediator = stage.mediator(block)
        mediator.exception = RuntimeError("boom")
        stage.interleaver.failed(mediator, "the block failed on stage 0: RuntimeError: boom")
        stage.exchange(src=0)
    else:
        stage.exchange(src=0)
        stage.interleaver.defer_exceptions = True
        mediator = stage.mediator(block)
        assert mediator.exception is not None and "failed on stage 0" in str(mediator.exception)
    stage.close()


def test_a_failed_copy_fails_the_peers_copy_waiting_on_it():
    run_two_ranks(_run_failed_block)


def _run_rounds(rank, world, rdv):
    """A per-step read of the later stage's value across three rounds: each
    round's value is taken at the next round's start, once that round's
    payload is back. The last round's is not taken on the earlier stage,
    which has no step left to take it at; the later stage's copy of the
    block, which has every value, completes and is the one collected from."""
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
            stage.exchange(src=1, reqs=("r",))
        # No fourth step start comes: the request is over.
        assert mediator.alive and mediator.pending.provider == B and mediator.lcls["vals"] == [0.0, 3.0]
        _sync()
    else:
        mediator = stage.mediator(block)
        for step in range(3):
            stage.fire(B, torch.full((3,), float(step)))
            stage.exchange(src=1, reqs=("r",))
        assert not mediator.alive and mediator.lcls["vals"] == [0.0, 3.0, 6.0]
        _sync()
    stage.close()


def test_a_per_step_read_follows_the_rounds():
    run_two_ranks(_run_rounds)


def _run_inplace(rank, world, rdv):
    """The owner's block changes the value in place right after reading it
    (as vLLM's fused norm does to its inputs). The peer must get the value as
    it was handed, before that change."""
    stage = Stage(rank, world, rdv, owners=OWNERS)
    block = (
        "a = Mediator.value('model.a.output')\n"
        "a.add_(100.0)\n"
        "b = Mediator.value('model.b.output')\n"
        "total = float(a.sum() + b.sum())\n"
    )
    if rank == 0:
        mediator = stage.mediator(block)
        stage.fire(A, torch.ones(3))
        assert mediator.pending.provider == B
        stage.exchange(src=0)
        _sync()
    else:
        stage.exchange(src=0)
        mediator = stage.mediator(block)
        # a arrived as it was handed on stage 0: ones, then this block's own add_.
        assert mediator.pending.provider == B
        assert torch.equal(mediator.lcls["a"], torch.full((3,), 101.0))
        stage.fire(B, torch.zeros(3))
        assert not mediator.alive and mediator.lcls["total"] == 303.0
        _sync()
    stage.close()


def test_a_value_travels_as_handed_before_the_block_changes_it():
    run_two_ranks(_run_inplace)


def _run_finished_forgets(rank, world, rdv):
    """A value the earlier stage's copy never takes, a later-stage value of
    the request's last round, leaves with the request."""
    stage = Stage(rank, world, rdv, owners=OWNERS)
    block = "b = Mediator.value('model.b.output'); total = float(b.sum())"
    if rank == 0:
        mediator = stage.mediator(block)
        stage.interleaver.rounds["r"] = 1
        stage.exchange(src=1, reqs=("r",))
        assert ("r", 0) in stage.interleaver.inbox and stage.interleaver.arrived == {"r": 1}
        stage.interleaver.finished("r")
        assert not stage.interleaver.inbox and not stage.interleaver.arrived and not stage.interleaver.rounds
        assert mediator.alive  # reported at the request's end, quietly, as the runner does
    else:
        mediator = stage.mediator(block)
        stage.fire(B, torch.ones(3))
        assert not mediator.alive
        stage.exchange(src=1, reqs=("r",))
    stage.close()


def test_what_a_finished_request_left_is_forgotten_with_it():
    run_two_ranks(_run_finished_forgets)
