"""What a step's entries look like on vLLM's transfers, and the reply wire's codec, in one process."""

import pickle
from collections import namedtuple

import pytest
import torch

from nnsight.modeling.vllm.pp_transport import (
    ERROR,
    META,
    VALUE,
    Kept,
    Link,
    decode,
    encode,
    pack,
    to_host,
    unpack,
)

Pair = namedtuple("Pair", "hidden residual")


def test_entries_ride_a_tensor_dict_and_leave_it_as_they_found_it():
    """Every tensor becomes a key of its own, since the transfer sends tensors
    directly and pickles the rest; unpacking removes all of them, so the
    dictionary is again what vLLM expects."""
    activations = {"hidden_states": torch.zeros(4, 8), "residual": torch.ones(4, 8)}
    served = Pair(torch.arange(6, dtype=torch.int64).reshape(2, 3), torch.full((3,), 1.5, dtype=torch.bfloat16))
    entries = [
        (VALUE, 0, "req-a", 0, "model.decoder_blocks.3.output", served),
        (VALUE, 0, "req-a", 1, "model.decoder_blocks.3.output", [torch.zeros(0), None, {"n": 2}]),
        (ERROR, 0, "req-b", 0, None, "the block failed on stage 0"),
    ]
    payload = pack(entries)
    assert META in payload and sum(1 for key in payload if key.startswith("nnsight.t")) == 3
    assert all(isinstance(payload[key], torch.Tensor) for key in payload if key != META)

    activations.update(payload)
    back = unpack(activations)
    assert set(activations) == {"hidden_states", "residual"}
    assert len(back) == 3
    kind, stage, req, ordinal, provider, value = back[0]
    assert (kind, stage, req, ordinal, provider) == (VALUE, 0, "req-a", 0, "model.decoder_blocks.3.output")
    assert isinstance(value, Pair) and torch.equal(value.hidden, served.hidden)
    assert value.residual.dtype == torch.bfloat16 and torch.equal(value.residual, served.residual)
    assert back[1][5][0].shape == (0,) and back[1][5][1] is None and back[1][5][2] == {"n": 2}
    assert back[2] == entries[2]


def test_a_dict_without_entries_unpacks_to_none_of_them():
    activations = {"hidden_states": torch.zeros(2)}
    assert unpack(activations) == [] and set(activations) == {"hidden_states"}


def test_a_non_contiguous_tensor_arrives_whole_over_the_reply_wire():
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    header, blob = encode({"value": value})
    back = decode(blob, int(header[0]))["value"]
    assert back.shape == (4, 3) and torch.equal(back, value)


def test_a_decoded_tensor_owns_its_storage():
    """A reply's tensors are copied out of the message, so a value kept for a
    request's life does not keep the whole message with it."""
    header, blob = encode({"small": torch.ones(1), "large": torch.zeros(4096)})
    envelope = decode(blob, int(header[0]))
    small = envelope["small"]
    assert small.untyped_storage().nbytes() == small.numel() * small.element_size()
    assert small.untyped_storage().data_ptr() != blob.untyped_storage().data_ptr()


def test_an_empty_envelope_has_no_data_region():
    header, blob = encode({"kind": "stop"})
    assert int(header[1]) == 0
    assert decode(blob, int(header[0])) == {"kind": "stop"}


def test_to_host_copies_every_tensor_and_keeps_the_structure():
    value = {"a": (torch.ones(2), 1), "b": [torch.zeros(1)]}
    host = to_host(value)
    assert host["a"][0] is not value["a"][0] and torch.equal(host["a"][0], value["a"][0])
    assert host["a"][1] == 1 and host["b"][0].device.type == "cpu"


def test_a_leaf_that_cannot_be_pickled_fails_before_anything_is_sent():
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        encode({"value": lambda: None})


def test_a_reply_nobody_waits_for_is_dropped():
    """A waiter that timed out is gone from the waiting set, so its reply,
    arriving later with a whole module state, is not kept."""
    link = Link(None, 0, 1)  # one rank: no peers, no threads
    link._waiting.add(7)
    link._file_reply(7, True, {"weight": torch.ones(2)})
    link._file_reply(8, True, {"weight": torch.zeros(4096)})
    assert set(link._replies) == {7}


def test_kept_values_live_with_their_requests_unless_pinned():
    kept = Kept()
    kept.put("model.lm_head.param.weight", "head", None)     # at load, no request: pinned
    kept.put("model.norm.state", "norm", "r1")
    assert kept.get("model.norm.state", "r2") == "norm"       # r2 now uses it too
    kept.release("r1")
    assert "model.norm.state" in kept                          # r2 still runs
    kept.release("r2")
    assert "model.norm.state" not in kept
    kept.release("r3")                                         # a request that kept nothing
    assert kept.get("model.lm_head.param.weight", "r4") == "head"
    kept.release("r4")
    assert "model.lm_head.param.weight" in kept                # pinned: outlives every request


def _stage_link(world):
    """What a stage's interleaver reads of its link here: the pipeline's size.
    A real Link to two peers would start their threads."""
    from types import SimpleNamespace

    return SimpleNamespace(world=world, peers=list(range(world - 1)))


def test_the_last_stage_sends_back_only_what_the_earlier_stages_lack():
    """The last of three stages got stage 0's and stage 1's entries forward.
    Stage 1's are needed back on stage 0; stage 0's are needed by no one
    before it, so they are not sent back to the stage that made them."""
    from nnsight.modeling.vllm.pp_interleaver import ROUND, PPInterleaver

    last = PPInterleaver(None, _stage_link(3), 2, torch.device("cpu"))
    first = (VALUE, 0, "req-a", 0, "model.decoder_blocks.1.output", torch.ones(2))
    middle = (VALUE, 1, "req-a", 0, "model.decoder_blocks.5.output", torch.zeros(2))
    last.receive([first, middle], forward=True)
    last.outbox.append((VALUE, 2, "req-a", 0, "model.output_projection.output", torch.full((2,), 3.0)))

    sent = last.flush_backward(["req-a"])
    assert [(kind, stage) for kind, stage, *_ in sent] == [(VALUE, 2), (VALUE, 1), (ROUND, 2)]
    assert last.flush_backward([]) == []


def test_a_relayed_value_is_sent_on_as_it_arrived():
    """The middle of three stages hands stage 0's value to its copy of the
    block, which changes it in place; stage 2 must still get it as stage 0
    served it."""
    from nnsight.modeling.vllm.pp_interleaver import PPInterleaver

    middle = PPInterleaver(None, _stage_link(3), 1, torch.device("cpu"))
    middle.receive([(VALUE, 0, "req-a", 0, "model.decoder_blocks.1.output", torch.ones(3))], forward=True)
    handed = middle.inbox[("req-a", 0)]["model.decoder_blocks.1.output"][0]
    handed.add_(100)

    (sent,) = middle.flush_forward()
    assert torch.equal(sent[5], torch.ones(3))


def test_the_scheduler_is_chosen_from_the_finished_config(monkeypatch):
    """vLLM settles async scheduling after the engine's arguments are read
    (it may pick Ray by itself), so the choice is made when the engine core
    builds the scheduler, from the config it hands over."""
    from types import SimpleNamespace

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    from nnsight.modeling.vllm.pp_scheduler import NNsightScheduler, pipeline_scheduler

    built = []
    for async_scheduling, expected in ((True, AsyncScheduler), (False, NNsightScheduler)):
        # vLLM's construction needs a real engine; record the call instead.
        monkeypatch.setattr(expected, "__init__", lambda self, *args, **kwargs: built.append((type(self), kwargs)))
        config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=async_scheduling))
        scheduler = pipeline_scheduler(vllm_config=config, block_size=16)
        assert type(scheduler) is expected
        assert built[-1] == (expected, {"vllm_config": config, "block_size": 16})


def test_every_pipeline_engine_gets_the_choosing_scheduler():
    from nnsight.modeling.vllm.vllm import VLLM

    def chosen(**kwargs):
        return VLLM._pipeline_kwargs(VLLM, kwargs).get("scheduler_cls")

    for kwargs in ({}, {"distributed_executor_backend": "ray"}, {"async_scheduling": False}):
        assert chosen(pipeline_parallel_size=2, **kwargs) == VLLM._SCHEDULER_CLS
    assert chosen() is None
    assert chosen(pipeline_parallel_size=2, scheduler_cls="my.Scheduler") == "my.Scheduler"


def test_the_scheduler_passes_on_what_the_engine_core_gives_schedule(monkeypatch):
    """vLLM 0.27 and later call schedule(throttle_prefills); 0.19 calls schedule()."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from nnsight.modeling.vllm.pp_scheduler import NNsightScheduler

    seen = []
    monkeypatch.setattr(Scheduler, "schedule", lambda self, *args, **kwargs: seen.append((args, kwargs)) or object())
    scheduler = NNsightScheduler.__new__(NNsightScheduler)
    scheduler._nnsight_pp = False
    scheduler.schedule()
    scheduler.schedule(False)
    assert seen == [((), {}), ((False,), {})]
