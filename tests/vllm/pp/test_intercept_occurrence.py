"""Occurrence tags on cross-stage reads, and the round guard on in-place
upstream serves. Single process; the listener is a recording stub."""

import sys

from nnsight.intervention.interleaver import Event, Mediator
from nnsight.modeling.vllm.lazy_remote_tensor import encode_pull_location
from nnsight.modeling.vllm.pp import PPModuleMap
from nnsight.modeling.vllm.pp_interleaver import PPInterleaver

SERVED = object()


class RecordingListener:
    def __init__(self):
        self.pulls = []

    def begin_pull(self, source_rank, provider, req_id=None):
        self.pulls.append((source_rank, provider, req_id))

        class Pull:
            ready = True

            def complete(self_inner, timeout=None):
                return SERVED

        return Pull()


def build(local_rank):
    module_map = PPModuleMap(2)
    module_map.set_derived_owners({"h.0": 0, "h.1": 1})
    listener = RecordingListener()
    interleaver = PPInterleaver(module_map, listener, local_rank)
    mediator = Mediator(compile("pass", "<t>", "exec"), {}, {})
    interleaver.mediators.append(mediator)
    mediator.interleaver = interleaver
    return interleaver, mediator, listener


def provider_of(result):
    (lazy,) = result
    return lazy._meta["provider_string"]


def test_pinned_step_tags_every_read_of_the_body():
    interleaver, mediator, _ = build(local_rank=1)
    mediator.iteration = 2
    first = interleaver.intercept(mediator, Event.VALUE, "model.h.0.output", ())
    second = interleaver.intercept(mediator, Event.VALUE, "model.h.0.output", ())
    assert provider_of(first).endswith(".i2")
    assert provider_of(second).endswith(".i2")


def test_relaxed_read_tags_with_the_requests_round():
    interleaver, mediator, _ = build(local_rank=1)
    mediator.iteration = None
    mediator.pp_req_id = "req"
    interleaver.rounds["req"] = 5
    interleaver.step = 11
    result = interleaver.intercept(mediator, Event.VALUE, "model.h.0.output", ())
    assert provider_of(result).endswith(".i5")


def test_relaxed_read_outside_the_engine_counts_forwards():
    interleaver, mediator, _ = build(local_rank=1)
    mediator.iteration = None
    interleaver.step = 7
    result = interleaver.intercept(mediator, Event.VALUE, "model.h.0.output", ())
    assert provider_of(result).endswith(".i7")


def test_upstream_pull_of_an_opened_round_serves_in_place():
    interleaver, mediator, listener = build(local_rank=1)
    mediator.pp_req_id = "req"
    interleaver.rounds["req"] = 3
    location = encode_pull_location(0, "req", "model.h.0.output.i3")
    result = interleaver.intercept(mediator, Event.VALUE, location, ())
    assert result == (SERVED,)
    assert listener.pulls == [(0, "model.h.0.output.i3", "req")]


def test_upstream_pull_of_a_future_round_parks():
    interleaver, mediator, listener = build(local_rank=1)
    mediator.pp_req_id = "req"
    interleaver.rounds["req"] = 3
    location = encode_pull_location(0, "req", "model.h.0.output.i4")
    result = interleaver.intercept(mediator, Event.VALUE, location, ())
    assert result is None, "the park stands; the serve point resumes it"
    assert listener.pulls == [(0, "model.h.0.output.i4", "req")]
    assert (id(mediator), location) in interleaver._pulls


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
