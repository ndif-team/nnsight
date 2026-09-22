"""The request table's two exits: what a finished request's registered block
saved, and what the engine let go of when the request left."""

import threading

import pytest
import torch

pytest.importorskip("vllm")

from nnsight.intervention.interleaver import Mediator
from nnsight.modeling.vllm.model_runners.GPUModelRunner import Request, Requests
from nnsight.tracing.tracer import mark


def _registered(requests, request_id, registration_id, lcls):
    """A request nobody traced, carrying one copy of a registered block."""
    requests.templates[registration_id] = object()
    request = Request(request_id)
    copy = Mediator(compile("pass", "<registered>", "exec"), {}, lcls)
    copy.batch_group = [0, 1]  # scheduled this step, so the step records it
    request.copies[registration_id] = copy
    requests.requests[request.id] = request
    return request


def test_saves_recorded_on_the_forward_thread_survive_a_collect_on_another():
    """Under Ray the collect arrives on another thread than the forward, and that
    thread's saved-id set is empty; the registered block's saved name still
    comes home."""
    requests = Requests()
    resid_probe = torch.arange(6.0)
    request = _registered(requests, "tenant-7-0000beef", "norm-watch", {"resid_probe": mark(resid_probe)})
    requests.record_saves()  # end of the step, on the forward thread

    collect = threading.Thread(target=requests.harvest, args=({request.id},))
    collect.start()
    collect.join()

    assert request.harvested["norm-watch"]["saves"]["resid_probe"] is resid_probe


def test_a_request_that_leaves_through_harvest_is_released():
    """An untraced request whose registered block saved nothing leaves when the
    scheduler reports it finished, before any collect; whatever else the engine
    keeps under its id is let go then too."""
    released = []
    requests = Requests(release=released.append)
    request = _registered(requests, "chat-00c0ffee", "quiet-edit", {})
    requests.record_saves()

    requests.harvest({request.id})

    assert request.id not in requests.requests
    assert released == [request.id]
