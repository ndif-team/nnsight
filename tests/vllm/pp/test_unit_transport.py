"""The wire codec, in one process."""

import pickle
from collections import namedtuple

import pytest
import torch

from nnsight.modeling.vllm.pp_transport import Kept, decode, encode, to_host

Pair = namedtuple("Pair", "hidden residual")


def roundtrip(envelope):
    header, blob = encode(envelope)
    assert header.dtype == torch.int64 and blob.dtype == torch.uint8
    return decode(blob, int(header[0]))


def test_tensors_of_several_dtypes_ride_beside_plain_leaves():
    value = (
        torch.arange(6, dtype=torch.int64).reshape(2, 3),
        None,
        {"h": torch.full((5,), 1.5, dtype=torch.bfloat16), "n": 3, "s": "text"},
        [torch.zeros(0), torch.tensor(True)],
    )
    back = roundtrip({"kind": "value", "value": value})["value"]
    assert torch.equal(back[0], value[0]) and back[1] is None
    assert back[2]["h"].dtype == torch.bfloat16 and torch.equal(back[2]["h"], value[2]["h"])
    assert back[2]["n"] == 3 and back[2]["s"] == "text"
    assert back[3][0].shape == (0,) and back[3][1].item() is True


def test_a_namedtuple_keeps_its_type_and_fields():
    value = Pair(torch.ones(2, 2), torch.full((2, 2), 2.0))
    back = roundtrip({"value": value})["value"]
    assert isinstance(back, Pair) and torch.equal(back.residual, value.residual)


def test_a_non_contiguous_tensor_arrives_whole():
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    back = roundtrip({"value": value})["value"]
    assert back.shape == (4, 3) and torch.equal(back, value)


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
