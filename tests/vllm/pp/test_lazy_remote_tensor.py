"""LazyRemoteTensor consumption semantics, single process, no wire.

The contract: creating, indexing, writing to, and saving a proxy pull
nothing; any real read (arithmetic, torch functions, comparisons, iteration,
attribute or metadata access) materializes it, exactly once, through
``Mediator.value`` on the encoded pull location. Tests either pre-set
``_real`` (consumption semantics) or monkeypatch ``Mediator.value``
(pull counting and location encoding).
"""

import pytest
import torch

from nnsight.intervention.interleaver import Mediator
from nnsight.modeling.vllm.lazy_remote_tensor import (
    LazyRemoteTensor,
    decode_pull_location,
    encode_pull_location,
)


def make_lazy(real=None, req_id=None):
    lazy = LazyRemoteTensor(
        source_rank=1,
        provider_string="model.h.5.output.i0",
        dtype=torch.float32,
        req_id=req_id,
    )
    if real is not None:
        lazy._real = real
    return lazy


@pytest.fixture
def pull_counter(monkeypatch):
    """Replace Mediator.value with a counting stub returning a fixed tensor."""
    calls = []

    def fake_value(location):
        calls.append(location)
        return torch.arange(6, dtype=torch.float32).reshape(2, 3)

    monkeypatch.setattr(Mediator, "value", staticmethod(fake_value))
    return calls


class TestNoTraffic:
    def test_creation_and_dtype_hint_pull_nothing(self):
        lazy = make_lazy()
        assert lazy.dtype == torch.float32
        assert lazy._real is None

    def test_setitem_is_absorbed(self):
        # The owning rank runs the same write line locally.
        lazy = make_lazy()
        lazy[:] = torch.zeros(2, 3)
        assert lazy._real is None

    def test_getitem_returns_deferred_child(self):
        lazy = make_lazy()
        child = lazy[0]
        assert isinstance(child, LazyRemoteTensor)
        assert child._parent is lazy
        assert lazy._real is None and child._real is None

    def test_save_is_not_defined_on_the_class(self):
        # The mounted object.save must win attribute lookup so a saved proxy
        # is marked by id and ships as the NOT_ON_THIS_RANK sentinel; a save
        # defined here would shadow it (and materializing would defeat the
        # point of the sentinel).
        assert "save" not in LazyRemoteTensor.__dict__

    def test_leaked_proxy_raises_outside_a_worker(self):
        lazy = make_lazy()
        with pytest.raises(ValueError, match="outside of interleaving"):
            lazy + 1

    def test_torch_function_accepts_a_child_of_a_forced_proxy(self):
        # Indexing into a cached parent never parks, so the guard must let a
        # resolved chain through.
        parent = make_lazy(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        assert torch.sum(parent[1]).item() == 12.0


class TestConsumptionMaterializes:
    def test_arithmetic_operators(self):
        real = torch.arange(3, dtype=torch.float32)
        lazy = make_lazy(real)
        assert torch.equal(lazy + 1, real + 1)
        assert torch.equal(1 + lazy, 1 + real)
        assert torch.equal(lazy - 1, real - 1)
        assert torch.equal(2 - lazy, 2 - real)
        assert torch.equal(lazy * 3, real * 3)
        assert torch.equal(lazy / 2, real / 2)
        assert torch.equal(-lazy, -real)

    def test_matmul(self):
        real = torch.eye(2)
        lazy = make_lazy(real)
        other = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.equal(lazy @ other, other)
        assert torch.equal(other @ lazy, other)

    def test_torch_function(self):
        real = torch.arange(4, dtype=torch.float32)
        lazy = make_lazy(real)
        assert torch.sum(lazy).item() == real.sum().item()

    def test_torch_function_with_lazy_among_args(self):
        real = torch.ones(2)
        lazy = make_lazy(real)
        assert torch.equal(torch.add(torch.zeros(2), lazy), real)

    def test_method_access(self):
        real = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        lazy = make_lazy(real)
        assert lazy.mean().item() == real.mean().item()
        assert lazy.view(-1).shape == (6,)

    def test_metadata_properties_reflect_the_real_value(self):
        real = torch.zeros(2, 3, dtype=torch.bfloat16)
        lazy = make_lazy(real)
        assert lazy.shape == (2, 3)
        # The hint said float32; the materialized value knows better.
        assert lazy.dtype == torch.bfloat16
        assert lazy.device == real.device

    def test_child_indexes_into_parent(self):
        real = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        parent = make_lazy(real)
        child = parent[1]
        assert torch.equal(child + 0, real[1])

    def test_tuple_real_names_the_indexing_fix(self):
        # Multi-output modules materialize to a (hidden, residual) tuple; a
        # tensor method on the tuple must say to index first.
        lazy = make_lazy((torch.zeros(2), torch.ones(2)))
        with pytest.raises(AttributeError, match="index it first"):
            lazy.mean()


class TestComparisons:
    # Without explicit comparison methods, == and != fall back to identity
    # (a plain bool) and the orderings raise, on the non-owning rank only,
    # so code branching on a comparison silently diverges between ranks.

    def test_comparisons_are_elementwise(self):
        real = torch.tensor([1.0, 2.0, 3.0])
        lazy = make_lazy(real)
        eq = lazy == torch.tensor([1.0, 0.0, 3.0])
        assert isinstance(eq, torch.Tensor) and eq.tolist() == [True, False, True]
        ne = lazy != torch.tensor([1.0, 0.0, 3.0])
        assert ne.tolist() == [False, True, False]
        assert (lazy < 2.5).tolist() == [True, True, False]
        assert (lazy <= 2.0).tolist() == [True, True, False]
        assert (lazy > 2.5).tolist() == [False, False, True]
        assert (lazy >= 2.0).tolist() == [False, True, True]

    def test_still_hashable(self):
        # Defining __eq__ clears the default hash; the proxy is tracked by
        # identity (the save set stores ids), so it must stay hashable.
        lazy = make_lazy()
        assert lazy in {lazy}


class TestIteration:
    # Without __iter__, Python iterates via __getitem__, which returns a
    # fresh lazy for every index and never raises IndexError, so tuple(lazy)
    # and unpacking spin forever on the non-owning rank.

    def test_iteration_over_tuple_terminates_and_yields_elements(self):
        hidden, residual = torch.zeros(2), torch.ones(2)
        lazy = make_lazy((hidden, residual))
        elements = list(lazy)
        assert len(elements) == 2
        assert torch.equal(elements[0], hidden)
        assert torch.equal(elements[1], residual)

    def test_unpacking(self):
        lazy = make_lazy((torch.zeros(2), torch.ones(2)))
        first, second = lazy
        assert torch.equal(second, torch.ones(2))

    def test_len(self):
        assert len(make_lazy((torch.zeros(2), torch.ones(2)))) == 2

    def test_iteration_over_tensor_matches_rows(self):
        real = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        lazy = make_lazy(real)
        rows = list(lazy)
        assert len(rows) == 3
        assert all(torch.equal(row, real[i]) for i, row in enumerate(rows))


class TestPullDiscipline:
    def test_repeated_consumption_pulls_once(self, pull_counter):
        lazy = make_lazy()
        total = (lazy + 0).sum().item()
        again = torch.sum(lazy).item()
        assert total == again == 15.0
        assert len(pull_counter) == 1

    def test_children_share_the_parent_pull(self, pull_counter):
        parent = make_lazy()
        row = parent[0] + 0
        other_row = parent[1] + 0
        assert torch.equal(row, torch.tensor([0.0, 1.0, 2.0]))
        assert torch.equal(other_row, torch.tensor([3.0, 4.0, 5.0]))
        assert len(pull_counter) == 1

    def test_pull_location_round_trips(self, pull_counter):
        lazy = make_lazy(req_id="req-7")
        lazy + 0
        assert decode_pull_location(pull_counter[0]) == (
            1, "req-7", "model.h.5.output.i0"
        )


class TestPullLocationCodec:
    def test_round_trip_without_req_id(self):
        location = encode_pull_location(2, None, "model.h.3.output.i4")
        assert decode_pull_location(location) == (2, None, "model.h.3.output.i4")

    def test_parks_own_occurrence_tag_is_stripped(self):
        # Mediator.event appends ".i{n}" to every parked location; the
        # provider inside the encoding already ends with its real tag, so
        # exactly one extra tag component must come off.
        location = encode_pull_location(0, "req-1", "model.h.3.output.i4")
        assert decode_pull_location(location + ".i0") == (
            0, "req-1", "model.h.3.output.i4"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
