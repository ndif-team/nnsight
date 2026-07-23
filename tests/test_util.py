from collections import OrderedDict, defaultdict, namedtuple

import pytest

from nnsight.util import apply


class Tensor:
    """A tiny stand-in leaf type so these tests don't depend on torch."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, Tensor) and other.value == self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"Tensor({self.value})"


def inc(t):
    return Tensor(t.value + 1)


class TestLeaf:
    def test_bare_instance(self):
        assert apply(Tensor(1), inc, Tensor) == Tensor(2)

    def test_non_matching_scalars_untouched(self):
        assert apply(5, inc, Tensor) == 5
        assert apply("hello", inc, Tensor) == "hello"
        assert apply(None, inc, Tensor) is None


class TestContainers:
    def test_list(self):
        assert apply([Tensor(1), 5], inc, Tensor) == [Tensor(2), 5]

    def test_dict(self):
        assert apply({"a": Tensor(1), "b": "x"}, inc, Tensor) == {"a": Tensor(2), "b": "x"}

    def test_nested(self):
        data = ((Tensor(1),), {"m": Tensor(2)})
        assert apply(data, inc, Tensor) == ((Tensor(2),), {"m": Tensor(3)})

    def test_set(self):
        assert apply({Tensor(1)}, inc, Tensor) == {Tensor(2)}

    def test_frozenset_type_preserved(self):
        out = apply(frozenset({Tensor(1)}), inc, Tensor)
        assert out == frozenset({Tensor(2)}) and isinstance(out, frozenset)

    def test_namedtuple_preserved(self):
        NT = namedtuple("NT", ["x", "y"])
        out = apply(NT(Tensor(1), 3), inc, Tensor)
        assert out == NT(Tensor(2), 3) and isinstance(out, NT)

    def test_ordereddict_type_preserved(self):
        out = apply(OrderedDict(a=Tensor(1)), inc, Tensor)
        assert isinstance(out, OrderedDict) and out["a"] == Tensor(2)

    def test_defaultdict_type_preserved(self):
        out = apply(defaultdict(int, {"a": Tensor(1)}), inc, Tensor)
        assert isinstance(out, defaultdict) and out["a"] == Tensor(2)

    def test_slice(self):
        out = apply(slice(Tensor(1), None, Tensor(2)), inc, Tensor)
        assert out == slice(Tensor(2), None, Tensor(3))


class TestNotInPlace:
    def test_list_original_untouched(self):
        original = [Tensor(1)]
        out = apply(original, inc, Tensor)
        assert original == [Tensor(1)] and out == [Tensor(2)] and out is not original

    def test_dict_original_untouched(self):
        original = {"a": Tensor(1)}
        out = apply(original, inc, Tensor)
        assert original == {"a": Tensor(1)} and out == {"a": Tensor(2)} and out is not original


class Holder:
    def __init__(self):
        self.t = Tensor(1)
        self.child = None


class TestObjectDescent:
    def test_default_n_zero_opaque(self):
        obj = Holder()
        out = apply(obj, inc, Tensor)
        assert out is obj and obj.t == Tensor(1)  # untouched

    def test_n_one_mutates_in_place(self):
        obj = Holder()
        out = apply(obj, inc, Tensor, n=1)
        assert out is obj and obj.t == Tensor(2)

    def test_n_one_does_not_reach_second_object(self):
        obj = Holder()
        obj.child = Holder()
        apply(obj, inc, Tensor, n=1)
        assert obj.t == Tensor(2)
        assert obj.child.t == Tensor(1)  # nested object left alone

    def test_n_two_reaches_second_object(self):
        obj = Holder()
        obj.child = Holder()
        apply(obj, inc, Tensor, n=2)
        assert obj.child.t == Tensor(2)

    def test_container_nesting_does_not_spend_budget(self):
        # An object reached through a container still counts as the first object.
        obj = Holder()
        apply([obj], inc, Tensor, n=1)
        assert obj.t == Tensor(2)
