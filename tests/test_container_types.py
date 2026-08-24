"""Tests for tensor traversal through container *subclasses*.

``nnsight.util.apply`` / ``applyn`` walk a module's inputs and outputs to find
tensors -- narrowing them to an invoke's batch slice, splicing a modified slice
back in, moving them to a device, casting a dtype, detaching them for a cache.

The builtin containers (``list``, ``tuple``, ``dict``) were the only ones
recursed into, which silently excluded every *subclass* of them. That mattered
because a HuggingFace ``ModelOutput`` is an ``OrderedDict`` subclass, and it is
what every top-level ``transformers`` module returns -- so ``model.output`` and
``model.transformer.output`` were skipped by all of the above, with no error.

These tests pin the traversal itself (``TestApply`` / ``TestApplyN``) and the
five user-visible behaviours that depend on it (``TestBatchedModelOutput``,
``TestCacheModelOutput``).
"""

import collections

import pytest
import torch

import nnsight
from nnsight.util import apply, applyn

# =============================================================================
# Helpers
# =============================================================================


def _mark(tensor: torch.Tensor) -> torch.Tensor:
    """A stand-in transformation: every visited tensor becomes all-ones."""
    return torch.ones_like(tensor)


def _all_marked(data) -> bool:
    """True when at least one tensor was found and every one of them is marked."""

    found = []

    def walk(value):
        if isinstance(value, torch.Tensor):
            found.append(bool((value == 1).all()))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(data)

    return len(found) > 0 and all(found)


def _rel_err(got: torch.Tensor, expected: torch.Tensor) -> float:
    """Scale-invariant error, comparing only the trailing (unpadded) positions.

    Batching an invoke alongside a longer prompt left-pads it, so the batched
    value carries extra leading positions. The real tokens are right-aligned.
    """

    length = min(got.shape[1], expected.shape[1])
    got, expected = got[:, -length:], expected[:, -length:]

    return ((got - expected).norm() / expected.norm()).item()


Point = collections.namedtuple("Point", ["x", "y"])


class ListSubclass(list):
    pass


class DictSubclass(dict):
    pass


def _model_output(**fields):
    from transformers.modeling_outputs import BaseModelOutputWithPast

    return BaseModelOutputWithPast(**fields)


# =============================================================================
# apply / applyn
# =============================================================================


class TestApply:
    """``apply`` must reach tensors inside container subclasses."""

    @pytest.mark.parametrize(
        "container",
        [
            pytest.param([torch.zeros(2)], id="list"),
            pytest.param((torch.zeros(2),), id="tuple"),
            pytest.param({"a": torch.zeros(2)}, id="dict"),
            pytest.param(
                collections.OrderedDict(a=torch.zeros(2)), id="OrderedDict"
            ),
            pytest.param(
                collections.defaultdict(list, a=torch.zeros(2)), id="defaultdict"
            ),
            pytest.param(Point(torch.zeros(2), torch.zeros(2)), id="namedtuple"),
            pytest.param(ListSubclass([torch.zeros(2)]), id="list_subclass"),
            pytest.param(DictSubclass(a=torch.zeros(2)), id="dict_subclass"),
            pytest.param([{"a": (torch.zeros(2),)}], id="nested"),
        ],
    )
    def test_reaches_tensors(self, container):
        assert _all_marked(apply(container, _mark, torch.Tensor))

    @pytest.mark.parametrize(
        "container",
        [
            pytest.param(collections.OrderedDict(a=torch.zeros(2)), id="OrderedDict"),
            pytest.param(Point(torch.zeros(2), torch.zeros(2)), id="namedtuple"),
            pytest.param(ListSubclass([torch.zeros(2)]), id="list_subclass"),
            pytest.param(DictSubclass(a=torch.zeros(2)), id="dict_subclass"),
        ],
    )
    def test_preserves_type(self, container):
        assert type(apply(container, _mark, torch.Tensor)) is type(container)

    def test_does_not_mutate_source_when_not_inplace(self):
        container = collections.OrderedDict(a=torch.zeros(2))

        apply(container, _mark, torch.Tensor)

        assert bool((container["a"] == 0).all())

    def test_inplace_mutates_source(self):
        container = collections.OrderedDict(a=torch.zeros(2))

        result = apply(container, _mark, torch.Tensor, inplace=True)

        assert result is container
        assert _all_marked(container)

    def test_torch_size_is_rebuilt_not_downgraded(self):
        # torch.Size is a tuple subclass holding no tensors: recursing into it
        # must be a no-op that still hands back a torch.Size.
        size = torch.Size([2, 3])

        result = apply(size, _mark, torch.Tensor)

        assert result == size
        assert isinstance(result, torch.Size)

    def test_torch_return_types_is_rebuilt(self):
        # torch.return_types.* are structsequences: a tuple subclass whose
        # constructor takes a single iterable rather than one arg per field.
        returned = torch.max(torch.zeros(3, 3), dim=0)

        result = apply(returned, _mark, torch.Tensor)

        assert type(result) is type(returned)
        assert _all_marked(result)

    def test_tuple_subclass_with_incompatible_constructor(self):
        class Fixed(tuple):
            def __new__(cls, a, b):
                return super().__new__(cls, (a, b))

        result = apply(Fixed(torch.zeros(2), torch.zeros(2)), _mark, torch.Tensor)

        # The concrete type cannot be rebuilt from an iterable, so a plain tuple
        # is the documented fallback -- but the tensors are still transformed.
        assert _all_marked(result)

    def test_non_container_objects_are_left_alone(self):
        class Opaque:
            def __init__(self):
                self.tensor = torch.zeros(2)

        opaque = Opaque()

        assert apply(opaque, _mark, torch.Tensor) is opaque
        assert bool((opaque.tensor == 0).all())

    def test_strings_are_not_traversed(self):
        assert apply("hello", _mark, torch.Tensor) == "hello"


class TestApplyN:
    """``applyn`` zips several containers of the same shape together."""

    def test_ordered_dict(self):
        a = collections.OrderedDict(x=torch.zeros(2))
        b = collections.OrderedDict(x=torch.ones(2))

        result = applyn([a, b], lambda p, q: p + q, torch.Tensor)

        assert type(result) is collections.OrderedDict
        assert _all_marked(result)

    def test_namedtuple(self):
        a = Point(torch.zeros(2), torch.zeros(2))
        b = Point(torch.ones(2), torch.ones(2))

        result = applyn([a, b], lambda p, q: p + q, torch.Tensor)

        assert type(result) is Point
        assert _all_marked(result)

    def test_list_subclass(self):
        a = ListSubclass([torch.zeros(2)])
        b = ListSubclass([torch.ones(2)])

        result = applyn([a, b], lambda p, q: p + q, torch.Tensor)

        assert type(result) is ListSubclass
        assert _all_marked(result)

    def test_key_missing_from_a_sibling_keeps_the_original(self):
        # `ModelOutput.keys()` only reports its non-None fields, so a
        # replacement value need not carry every key the original has.
        a = collections.OrderedDict(x=torch.zeros(2), y=torch.zeros(2))
        b = collections.OrderedDict(x=torch.ones(2))

        result = applyn([a, b], lambda p, q: p + q, torch.Tensor)

        assert bool((result["x"] == 1).all())
        assert bool((result["y"] == 0).all())


# =============================================================================
# ModelOutput through the batching path
# =============================================================================


@pytest.mark.usefixtures("gpt2")
class TestBatchedModelOutput:
    """A ``ModelOutput`` must be narrowed and spliced like a tuple output.

    ``model.output`` and ``model.transformer.output`` are ``ModelOutput``
    instances. Before container subclasses were traversed, an invoke read the
    *whole* batch through them, and a write through them landed on every other
    invoke's rows.
    """

    PROMPT_A = "The Eiffel Tower is in the city of"
    PROMPT_B = "Madison Square Garden is located in the city of"

    @torch.no_grad()
    def test_read_is_narrowed_to_the_invoke(self, gpt2: nnsight.LanguageModel):
        with gpt2.trace() as tracer:
            with tracer.invoke(self.PROMPT_A):
                a_inner = gpt2.transformer.output.save()
                a_top = gpt2.output.save()
            with tracer.invoke(self.PROMPT_B):
                b_inner = gpt2.transformer.output.save()
                b_top = gpt2.output.save()

        for value in (a_inner.last_hidden_state, a_top.logits):
            assert value.shape[0] == 1
        for value in (b_inner.last_hidden_state, b_top.logits):
            assert value.shape[0] == 1

        # Each invoke gets its own narrowed view, not one shared object.
        assert a_top.logits is not b_top.logits

    @torch.no_grad()
    def test_read_matches_the_same_prompt_run_alone(
        self, gpt2: nnsight.LanguageModel
    ):
        with gpt2.trace(self.PROMPT_A):
            reference = gpt2.output.logits.save()
        with gpt2.trace(self.PROMPT_B):
            other_reference = gpt2.output.logits.save()

        with gpt2.trace() as tracer:
            with tracer.invoke(self.PROMPT_A):
                batched = gpt2.output.logits.save()
            with tracer.invoke(self.PROMPT_B):
                gpt2.output.save()

        assert _rel_err(batched, reference) < 1e-4
        # Negative control: a wrong slice would look like the other prompt.
        assert _rel_err(batched, other_reference) > 1e-2

    @torch.no_grad()
    def test_write_does_not_leak_into_other_invokes(
        self, gpt2: nnsight.LanguageModel
    ):
        with gpt2.trace() as tracer:
            with tracer.invoke(self.PROMPT_A):
                base_a = gpt2.lm_head.output.save()
            with tracer.invoke(self.PROMPT_B):
                base_b = gpt2.lm_head.output.save()

        with gpt2.trace() as tracer:
            with tracer.invoke(self.PROMPT_A):
                gpt2.transformer.output.last_hidden_state[:] = 0.0
                edited_a = gpt2.lm_head.output.save()
            with tracer.invoke(self.PROMPT_B):
                edited_b = gpt2.lm_head.output.save()

        assert not torch.allclose(edited_a, base_a, atol=1e-3)
        assert torch.allclose(edited_b, base_b, atol=1e-3)

    @torch.no_grad()
    def test_attribute_and_item_stay_in_sync(self, gpt2: nnsight.LanguageModel):
        # ModelOutput mirrors items onto attributes; rebuilding it through
        # __setitem__ keeps that invariant, so `.logits` is the narrowed value.
        with gpt2.trace() as tracer:
            with tracer.invoke(self.PROMPT_A):
                top = gpt2.output.save()
            with tracer.invoke(self.PROMPT_B):
                gpt2.output.save()

        assert type(top).__name__.endswith("Output") or "Output" in type(top).__name__
        assert top.logits is top["logits"]

    @torch.no_grad()
    def test_empty_invoke_still_sees_the_full_batch(
        self, gpt2: nnsight.LanguageModel
    ):
        with gpt2.trace() as tracer:
            with tracer.invoke(self.PROMPT_A):
                pass
            with tracer.invoke(self.PROMPT_B):
                pass
            with tracer.invoke():
                full = gpt2.output.save()

        assert full.logits.shape[0] == 2


# =============================================================================
# ModelOutput through the cache path
# =============================================================================


@pytest.mark.cache
@pytest.mark.usefixtures("gpt2")
class TestCacheModelOutput:
    """``tracer.cache`` transformations must reach inside a ``ModelOutput``."""

    @torch.no_grad()
    def test_dtype_is_applied(self, gpt2: nnsight.LanguageModel):
        with gpt2.trace("hello world") as tracer:
            cache = tracer.cache(
                modules=[gpt2.transformer], dtype=torch.float16
            ).save()

        output = cache["model.transformer"].output

        assert output.last_hidden_state.dtype == torch.float16

    @torch.no_grad()
    def test_device_is_applied(self, gpt2: nnsight.LanguageModel):
        with gpt2.trace("hello world") as tracer:
            cache = tracer.cache(modules=[gpt2.transformer], device="cpu").save()

        output = cache["model.transformer"].output

        assert output.last_hidden_state.device.type == "cpu"

    def test_detach_is_applied(self, gpt2: nnsight.LanguageModel):
        # detach defaults to True; a ModelOutput that skipped it kept the whole
        # autograd graph alive for as long as the cache was held.
        with gpt2.trace("hello world") as tracer:
            cache = tracer.cache(modules=[gpt2.transformer]).save()

        output = cache["model.transformer"].output

        assert output.last_hidden_state.grad_fn is None
        assert not output.last_hidden_state.requires_grad

    @torch.no_grad()
    def test_narrowed_per_invoke(self, gpt2: nnsight.LanguageModel):
        with gpt2.trace() as tracer:
            with tracer.invoke("The Eiffel Tower is in the city of"):
                first = tracer.cache(modules=[gpt2.transformer]).save()
            with tracer.invoke("Madison Square Garden is located in the city of"):
                second = tracer.cache(modules=[gpt2.transformer]).save()

        for cache in (first, second):
            output = cache["model.transformer"].output
            assert output.last_hidden_state.shape[0] == 1
