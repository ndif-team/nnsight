"""Tests for the ``eproperty.transform`` callback.

``transform`` complements ``preprocess``: when ``preprocess`` returns a
*derived* value (clone, view, reshape), in-place edits the user makes are
invisible to the running model. ``transform`` runs on the mediator side
after the worker yields control and ``batcher.swap``s its return value back
into the model so those edits propagate.

Covered:
- Identity transform with a clone preprocess: in-place edits propagate.
- Without transform (preprocess only): in-place edits do NOT propagate
  (regression / control case).
- Reshape-style transform mimicking per-head attention access.
- Transform may return a different tensor than what was given (full
  replacement).
- Per-access one-shot semantics: two sequential reads each get their own
  fresh transform binding.
- Transform without preprocess (rare but supported): receives the raw value.
"""

import pytest
import torch

from nnsight import NNsight
from nnsight.intervention.envoy import Envoy, eproperty
from nnsight.intervention.hooks import requires_output


# ---------------------------------------------------------------------------
# Tiny model used by all tests
# ---------------------------------------------------------------------------


class _Tiny(torch.nn.Module):
    """Identity-ish model so it's easy to compare input → output."""

    def __init__(self, in_features: int = 4, out_features: int = 4):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        # Make fc act like identity so we can reason about values directly.
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(out_features, in_features))

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Envoy variants
# ---------------------------------------------------------------------------


class _CloneOnlyModel(NNsight):
    """Clones value on read but registers no transform."""

    @eproperty(key="output")
    @requires_output
    def cloned(self): ...

    @cloned.preprocess
    def cloned(self, value):
        return value.clone()


class _CloneAndTransformModel(NNsight):
    """Clones on read, transform returns the (mutated) clone unchanged."""

    @eproperty(key="output")
    @requires_output
    def thing(self): ...

    @thing.preprocess
    def thing(self, value):
        return value.clone()

    @thing.transform
    @staticmethod
    def thing(value):
        return value


class _ReshapeHeadsModel(NNsight):
    """Mimics splitting attention heads.

    Treats the last dim as ``n_heads * head_dim`` and exposes
    ``[B, n_heads, head_dim]`` so the user can edit a single "head" — the
    transform reshapes back to ``[B, H]`` before swapping into the model.
    """

    n_heads = 2

    @eproperty(key="output")
    @requires_output
    def heads(self): ...

    @heads.preprocess
    def heads(self, value):
        B, H = value.shape
        return value.clone().view(B, self.n_heads, H // self.n_heads)

    @heads.transform
    @staticmethod
    def heads(value):
        B, n_heads, head_dim = value.shape
        return value.reshape(B, n_heads * head_dim)


class _ReplaceModel(NNsight):
    """Transform returns a freshly constructed tensor (full replacement)."""

    @eproperty(key="output")
    @requires_output
    def thing(self): ...

    @thing.preprocess
    def thing(self, value):
        return value.clone()

    @thing.transform
    @staticmethod
    def thing(value):
        return torch.full_like(value, 7.0)


class _NoPreprocessModel(NNsight):
    """Transform with no preprocess: the raw model value is bound."""

    @eproperty(key="output")
    @requires_output
    def thing(self): ...

    @thing.transform
    @staticmethod
    def thing(value):
        return value * 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_clone_without_transform_does_not_propagate():
    """Sanity / control: cloning in preprocess hides user edits from the model."""
    model = _CloneOnlyModel(_Tiny())
    x = torch.arange(4, dtype=torch.float32).unsqueeze(0)  # [[0, 1, 2, 3]]

    with model.trace(x):
        clone = model.cloned.save()
        clone[:] = 0
        out = model.output.save()

    # User's clone was zeroed locally...
    assert torch.equal(clone, torch.zeros_like(clone))
    # ...but the model's downstream output is unaffected.
    assert torch.equal(out, x)


@torch.no_grad()
def test_clone_plus_transform_propagates():
    """Identity transform makes in-place edits to the clone reach the model."""
    model = _CloneAndTransformModel(_Tiny())
    x = torch.arange(4, dtype=torch.float32).unsqueeze(0)

    with model.trace(x):
        thing = model.thing.save()
        thing[:] = 0
        out = model.output.save()

    assert torch.equal(thing, torch.zeros_like(thing))
    # Output now reflects the user's in-place edit.
    assert torch.equal(out, torch.zeros_like(out))


@torch.no_grad()
def test_reshape_transform_per_head_edit():
    """Edit a single 'head' through a reshape view; model reflects the edit."""
    model = _ReshapeHeadsModel(_Tiny(in_features=4, out_features=4))
    x = torch.arange(4, dtype=torch.float32).unsqueeze(0)  # [[0, 1, 2, 3]]

    with model.trace(x):
        heads = model.heads.save()  # shape [1, 2, 2]
        # Zero out only the second head.
        heads[:, 1] = 0
        out = model.output.save()

    assert heads.shape == (1, 2, 2)
    # First head untouched, second head zeroed in the user's view.
    assert torch.equal(heads[0, 0], torch.tensor([0.0, 1.0]))
    assert torch.equal(heads[0, 1], torch.tensor([0.0, 0.0]))
    # Model output: head 0 (positions 0–1) preserved, head 1 (positions 2–3) zeroed.
    assert torch.equal(out, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))


@torch.no_grad()
def test_transform_can_replace_value_entirely():
    """Transform return value, not the user's mutated clone, is what swaps in."""
    model = _ReplaceModel(_Tiny())
    x = torch.arange(4, dtype=torch.float32).unsqueeze(0)

    with model.trace(x):
        thing = model.thing.save()
        # User doesn't even mutate; transform returns full_like(value, 7).
        out = model.output.save()

    # User's view is the original cloned value.
    assert torch.equal(thing, x)
    # Model sees the transform's return value.
    assert torch.equal(out, torch.full_like(out, 7.0))


class _ChildThingEnvoy(Envoy):
    """Non-root Envoy with the same clone+identity-transform `.thing`."""

    @eproperty(key="output")
    @requires_output
    def thing(self): ...

    @thing.preprocess
    def thing(self, value):
        return value.clone()

    @thing.transform
    @staticmethod
    def thing(value):
        return value


@torch.no_grad()
def test_transform_is_one_shot_per_access():
    """Two sequential reads each get their own transform binding.

    The first read zeroes one submodule's output; the second read on the
    next submodule writes 5s. If transforms leaked across reads, the second
    read's transform would fire with the first read's preprocessed value.
    """

    class _Passthrough(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Identity()
            self.b = torch.nn.Identity()

        def forward(self, x):
            return self.b(self.a(x))

    # Use the `envoys=` mapping so both Identity submodules get the
    # transform-equipped Envoy subclass.
    model = NNsight(_Passthrough(), envoys={torch.nn.Identity: _ChildThingEnvoy})

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    with model.trace(x):
        ta = model.a.thing.save()
        ta[:] = 0
        tb = model.b.thing.save()
        tb[:] = 5
        out = model.output.save()

    assert torch.equal(ta, torch.zeros_like(ta))
    assert torch.equal(tb, torch.full_like(tb, 5.0))
    # Final output reflects b's swap (the most recent), proving b's transform
    # fired with b's preprocessed value (not a stale binding from a).
    assert torch.equal(out, torch.full_like(out, 5.0))


@torch.no_grad()
def test_transform_without_preprocess():
    """Transform with no preprocess gets the raw model-side value."""
    model = _NoPreprocessModel(_Tiny())
    x = torch.arange(4, dtype=torch.float32).unsqueeze(0)

    with model.trace(x):
        _ = model.thing  # trigger transform; we don't even need the value
        out = model.output.save()

    # Transform returned `value * 0`, so the model output is zeros.
    assert torch.equal(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# Integration: attention-head splitting via `envoys=` mapping + transform
# ---------------------------------------------------------------------------
#
# The canonical real-world use case for `transform`. Combines:
#   - `envoys={...}`: route a specific torch module type to a custom Envoy
#     subclass on the entire tree.
#   - `preprocess` + `transform`: expose the attention output as
#     [B, n_heads, S, head_dim] so users can edit a single head, and reshape
#     back to [B, S, hidden] so the model continues with the user-edited heads.


class _AttnLike(torch.nn.Module):
    """Mimics the shape contract of an attention output projection.

    Real attention does QK/V/scaled-dot-product internally; for the test all
    that matters is that the module's output is shaped `[B, S, hidden]` where
    `hidden == n_heads * head_dim`. We use an identity-weighted Linear so the
    output equals the input verbatim — easy to write expectations against.
    """

    n_heads = 4
    head_dim = 3

    def __init__(self):
        super().__init__()
        hidden = self.n_heads * self.head_dim
        self.proj = torch.nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden))

    def forward(self, x):
        return self.proj(x)


class _AttnHeadsEnvoy(Envoy):
    """Exposes `.heads` as a list of `[B, S, head_dim]` views into the
    attention output.

    No clone, no transform: each list entry is a view into the underlying
    `[B, S, hidden]` tensor that the model also uses downstream, so in-place
    edits to a head propagate naturally.
    """

    @eproperty(key="output")
    @requires_output
    def heads(self): ...

    @heads.preprocess
    def heads(self, value):
        # [B, S, hidden] viewed as [B, S, n_heads, head_dim], then unbound
        # along the head dim into a tuple of [B, S, head_dim] views.
        n_heads = self._module.n_heads
        B, S, H = value.shape
        return list(value.view(B, S, n_heads, H // n_heads).unbind(dim=2))


class _AttnBlock(torch.nn.Module):
    """attn -> identity downstream, so block output == (possibly swapped) attn output."""

    def __init__(self):
        super().__init__()
        self.attn = _AttnLike()
        self.down = torch.nn.Identity()

    def forward(self, x):
        return self.down(self.attn(x))


@torch.no_grad()
def test_attention_heads_split_via_envoys():
    """End-to-end: route _AttnLike → _AttnHeadsEnvoy via the envoys mapping,
    then use preprocess to expose attention output as a list of per-head
    views. In-place edits to a view mutate the underlying model tensor
    directly — no transform / clone needed.
    """
    model = NNsight(_AttnBlock(), envoys={_AttnLike: _AttnHeadsEnvoy})

    n_heads, head_dim = _AttnLike.n_heads, _AttnLike.head_dim
    hidden = n_heads * head_dim
    B, S = 1, 2
    x = torch.arange(B * S * hidden, dtype=torch.float32).view(B, S, hidden)

    # The envoys mapping placed _AttnHeadsEnvoy on the attn submodule.
    assert type(model.attn) is _AttnHeadsEnvoy
    # Block submodule didn't match the mapping — falls back to base Envoy.
    assert type(model.down) is Envoy

    # Per-head view of the original input, used for expected-value construction.
    x_per_head = x.view(B, S, n_heads, head_dim).unbind(dim=2)

    # 1) Shape: preprocess returns a list of per-head views.
    with model.trace(x):
        heads = model.attn.heads.save()
        out_unmodified = model.output.save()

    assert isinstance(heads, list)
    assert len(heads) == n_heads
    for h in range(n_heads):
        assert heads[h].shape == (B, S, head_dim)
        assert torch.equal(heads[h], x_per_head[h])
    # No edits → model output unchanged from the pure forward pass.
    assert torch.equal(out_unmodified, x)

    # 2) Zero out a single head via in-place edit on the view.
    with model.trace(x):
        heads = model.attn.heads.save()
        heads[1][:] = 0
        out_one_head = model.output.save()

    out_per_head = out_one_head.view(B, S, n_heads, head_dim).unbind(dim=2)
    for h in range(n_heads):
        if h == 1:
            assert torch.equal(out_per_head[h], torch.zeros(B, S, head_dim))
        else:
            assert torch.equal(out_per_head[h], x_per_head[h])

    # 3) Edit multiple heads independently.
    with model.trace(x):
        heads = model.attn.heads.save()
        heads[0][:] = 0
        heads[2][:] = 0
        out_multi = model.output.save()

    out_per_head = out_multi.view(B, S, n_heads, head_dim).unbind(dim=2)
    for h in range(n_heads):
        if h in (0, 2):
            assert torch.equal(out_per_head[h], torch.zeros(B, S, head_dim))
        else:
            assert torch.equal(out_per_head[h], x_per_head[h])

    # 4) Replace one head's values with arbitrary content.
    replacement = torch.tensor([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]])
    with model.trace(x):
        heads = model.attn.heads.save()
        heads[3][0] = replacement
        out_replaced = model.output.save()

    out_per_head = out_replaced.view(B, S, n_heads, head_dim).unbind(dim=2)
    assert torch.equal(out_per_head[3][0], replacement)
    for h in (0, 1, 2):
        assert torch.equal(out_per_head[h], x_per_head[h])
