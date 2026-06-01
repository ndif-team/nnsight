"""Regression tests for ``nnsight.util.apply`` / ``applyn`` over
dict-subclass containers (HuggingFace ``ModelOutput``).

The batcher's ``narrow``/``swap`` path uses ``apply`` to recurse into
structured activations and slice only the per-request rows. Before the
fix, ``apply`` used strict ``type(data) == dict`` equality, so
``CausalLMOutputWithPast`` (a subclass of ``OrderedDict``) fell through
to the terminal ``return data`` branch — tensors inside were never
narrowed. User-visible symptom: ``model.output.logits.save()`` in a
batched server returned the full concatenated batch instead of the
per-request row.

These tests pin the contract so the regression can't return silently.
"""

from __future__ import annotations

import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from nnsight.util import apply, applyn


def _narrow(t: torch.Tensor) -> torch.Tensor:
    return t.narrow(0, 0, 1)


class TestApplyModelOutput:
    def test_causal_lm_output_is_dict_subclass(self):
        """Sanity-check the precondition that motivated the fix."""
        out = CausalLMOutputWithPast(logits=torch.zeros(4, 3, 10))
        assert type(out) is not dict, "subclass, not strict dict"
        assert isinstance(out, dict), "but still a dict-like"

    def test_apply_narrows_logits_inside_model_output(self):
        """The root-envoy bug this regression targets."""
        out = CausalLMOutputWithPast(logits=torch.zeros(19, 5, 100))
        res = apply(out, _narrow, torch.Tensor)
        assert res.logits.shape == (1, 5, 100), (
            f"logits not narrowed: got {tuple(res.logits.shape)}"
        )

    def test_apply_preserves_model_output_class(self):
        """Class identity matters — downstream ``out.logits`` attribute
        access depends on ``CausalLMOutputWithPast.__getattr__``.
        A plain-dict fallback would break user code.
        """
        out = CausalLMOutputWithPast(logits=torch.zeros(4, 3, 10))
        res = apply(out, _narrow, torch.Tensor)
        assert type(res) is CausalLMOutputWithPast
        # Attribute access must keep working post-narrow.
        assert res.logits is not None

    def test_apply_handles_none_fields(self):
        """ModelOutput fields default to ``None``; reconstruction
        must not choke on them.
        """
        out = CausalLMOutputWithPast(
            logits=torch.zeros(4, 3, 10),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        res = apply(out, _narrow, torch.Tensor)
        assert res.logits.shape == (1, 3, 10)
        assert res.past_key_values is None

    def test_apply_narrows_base_model_output_too(self):
        """Not specific to CausalLMOutputWithPast — any ModelOutput
        subclass should narrow. BaseModelOutputWithPast has
        ``last_hidden_state`` instead of ``logits``.
        """
        out = BaseModelOutputWithPast(last_hidden_state=torch.zeros(8, 2, 20))
        res = apply(out, _narrow, torch.Tensor)
        assert type(res) is BaseModelOutputWithPast
        assert res.last_hidden_state.shape == (1, 2, 20)

    def test_apply_narrows_nested_tensors_list_inside_model_output(self):
        """hidden_states is a tuple of tensors in HF outputs — must
        still be narrowed via the tuple branch.
        """
        out = CausalLMOutputWithPast(
            logits=torch.zeros(4, 3, 10),
            hidden_states=(
                torch.zeros(4, 3, 20),
                torch.zeros(4, 3, 20),
                torch.zeros(4, 3, 20),
            ),
        )
        res = apply(out, _narrow, torch.Tensor)
        assert res.logits.shape == (1, 3, 10)
        assert isinstance(res.hidden_states, tuple)
        assert all(h.shape == (1, 3, 20) for h in res.hidden_states), (
            f"nested tuple not narrowed: shapes {[h.shape for h in res.hidden_states]}"
        )

    def test_apply_plain_dict_unchanged(self):
        """The existing plain-dict fast path must still work — the fix
        added a new ``elif isinstance(data, dict)`` branch AFTER the
        strict ``type(data) == dict`` branch.
        """
        d = {"a": torch.zeros(4, 10), "b": torch.zeros(4, 10)}
        res = apply(d, _narrow, torch.Tensor)
        assert type(res) is dict
        assert res["a"].shape == (1, 10)
        assert res["b"].shape == (1, 10)


class TestApplyNModelOutput:
    def test_applyn_swap_slice_in_model_output(self):
        """``applyn`` (2-arg version used by ``swap``) must recurse into
        dict subclasses too. Otherwise ``envoy.output.logits = ...``
        assignment on the root envoy silently fails.
        """
        current = CausalLMOutputWithPast(logits=torch.zeros(4, 3, 10))
        swap_value = CausalLMOutputWithPast(logits=torch.ones(1, 3, 10))

        def _swap_batch(current_t, swap_t):
            new = current_t.clone()
            new.narrow(0, 0, 1).copy_(swap_t)
            return new

        res = applyn([current, swap_value], _swap_batch, torch.Tensor)
        assert type(res) is CausalLMOutputWithPast
        assert res.logits.shape == (4, 3, 10)
        # First row should now be ones (the swapped slice).
        assert torch.all(res.logits[0] == 1.0)
        # Remaining rows unchanged.
        assert torch.all(res.logits[1:] == 0.0)
