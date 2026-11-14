import gc
import math
import sys
import warnings
import weakref

import pytest
import torch

from pathlib import Path

import nnsight
import torch.nn as nn
import torch.nn.functional as F

from nnsight.intervention import kernel, sites
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import OutOfOrderError
from nnsight.intervention.sites import Provenance
from nnsight.intervention.source import SourceNotAvailable


def _reference(q, k, mask=None, is_causal=False, scale=None):
    """Compute reference fp32 scores, softmax probabilities, and logsumexp over the kernel's inputs (expanding `k` for GQA)."""
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, 1) if groups > 1 else k
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = (q.float() @ k.float().transpose(-1, -2)) * scale
    if is_causal:
        causal = torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.float()
    return scores, torch.softmax(scores, -1), torch.logsumexp(scores, -1)


class Mixer(nn.Module):
    """Minimal attention module that calls `F.scaled_dot_product_attention` directly."""

    def __init__(self, width=16, q_heads=4, kv_heads=4, causal=True):
        super().__init__()
        self.q_heads, self.kv_heads, self.causal = q_heads, kv_heads, causal
        self.head = width // q_heads
        self.to_q = nn.Linear(width, q_heads * self.head)
        self.to_kv = nn.Linear(width, 2 * kv_heads * self.head)
        self.merge = nn.Linear(q_heads * self.head, width)

    def split(self, t, heads):
        return t.view(t.shape[0], t.shape[1], heads, self.head).transpose(1, 2)

    def forward(self, x, bias=None):
        q = self.split(self.to_q(x), self.q_heads)
        k, v = self.to_kv(x).chunk(2, -1)
        k, v = self.split(k, self.kv_heads), self.split(v, self.kv_heads)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=self.causal and bias is None, enable_gqa=self.q_heads != self.kv_heads
        )
        return self.merge(out.transpose(1, 2).flatten(2))


class Twice(nn.Module):
    """Module containing two sequential attention submodules with distinct kernel configurations."""

    def __init__(self):
        super().__init__()
        self.first = Mixer()
        self.second = Mixer(causal=False)

    def forward(self, x):
        return self.second(self.first(x))



@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(2, 6, 16)


def _inputs(mixer, x):
    q = mixer.split(mixer.to_q(x), mixer.q_heads)
    k, _ = mixer.to_kv(x).chunk(2, -1)
    return q, mixer.split(k, mixer.kv_heads)


class TestSites:
    def test_probs_match_fp32_softmax_and_carry_provenance(self, x):
        mixer = Mixer()
        envoy = Envoy(mixer)
        with envoy.trace(x):
            probs = nnsight.save(envoy.kernel.probs())
        _, expected, _ = _reference(*_inputs(mixer, x), is_causal=True)
        assert probs.shape == (2, 4, 6, 6)
        assert torch.allclose(probs, expected, atol=1e-6)
        assert (torch.triu(probs, 1) == 0).all()
        assert isinstance(probs.provenance, Provenance)
        assert probs.provenance.kind == "exact reconstruction"
        assert "scaled_dot_product" in probs.provenance.kernel

    def test_lse_is_the_kernels_export(self, x):
        mixer = Mixer()
        envoy = Envoy(mixer)
        with envoy.trace(x):
            lse = nnsight.save(envoy.kernel.lse())
        _, _, expected = _reference(*_inputs(mixer, x), is_causal=True)
        assert lse.shape == (2, 4, 6)
        assert torch.allclose(lse, expected, atol=1e-5)
        assert lse.provenance.kind == "exact export"

    def test_scores_carry_the_mask(self, x):
        mixer = Mixer()
        envoy = Envoy(mixer)
        with envoy.trace(x):
            scores = nnsight.save(envoy.kernel.scores())
        expected, _, _ = _reference(*_inputs(mixer, x), is_causal=True)
        assert torch.equal(torch.isinf(scores), torch.isinf(expected))
        finite = ~torch.isinf(expected)
        assert torch.allclose(scores[finite], expected[finite], atol=1e-6)

    def test_grouped_queries_read_their_key_head(self, x):
        mixer = Mixer(q_heads=4, kv_heads=2)
        envoy = Envoy(mixer)
        with envoy.trace(x):
            probs = nnsight.save(envoy.kernel.probs(heads=3))
        _, expected, _ = _reference(*_inputs(mixer, x), is_causal=True)
        assert probs.shape == (2, 6, 6)
        assert torch.allclose(probs, expected[:, 3], atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
    def test_explicit_mask_is_applied(self, x, dtype):
        mixer = Mixer(causal=False)
        envoy = Envoy(mixer)
        allowed = torch.rand(2, 1, 6, 6) > 0.3
        allowed[..., 0] = True  # every row keeps one key, so no row is all -inf
        bias = allowed if dtype is torch.bool else torch.zeros(2, 1, 6, 6).masked_fill(~allowed, float("-inf"))
        with envoy.trace(x, bias):
            probs = nnsight.save(envoy.kernel.probs())
        _, expected, _ = _reference(*_inputs(mixer, x), mask=bias)
        assert torch.allclose(probs, expected, atol=1e-6)
        assert (probs[~allowed.expand_as(probs)] == 0).all()

    def test_output_is_unchanged_by_a_request(self, x):
        envoy = Envoy(Mixer())
        with envoy.trace(x):
            plain = nnsight.save(envoy.output)
        with envoy.trace(x):
            envoy.kernel.probs()
            served = nnsight.save(envoy.output)
        assert torch.equal(plain, served)

    def test_each_module_serves_its_own_kernel(self, x):
        twice = Twice()
        envoy = Envoy(twice)
        with envoy.trace(x):
            first = nnsight.save(envoy.first.kernel.probs())
            second = nnsight.save(envoy.second.kernel.probs())
        assert (torch.triu(first, 1) == 0).all()
        assert not (torch.triu(second, 1) == 0).all()

    def test_probs_equal_exp_of_scores_minus_lse(self, x):
        mixer = Mixer()
        envoy = Envoy(mixer)
        with envoy.trace(x):
            scores = nnsight.save(envoy.kernel.scores(heads=0, query=-1))
            lse = nnsight.save(envoy.kernel.lse(heads=0, query=-1))
            probs = nnsight.save(envoy.kernel.probs(heads=0, query=-1))
        assert torch.allclose(torch.exp(scores - lse.unsqueeze(-1)), probs, atol=1e-6)


class TestIndexing:
    def test_int_drops_the_axis_and_negative_counts_from_the_end(self, x):
        mixer = Mixer()
        envoy = Envoy(mixer)
        with envoy.trace(x):
            probs = nnsight.save(envoy.kernel.probs(heads=-1, query=-1, key=2))
        _, expected, _ = _reference(*_inputs(mixer, x), is_causal=True)
        assert probs.shape == (2,)
        assert torch.allclose(probs, expected[:, 3, -1, 2], atol=1e-6)

    def test_slice_and_list_keep_their_axes(self, x):
        mixer = Mixer()
        envoy = Envoy(mixer)
        with envoy.trace(x):
            probs = nnsight.save(envoy.kernel.probs(heads=[0, 3], query=slice(-2, None), key=torch.tensor([1, 4])))
        _, expected, _ = _reference(*_inputs(mixer, x), is_causal=True)
        assert probs.shape == (2, 2, 2, 2)
        assert torch.allclose(probs, expected[:, [0, 3]][:, :, -2:][..., [1, 4]], atol=1e-6)

    def test_distinct_specs_in_one_block_are_served_separately(self, x):
        mixer = Mixer()
        envoy = Envoy(mixer)
        with envoy.trace(x):
            one = nnsight.save(envoy.kernel.probs(heads=1))
            as_slice = nnsight.save(envoy.kernel.probs(heads=slice(1, 2)))
        assert one.shape == (2, 6, 6)
        assert as_slice.shape == (2, 1, 6, 6)
        assert torch.equal(one, as_slice[:, 0])

    @pytest.mark.parametrize("spec", [{"heads": 4}, {"heads": -5}, {"query": [0, 6]}])
    def test_out_of_range_index_raises(self, x, spec):
        envoy = Envoy(Mixer())
        with pytest.raises(IndexError, match="out of range"):
            with envoy.trace(x):
                envoy.kernel.probs(**spec)

    def test_unsupported_index_type_raises_at_the_request(self, x):
        envoy = Envoy(Mixer())
        with pytest.raises(TypeError, match="not an int, slice, list or tensor"):
            with envoy.trace(x):
                envoy.kernel.probs(heads="all")

    @pytest.mark.parametrize("spec", [{"heads": True}, {"heads": torch.tensor([0.5, 2.0])}])
    def test_bool_and_float_indices_are_refused(self, x, spec):
        envoy = Envoy(Mixer())
        with pytest.raises(TypeError):
            with envoy.trace(x):
                envoy.kernel.probs(**spec)

    def test_index_like_scalars_and_boolean_masks_follow_torch(self, x):
        import numpy as np

        mixer = Mixer()
        envoy = Envoy(mixer)
        with envoy.trace(x):
            scalar = nnsight.save(envoy.kernel.probs(heads=torch.tensor(2), query=np.int64(-1)))
            picked = nnsight.save(envoy.kernel.probs(heads=torch.tensor([True, False, False, True]), query=-1))
        _, expected, _ = _reference(*_inputs(mixer, x), is_causal=True)
        assert scalar.shape == (2, 6)
        assert torch.allclose(scalar, expected[:, 2, -1], atol=1e-6)
        assert picked.shape == (2, 2, 6)
        assert torch.allclose(picked, expected[:, [0, 3], -1], atol=1e-6)

    def test_unknown_site_raises_naming_the_available_ones(self, x):
        envoy = Envoy(Mixer())
        with pytest.raises(ValueError, match="no site 'entropy'; available: lse, scores, probs"):
            with envoy.trace(x):
                envoy.kernel.entropy()

    def test_unknown_spec_key_raises(self, x):
        envoy = Envoy(Mixer())
        with pytest.raises(ValueError, match="takes heads, query"):
            with envoy.trace(x):
                envoy.kernel.lse(key=0)

    def test_large_unsliced_request_warns(self, monkeypatch, x):
        monkeypatch.setattr(sites, "_LARGE_SITE_BYTES", 100)
        envoy = Envoy(Mixer())
        with pytest.warns(UserWarning, match="materializes"):
            with envoy.trace(x):
                envoy.kernel.probs()


class TestOrdering:
    def test_request_after_module_output_raises_out_of_order(self, x):
        envoy = Envoy(Mixer())
        with pytest.raises(OutOfOrderError):
            with envoy.trace(x):
                envoy.output
                envoy.kernel.probs()

    def test_request_outside_a_trace_raises(self):
        envoy = Envoy(Mixer())
        with pytest.raises(ValueError, match="outside of interleaving"):
            envoy.kernel.probs()

    def test_submodule_reads_around_the_request_on_a_fresh_module(self, x):
        mixer = Mixer()
        envoy = Envoy(mixer)
        # Reading a submodule that executes before the kernel, followed by the kernel site, works on the first trace.
        with envoy.trace(x):
            projected = nnsight.save(envoy.to_q.output)
            probs = nnsight.save(envoy.kernel.probs(heads=0))
        assert torch.allclose(projected, mixer.to_q(x))
        assert probs.shape == (2, 6, 6)
        # Requesting the kernel site first, followed by a submodule that executes after the kernel, also works.
        with envoy.trace(x):
            probs = nnsight.save(envoy.kernel.probs(heads=0))
            merged = nnsight.save(envoy.merge.output)
        assert probs.shape == (2, 6, 6) and merged.shape == (2, 6, 16)
        # Reading a submodule that executes after the kernel and then requesting the site raises `OutOfOrderError`.
        with pytest.raises(OutOfOrderError, match="already ran past"):
            with envoy.trace(x):
                envoy.merge.output
                envoy.kernel.probs(heads=0)

    def test_request_on_a_module_without_a_kernel_is_never_served(self, x):
        envoy = Envoy(nn.Sequential(nn.Linear(16, 16), nn.ReLU()))
        with pytest.raises(OutOfOrderError, match="already ran past it"):
            with envoy.trace(x):
                envoy.kernel.probs()

    def test_served_values_carry_no_gradient(self, x):
        envoy = Envoy(Mixer())
        with envoy.trace(x.requires_grad_()):
            probs = nnsight.save(envoy.kernel.probs(heads=0))
            lse = nnsight.save(envoy.kernel.lse())
        assert not probs.requires_grad and not lse.requires_grad


class TestMasks:
    def test_float_mask_at_the_dtype_minimum_masks_exactly(self, x):
        # Transformers constructs additive float masks using the dtype's minimum representable value.
        mixer = Mixer(causal=False)
        envoy = Envoy(mixer)
        allowed = torch.ones(6, 6, dtype=torch.bool).tril()
        bias = torch.zeros(6, 6).masked_fill(~allowed, torch.finfo(torch.float32).min)
        with envoy.trace(x, bias):
            probs = nnsight.save(envoy.kernel.probs())
            scores = nnsight.save(envoy.kernel.scores(heads=0, query=0))
        assert (probs[..., ~allowed] == 0).all()
        assert torch.allclose(probs.sum(-1), torch.ones(2, 4, 6), atol=1e-6)
        assert scores[1:].min() <= torch.finfo(torch.float32).min / 2  # the float mask value is added to scores

    @pytest.mark.parametrize("masked", [None, torch.finfo(torch.float16).min, torch.finfo(torch.float32).min])
    def test_a_row_with_every_key_masked_is_zero(self, x, masked):
        # Test boolean masks and additive masks using float16 or float32 minimums.
        mixer = Mixer(causal=False)
        envoy = Envoy(mixer)
        allowed = torch.ones(2, 1, 6, 6, dtype=torch.bool)
        allowed[1, :, :2] = False  # mask out all keys for the first two queries of batch item 1 (padding queries)
        bias = allowed if masked is None else torch.zeros(2, 1, 6, 6).masked_fill(~allowed, masked)
        with envoy.trace(x, bias):
            probs = nnsight.save(envoy.kernel.probs())
        assert (probs[1, :, :2] == 0).all()
        assert torch.allclose(probs[0].sum(-1), torch.ones(4, 6), atol=1e-6)
        assert torch.allclose(probs[1, :, 2:].sum(-1), torch.ones(4, 4), atol=1e-6)

    def test_math_fallback_is_served_as_a_replay(self, x):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        mixer = Mixer()
        envoy = Envoy(mixer)
        with sdpa_kernel(SDPBackend.MATH):
            with envoy.trace(x):
                probs = nnsight.save(envoy.kernel.probs(heads=1))
                lse = nnsight.save(envoy.kernel.lse(heads=1))
        _, expected, expected_lse = _reference(*_inputs(mixer, x), is_causal=True)
        assert probs.provenance.kind == "replay"
        assert probs.provenance.kernel == "math backend (fp32)"
        assert lse.provenance.kind == "replay"
        assert torch.allclose(probs, expected[:, 1], atol=1e-6)
        assert torch.allclose(lse, expected_lse[:, 1], atol=1e-5)


class TestBatching:
    def test_each_invoke_sees_its_own_rows(self, gpt2):
        attn = gpt2.transformer.h[1].attn
        short, long = "Hello there", "Hello there, how are you today"
        with gpt2.trace() as tracer:
            with tracer.invoke(short):
                first = attn.kernel.probs(heads=2, query=-1).save()
            with tracer.invoke(long):
                second = attn.kernel.probs(heads=2, query=-1).save()
                again = attn.kernel.probs(heads=2, query=-1).save()
        with gpt2.trace(short):
            solo = attn.kernel.probs(heads=2, query=-1).save()
        width = gpt2.tokenizer(long, return_tensors="pt").input_ids.shape[1]
        assert first.shape == second.shape == (1, width)
        assert torch.equal(second, again)
        assert first.provenance.kind == "exact reconstruction"
        # The shorter prompt is left-padded: padding keys receive zero probability,
        # and the unpadded suffix matches a standalone trace of the short prompt.
        assert (first[0, : width - solo.shape[1]] == 0).all()
        assert torch.allclose(first[0, width - solo.shape[1] :], solo[0], atol=1e-4)

    def test_invokes_with_different_specs(self, gpt2):
        attn = gpt2.transformer.h[1].attn
        with gpt2.trace() as tracer:
            with tracer.invoke("Hello there"):
                lse = attn.kernel.lse().save()
            with tracer.invoke("Hello there, how are you today"):
                probs = attn.kernel.probs(query=-1).save()
        assert lse.shape[:2] == (1, gpt2.config.n_head)
        assert probs.shape[:2] == (1, gpt2.config.n_head)
        assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-6)

    def test_a_stop_in_one_invoke_still_serves_the_other(self, gpt2):
        attn = gpt2.transformer.h[0].attn
        with gpt2.trace() as tracer:
            with tracer.invoke("Hello there"):
                lse = attn.kernel.lse(heads=0).save()
                tracer.stop()
            with tracer.invoke("Hello there, how are you today"):
                probs = attn.kernel.probs(heads=0, query=-1).save()
        assert lse.shape[0] == 1 and probs.shape[0] == 1
        assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-6)


class TestMemory:
    def test_module_is_freed_after_a_request(self, x):
        def build():
            envoy = Envoy(Mixer())
            with envoy.trace(x):
                envoy.kernel.probs()
            return weakref.ref(envoy._module)

        ref = build()
        gc.collect()
        assert ref() is None


class TestCatalog:
    def test_every_catalog_row_is_in_the_supported_kernels_table(self):
        from nnsight.intervention import attention
        from nnsight.modeling.kernels import transformers as catalog

        table = (Path(__file__).parent.parent / "docs" / "usage" / "kernel.md").read_text().split("## Supported kernels")[1].split("\n## ")[0]
        rows = [*attention.CATALOG, *catalog.CATALOG, *catalog.FLASH]
        assert len({spec.name for spec in rows}) == len(rows)
        for spec in rows:
            assert f"| `{spec.name}` | {spec.family} |" in table, spec.name
        assert table.count("\n| `") == len(rows)


class TestShims:
    """Verify that kernel shims are installed at registered holders only while at least one trace is open."""

    def test_kernel_function_is_shimmed_while_any_trace_is_open(self):
        original = torch._C._nn.scaled_dot_product_attention
        assert F.scaled_dot_product_attention is original
        first, second = Envoy(Mixer()).interleaver, Envoy(Mixer()).interleaver
        with first:
            shim = F.scaled_dot_product_attention
            assert shim is not original and kernel.unshimmed(shim) is original
            assert torch._C._nn.scaled_dot_product_attention is original  # PyTorch's C++ dispatch table is not patched
            with second:
                assert F.scaled_dot_product_attention is shim
            assert F.scaled_dot_product_attention is shim  # outer trace is still open
        assert F.scaled_dot_product_attention is original
        assert torch._C._nn.scaled_dot_product_attention is original

    def test_worker_error_before_first_park_leaves_shims_uninstalled(self, x):
        original = F.scaled_dot_product_attention
        envoy = Envoy(Mixer())
        with pytest.raises(TypeError, match="not an int, slice, list or tensor"):
            with envoy.trace(x):
                envoy.kernel.probs(heads="all")  # raises during worker startup, before `Interleaver.__enter__` installs shims
        assert F.scaled_dot_product_attention is original
        assert kernel._open == 0

    def test_register_patches_explicit_holders_only_while_a_trace_is_open(self):
        def scan(t):
            return t.cumsum(1)

        original = scan
        table = {"scan": scan}

        def through_cell():
            return scan

        cell = through_cell.__closure__[0]  # closure cell holding `scan`
        kernel.register(original, _no_sites, kernel.Item(table, "scan"), kernel.Attr(cell, "cell_contents"))
        try:
            kernel.register(original, _no_sites, kernel.Item(table, "scan"))  # duplicate holder is deduplicated
            assert len(kernel._KERNELS[original].holders) == 2
            with Envoy(Mixer()).interleaver:
                assert kernel.unshimmed(table["scan"]) is original and cell.cell_contents is table["scan"]
            assert table["scan"] is original and cell.cell_contents is original
        finally:
            kernel._KERNELS.pop(original)

    def test_register_while_a_trace_is_open_patches_new_holders_immediately(self):
        def scan(t):
            return t.cumsum(1)

        table = {"scan": scan}
        kernel.register(scan, _no_sites)
        try:
            with Envoy(Mixer()).interleaver:
                kernel.register(scan, _no_sites, kernel.Item(table, "scan"))
                assert table["scan"] is kernel._KERNELS[scan].shim
            assert table["scan"] is scan
        finally:
            kernel._KERNELS.pop(scan)

    def test_holder_overwritten_externally_is_not_clobbered(self):
        def scan(t):
            return t.cumsum(1)

        def other(t):
            return t.cumsum(1)

        table = {"before": other, "during": scan}
        kernel.register(scan, _no_sites, kernel.Item(table, "before"), kernel.Item(table, "during"))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Envoy(Mixer()).interleaver:  # holder already containing `other` is neither patched nor warned about
                    assert table["before"] is other and table["during"] is kernel._KERNELS[scan].shim
            with pytest.warns(UserWarning, match="no longer holds the kernel shim"):
                with Envoy(Mixer()).interleaver:
                    table["during"] = other  # overwritten while the trace is open
            assert table["before"] is other and table["during"] is other
        finally:
            kernel._KERNELS.pop(scan)

    def test_registering_while_a_trace_is_open_unwraps_the_shim(self):
        original = torch._C._nn.scaled_dot_product_attention
        entries = len(kernel._KERNELS)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Envoy(Mixer()).interleaver:
                kernel.register(F.scaled_dot_product_attention, kernel._KERNELS[original].serve)  # passes the installed shim
                assert len(kernel._KERNELS) == entries and kernel.unshimmed(F.scaled_dot_product_attention) is original
        assert F.scaled_dot_product_attention is original

    def test_envoy_called_with_hook_false_does_not_serve_sites(self, x):
        class Caller(nn.Module):
            """Calls a helper `Envoy` on the same interleaver with `hook=False` (the default) inside its forward pass."""

            def __init__(self, helper):
                super().__init__()
                self.helper_envoy = [helper]  # stored in a list so `helper` is not registered as a submodule
                self.proj = nn.Linear(16, 16)

            def forward(self, x):
                return self.proj(self.helper_envoy[0](x))

        interleaver = Envoy(nn.Identity()).interleaver
        helper = Envoy(Mixer(), path="helper", interleaver=interleaver)
        caller = Envoy(Caller(helper), path="caller", interleaver=interleaver)
        with pytest.raises(OutOfOrderError, match="already ran past"):
            with caller.trace(x):
                caller.kernel.probs(heads=0)

    def test_worker_kernel_call_is_not_served_as_the_model_call(self, gpt2):
        attn = gpt2.transformer.h[2].attn
        other = torch.randn(1, 3, 5, 4)
        with gpt2.trace() as tracer:
            with tracer.invoke("The quick brown fox"):
                probs = attn.kernel.probs(heads=5, query=-1).save()
            with tracer.invoke("The quick brown fox"):
                attn.c_attn.output  # resumes inside `attn.forward` while the first invoke waits on `attn.kernel`
                own = F.scaled_dot_product_attention(other, other, other).save()
        with gpt2.trace("The quick brown fox"):
            solo = attn.kernel.probs(heads=5, query=-1).save()
        assert torch.allclose(probs, solo, atol=1e-6)
        assert torch.allclose(own, F.scaled_dot_product_attention(other, other, other))

    def test_module_called_by_a_worker_with_hook_true_serves_sites(self, gpt2):
        # Module outside the model's forward pass, invoked from one worker with `hook=True` while another worker waits on its `.kernel`.
        torch.manual_seed(0)
        mixer = Mixer(width=gpt2.config.n_embd, q_heads=4, kv_heads=4).to(next(gpt2._module.parameters()).device)
        adapter = Envoy(mixer, path="adapter", interleaver=gpt2.interleaver)
        with gpt2.trace() as tracer:
            with tracer.invoke("The quick brown fox"):
                inputs = adapter.inputs.save()
                probs = adapter.kernel.probs(heads=0).save()
            with tracer.invoke("The quick brown fox"):
                adapter(gpt2.transformer.h[0].output, hook=True)  # tensor output in transformers 5
        hidden = inputs[0][0]
        _, expected, _ = _reference(*_inputs(adapter._module, hidden), is_causal=True)
        assert probs.shape == (1, hidden.shape[1], hidden.shape[1])
        assert torch.allclose(probs, expected[:, 0], atol=1e-5)

    def test_compiled_module_compiles_without_graph_breaks_and_serves_no_sites(self, x):
        from torch._dynamo.utils import counters

        mixer = Mixer()
        plain = mixer(x)
        envoy = Envoy(torch.compile(mixer))
        torch._dynamo.reset()
        counters.clear()
        with envoy.trace(x):
            out = nnsight.save(envoy.output)
        assert torch.allclose(out, plain, atol=1e-6)
        assert dict(counters["graph_break"]) == {}
        # Inside the compiled region the kernel runs without frame bookkeeping, so no site is served on either the
        # inner module or the `OptimizedModule` wrapper returned by `torch.compile`.
        for module in (envoy._orig_mod, envoy):
            with pytest.raises(OutOfOrderError, match="already ran past"):
                with envoy.trace(x):
                    module.kernel.probs(heads=0)
        with envoy.trace(x):
            out = nnsight.save(envoy.output)
        assert torch.allclose(out, plain, atol=1e-6)
        assert dict(counters["graph_break"]) == {}

    def test_eager_kernel_call_inside_compiled_module_serves_enclosing_eager_module(self, x):
        @torch._dynamo.disable
        def attend(q, k, v, causal):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        class Excluding(Mixer):
            """Compiled module whose kernel call runs inside a `@torch._dynamo.disable` helper."""

            def forward(self, x):
                q = self.split(self.to_q(x), self.q_heads)
                k, v = self.to_kv(x).chunk(2, -1)
                out = attend(q, self.split(k, self.kv_heads), self.split(v, self.kv_heads), self.causal)
                return self.merge(out.transpose(1, 2).flatten(2))

        mixer = Excluding()
        plain = mixer(x)
        envoy = Envoy(torch.compile(mixer))
        torch._dynamo.reset()
        # The compiled inner module pushes no frame, so requesting `.kernel` on `_orig_mod` is not served.
        with pytest.raises(OutOfOrderError, match="already ran past"):
            with envoy.trace(x):
                envoy._orig_mod.kernel.probs(heads=0)
        # The call is attributed to the innermost module running eagerly, which is the outer `OptimizedModule` wrapper.
        with envoy.trace(x):
            probs = nnsight.save(envoy.kernel.probs(heads=0))
            out = nnsight.save(envoy.output)
        _, expected, _ = _reference(*_inputs(mixer, x), is_causal=True)
        assert torch.allclose(probs, expected[:, 0], atol=1e-6)
        assert torch.allclose(out, plain, atol=1e-6)


class TestTransformers:
    def test_submodule_read_then_site_on_first_trace(self, gpt2):
        attn = gpt2.transformer.h[2].attn
        with gpt2.trace("The quick brown fox"):
            qkv = attn.c_attn.output.save()
            probs = attn.kernel.probs(heads=5, query=-1).save()
            logits = gpt2.output.logits.save()
        with gpt2.trace("The quick brown fox"):
            plain = gpt2.output.logits.save()
        assert qkv.shape[-1] == 3 * gpt2.config.n_embd
        assert probs.shape == (1, qkv.shape[1])
        assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-6)
        assert torch.equal(logits, plain)

    def test_iter_requests_are_served_at_their_own_step(self, gpt2):
        attn = gpt2.transformer.h[0].attn
        prompt = "The quick brown fox jumps"
        width = gpt2.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        with gpt2.generate(prompt, max_new_tokens=4, do_sample=False) as tracer:
            rows = nnsight.save([])
            for step in tracer.iter[0:4]:
                if step % 2 == 0:
                    gpt2.lm_head.output  # step without a kernel site request
                else:
                    rows.append(attn.kernel.probs(heads=0, query=-1))
        # Step n attends over the prompt plus n previously generated keys.
        assert [row.shape[-1] for row in rows] == [width + 1, width + 3]
        assert all(row.shape[:-1] == (1,) for row in rows)

    def test_source_around_the_kernel_and_a_site_in_one_trace(self, gpt2):
        attn = gpt2.transformer.h[2].attn
        with gpt2.trace("The quick brown fox"):
            inner = attn.source.attention_interface_1.source
            inputs = inner.torch_nn_functional_scaled_dot_product_attention_0.inputs.save()  # before the kernel call
            probs = attn.kernel.probs(heads=5, query=-1).save()  # during the kernel call
            out = inner.torch_nn_functional_scaled_dot_product_attention_0.output.save()
        q, k, _ = inputs[0][:3]
        _, expected, _ = _reference(q, k, is_causal=inputs[1]["is_causal"], mask=inputs[1]["attn_mask"], scale=inputs[1]["scale"])
        assert torch.allclose(probs, expected[:, 5, -1], atol=1e-6)
        assert out.shape == q.shape

    def test_source_on_shimmed_builtin_kernel_raises_source_not_available(self, gpt2):
        attn = gpt2.transformer.h[2].attn
        with pytest.raises(SourceNotAvailable, match="builtin"):
            with gpt2.trace("The quick brown fox"):
                attn.source.attention_interface_1.source.torch_nn_functional_scaled_dot_product_attention_0.source

    def test_matches_the_eager_implementation(self, gpt2, eager_gpt2):
        with eager_gpt2.trace("The quick brown fox"):
            weights = eager_gpt2.transformer.h[2].attn.output[1].save()
        with gpt2.trace("The quick brown fox"):
            probs = gpt2.transformer.h[2].attn.kernel.probs().save()
        assert torch.allclose(probs, weights, atol=1e-6)

    def test_eager_attention_serves_the_weights_it_returned(self, eager_gpt2):
        attn = eager_gpt2.transformer.h[2].attn
        with eager_gpt2.trace("The quick brown fox"):
            probs = attn.kernel.probs().save()
            row = attn.kernel.probs(heads=5, query=-1).save()
            lse = attn.kernel.lse().save()
            scores = attn.kernel.scores(heads=[1, 7]).save()
            weights = attn.output[1].save()
            logits = eager_gpt2.output.logits.save()
        with eager_gpt2.trace("The quick brown fox"):
            plain = eager_gpt2.output.logits.save()
        assert torch.equal(probs, weights) and torch.equal(row, weights[:, 5, -1])
        assert probs.provenance.kind == "exact export" and probs.provenance.kernel == "modeling_gpt2.eager_attention_forward"
        assert lse.provenance.kind == "replay" and scores.provenance.kind == "replay"
        assert torch.allclose(torch.exp(scores - lse[:, [1, 7]].unsqueeze(-1)), weights[:, [1, 7]], atol=1e-6)
        assert torch.equal(logits, plain)

    def test_eager_sites_with_source_on_the_attention_module(self, eager_gpt2):
        attn = eager_gpt2.transformer.h[2].attn
        # `.source` on the module runs its forward with a copy of the modeling module's
        # globals dict; `eager_attention_forward` is read from that copy and reaches the shim.
        with eager_gpt2.trace("The quick brown fox"):
            interface = attn.source.attention_interface_1  # instruments the forward before it runs
            probs = attn.kernel.probs(heads=0).save()  # served inside the interface call
            returned = interface.output[1].save()
        assert torch.equal(probs, returned[:, 0])
        # Drilling into the kernel function's body takes precedence on that call.
        with eager_gpt2.trace("The quick brown fox"):
            softmax = attn.source.attention_interface_1.source.nn_functional_softmax_0.output.save()
            with pytest.raises(ValueError, match="drilled with .source"):
                attn.kernel.probs()
        assert torch.equal(softmax, returned)

    def test_eager_grouped_query_scores_replay_the_returned_weights(self):
        from transformers import Qwen3NextForCausalLM

        torch.manual_seed(0)
        config = _gated_delta_config()
        config._attn_implementation = "eager"
        model = _wrap(Qwen3NextForCausalLM(config))
        attn = model.model.layers[1].self_attn  # 4 query heads over 2 key heads
        with model.trace(_ids(9, batch=2)):
            probs = attn.kernel.probs().save()
            lse = attn.kernel.lse().save()
            scores = attn.kernel.scores(heads=[1, 3]).save()
            weights = attn.output[1].save()
        assert torch.equal(probs, weights) and probs.shape == (2, 4, 9, 9)
        assert probs.provenance.kernel == "modeling_qwen3_next.eager_attention_forward"
        assert torch.allclose(torch.exp(scores - lse[:, [1, 3]].unsqueeze(-1)), weights[:, [1, 3]], atol=1e-6)

    def test_eager_softcap_serves_the_weights_but_not_a_replay(self):
        from transformers import Gemma2Config, Gemma2ForCausalLM

        torch.manual_seed(0)
        config = Gemma2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=256)
        config._attn_implementation = "eager"
        model = _wrap(Gemma2ForCausalLM(config))
        attn = model.model.layers[0].self_attn
        with model.trace(_ids(7, batch=2)):
            probs = attn.kernel.probs(heads=0).save()
            with pytest.raises(ValueError, match="received softcap=50.0"):
                attn.kernel.lse()
            weights = attn.output[1].save()
        assert torch.equal(probs, weights[:, 0]) and probs.provenance.kind == "exact export"

    def test_eager_attention_sinks_serve_the_weights_but_not_a_replay(self):
        from transformers import GptOssConfig, GptOssForCausalLM

        torch.manual_seed(0)
        config = GptOssConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            vocab_size=256, num_local_experts=2, num_experts_per_tok=1, layer_types=["full_attention"],
        )
        config._attn_implementation = "eager"
        model = _wrap(GptOssForCausalLM(config))
        attn = model.model.layers[0].self_attn
        with model.trace(_ids(7, batch=2)):
            probs = attn.kernel.probs().save()
            with pytest.raises(ValueError, match="attention sinks"):
                attn.kernel.scores()
            weights = attn.output[1].save()
        assert torch.equal(probs, weights) and probs.sum(-1).max() < 1

    def test_source_copies_are_held_only_while_they_live(self, eager_gpt2, gated_delta):
        import transformers.models.gpt2.modeling_gpt2 as modeling

        entry = kernel._KERNELS[modeling.eager_attention_forward]
        attn = eager_gpt2.transformer.h[3].attn

        def live():
            return sum(holder.alive for holder in entry.holders)

        def drill():
            with eager_gpt2.trace("The quick brown fox"):
                attn.source.attention_interface_1.source.nn_functional_softmax_0.output.save()

        drill()
        gc.collect()
        settled = live()
        for _ in range(5):
            drill()
        gc.collect()
        assert live() == settled  # the copy created by each drill expires when the trace exits
        # `record` builds a temporary function per kernel call; its holder expires after the call.
        mixer = gated_delta.model.layers[0].linear_attn
        qwen = kernel._KERNELS[sys.modules[type(mixer._module).__module__].eager_attention_forward]
        for _ in range(3):
            with gated_delta.trace(_ids(70)):
                mixer.kernel.states(tokens=-1).save()
        gc.collect()
        assert sum(holder.alive for holder in qwen.holders) == 1

    def test_a_kernel_registered_after_source_reaches_the_instrumented_copy(self, x):
        envoy = Envoy(_LateUser())
        with envoy.trace(x):
            envoy.source._late_scan_0.output.save()  # instruments the forward before `_late_scan` is registered
        kernel.register(_late_scan, _no_sites)
        try:
            with envoy.trace(x):
                with pytest.raises(ValueError, match="keeps nothing to read"):
                    envoy.kernel.anything()
        finally:
            kernel._KERNELS.pop(_late_scan)


@pytest.fixture(scope="module")
def eager_gpt2():
    from nnsight.modeling.transformers import TransformersModel

    return TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True, attn_implementation="eager")


def _wrap(module):
    from transformers import AutoTokenizer

    from nnsight.modeling.transformers import TransformersModel

    # Use GPT-2's tokenizer with "!" (token ID 0, inside the small test vocab) as
    # the pad token so invokes of different lengths can be batched.
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = "!"
    return TransformersModel(module.eval(), task="text-generation", tokenizer=tokenizer)


def _gated_delta_config():
    from transformers import Qwen3NextConfig

    return Qwen3NextConfig(
        hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        linear_num_value_heads=6, linear_num_key_heads=3, linear_key_head_dim=8, linear_value_head_dim=12,
        linear_conv_kernel_dim=4, vocab_size=256, layer_types=["linear_attention", "full_attention"],
        num_experts=0, decoder_sparse_step=0, mlp_only_layers=[0, 1], moe_intermediate_size=32,
    )


def _gated_delta_module(**to):
    """Build a two-layer Qwen3-Next model whose first layer uses gated delta rule over 6 value heads and 3 key heads (K=8, V=12)."""
    from transformers import Qwen3NextForCausalLM

    torch.manual_seed(0)
    module = Qwen3NextForCausalLM(_gated_delta_config()).to(**to)
    # Default random initialization of `A_log` produces a per-token decay near
    # 1e-5, which zeroes out the entering chunk boundary within a few tokens.
    # Setting `A_log` to `log(0.05)` gives a per-token decay near 0.97 so the
    # entering boundary state contributes noticeably to every derived state in the chunk.
    module.model.layers[0].linear_attn.A_log.data.fill_(math.log(0.05))
    return module


@pytest.fixture(scope="module")
def gated_delta():
    return _wrap(_gated_delta_module())


@pytest.fixture(scope="module")
def mamba2():
    """Build a two-layer Mamba2 model with 16-token chunks and random weights."""
    from transformers import Mamba2Config, Mamba2ForCausalLM

    config = Mamba2Config(
        hidden_size=64, num_hidden_layers=2, num_heads=4, head_dim=16, state_size=8, n_groups=1, expand=1,
        conv_kernel=4, chunk_size=16, vocab_size=256,
    )
    torch.manual_seed(0)
    return _wrap(Mamba2ForCausalLM(config))


@pytest.fixture(scope="module")
def mamba1():
    """Build a two-layer Mamba1 model (per-token selective scan) with 24 channels, state size 8, and random weights."""
    from transformers import MambaConfig, MambaForCausalLM

    config = MambaConfig(hidden_size=12, state_size=8, num_hidden_layers=2, conv_kernel=4, time_step_rank=4, vocab_size=256)
    torch.manual_seed(0)
    return _wrap(MambaForCausalLM(config))


@pytest.fixture(scope="module")
def kimi():
    """Build a two-layer Kimi Linear model whose first layer runs Kimi delta attention over 4 heads (K = V = 8)."""
    from transformers import KimiLinearConfig, KimiLinearForCausalLM

    config = KimiLinearConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, vocab_size=256,
        layer_types=["linear_attention", "full_attention"], mlp_layer_types=["dense", "dense"],
        linear_num_heads=4, linear_head_dim=8, linear_conv_kernel_dim=4, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    torch.manual_seed(0)
    module = KimiLinearForCausalLM(config)
    # Keep per-channel decay near 0.95 so the chunk boundary state remains visible across the chunk.
    module.model.layers[0].self_attn.forget_gate.A_log.data.fill_(math.log(0.05))
    return _wrap(module)


def _ids(length, batch=3):
    torch.manual_seed(1)
    return torch.randint(0, 256, (batch, length))


def _hf_state_after(inputs, token):
    """Run transformers' per-token gated delta reference over the first ``token + 1`` tokens and return its final state."""
    import transformers.models.qwen3_next.modeling_qwen3_next as modeling

    args, kwargs = inputs
    query, key, value, g, beta = (t[:, : token + 1] for t in (*args[:3], kwargs["g"], kwargs["beta"]))
    _, final = modeling.torch_recurrent_gated_delta_rule.__wrapped__(
        query, key, value, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=kwargs["use_qk_l2norm_in_kernel"]
    )
    return final


def _late_scan(t):
    return t.cumsum(1)


class _LateUser(nn.Module):
    """Module whose forward calls `_late_scan` as a bare global of this test module."""

    def forward(self, t):
        return _late_scan(t)


def _no_sites(fn, args, kwargs):
    return fn(*args, **kwargs), sites.Unavailable("this call keeps nothing to read")


def _plain_scan(x):
    return x.cumsum(1)


kernel.register(_plain_scan, _no_sites)


class Scan(nn.Module):
    def forward(self, x):
        return _plain_scan(x)


class TestRecurrent:
    def test_sites_are_served_when_the_kernels_package_wraps_the_reference(self, monkeypatch):
        import transformers.models.qwen3_next.modeling_qwen3_next as modeling

        def module_around(func):
            """What ``kernels.use_kernel_forward_from_hub`` returns for a function: a module whose forward closes over it as ``func``."""

            class Func(nn.Module):
                def forward(self, *args, **kwargs):
                    return func(*args, **kwargs)

            return Func()

        monkeypatch.setattr(modeling, "torch_chunk_gated_delta_rule", module_around(modeling.torch_chunk_gated_delta_rule))
        model = _wrap(_gated_delta_module())
        with model.trace(_ids(150)):
            boundaries = model.model.layers[0].linear_attn.kernel.chunk_states().save()
        assert boundaries.shape == (3, 4, 6, 8, 12)
        assert boundaries.provenance.kind == "exact export"

    def test_gated_delta_chunk_states_match_the_per_token_reference(self, gated_delta):
        mixer = gated_delta.model.layers[0].linear_attn
        ids = _ids(150)  # two full 64-token chunks followed by a partial chunk
        with gated_delta.trace(ids):
            plain = gated_delta.output.logits.save()
        with gated_delta.trace(ids):
            inputs = mixer.source.torch_chunk_gated_delta_rule_0.inputs.save()
            boundaries = mixer.kernel.chunk_states().save()
            logits = gated_delta.output.logits.save()
        assert boundaries.shape == (3, 4, 6, 8, 12)
        assert boundaries.provenance.kind == "exact export"
        assert boundaries.dtype is torch.float32
        assert torch.equal(plain, logits)
        assert torch.equal(boundaries[:, 0], torch.zeros_like(boundaries[:, 0]))
        for index, token in ((1, 63), (2, 127), (3, 149)):
            assert torch.allclose(boundaries[:, index], _hf_state_after(inputs, token), atol=1e-6)

    def test_gated_delta_states_inside_a_chunk_are_derived(self, gated_delta):
        mixer = gated_delta.model.layers[0].linear_attn
        ids = _ids(150)
        with gated_delta.trace(ids):
            inputs = mixer.source.torch_chunk_gated_delta_rule_0.inputs.save()
            states = mixer.kernel.states(tokens=[5, 63, 100], heads=0).save()
            last = mixer.kernel.states(tokens=-1).save()
            none = mixer.kernel.states(tokens=[]).save()
        assert states.shape == (3, 3, 8, 12)
        assert states.provenance.kind == "derived"
        assert "relative" in states.provenance.note
        assert last.shape == (3, 6, 8, 12)
        assert last.provenance.kind == "exact export"
        assert none.shape == (3, 0, 6, 8, 12)
        for i, token in enumerate([5, 63, 100]):
            assert torch.allclose(states[:, i], _hf_state_after(inputs, token)[:, 0], atol=1e-6)

    def test_decoding_states_come_from_the_per_token_kernel(self, gated_delta):
        mixer = gated_delta.model.layers[0].linear_attn
        ids = _ids(70, batch=1)
        with gated_delta.generate(ids, max_new_tokens=3, do_sample=False) as tracer:
            steps = nnsight.save([])
            for _ in tracer.iter[0:3]:
                steps.append(mixer.kernel.states(tokens=-1))
            generated = tracer.result.save()
        assert [step.provenance.kernel for step in steps] == [
            "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "torch_recurrent_gated_delta_rule"
        ]
        assert all(step.provenance.kind == "exact export" for step in steps)
        # The state after decode step 1 matches the final state of a 71-token prefill over the same tokens.
        with gated_delta.trace(generated[:, :71]):
            final = mixer.kernel.chunk_states(chunks=-1).save()
        assert torch.allclose(steps[1], final, atol=1e-6)

    def test_invokes_of_different_lengths_read_their_own_last_state(self, gated_delta):
        mixer = gated_delta.model.layers[0].linear_attn
        long, short = _ids(100, batch=1), _ids(70, batch=1)
        with gated_delta.trace() as tracer:
            with tracer.invoke(long):
                a = mixer.kernel.states(tokens=-1).save()
            with tracer.invoke(short):
                b = mixer.kernel.states(tokens=-1).save()
        with gated_delta.trace(long):
            a_alone = mixer.kernel.chunk_states(chunks=-1).save()
        with gated_delta.trace(short):
            b_alone = mixer.kernel.chunk_states(chunks=-1).save()
        assert torch.allclose(a, a_alone, atol=1e-6)
        assert torch.allclose(b, b_alone, atol=1e-6)

    def test_source_still_drills_into_a_registered_kernel(self, gated_delta):
        mixer = gated_delta.model.layers[0].linear_attn
        with gated_delta.trace(_ids(150)) as tracer:
            loop = mixer.source.torch_chunk_gated_delta_rule_0.source.implementation_0.source
            states = nnsight.save([])
            for _ in tracer.iter[:3]:
                states.append(loop.last_recurrent_state_2.output)
        assert [s.shape for s in states] == [(3, 6, 8, 12)] * 3
        with gated_delta.trace(_ids(150)):
            boundaries = mixer.kernel.chunk_states(chunks=[1, 2, 3]).save()
        assert torch.allclose(torch.stack(states, 1), boundaries, atol=1e-6)

    def test_source_into_kernel_body_and_site_on_same_call_raises(self, gated_delta):
        mixer = gated_delta.model.layers[0].linear_attn
        with gated_delta.trace(_ids(150)):
            loop = mixer.source.torch_chunk_gated_delta_rule_0.source.implementation_0.source
            state = loop.last_recurrent_state_2.output.save()  # read an operation inside the kernel's body
            with pytest.raises(ValueError, match="drilled with .source"):
                mixer.kernel.chunk_states(chunks=-1)
        assert state.shape == (3, 6, 8, 12)

    def test_model_built_while_a_trace_is_open_unwraps_the_shim(self):
        import transformers.models.qwen3_next.modeling_qwen3_next as modeling

        from nnsight.modeling.kernels import recurrent
        from nnsight.modeling.kernels.transformers import _hub_wrappers

        cell = next(cells for cells in _hub_wrappers(modeling) if cells["func_name"].cell_contents == "chunk_gated_delta_rule")["implementation"]
        torch_function = cell.cell_contents
        with Envoy(Mixer()).interleaver:
            model = _wrap(_gated_delta_module())  # the catalog is registered while the cell contains the shim
        assert cell.cell_contents is torch_function
        assert kernel._KERNELS[torch_function].serve is recurrent.serve_gated_delta_chunk
        mixer = model.model.layers[0].linear_attn
        with model.trace(_ids(150)):
            states = mixer.kernel.chunk_states(chunks=-1).save()
        assert states.shape == (3, 6, 8, 12)

    def test_mamba2_states_match_an_fp64_scan(self, mamba2):
        mixer = mamba2.backbone.layers[0].mixer
        ids = _ids(100)
        with mamba2.trace(ids):
            plain = mamba2.output.logits.save()
        with mamba2.trace(ids):
            inputs = mixer.source.mamba2_chunk_scan_0.inputs.save()
            boundaries = mixer.kernel.chunk_states(chunks=[1, -1]).save()
            states = mixer.kernel.states(tokens=[3, 15, 50], heads=2).save()
            logits = mamba2.output.logits.save()
        assert torch.equal(plain, logits)
        assert boundaries.shape == (3, 2, 4, 16, 8)
        assert states.shape == (3, 3, 16, 8)
        args, kwargs = inputs
        x, dt, A, B, _ = (value.double() for value in args[:5])
        dt = torch.nn.functional.softplus(dt + kwargs["dt_bias"].double()).clamp(*kwargs["dt_limit"])
        B = B.repeat_interleave(x.shape[2] // B.shape[2], 2)
        state = torch.zeros(3, 4, 16, 8, dtype=torch.float64, device=x.device)
        after = {}
        for t in range(100):
            state = state * torch.exp(dt[:, t] * A)[..., None, None] + (x[:, t] * dt[:, t, :, None]).unsqueeze(-1) * B[:, t].unsqueeze(-2)
            after[t] = state.clone()
        assert torch.allclose(boundaries[:, 0].double(), after[15], atol=1e-7)
        assert torch.allclose(boundaries[:, 1].double(), after[99], atol=1e-7)
        for i, token in enumerate([3, 15, 50]):
            assert torch.allclose(states[:, i].double(), after[token][:, 2], atol=1e-7)

    def test_mamba2_decoding_states_come_from_the_cache_update(self, mamba2):
        mixer = mamba2.backbone.layers[0].mixer
        ids = _ids(70, batch=1)
        with mamba2.generate(ids, max_new_tokens=3, do_sample=False) as tracer:
            steps = nnsight.save([])
            for _ in tracer.iter[0:3]:
                steps.append(mixer.kernel.states(tokens=-1))
            generated = tracer.result.save()
        assert [step.provenance.kernel for step in steps] == [
            "mamba2_chunk_scan", "mamba2_selective_state_update", "mamba2_selective_state_update"
        ]
        assert all(step.provenance.kind == "exact export" for step in steps)
        with mamba2.trace(generated[:, :71]):
            final = mixer.kernel.chunk_states(chunks=-1).save()
        assert torch.allclose(steps[1], final, atol=1e-6)

    def test_a_scan_serves_shapes(self, mamba2):
        mixer = mamba2.backbone.layers[0].mixer
        with mamba2.scan(_ids(37)):
            boundaries = mixer.kernel.chunk_states(chunks=slice(1, 3)).save()
            states = mixer.kernel.states(tokens=[0, 20, 36], heads=1).save()
        assert boundaries.shape == (3, 2, 4, 16, 8)
        assert states.shape == (3, 3, 16, 8)
        assert states.provenance.kind == "derived"
        with pytest.raises(TypeError, match="created under scan"):
            with mamba2.scan(_ids(37)):
                mixer.kernel.states(tokens=torch.tensor([0, 20]))

    def test_mamba1_states_are_the_loop_states_and_match_an_fp64_scan(self, mamba1):
        mixer = mamba1.backbone.layers[0].mixer
        ids = _ids(37)
        with mamba1.trace(ids):
            plain = mamba1.output.logits.save()
        with mamba1.trace(ids):
            inputs = mixer.source.mamba_selective_scan_0.inputs.save()
            states = mixer.kernel.states(tokens=[0, 20, 36], heads=slice(0, 5)).save()
            every = mixer.kernel.chunk_states().save()
            logits = mamba1.output.logits.save()
        assert torch.equal(plain, logits)
        assert states.shape == (3, 3, 5, 8)
        assert states.provenance.kind == "exact export"
        assert states.provenance.kernel == "mamba_selective_scan"
        assert every.shape == (3, 38, 24, 8)
        args, kwargs = inputs
        x, dt, A, B, _ = (value.double() for value in args[:5])
        dt = torch.nn.functional.softplus(dt + kwargs["delta_bias"].double()[:, None])
        state = torch.zeros(3, 24, 8, dtype=torch.float64, device=x.device)
        after = {}
        for t in range(37):
            state = torch.exp(A[None] * dt[:, :, t, None]) * state + dt[:, :, t, None] * B[:, None, :, t] * x[:, :, t, None]
            after[t] = state.clone()
        for i, token in enumerate([0, 20, 36]):
            assert torch.allclose(states[:, i].double(), after[token][:, :5], atol=1e-6)
        assert torch.allclose(every[:, 37].double(), after[36], atol=1e-6)

    def test_mamba1_decoding_states_come_from_the_cache_update(self, mamba1):
        mixer = mamba1.backbone.layers[0].mixer
        ids = _ids(20, batch=1)
        with mamba1.generate(ids, max_new_tokens=3, do_sample=False) as tracer:
            steps = nnsight.save([])
            for _ in tracer.iter[0:3]:
                steps.append(mixer.kernel.states(tokens=-1))
            generated = tracer.result.save()
        assert [step.provenance.kernel for step in steps] == [
            "mamba_selective_scan", "mamba_selective_state_update", "mamba_selective_state_update"
        ]
        with mamba1.trace(generated[:, :21]):
            final = mixer.kernel.states(tokens=-1).save()
        assert torch.allclose(steps[1], final, atol=1e-6)

    def test_kimi_delta_states_inside_a_chunk_match_the_per_token_reference(self, kimi):
        import transformers.models.kimi_linear.modeling_kimi_linear as modeling

        mixer = kimi.model.layers[0].self_attn
        ids = _ids(150)  # two full 64-token chunks followed by a partial chunk
        with kimi.trace(ids):
            plain = kimi.output.logits.save()
        with kimi.trace(ids):
            inputs = mixer.source.chunk_kimi_delta_attention_0.inputs.save()
            boundaries = mixer.kernel.chunk_states().save()
            states = mixer.kernel.states(tokens=[5, 63, 100], heads=1).save()
            logits = kimi.output.logits.save()
        assert torch.equal(plain, logits)
        assert boundaries.shape == (3, 4, 4, 8, 8)
        assert states.shape == (3, 3, 8, 8)
        assert states.provenance.kind == "derived"
        assert states.provenance.kernel == "chunk_kimi_delta_attention"
        args, kwargs = inputs
        query, key, value = args[:3]

        def reference(token):
            _, final = modeling.recurrent_kimi_delta_attention.__wrapped__(
                query[:, : token + 1], key[:, : token + 1], value[:, : token + 1], kwargs["g"][:, : token + 1],
                kwargs["beta"][:, : token + 1], None, True, use_qk_l2norm_in_kernel=kwargs["use_qk_l2norm_in_kernel"],
            )
            return final

        for index, token in ((1, 63), (2, 127), (3, 149)):
            assert torch.allclose(boundaries[:, index], reference(token), atol=1e-6)
        for i, token in enumerate([5, 63, 100]):
            assert torch.allclose(states[:, i], reference(token)[:, 1], atol=1e-6)

    def test_kimi_decoding_states_come_from_the_per_token_kernel(self, kimi):
        mixer = kimi.model.layers[0].self_attn
        ids = _ids(70, batch=1)
        with kimi.generate(ids, max_new_tokens=3, do_sample=False) as tracer:
            steps = nnsight.save([])
            for _ in tracer.iter[0:3]:
                steps.append(mixer.kernel.states(tokens=-1))
            generated = tracer.result.save()
        assert [step.provenance.kernel for step in steps] == [
            "chunk_kimi_delta_attention", "recurrent_kimi_delta_attention", "recurrent_kimi_delta_attention"
        ]
        with kimi.trace(generated[:, :71]):
            final = mixer.kernel.states(tokens=-1).save()
        assert torch.allclose(steps[1], final, atol=1e-6)

    def test_unknown_spec_key_names_the_site_axes(self, mamba2):
        mixer = mamba2.backbone.layers[0].mixer
        with pytest.raises(ValueError, match="takes chunks, heads"):
            with mamba2.trace(_ids(20)):
                mixer.kernel.chunk_states(tokens=0)

    def test_a_call_without_sites_raises_its_reason_at_the_request(self):
        envoy = Envoy(Scan())
        with pytest.raises(ValueError, match="keeps nothing to read"):
            with envoy.trace(torch.ones(2, 5)):
                envoy.kernel.states()


@pytest.fixture(scope="module", params=["grouped_mm", "batched_mm"])
def moe(request):
    """Build a one-layer Qwen2-MoE model (5 experts, top-2 routing) parameterized over `grouped_mm` and `batched_mm`."""
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    config = Qwen2MoeConfig(
        hidden_size=32, intermediate_size=64, moe_intermediate_size=24, shared_expert_intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, num_experts=5, num_experts_per_tok=2,
        vocab_size=256, decoder_sparse_step=1,
    )
    torch.manual_seed(0)
    module = Qwen2MoeForCausalLM(config)
    module.config._experts_implementation = request.param
    return _wrap(module)


class Pool(nn.Module):
    """Minimal ungated MoE experts module (`act(up @ x)` followed by `down`) with 3 experts, hidden size 8, and intermediate size 12."""

    def __init__(self, config):
        super().__init__()
        # Row dimensions are multiples of 16 bytes as required by PyTorch's grouped matmul.
        self.num_experts, self.hidden_dim, self.intermediate_dim = 3, 8, 12
        self.up_proj = nn.Parameter(torch.randn(3, 12, 8))
        self.down_proj = nn.Parameter(torch.randn(3, 8, 12))
        self.act_fn = F.gelu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        raise NotImplementedError("the interface dispatches to a fused forward")


def _pool():
    from types import SimpleNamespace

    from transformers.integrations import moe
    from transformers.integrations.moe import use_experts_implementation

    from nnsight.modeling.kernels import experts

    # Register `grouped_mm_experts_forward` for standalone `Envoy` tests that do not construct a `TransformersModel`;
    # `use_experts_implementation` looks up the implementation in `ALL_EXPERTS_FUNCTIONS`.
    kernel.register(
        moe.grouped_mm_experts_forward, experts.serve_grouped_mm, kernel.Item(moe.ALL_EXPERTS_FUNCTIONS._global_mapping, "grouped_mm")
    )
    torch.manual_seed(2)
    return Envoy(use_experts_implementation(Pool, has_gate=False)(SimpleNamespace(_experts_implementation="grouped_mm")))


def _per_pair(experts_module, x, index):
    """Compute each `(token, slot)` pair's intermediate activation and down projection sequentially in fp64 on CPU."""
    x, index = x.double().cpu(), index.cpu()
    gate_up, down = experts_module.gate_up_proj.double().cpu(), experts_module.down_proj.double().cpu()
    act = torch.zeros(*index.shape, experts_module.intermediate_dim, dtype=torch.float64)
    out = torch.zeros(*index.shape, experts_module.hidden_dim, dtype=torch.float64)
    for t in range(index.shape[0]):
        for k in range(index.shape[1]):
            e = index[t, k]
            gate, up = F.linear(x[t], gate_up[e]).chunk(2, -1)
            act[t, k] = experts_module.act_fn(gate) * up
            out[t, k] = F.linear(act[t, k], down[e])
    return act, out


def _dense_flash(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False):
    """Stand-in with FlashAttention 2's dense signature and return convention, computed in fp32 with the mask aligned bottom right."""
    out, lse = _flash_math(q, k, v, softmax_scale, causal)
    return (out, lse, None) if return_attn_probs else out


def _flash_math(q, k, v, softmax_scale, causal):
    """The stand-ins' shared computation, kept out of the kernel registry so a packed call does not shim its per-span calls."""
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    keys = k.float().repeat_interleave(q.shape[2] // k.shape[2], 2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), keys) * scale
    if causal:
        rows, cols = torch.arange(q.shape[1], device=q.device)[:, None], torch.arange(k.shape[1], device=q.device)[None, :]
        scores = scores.masked_fill(cols > rows + (k.shape[1] - q.shape[1]), float("-inf"))
    values = v.float().repeat_interleave(q.shape[2] // v.shape[2], 2)
    out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(scores, -1), values).to(q.dtype)
    return out, torch.logsumexp(scores, -1)


def _packed_flash(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False, block_table=None):
    """Stand-in with FlashAttention 2's varlen signature: one dense call per span, rows concatenated and lse as ``[heads, rows]``."""
    outs, lses = [], []
    for i in range(len(cu_seqlens_q) - 1):
        rows, cols = slice(cu_seqlens_q[i], cu_seqlens_q[i + 1]), slice(cu_seqlens_k[i], cu_seqlens_k[i + 1])
        out, lse = _flash_math(q[None, rows], k[None, cols], v[None, cols], softmax_scale, causal)
        outs.append(out[0])
        lses.append(lse[0])
    out = torch.cat(outs)
    return (out, torch.cat(lses, -1), None) if return_attn_probs else out


def _batch(total, start, size):
    """A `sites.Batch` narrowing as the base batcher does: rows ``[start, start + size)`` of a dim 0 equal to ``total``, else every row."""
    return sites.Batch(total, start, size, lambda rows: rows[start : start + size] if rows.shape[0] == total else rows)


@pytest.fixture
def stand_in_flash(monkeypatch):
    """Make transformers load the CPU stand-ins as its FlashAttention 2 functions, restoring its globals afterwards."""
    from transformers import modeling_flash_attention_utils as utils

    for name in ("_loaded_implementation", "_flash_fn", "_flash_varlen_fn", "_flash_with_kvcache_fn", "_pad_fn", "_unpad_fn", "_process_flash_kwargs_fn", "_process_with_kvcache_kwargs_fn"):
        monkeypatch.setattr(utils, name, vars(utils)[name])
    monkeypatch.setattr(utils, "_lazy_imports", lambda *args, **kwargs: (_dense_flash, _packed_flash, None, utils._pad_input, utils._unpad_input))
    return utils


def _llama(implementation):
    """Build a two-layer Llama (4 query heads, 2 key-value heads) with the same weights for every ``implementation``."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    torch.manual_seed(0)
    module = LlamaForCausalLM(config)
    # Set after construction: constructing under flash_attention_2 checks for the package and a GPU.
    module.config._attn_implementation = implementation
    return module


class TestFlash:
    @pytest.mark.parametrize("before", ["switched after wrapping", "another implementation loaded last"])
    def test_sites_follow_the_implementation_loaded_when_the_trace_starts(self, stand_in_flash, before):
        if before == "switched after wrapping":
            model = _wrap(_llama("sdpa"))
            model._module.config._attn_implementation = "flash_attention_2"
        else:
            other = lambda *args, **kwargs: None  # noqa: E731
            stand_in_flash._loaded_implementation, stand_in_flash._flash_fn, stand_in_flash._flash_varlen_fn = "flash_attention_3", other, other
            model = _wrap(_llama("flash_attention_2"))
        reference = _wrap(_llama("sdpa"))
        ids = _ids(6, batch=1).clamp(min=1)
        with model.trace(ids):
            probs = model.model.layers[0].self_attn.kernel.probs().save()
        with reference.trace(ids):
            expected = reference.model.layers[0].self_attn.kernel.probs().save()
        assert probs.provenance.kernel == "_dense_flash"
        assert torch.allclose(probs, expected, atol=1e-5)

    def test_beam_search_gives_each_invoke_every_beam_as_sdpa_does(self, stand_in_flash):
        first, second = _ids(6, batch=2).clamp(min=1).chunk(2)

        def read(model):
            saved = []
            with model.generate(max_new_tokens=1, num_beams=2, pad_token_id=0, eos_token_id=None) as tracer:
                for prompt in (first, second):
                    with tracer.invoke(prompt):
                        saved.append(model.model.layers[0].self_attn.kernel.probs().save())
            return saved

        # generate expands the two prompts into four beam rows, which nnsight does not attribute to invokes.
        for got, expected in zip(read(_wrap(_llama("flash_attention_2"))), read(_wrap(_llama("sdpa")))):
            assert got.shape == expected.shape == (4, 4, 6, 6)
            assert torch.allclose(got, expected, atol=1e-5)

    def test_sequences_packed_into_one_row_are_served_separately(self, stand_in_flash):
        model, reference = _wrap(_llama("flash_attention_2")), _wrap(_llama("sdpa"))
        ids = _ids(8, batch=1).clamp(min=1)
        # position_ids restarting at 0 pack two 4-token sequences into one row.
        with model.trace({"input_ids": ids, "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])}):
            probs = model.model.layers[0].self_attn.kernel.probs().save()
        assert probs.provenance.kernel == "_packed_flash"
        assert probs.shape == (2, 4, 4, 4)
        for index, sequence in enumerate(ids.chunk(2, dim=1)):
            with reference.trace(sequence):
                expected = reference.model.layers[0].self_attn.kernel.probs().save()
            assert torch.allclose(probs[index], expected[0], atol=1e-5)

    def test_the_padded_varlen_logsumexp_of_flash_attn_before_2_6_is_refused(self):
        from nnsight.modeling.kernels import flash

        def padded_flash(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False, block_table=None):
            out, lse, _ = _packed_flash(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=softmax_scale, causal=causal, return_attn_probs=True)
            padded = lse.new_zeros(len(cu_seqlens_q) - 1, lse.shape[0], max_seqlen_q)  # flash_attn 2.3.3 to 2.5.9: [batch, heads, max_seqlen_q]
            return (out, padded, None) if return_attn_probs else out

        torch.manual_seed(0)
        q, k, v = torch.randn(8, 2, 4), torch.randn(8, 2, 4), torch.randn(8, 2, 4)
        spans = torch.tensor([0, 5, 8])
        output, reader = flash.serve(padded_flash, (q, k, v, spans, spans, 5, 5), {"causal": True})
        assert torch.equal(output, _packed_flash(q, k, v, spans, spans, 5, 5, causal=True))
        with pytest.raises(ValueError, match="flash_attn 2.6.0 or later"):
            reader.read("probs", {}, None)

    def test_dense_call_serves_bottom_right_causal_probs_per_prompt(self):
        from nnsight.modeling.kernels import flash

        torch.manual_seed(0)
        q, k, v = torch.randn(2, 3, 4, 8), torch.randn(2, 7, 2, 8), torch.randn(2, 7, 2, 8)
        output, reader = flash.serve(_dense_flash, (q, k, v), {"causal": True})
        probs, provenance = reader.read("probs", {}, _batch(2, 1, 1))
        _, lse, _ = _dense_flash(q, k, v, causal=True, return_attn_probs=True)
        keys = k.repeat_interleave(2, 2)
        expected = torch.softmax(torch.einsum("bqhd,bkhd->bhqk", q, keys) * 8**-0.5, -1)
        assert torch.equal(output, _dense_flash(q, k, v, causal=True))
        assert probs.shape == (1, 4, 3, 7)
        assert (probs[0, :, 0, 5:] == 0).all() and (probs[0, :, 0, :5] > 0).all()  # row 0 of 3 sees keys 0..4 of 7
        assert torch.allclose(probs.sum(-1), torch.ones(1, 4, 3), atol=1e-5)
        assert torch.allclose(probs[0, :, -1], expected[1, :, -1], atol=1e-6)  # the last row is unmasked
        assert provenance.kind == "exact reconstruction" and provenance.kernel == "_dense_flash"
        assert torch.equal(reader.read("lse", {"heads": 3}, _batch(2, 0, 2))[0], lse[:, 3])

    def test_packed_call_serves_each_prompt_without_padding_rows(self):
        from nnsight.modeling.kernels import flash

        torch.manual_seed(0)
        q, k, v = torch.randn(8, 2, 4), torch.randn(8, 2, 4), torch.randn(8, 2, 4)
        spans = torch.tensor([0, 5, 8])
        output, reader = flash.serve(_packed_flash, (q, k, v, spans, spans, 5, 5), {"causal": True})
        short, _ = reader.read("probs", {}, _batch(2, 1, 1))
        expected, _, _ = _reference(q[5:].transpose(0, 1)[None], k[5:].transpose(0, 1)[None], is_causal=True)
        assert torch.equal(output, _packed_flash(q, k, v, spans, spans, 5, 5, causal=True))
        assert short.shape == (1, 2, 3, 3)
        assert torch.allclose(short, torch.softmax(expected, -1), atol=1e-6)
        assert reader.read("scores", {"query": -1, "key": 0}, _batch(2, 0, 1))[0].shape == (1, 2)
        with pytest.raises(ValueError, match="different lengths"):
            reader.read("probs", {}, _batch(2, 0, 2))

    def test_an_argument_outside_the_modelled_set_makes_the_sites_unavailable(self):
        from nnsight.modeling.kernels import flash

        q, k, v = torch.randn(1, 4, 2, 4), torch.randn(1, 4, 2, 4), torch.randn(1, 4, 2, 4)
        output, reader = flash.serve(_dense_flash, (q, k, v), {"softcap": 30.0})
        assert torch.equal(output, _dense_flash(q, k, v))
        with pytest.raises(ValueError, match="sets softcap"):
            reader.read("probs", {}, None)


class TestExperts:
    def test_pair_buffers_match_a_per_pair_computation(self, moe):
        experts = moe.model.layers[0].mlp.experts
        ids = _ids(7)
        with moe.trace(ids):
            plain = moe.output.logits.save()
        with moe.trace(ids):
            inputs = experts.inputs.save()
            inp = experts.kernel.expert_input().save()
            act = experts.kernel.expert_activation().save()
            out = experts.kernel.expert_output().save()
            routed = experts.output.save()
            logits = moe.output.logits.save()
        assert torch.equal(plain, logits)
        assert inp.shape == (3, 7, 2, 32) and act.shape == (3, 7, 2, 24) and out.shape == (3, 7, 2, 32)
        assert out.provenance.kind == "exact export"
        assert out.provenance.kernel == f"{moe._module.config._experts_implementation}_experts_forward"
        x, index, weights = inputs[0]
        assert torch.equal(inp.reshape(21, 2, 32), x.unsqueeze(1).expand(-1, 2, -1))
        ref_act, ref_out = _per_pair(moe._module.model.layers[0].mlp.experts, x, index)
        assert torch.allclose(act.reshape(21, 2, 24).double().cpu(), ref_act, rtol=1e-5, atol=1e-7)
        assert torch.allclose(out.reshape(21, 2, 32).double().cpu(), ref_out, rtol=1e-5, atol=1e-7)
        # Summing the down-projected rows weighted by `top_k_weights` reproduces `experts.output`.
        assert torch.allclose((out.reshape(21, 2, 32) * weights.unsqueeze(-1)).sum(1), routed, atol=1e-6)

    def test_expert_and_slot_selections(self, moe):
        experts = moe.model.layers[0].mlp.experts
        ids = _ids(7)
        with moe.trace(ids):
            index = experts.inputs[0][1].save()
            whole = experts.kernel.expert_output().save()
            third = experts.kernel.expert_output(expert=3).save()
            last_expert = experts.kernel.expert_output(expert=-1).save()
            last_slot1 = experts.kernel.expert_activation(tokens=-1, slot=1).save()
            tail = experts.kernel.expert_input(tokens=slice(-2, None), slot=0).save()
            none = experts.kernel.expert_output(tokens=[]).save()
        mask = (index.view(3, 7, 2) == 3).unsqueeze(-1)
        assert mask.any()
        assert torch.equal(third, (whole * mask).sum(2))
        assert torch.equal(last_expert, (whole * (index.view(3, 7, 2) == 4).unsqueeze(-1)).sum(2))
        assert third.shape == (3, 7, 32) and last_slot1.shape == (3, 24) and tail.shape == (3, 2, 32)
        assert none.shape == (3, 0, 2, 32)
        assert "0 for a token not routed" in third.provenance.note

    def test_expert_index_is_one_int_in_range(self, moe):
        experts = moe.model.layers[0].mlp.experts
        with pytest.raises(TypeError, match="one expert index"):
            with moe.trace(_ids(3)):
                experts.kernel.expert_output(expert=[0, 1])
        with pytest.raises(IndexError, match="out of range for 5 experts"):
            with moe.trace(_ids(3)):
                experts.kernel.expert_output(expert=5)
        with pytest.raises(ValueError, match="slot= or expert=, not both"):
            with moe.trace(_ids(3)):
                experts.kernel.expert_output(expert=0, slot=0)

    def test_experts_without_a_gate_serve_their_activation(self):
        pool = _pool()
        x, index, weights = torch.randn(6, 8), torch.tensor([[0, 2], [2, 1], [1, 0], [0, 1], [2, 0], [1, 2]]), torch.rand(6, 2)
        with pool.trace(x, index, weights):
            act = pool.kernel.expert_activation().save()
            out = pool.kernel.expert_output(expert=2).save()
            routed = pool.output.save()
        # A standalone `Envoy` treats its entire input as a single batch row.
        assert act.shape == (1, 6, 2, 12) and out.shape == (1, 6, 8)
        act, out = act[0], out[0]
        module = pool._module
        for t in range(6):
            for k in range(2):
                e = index[t, k]
                reference = F.gelu(module.up_proj[e].double() @ x[t].double())
                assert torch.allclose(act[t, k].double(), reference, rtol=1e-5, atol=1e-5)
                if e == 2:
                    assert torch.allclose(out[t].double(), module.down_proj[e].double() @ reference, rtol=1e-5, atol=1e-5)
        assert torch.allclose(routed, (torch.stack([module.down_proj[index[t, k]] @ act[t, k] for t in range(6) for k in range(2)]).view(6, 2, 8) * weights.unsqueeze(-1)).sum(1), atol=1e-6)

    def test_parallel_experts_are_refused_at_the_request(self):
        pool = _pool()
        pool._module._is_expert_parallel = True
        with pytest.raises(ValueError, match="tensor or expert parallelism"):
            with pool.trace(torch.randn(4, 8), torch.tensor([[0, 1]] * 4), torch.rand(4, 2)):
                pool.kernel.expert_output()

    def test_a_scan_serves_shapes(self, moe):
        if moe._module.config._experts_implementation == "grouped_mm":
            pytest.skip("torch's grouped matmul has no fake-tensor kernel for fp32 inputs; the scan fails before any site")
        experts = moe.model.layers[0].mlp.experts
        with moe.scan(_ids(5)):
            out = experts.kernel.expert_output(expert=1).save()
            whole = experts.kernel.expert_activation().save()
        assert out.shape == (3, 5, 32) and whole.shape == (3, 5, 2, 24)
        with moe.scan(_ids(5)):
            logits = moe.output.logits.save()
        assert logits.shape == (3, 5, 256)

    def test_invokes_read_their_own_rows_and_decode_steps_are_served(self, moe):
        experts = moe.model.layers[0].mlp.experts
        short, long = _ids(4, batch=1), _ids(7, batch=2)
        with moe.trace() as tracer:
            with tracer.invoke(short):
                a = experts.kernel.expert_output().save()
            with tracer.invoke(long):
                b = experts.kernel.expert_output().save()
        with moe.trace(long):
            b_alone = experts.kernel.expert_output().save()
        with moe.trace(short):
            a_alone = experts.kernel.expert_output().save()
        assert a.shape == (1, 7, 2, 32) and b.shape == (2, 7, 2, 32)
        # Because the short invoke is left-padded to length 7, its 4 real tokens sit at the end of the sequence axis.
        assert torch.allclose(b, b_alone, atol=1e-6)
        assert torch.allclose(a[:, -4:], a_alone, atol=1e-6)
        with moe.generate(long, max_new_tokens=2, min_new_tokens=2, do_sample=False) as tracer:
            steps = nnsight.save([])
            for _ in tracer.iter[0:2]:
                steps.append(experts.kernel.expert_output(expert=0))
        assert [step.shape for step in steps] == [(2, 7, 32), (2, 1, 32)]


class TestWorkloads:
    """End-to-end interpretability workloads executed through `.kernel` sites."""

    def test_induction_heads_score_high_on_a_repeated_sequence(self, gpt2):
        torch.manual_seed(0)
        half = torch.randint(1000, 20000, (1, 24))
        with gpt2.trace(torch.cat([half, half], 1)):
            rows = nnsight.save([])
            for block in gpt2.transformer.h:
                rows.append(block.attn.kernel.probs(query=slice(24, None)))  # [1, heads, 24, 48]
        # Query `24 + i` is the second copy of token `i`; an induction head attends to key `i + 1`.
        queries, keys = torch.arange(24), torch.arange(24) + 1
        scores = torch.stack([row[0][:, queries, keys].mean(-1) for row in rows])  # [layers, heads]
        assert scores.shape == (12, 12)
        assert scores[5, 5] > 0.9 and scores[7, 10] > 0.8  # known induction heads in GPT-2
        assert scores[:2].max() < 0.05  # no induction heads in the first two layers

    def test_attention_sink_mass_sits_on_the_first_key(self, gpt2):
        prompt = "The quick brown fox jumps over the lazy dog because it wanted to reach the other side of the river"
        with gpt2.trace(prompt):
            sinks = nnsight.save([])
            for i, block in enumerate(gpt2.transformer.h):
                sinks.append(block.attn.kernel.probs(key=0))  # [1, heads, queries]
                if i == 10:
                    full = block.attn.kernel.probs().save()
        mass = torch.stack([sink[0, :, 1:].mean(-1) for sink in sinks])  # [layers, heads], queries after position 0
        assert torch.allclose(sinks[10][0], full[0, :, :, 0], atol=1e-6)
        assert (mass[10] > 0.5).all()
        assert mass[0].mean() < 0.3

    def test_a_retrieval_head_copies_during_generation(self, gpt2):
        torch.manual_seed(0)
        half = torch.randint(1000, 20000, (1, 12))
        ids = torch.cat([half, half[:, :3]], 1)  # prompt ends with the start of the second copy
        attn = gpt2.transformer.h[5].attn
        with gpt2.generate(ids, max_new_tokens=6, do_sample=False) as tracer:
            rows = nnsight.save([])
            for _ in tracer.iter[0:6]:
                rows.append(attn.kernel.probs(heads=5, query=-1))  # [1, prompt + step]
            out = tracer.result.save()
        assert torch.equal(out[0, 15:].cpu(), half[0, 3:9])
        # At each decode step, head 5's top key points to the token after the earlier match, which is the token generated.
        for step, row in enumerate(rows):
            assert row.shape == (1, 15 + step)
            assert out[0, row[0].argmax()] == out[0, 15 + step]

    def test_knocking_out_one_key_through_the_mask_shows_in_the_site(self, gpt2):
        prompt = "The quick brown fox jumps"
        attn = gpt2.transformer.h[5].attn
        with gpt2.trace(prompt):
            base = attn.kernel.probs(query=-1).save()  # [1, heads, keys]
            plain = gpt2.output.logits.save()
        head = base[0, :, 1].argmax()
        with gpt2.trace(prompt):
            # For an unpadded prompt, GPT-2 passes `None` as the mask and relies on causal masking.
            # Replace it with an explicit causal bias that also sets key 1 to `-inf` on the last query row.
            args, kwargs = attn.source.attention_interface_1.inputs
            q = args[1]
            n = q.shape[2]
            bias = torch.zeros(n, n, device=q.device).masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
            bias[-1, 1] = float("-inf")
            attn.source.attention_interface_1.inputs = ((*args[:4], bias), kwargs)
            knocked = attn.kernel.probs(heads=head, query=-1).save()
            logits = gpt2.output.logits.save()
        expected = base[0, head].clone()
        expected[1] = 0
        assert knocked[0, 1] == 0
        assert torch.allclose(knocked[0], expected / expected.sum(), atol=1e-6)
        assert not torch.equal(plain, logits)

    def test_the_state_retrieves_a_token_best_right_after_its_write(self, gated_delta):
        mixer = gated_delta.model.layers[0].linear_attn
        with gated_delta.trace(_ids(100, batch=2)):
            inputs = mixer.source.torch_chunk_gated_delta_rule_0.inputs.save()
            states = mixer.kernel.states(tokens=slice(None)).save()  # [batch, 100, heads, K, V]
        (_, key, value, *_), _ = inputs
        key, value = key.float(), value.float()
        # Preprocess keys as the kernel does: L2-normalize and repeat each key head across its two value heads.
        k = F.normalize(key, dim=-1).repeat_interleave(value.shape[2] // key.shape[2], 2)
        read = torch.einsum("bshkv,bthk->bsthv", states, k)  # readout of token t's key from the state after token s
        err = (read - value[:, None]).norm(dim=-1)  # [batch, state, token, heads]
        t = torch.arange(1, 100)
        assert (err[:, t, t] < err[:, t - 1, t]).all()  # writing token t at step t reduces its readout error
        now, later = err[:, torch.arange(70), torch.arange(70)], err[:, torch.arange(30, 100), torch.arange(70)]
        assert (later > now).float().mean() > 0.95  # 30 tokens later, readout error has increased for over 95% of pairs

    def test_ablating_one_expert_from_its_rows_matches_zeroing_its_weights(self, moe):
        experts = moe.model.layers[0].mlp.experts
        ids = _ids(7)
        with moe.trace(ids):
            index, weights = experts.inputs[0][1], experts.inputs[0][2]  # [tokens, top_k]
            rows = experts.kernel.expert_output(expert=3)  # [batch, seq, hidden], 0 where not routed to expert 3
            weight = (weights * (index == 3)).sum(-1)  # [tokens]
            experts.output[:] = experts.output - weight[:, None] * rows.flatten(0, 1)
            ablated = moe.output.logits.save()
        down = experts._module.down_proj
        saved = down.data[3].clone()
        down.data[3].zero_()
        try:
            with moe.trace(ids):
                zeroed = moe.output.logits.save()
        finally:
            down.data[3].copy_(saved)
        with moe.trace(ids):
            plain = moe.output.logits.save()
        assert torch.allclose(ablated, zeroed, atol=1e-6)
        assert not torch.equal(plain, zeroed)

    def test_collecting_one_experts_activations_over_invokes_of_different_lengths(self, moe):
        experts = moe.model.layers[0].mlp.experts
        short, long = _ids(4, batch=1), _ids(7, batch=2)
        with moe.trace() as tracer:
            with tracer.invoke(short):
                index = experts.inputs[0][1].save()  # flattened over the padded batch: [3 * 7, top_k]
                a = experts.kernel.expert_activation(expert=2).save()  # [1, 7, intermediate], left-padded
                a_slots = experts.kernel.expert_activation().save()  # [1, 7, top_k, intermediate]
            with tracer.invoke(long):
                b = experts.kernel.expert_activation(expert=2).save()
                b_slots = experts.kernel.expert_activation().save()
        routed = index.view(3, 7, 2) == 2  # [batch, seq, top_k]
        for act, slots, mask in ((a, a_slots, routed[:1]), (b, b_slots, routed[1:])):
            hit = mask.any(-1)
            assert act.shape == (*slots.shape[:2], slots.shape[3])
            assert hit.sum() > 0
            assert torch.equal(act[hit], slots[mask])  # `expert=2` at routed tokens matches the corresponding slot rows
            assert (act[~hit] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused CUDA kernels")
class TestCuda:
    @pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (8, 2)])
    def test_flash_probs_match_fp32_reference(self, q_heads, kv_heads):
        torch.manual_seed(0)
        mixer = Mixer(width=512, q_heads=q_heads, kv_heads=kv_heads).cuda().to(torch.bfloat16)
        x = torch.randn(2, 40, 512, device="cuda", dtype=torch.bfloat16)
        envoy = Envoy(mixer)
        with envoy.trace(x):
            probs = nnsight.save(envoy.kernel.probs(heads=[1, 6], query=-1))
            lse = nnsight.save(envoy.kernel.lse())
        with torch.no_grad():
            _, expected, expected_lse = _reference(*_inputs(mixer, x), is_causal=True)
        assert probs.provenance.kernel.startswith("aten._scaled_dot_product")
        assert torch.allclose(probs, expected[:, [1, 6], -1], atol=1e-5)
        assert torch.allclose(lse, expected_lse, atol=1e-4)
        assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-5)

    @pytest.mark.parametrize("backend", ["EFFICIENT_ATTENTION", "CUDNN_ATTENTION"])
    def test_kernels_that_write_lse_on_request_leave_the_output_unchanged(self, backend):
        # Memory-efficient and cuDNN attention only write `lse` when `compute_log_sumexp=True`;
        # verify that requesting `probs` under `no_grad` leaves the module's output bit-identical.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.manual_seed(0)
        mixer = Mixer(width=512, q_heads=8, kv_heads=8).cuda().to(torch.bfloat16)
        x = torch.randn(1, 128, 512, device="cuda", dtype=torch.bfloat16)
        envoy = Envoy(mixer)
        with torch.no_grad(), sdpa_kernel(getattr(SDPBackend, backend)):
            with envoy.trace(x):
                plain = nnsight.save(envoy.output)
            with envoy.trace(x):
                probs = nnsight.save(envoy.kernel.probs(heads=0, query=-1))
                served = nnsight.save(envoy.output)
        assert torch.equal(plain, served)
        assert backend.split("_")[0].lower() in probs.provenance.kernel
        assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-5)

    def test_flash_attention_2_sites_match_the_sdpa_sites_on_the_same_weights(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        def build(implementation):
            config = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
            config._attn_implementation = implementation
            torch.manual_seed(0)
            return _wrap(LlamaForCausalLM(config).cuda().to(torch.bfloat16))

        try:
            flash = build("flash_attention_2")
        except ImportError:
            pytest.skip("FlashAttention 2 is not installed")
        sdpa = build("sdpa")
        long, short = _ids(12, batch=1).cuda(), _ids(7, batch=1).cuda()

        def read(model, *prompts):
            attn = model.model.layers[0].self_attn
            saved = []
            with model.trace() as tracer:
                for prompt in prompts:
                    with tracer.invoke(prompt):
                        saved.append(nnsight.save(attn.kernel.probs()))
            return saved

        # The padded two-invoke batch runs through the packed kernel; each invoke gets its own prompt's tokens.
        packed_long, packed_short = read(flash, long, short)
        (dense_long,), (dense_short,) = read(sdpa, long), read(sdpa, short)
        assert packed_long.provenance.kernel == "flash_attn_varlen_func"
        assert packed_short.shape == (1, 4, 7, 7)
        assert torch.allclose(packed_long, dense_long, atol=1e-5)
        assert torch.allclose(packed_short, dense_short, atol=1e-5)
        (single,) = read(flash, long)
        assert single.provenance.kernel == "flash_attn_func"
        assert torch.allclose(single, dense_long, atol=1e-5)
        # A model loaded under SDPA and switched afterwards is served by the next trace.
        switched = build("sdpa")
        switched._module.set_attn_implementation("flash_attention_2")
        (after_switch,) = read(switched, long)
        assert after_switch.provenance.kernel == "flash_attn_func"
        assert torch.allclose(after_switch, dense_long, atol=1e-5)

        def beams(model):
            saved = []
            with model.generate(max_new_tokens=1, num_beams=2, pad_token_id=0, eos_token_id=None) as tracer:
                for prompt in (long, long.flip(1)):
                    with tracer.invoke(prompt):
                        saved.append(nnsight.save(model.model.layers[0].self_attn.kernel.probs()))
            return saved

        # Beam search runs four rows for two invokes; each invoke gets all four, as under SDPA.
        for got, expected in zip(beams(flash), beams(sdpa)):
            assert got.shape == (4, 4, 12, 12)
            assert torch.allclose(got, expected, atol=1e-5)

    def test_fla_chunk_states_match_the_reference_loop(self):
        pytest.importorskip("fla")
        model = _wrap(_gated_delta_module(device="cuda", dtype=torch.bfloat16))
        mixer = model.model.layers[0].linear_attn
        ids = _ids(150).cuda()
        with model.trace(ids):
            plain = model.output.logits.save()
        with model.trace(ids):
            inputs = mixer.source.torch_chunk_gated_delta_rule_0.inputs.save()
            boundaries = mixer.kernel.chunk_states().save()
            inside = mixer.kernel.states(tokens=[5, 100], heads=1).save()
            logits = model.output.logits.save()
        assert torch.equal(plain, logits)
        assert boundaries.provenance.kernel == "chunk_gated_delta_rule"
        assert boundaries.dtype is torch.float32 and "bfloat16" in boundaries.provenance.note
        assert boundaries.shape == (3, 4, 6, 8, 12)
        # Because `fla` stores inter-chunk states in bf16, each exported boundary
        # (and the fp32 final state derived from them) matches the fp32 reference within bf16 rounding.
        for index, token in ((1, 63), (2, 127), (3, 149)):
            reference = _hf_state_after(inputs, token)
            assert torch.allclose(boundaries[:, index], reference, atol=1e-2 * reference.abs().max().item())
        assert inside.provenance.kind == "derived"
        for i, token in enumerate([5, 100]):
            reference = _hf_state_after(inputs, token)[:, 1]
            assert torch.allclose(inside[:, i], reference, atol=1e-2 * reference.abs().max().item())
