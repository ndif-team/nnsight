import gc
import math
import weakref

import pytest
import torch

import nnsight
import torch.nn as nn
import torch.nn.functional as F

from nnsight.intervention import kernel
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import OutOfOrderError
from nnsight.intervention.kernel import Provenance


def _reference(q, k, mask=None, is_causal=False, scale=None):
    """fp32 softmax over the kernel's own inputs; `k` is expanded for grouped queries."""
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
    """Attention that calls the fused kernel directly, with its own projection names."""

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


def attention_through(module, q, k, v, bias, **kwargs):
    """A library-style attention interface the model picks at run time."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=bias is None), None


class Routed(nn.Module):
    """Attention that reaches the kernel through an interface function chosen in the forward."""

    def __init__(self, width=16, heads=2):
        super().__init__()
        self.heads, self.head = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)

    def forward(self, x):
        q, k, v = (t.view(*t.shape[:2], self.heads, self.head).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))
        interface = attention_through
        out, _ = interface(self, q, k, v, None)
        return out.transpose(1, 2).flatten(2)


class Twice(nn.Module):
    """Two kernel calls in one forward: the second over the first's output."""

    def __init__(self):
        super().__init__()
        self.first = Mixer()
        self.second = Mixer(causal=False)

    def forward(self, x):
        return self.second(self.first(x))


# Registered once, as a model class registers the interfaces its checkpoints use.
kernel.through(attention_through)


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

    def test_interface_function_is_drilled_through(self, x):
        routed = Routed()
        envoy = Envoy(routed)
        with envoy.trace(x):
            probs = nnsight.save(envoy.kernel.probs(heads=1, query=-1))
        q, k, _ = (t.view(*t.shape[:2], 2, 8).transpose(1, 2) for t in routed.qkv(x).chunk(3, -1))
        _, expected, _ = _reference(q, k, is_causal=True)
        assert torch.allclose(probs, expected[:, 1, -1], atol=1e-6)

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
        monkeypatch.setattr(kernel, "_LARGE_SITE_BYTES", 100)
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

    def test_submodule_read_before_request_needs_the_module_instrumented_first(self, x):
        mixer = Mixer()
        envoy = Envoy(mixer)
        with pytest.raises(OutOfOrderError, match="already running without an instrumented forward"):
            with envoy.trace(x):
                envoy.to_q.output
                envoy.kernel.probs(heads=0)
        # Accessing `.kernel` before the trace instruments the forward, so the
        # same block then works: this call runs the instrumented body.
        envoy.kernel
        with envoy.trace(x):
            projected = nnsight.save(envoy.to_q.output)
            probs = nnsight.save(envoy.kernel.probs(heads=0))
        assert torch.allclose(projected, mixer.to_q(x))
        assert probs.shape == (2, 6, 6)

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
        # transformers builds float masks at the dtype's minimum rather than -inf.
        mixer = Mixer(causal=False)
        envoy = Envoy(mixer)
        allowed = torch.ones(6, 6, dtype=torch.bool).tril()
        bias = torch.zeros(6, 6).masked_fill(~allowed, torch.finfo(torch.float32).min)
        with envoy.trace(x, bias):
            probs = nnsight.save(envoy.kernel.probs())
            scores = nnsight.save(envoy.kernel.scores(heads=0, query=0))
        assert (probs[..., ~allowed] == 0).all()
        assert torch.allclose(probs.sum(-1), torch.ones(2, 4, 6), atol=1e-6)
        assert scores[1:].min() <= torch.finfo(torch.float32).min / 2  # the mask's own value, added

    @pytest.mark.parametrize("masked", [None, torch.finfo(torch.float16).min, torch.finfo(torch.float32).min])
    def test_a_row_with_every_key_masked_is_zero(self, x, masked):
        # Boolean, or additive at the minimum of float16 / float32 as transformers builds them.
        mixer = Mixer(causal=False)
        envoy = Envoy(mixer)
        allowed = torch.ones(2, 1, 6, 6, dtype=torch.bool)
        allowed[1, :, :2] = False  # the second row's first two queries see nothing (pad queries)
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
        # The shorter prompt is left-padded: its rows put nothing on the pad keys
        # and match a solo run on the real ones.
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


class TestTransformers:
    def test_drills_through_the_attention_interface(self, gpt2):
        attn = gpt2.transformer.h[2].attn
        attn.kernel  # instrumented before the trace, so a read inside the module may come first
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
                    gpt2.lm_head.output  # a step with no kernel request
                else:
                    rows.append(attn.kernel.probs(heads=0, query=-1))
        # Step n sees the prompt plus n generated keys.
        assert [row.shape[-1] for row in rows] == [width + 1, width + 3]
        assert all(row.shape[:-1] == (1,) for row in rows)

    def test_matches_the_eager_implementation(self, gpt2):
        from nnsight.modeling.transformers import TransformersModel

        eager = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True, attn_implementation="eager")
        with eager.trace("The quick brown fox"):
            weights = eager.transformer.h[2].attn.output[1].save()
        with gpt2.trace("The quick brown fox"):
            probs = gpt2.transformer.h[2].attn.kernel.probs().save()
        assert torch.allclose(probs, weights, atol=1e-6)


def _wrap(module):
    from transformers import AutoTokenizer

    from nnsight.modeling.transformers import TransformersModel

    # gpt2's tokenizer, with a pad token inside the tiny vocabulary (id 0), so
    # invokes of different lengths can be batched.
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
    """A two-layer Qwen3-Next whose first layer is a gated delta rule over 6 value heads, 3 key heads, K=8, V=12."""
    from transformers import Qwen3NextForCausalLM

    torch.manual_seed(0)
    module = Qwen3NextForCausalLM(_gated_delta_config()).to(**to)
    # The random A_log gives a decay near 1e-5 per token, which erases a chunk's
    # boundary state within a few tokens; a decay near 0.97 keeps the boundary
    # visible in every derived state, so a wrong boundary fails the comparison.
    module.model.layers[0].linear_attn.A_log.data.fill_(math.log(0.05))
    return module


@pytest.fixture(scope="module")
def gated_delta():
    return _wrap(_gated_delta_module())


@pytest.fixture(scope="module")
def mamba2():
    """A two-layer Mamba2 with 16-token chunks, random weights."""
    from transformers import Mamba2Config, Mamba2ForCausalLM

    config = Mamba2Config(
        hidden_size=64, num_hidden_layers=2, num_heads=4, head_dim=16, state_size=8, n_groups=1, expand=1,
        conv_kernel=4, chunk_size=16, vocab_size=256,
    )
    torch.manual_seed(0)
    return _wrap(Mamba2ForCausalLM(config))


@pytest.fixture(scope="module")
def mamba1():
    """A two-layer Mamba (the per-token selective scan) with 24 channels and an 8-wide state, random weights."""
    from transformers import MambaConfig, MambaForCausalLM

    config = MambaConfig(hidden_size=12, state_size=8, num_hidden_layers=2, conv_kernel=4, time_step_rank=4, vocab_size=256)
    torch.manual_seed(0)
    return _wrap(MambaForCausalLM(config))


@pytest.fixture(scope="module")
def kimi():
    """A two-layer Kimi Linear whose first layer is a Kimi delta attention over 4 heads of K = V = 8, random weights."""
    from transformers import KimiLinearConfig, KimiLinearForCausalLM

    config = KimiLinearConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, vocab_size=256,
        layer_types=["linear_attention", "full_attention"], mlp_layer_types=["dense", "dense"],
        linear_num_heads=4, linear_head_dim=8, linear_conv_kernel_dim=4, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    torch.manual_seed(0)
    module = KimiLinearForCausalLM(config)
    # As on the gated delta rule fixture: a decay near 0.95 per channel keeps the
    # chunk boundary visible in every derived state.
    module.model.layers[0].self_attn.forget_gate.A_log.data.fill_(math.log(0.05))
    return _wrap(module)


def _ids(length, batch=3):
    torch.manual_seed(1)
    return torch.randint(0, 256, (batch, length))


def _hf_state_after(inputs, token):
    """transformers' per-token reference run over the first ``token + 1`` positions; the state it ends in."""
    import transformers.models.qwen3_next.modeling_qwen3_next as modeling

    args, kwargs = inputs
    query, key, value, g, beta = (t[:, : token + 1] for t in (*args[:3], kwargs["g"], kwargs["beta"]))
    _, final = modeling.torch_recurrent_gated_delta_rule.__wrapped__(
        query, key, value, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=kwargs["use_qk_l2norm_in_kernel"]
    )
    return final


def _no_sites(fn, args, kwargs):
    return fn(*args, **kwargs), kernel.Unavailable("this call keeps nothing to read")


def _plain_scan(x):
    return x.cumsum(1)


kernel.register(_plain_scan, _no_sites)


class Scan(nn.Module):
    def forward(self, x):
        return _plain_scan(x)


class TestRecurrent:
    def test_gated_delta_chunk_states_match_the_per_token_reference(self, gated_delta):
        mixer = gated_delta.model.layers[0].linear_attn
        ids = _ids(150)  # three chunks of 64, the last partial
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
        # The state after decode step 1 is the state a prefill over the same 71 tokens ends in.
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
        ids = _ids(150)  # three chunks of 64, the last partial
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
    """A one-layer Qwen2-MoE with 5 experts, 2 per token, random weights, on each of transformers' two fused experts paths."""
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
    """Experts without a gate on transformers' experts interface: ``act(up @ x)`` then ``down``, 3 experts, hidden 8, intermediate 12."""

    def __init__(self, config):
        super().__init__()
        # Row widths are multiples of 16 bytes, as torch's grouped matmul requires.
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

    from nnsight.intervention import experts

    # What `TransformersModel` registers at construction, for a module traced through a plain Envoy.
    kernel.register(moe.grouped_mm_experts_forward, experts.serve_grouped_mm)
    torch.manual_seed(2)
    return Envoy(use_experts_implementation(Pool, has_gate=False)(SimpleNamespace(_experts_implementation="grouped_mm")))


def _per_pair(experts_module, x, index):
    """Each (token, slot) pair's activation and down projection, computed one pair at a time in fp64 on the CPU."""
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


class TestExperts:
    def test_pair_buffers_match_a_per_pair_computation(self, moe):
        experts = moe.model.layers[0].mlp.experts
        experts.kernel  # instrument before the trace, so the module's inputs may be read first
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
        # The module's output is the routing-weighted sum of these rows.
        assert torch.allclose((out.reshape(21, 2, 32) * weights.unsqueeze(-1)).sum(1), routed, atol=1e-6)

    def test_expert_and_slot_selections(self, moe):
        experts = moe.model.layers[0].mlp.experts
        experts.kernel
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
        # A plain Envoy's batcher counts its input as one row.
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
        # A matmul over 3 rows and over 2 rows may block differently on a GPU; the
        # short invoke is left-padded, so its own tokens are the last four positions.
        assert torch.allclose(b, b_alone, atol=1e-6)
        assert torch.allclose(a[:, -4:], a_alone, atol=1e-6)
        with moe.generate(long, max_new_tokens=2, min_new_tokens=2, do_sample=False) as tracer:
            steps = nnsight.save([])
            for _ in tracer.iter[0:2]:
                steps.append(experts.kernel.expert_output(expert=0))
        assert [step.shape for step in steps] == [(2, 7, 32), (2, 1, 32)]


class TestWorkloads:
    """One interpretability method per test, run through the sites and checked by the method's own invariant."""

    def test_induction_heads_score_high_on_a_repeated_sequence(self, gpt2):
        torch.manual_seed(0)
        half = torch.randint(1000, 20000, (1, 24))
        with gpt2.trace(torch.cat([half, half], 1)):
            rows = nnsight.save([])
            for block in gpt2.transformer.h:
                rows.append(block.attn.kernel.probs(query=slice(24, None)))  # [1, heads, 24, 48]
        # Query 24 + i is the second copy of token i; an induction head puts its mass on key i + 1.
        queries, keys = torch.arange(24), torch.arange(24) + 1
        scores = torch.stack([row[0][:, queries, keys].mean(-1) for row in rows])  # [layers, heads]
        assert scores.shape == (12, 12)
        assert scores[5, 5] > 0.9 and scores[7, 10] > 0.8  # gpt2's induction heads
        assert scores[:2].max() < 0.05  # none in the first two layers

    def test_attention_sink_mass_sits_on_the_first_key(self, gpt2):
        prompt = "The quick brown fox jumps over the lazy dog because it wanted to reach the other side of the river"
        with gpt2.trace(prompt):
            sinks = nnsight.save([])
            for i, block in enumerate(gpt2.transformer.h):
                sinks.append(block.attn.kernel.probs(key=0))  # [1, heads, queries]
                if i == 10:
                    full = block.attn.kernel.probs().save()
        mass = torch.stack([sink[0, :, 1:].mean(-1) for sink in sinks])  # [layers, heads], queries after the first
        assert torch.allclose(sinks[10][0], full[0, :, :, 0], atol=1e-6)
        assert (mass[10] > 0.5).all()
        assert mass[0].mean() < 0.3

    def test_a_retrieval_head_copies_during_generation(self, gpt2):
        torch.manual_seed(0)
        half = torch.randint(1000, 20000, (1, 12))
        ids = torch.cat([half, half[:, :3]], 1)  # the second copy has begun; greedy decoding continues it
        attn = gpt2.transformer.h[5].attn
        with gpt2.generate(ids, max_new_tokens=6, do_sample=False) as tracer:
            rows = nnsight.save([])
            for _ in tracer.iter[0:6]:
                rows.append(attn.kernel.probs(heads=5, query=-1))  # [1, prompt + step]
            out = tracer.result.save()
        assert torch.equal(out[0, 15:].cpu(), half[0, 3:9])
        # At each step the head's top key is the position after the previous occurrence, whose token is the one generated.
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
            # gpt2 hands the interface no mask for an unpadded prompt (the kernel runs causal);
            # give it a causal bias with key 1 forbidden for the last query.
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
        # The kernel's keys: l2-normalized, one key head shared by two value heads.
        k = F.normalize(key, dim=-1).repeat_interleave(value.shape[2] // key.shape[2], 2)
        read = torch.einsum("bshkv,bthk->bsthv", states, k)  # what the state after token s returns for token t's key
        err = (read - value[:, None]).norm(dim=-1)  # [batch, state, token, heads]
        t = torch.arange(1, 100)
        assert (err[:, t, t] < err[:, t - 1, t]).all()  # the write at t lowers the error on token t
        now, later = err[:, torch.arange(70), torch.arange(70)], err[:, torch.arange(30, 100), torch.arange(70)]
        assert (later > now).float().mean() > 0.95  # thirty tokens on, the state has mostly forgotten it

    def test_ablating_one_expert_from_its_rows_matches_zeroing_its_weights(self, moe):
        experts = moe.model.layers[0].mlp.experts
        experts.kernel  # instrument before the trace, so the module's inputs may be read first
        ids = _ids(7)
        with moe.trace(ids):
            index, weights = experts.inputs[0][1], experts.inputs[0][2]  # [tokens, top_k]
            rows = experts.kernel.expert_output(expert=3)  # [batch, seq, hidden], 0 where the token is not routed to 3
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
        experts.kernel
        short, long = _ids(4, batch=1), _ids(7, batch=2)
        with moe.trace() as tracer:
            with tracer.invoke(short):
                index = experts.inputs[0][1].save()  # flat over the padded batch, [3 * 7, top_k]
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
            assert torch.equal(act[hit], slots[mask])  # the dense form holds exactly the routed rows, in token order
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
        # These two take a compute_log_sumexp flag; the output the model gets is the
        # one computed with the flag as the model passed it, under no_grad too.
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
        # The kernel works in bf16 between chunks, so every boundary (the fp32
        # final state included) sits within bf16 rounding of the fp32 recurrence.
        for index, token in ((1, 63), (2, 127), (3, 149)):
            reference = _hf_state_after(inputs, token)
            assert torch.allclose(boundaries[:, index], reference, atol=1e-2 * reference.abs().max().item())
        assert inside.provenance.kind == "derived"
        for i, token in enumerate([5, 100]):
            reference = _hf_state_after(inputs, token)[:, 1]
            assert torch.allclose(inside[:, i], reference, atol=1e-2 * reference.abs().max().item())
