"""Tracing an engine that replays CUDA graphs, at declared taps.

The engine is built with ``taps=`` and so without ``enforce_eager``: the tapped
locations are recorded into vLLM's breakable graphs and served on replay, every
other location is not. Reads are checked against the eager gpt2 fixture, edits
by their effect on the prediction, and an untapped read by the error its request
comes home with.
"""

import pytest
import torch

pytest.importorskip("vllm")

GPU_COUNT = torch.cuda.device_count()


@pytest.fixture(scope="module")
def vllm_gpt2_taps():
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    return VLLM(
        "gpt2",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1,
        dispatch=True,
        taps=["transformer.h.*.output", "transformer.h.8.mlp.input"],
    )


def test_taps_resolve_to_every_layer(vllm_gpt2_taps):
    assert "model.transformer.h.0.output" in vllm_gpt2_taps.taps
    assert "model.transformer.h.11.output" in vllm_gpt2_taps.taps
    assert "model.transformer.h.8.mlp.input" in vllm_gpt2_taps.taps
    # `*` is one segment: the block's sublayers are not tapped.
    assert "model.transformer.h.8.mlp.output" not in vllm_gpt2_taps.taps


def test_unresolvable_tap_is_refused():
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    with pytest.raises(ValueError, match="names no module"):
        VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True, taps=["transformer.nope.output"])


@torch.no_grad()
def test_read_matches_eager(vllm_gpt2_taps, vllm_gpt2, ET_prompt):
    with vllm_gpt2_taps.trace(ET_prompt, temperature=0.0, top_p=1):
        tapped = vllm_gpt2_taps.transformer.h[6].output.clone().save()
    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
        eager = vllm_gpt2.transformer.h[6].output.clone().save()

    assert tapped.shape == eager.shape
    assert torch.allclose(tapped.float(), eager.float(), atol=1e-2, rtol=1e-2)


@torch.no_grad()
def test_in_place_edit_lands(vllm_gpt2_taps, ET_prompt):
    with vllm_gpt2_taps.trace(ET_prompt, temperature=0.0, top_p=1):
        clean = vllm_gpt2_taps.logits.clone().save()
    with vllm_gpt2_taps.trace(ET_prompt, temperature=0.0, top_p=1):
        vllm_gpt2_taps.transformer.h[8].output[:] = 0
        edited = vllm_gpt2_taps.logits.clone().save()

    assert vllm_gpt2_taps.tokenizer.decode(clean.argmax(dim=-1)) == " Paris"
    assert not torch.equal(clean, edited)


@torch.no_grad()
def test_replacement_is_copied_back(vllm_gpt2_taps, ET_prompt):
    with vllm_gpt2_taps.trace(ET_prompt, temperature=0.0, top_p=1):
        vllm_gpt2_taps.transformer.h[8].output = torch.zeros_like(vllm_gpt2_taps.transformer.h[8].output)
        after = vllm_gpt2_taps.transformer.h[9].output.clone().save()
    with vllm_gpt2_taps.trace(ET_prompt, temperature=0.0, top_p=1):
        clean = vllm_gpt2_taps.transformer.h[9].output.clone().save()

    assert not torch.equal(after, clean)


@torch.no_grad()
def test_per_step_reads_under_iter(vllm_gpt2_taps, MSG_prompt):
    with vllm_gpt2_taps.trace(MSG_prompt, temperature=0.0, top_p=1, max_tokens=3) as tracer:
        hiddens = list().save()
        for _ in tracer.iter[:3]:
            hiddens.append(vllm_gpt2_taps.transformer.h[11].output.clone())

    assert len(hiddens) == 3
    assert hiddens[0].shape[0] > 1 and hiddens[1].shape[0] == 1  # prefill, then one decode row
    assert not torch.equal(hiddens[1], hiddens[2])


@torch.no_grad()
def test_untapped_location_is_named(vllm_gpt2_taps, ET_prompt):
    # A worker's error is deferred and re-raised at the client as a RuntimeError
    # carrying the original message (see intervention.errors.raise_deferred).
    with pytest.raises(RuntimeError, match="is not a tap on this engine"):
        with vllm_gpt2_taps.trace(ET_prompt, temperature=0.0, top_p=1):
            vllm_gpt2_taps.transformer.h[8].mlp.output.save()


def test_a_wrong_shaped_swap_errors_its_request_only(vllm_gpt2_taps, ET_prompt):
    # The batcher's widen refuses a replacement that does not fit the rows, inside
    # the worker's handoff, so the error is this request's; the engine — which
    # other tenants share — stays up and serves the next trace.
    with pytest.raises(RuntimeError):
        with vllm_gpt2_taps.trace(ET_prompt, temperature=0.0, top_p=1):
            out = vllm_gpt2_taps.transformer.h[6].output
            vllm_gpt2_taps.transformer.h[6].output = out[..., :8]
    with vllm_gpt2_taps.trace(ET_prompt, temperature=0.0, top_p=1):
        after = vllm_gpt2_taps.transformer.h[6].output.clone().save()
    assert after.shape[-1] == 768


def test_untapped_location_is_named_on_a_long_prompt(vllm_gpt2_taps):
    # A prompt longer than the largest captured graph runs eagerly, so the
    # controllers do reach an untapped location that step. It is still refused:
    # what a tapped engine serves must not depend on prompt length.
    prompt = "word " * 700
    n = len(vllm_gpt2_taps.tokenizer(prompt)["input_ids"])
    assert 520 < n < 1000, n  # past min(2 * max_num_seqs, 512), within gpt2's context
    with pytest.raises(RuntimeError, match="is not a tap on this engine"):
        with vllm_gpt2_taps.trace(prompt, temperature=0.0, top_p=1, max_tokens=2):
            vllm_gpt2_taps.transformer.h[8].mlp.output.save()


def test_enforce_eager_contradicting_taps_is_refused():
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    with pytest.raises(ValueError, match="enforce_eager=True contradicts taps"):
        VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True, taps=["transformer.h.*.output"], enforce_eager=True)
    with pytest.raises(ValueError, match="enforce_eager=False contradicts taps"):
        VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True, enforce_eager=False)


# --- taps on operations inside a forward (`.source`) ---------------------------


@pytest.fixture(scope="module")
def vllm_gpt2_source_taps():
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    return VLLM(
        "gpt2",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1,
        dispatch=True,
        taps=["transformer.h.3.attn.source.qkv_chunk_0.output", "transformer.h.*.output"],
    )


def test_source_tap_resolves(vllm_gpt2_source_taps):
    assert "model.transformer.h.3.attn.source.qkv_chunk_0.output" in vllm_gpt2_source_taps.taps
    assert "model.transformer.h.3.output" in vllm_gpt2_source_taps.taps


@torch.no_grad()
def test_source_tap_read_matches_eager(vllm_gpt2_source_taps, vllm_gpt2, ET_prompt):
    """The op's value on replay is the eager engine's value."""
    with vllm_gpt2_source_taps.trace(ET_prompt, temperature=0.0, top_p=1):
        tapped = vllm_gpt2_source_taps.transformer.h[3].attn.source.qkv_chunk_0.output[0].clone().save()
    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
        eager = vllm_gpt2.transformer.h[3].attn.source.qkv_chunk_0.output[0].clone().save()

    assert tapped.shape == eager.shape and tapped.shape[-1] == 768
    assert torch.allclose(tapped.float(), eager.float(), atol=1e-2, rtol=1e-2)


@torch.no_grad()
def test_source_tap_per_step_under_iter(vllm_gpt2_source_taps, MSG_prompt):
    with vllm_gpt2_source_taps.trace(MSG_prompt, temperature=0.0, max_tokens=4, ignore_eos=True) as tracer:
        qs = list().save()
        for _ in tracer.iter[:4]:
            qs.append(vllm_gpt2_source_taps.transformer.h[3].attn.source.qkv_chunk_0.output[0][-1].clone())
        result = tracer.result.save()

    assert len(qs) == 4 and all(q.shape == (768,) for q in qs)
    assert not torch.equal(qs[1], qs[2])
    assert len(result.outputs[0].token_ids) == 4


@torch.no_grad()
def test_source_tap_in_place_edit_lands(vllm_gpt2_source_taps, ET_prompt):
    """Zeroing q at the op changes what the engine predicts."""
    model = vllm_gpt2_source_taps
    with model.trace(ET_prompt, temperature=0.0, top_p=1):
        plain = model.logits.clone().save()
    with model.trace(ET_prompt, temperature=0.0, top_p=1):
        model.transformer.h[3].attn.source.qkv_chunk_0.output[0][:] = 0
        edited = model.logits.clone().save()

    assert not torch.allclose(plain.float(), edited.float())


def test_source_tap_on_a_missing_op_is_refused():
    """The op is checked when the worker loads, so a typo is a load error, not a parked request."""
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    with pytest.raises(Exception):
        VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True,
             taps=["transformer.h.3.attn.source.nope_0.output"])
