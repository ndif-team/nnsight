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
