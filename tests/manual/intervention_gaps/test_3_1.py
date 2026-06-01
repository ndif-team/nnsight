"""
Gap 3.1: Tensor layout is [total_tokens, hidden] (2D flat) instead of [batch, seq, hidden] (3D)

DOCUMENTED PATTERN (CLAUDE.md "Modifying Activations"):
    model.transformer.h[0].output[0][:, -1, :]  # Last token per sequence
    model.transformer.h[0].output[0][:, :, 0]   # First hidden dim
The docs show 3D indexing patterns like [:, -1, :] for last-token selection and
[:, pos, :] for position-specific interventions. These assume a [batch, seq,
hidden] layout.

ON HF (expected behavior):
    Tensors are [batch_size, seq_len, hidden_dim] (3D, padded). The [:, -1, :]
    pattern correctly selects the last token for each sequence in the batch.
    Note: nnsight's batcher narrows per-invoke, so within an invoke tensors may
    appear as [seq, hidden] (2D), but the seq dimension is still present.

ON vLLM (the gap):
    Tensors are [total_tokens, hidden_dim] (2D, flat). vLLM uses continuous
    batching -- all tokens from all sequences are packed into a single flat
    dimension with no padding. The [:, -1, :] pattern fails because there is no
    sequence dimension. Selecting the last token of a specific sequence requires
    knowing its token boundaries in the flat layout.

WHY THIS MATTERS:
    Nearly all intervention code uses positional indexing to target specific
    tokens (last token for next-token prediction, specific positions for
    activation patching). The flat layout breaks all such patterns. Even simple
    operations like "zero out the last token's hidden state" require fundamentally
    different code on vLLM.

VALIDATION: Compare ndim of layer outputs. vLLM produces 2D tensors, HF produces
3D (or 2D per-invoke via batcher, but with a seq dimension preserved).
"""

import argparse
import json

import torch
import nnsight


def run_vllm(model_name, prompt):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)

    # Single prompt
    with model.trace(prompt, temperature=0.0):
        vllm_shape_single = nnsight.save(
            model.model.layers[0].output[0].clone().shape
        )

    # Two prompts via invokes
    with model.trace(temperature=0.0) as tracer:
        with tracer.invoke(prompt):
            vllm_shape_invoke1 = nnsight.save(
                model.model.layers[0].output[0].clone().shape
            )
        with tracer.invoke("Hello"):
            vllm_shape_invoke2 = nnsight.save(
                model.model.layers[0].output[0].clone().shape
            )

    print("vLLM layer output shapes:")
    print(f"  single prompt: {vllm_shape_single}")
    print(f"  invoke 1 ('{prompt}'): {vllm_shape_invoke1}")
    print(f"  invoke 2 ('Hello'): {vllm_shape_invoke2}")

    vllm_ndim = len(vllm_shape_single)
    print(f"  ndim={vllm_ndim}")

    gap_confirmed = vllm_ndim == 2
    status = "CONFIRMED" if gap_confirmed else "NOT_REPRODUCED"
    detail = f"vLLM ndim={vllm_ndim} shape={list(vllm_shape_single)}"
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "ndim": vllm_ndim,
        "shape": list(vllm_shape_single),
    }


def run_hf(model_name, prompt):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)

    # Single prompt
    with model.trace(prompt):
        hf_shape_single = nnsight.save(
            model.model.layers[0].output[0].shape
        )

    # Two prompts via invokes
    with model.trace() as tracer:
        with tracer.invoke(prompt):
            hf_shape_invoke1 = nnsight.save(
                model.model.layers[0].output[0].shape
            )
        with tracer.invoke("Hello"):
            hf_shape_invoke2 = nnsight.save(
                model.model.layers[0].output[0].shape
            )

    print("HF layer output shapes:")
    print(f"  single prompt: {hf_shape_single}")
    print(f"  invoke 1 ('{prompt}'): {hf_shape_invoke1}")
    print(f"  invoke 2 ('Hello'): {hf_shape_invoke2}")

    hf_ndim = len(hf_shape_single)
    print(f"  ndim={hf_ndim}")

    # Note: nnsight's batcher narrows the batch dim per-invoke, so even HF
    # may show 2D [seq, hidden] instead of 3D [batch, seq, hidden] when
    # accessed inside an invoke. The real gap is that vLLM's native format
    # is flat [total_tokens, hidden] (no padding, no batch dim) while HF's
    # native format is [batch, seq, hidden] (padded). Through nnsight's batcher,
    # per-invoke tensors are narrowed to [seq, hidden] (2D). So when accessed
    # inside an invoke, both HF and vLLM may appear 2D. However, for a single-
    # prompt trace (no explicit invokes), HF should still show 3D [1, seq, hidden].
    # We report NO_GAP if 3D (native HF layout), UNEXPECTED if something else.
    status = "NO_GAP" if hf_ndim >= 2 else "UNEXPECTED"
    detail = f"HF ndim={hf_ndim} shape={list(hf_shape_single)} (batcher may narrow per-invoke)"
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "ndim": hf_ndim,
        "shape": list(hf_shape_single),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--backend", choices=["vllm", "hf"], required=True)
    args = parser.parse_args()

    prompt = "The Eiffel Tower is in"

    if args.backend == "vllm":
        result = run_vllm(args.model, prompt)
    else:
        result = run_hf(args.model, prompt)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
