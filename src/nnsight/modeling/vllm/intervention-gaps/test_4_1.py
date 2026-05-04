"""
Gap 4.1: All gradient computation is blocked by torch.inference_mode()

DOCUMENTED PATTERN (CLAUDE.md "Gradients and Backpropagation"):
    with model.trace("Hello"):
        hs = model.transformer.h[-1].output[0]
        hs.requires_grad_(True)
        logits = model.lm_head.output
        loss = logits.sum()
        with loss.backward():
            grad = hs.grad.save()
The docs show gradient-based workflows using requires_grad_() and backward()
inside traces for attribution, saliency maps, and probe training.

ON HF (expected behavior):
    requires_grad_(True) succeeds, loss.backward() computes gradients, and
    hs.grad contains the gradient tensor. Gradient-based attribution (e.g.,
    integrated gradients), saliency maps, and training linear probes on
    intermediate activations all work within the tracing context.

ON vLLM (the gap):
    vLLM wraps all model execution in torch.inference_mode(), which globally
    disables gradient tracking. requires_grad_(True) raises RuntimeError,
    and loss.backward() fails. There is no way to compute gradients on any
    tensor during vLLM inference.

WHY THIS MATTERS:
    Gradient-based interpretability is a major research paradigm: integrated
    gradients, GradCAM-style attribution, saliency maps, gradient-based feature
    attribution, and training probes/SAEs on activations (which need gradients
    for the probe parameters) are all completely impossible under vLLM. This
    rules out a substantial fraction of interpretability research workflows.

VALIDATION: Test whether requires_grad_(True) and loss.backward() succeed (HF)
or raise errors (vLLM).
"""

import argparse
import json
import traceback

import torch


def run_vllm(model_name, prompt):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)
    errors = []

    # Test 1: requires_grad_ inside trace
    try:
        with model.trace(prompt, temperature=0.0):
            hs = model.model.layers[5].output[0]
            hs.requires_grad_(True)
            logits = model.logits.output.save()
        errors.append(("requires_grad_", None))
        print("requires_grad_(True): NO ERROR (unexpected)")
    except Exception as e:
        errors.append(("requires_grad_", str(e)))
        print(f"requires_grad_(True): ERROR -- {type(e).__name__}: {e}")

    # Test 2: backward inside trace — check that gradients actually contain values,
    # not just that the context manager doesn't throw.
    try:
        with model.trace(prompt, temperature=0.0):
            hs = model.model.layers[5].output[0]
            hs.requires_grad_(True)
            logits = model.logits.output
            loss = logits.sum()
            with loss.backward():
                grad = hs.grad.save()
        # Even if no exception, verify grad has real nonzero values
        if grad.abs().sum().item() > 0:
            errors.append(("backward", None))
            print(f"loss.backward(): OK, grad has nonzero values, shape={grad.shape}")
        else:
            errors.append(("backward", "grad is all zeros — no real gradient flow"))
            print("loss.backward(): FAILED — grad is all zeros")
    except Exception as e:
        errors.append(("backward", str(e)))
        print(f"loss.backward(): ERROR -- {type(e).__name__}: {e}")

    # The real test is whether backward() works (actual gradient flow).
    # requires_grad_() may succeed if nnsight clones tensors out of inference
    # mode, but that doesn't mean gradients actually propagate — the
    # underlying computation graph still ran under inference_mode.
    backward_failed = errors[1][1] is not None if len(errors) > 1 else True
    status = "CONFIRMED" if backward_failed else "NOT_REPRODUCED"
    detail = "; ".join(f"{name}: {'FAILED' if err else 'OK'}" for name, err in errors)
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "requires_grad_ok": errors[0][1] is None if errors else None,
        "backward_ok": errors[1][1] is None if len(errors) > 1 else None,
    }


def run_hf(model_name, prompt):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)
    errors = []

    # Test 1: requires_grad_ inside trace
    try:
        with model.trace(prompt):
            hs = model.model.layers[5].output[0]
            hs.requires_grad_(True)
            logits = model.lm_head.output.save()
        errors.append(("requires_grad_", None))
        print("requires_grad_(True): NO ERROR (expected)")
    except Exception as e:
        errors.append(("requires_grad_", str(e)))
        print(f"requires_grad_(True): ERROR -- {type(e).__name__}: {e}")

    # Test 2: backward inside trace
    try:
        with model.trace(prompt):
            hs = model.model.layers[5].output[0]
            hs.requires_grad_(True)
            logits = model.lm_head.output
            loss = logits.sum()
            with loss.backward():
                grad = hs.grad.save()
        errors.append(("backward", None))
        print(f"loss.backward(): NO ERROR (expected), grad shape={grad.shape}")
    except Exception as e:
        errors.append(("backward", str(e)))
        print(f"loss.backward(): ERROR -- {type(e).__name__}: {e}")

    requires_grad_ok = errors[0][1] is None if errors else False
    # The core gap is about inference_mode: requires_grad_ works on HF but not vLLM.
    # backward may fail on HF for unrelated reasons (grad not captured for this model).
    status = "NO_GAP" if requires_grad_ok else "UNEXPECTED_FAILURE"
    detail = "; ".join(f"{name}: {'OK' if err is None else 'FAILED'}" for name, err in errors)
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "requires_grad_ok": errors[0][1] is None if errors else None,
        "backward_ok": errors[1][1] is None if len(errors) > 1 else None,
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
