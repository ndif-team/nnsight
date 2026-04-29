"""
Gap 4.2: Source tracing shows only 1 trivial delegate op for fused modules

DOCUMENTED PATTERN (CLAUDE.md "Source Tracing"):
    print(model.transformer.h[0].attn.source)  # Shows all operations
    # Output shows operation names like attention_interface_0, self_c_proj_0, etc.
The docs show .source revealing the internal operations of a module's forward
method, enabling fine-grained intervention at any point inside the computation.

ON HF (expected behavior):
    Source tracing reveals rich internal structure: input_layernorm shows 6+
    operations (multiply, rsqrt, mean, etc.), self_attn shows 16+ operations
    (projections, reshape, attention, output projection), and act_fn shows its
    activation function internals. Users can hook into any intermediate operation.

ON vLLM (the gap):
    Fused modules like input_layernorm, act_fn (SiluAndMul), and the attention
    kernel have Python-level forward methods that are trivial wrappers:
        return self._forward_method(*args, **kwargs)
    Source tracing on these modules shows only 1 operation -- the delegate call
    to the C/CUDA kernel. The actual computation (normalization, activation,
    attention) happens entirely in compiled code and is invisible to .source.

WHY THIS MATTERS:
    Source tracing is nnsight's mechanism for fine-grained intervention inside
    modules -- e.g., hooking between the query projection and the attention
    computation, or intercepting the intermediate result after RMS normalization
    but before scaling. When the real computation is hidden inside a fused CUDA
    kernel, users cannot inspect or modify any intermediate values within these
    modules, limiting intervention granularity.

VALIDATION: Count operations (-> arrows) in .source output for input_layernorm,
self_attn, act_fn, and decoder_layer. HF shows many ops; vLLM fused modules
show 1 (the delegate call).
"""

import argparse
import json

import torch


def run_vllm(model_name):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)
    findings = {}

    modules_to_test = [
        ("input_layernorm", model.model.layers[0].input_layernorm, True),
        ("self_attn", model.model.layers[0].self_attn, False),
        ("act_fn", model.model.layers[0].mlp.act_fn, True),
        ("decoder_layer", model.model.layers[0], False),
    ]
    for name, mod, is_fused in modules_to_test:
        try:
            src = mod.source
            src_str = str(src)
            op_count = src_str.count("->")
            print(f"\n{name}.source ({op_count} ops, fused={is_fused}):")
            print(src_str[:500])
            findings[name] = {
                "accessible": True,
                "content_length": len(src_str),
                "op_count": op_count,
                "is_fused": is_fused,
            }
        except Exception as e:
            print(f"{name}.source: ERROR -- {type(e).__name__}: {e}")
            findings[name] = {"accessible": False, "error": str(e), "is_fused": is_fused}

    # The real test: fused modules (input_layernorm, act_fn) have a trivial
    # Python wrapper "return self._forward_method(*args, **kwargs)" that
    # delegates to a C/CUDA kernel. Source tracing "works" (returns non-empty)
    # but the content has only 1 operation (the delegate call) — not meaningful.
    fused_trivial = any(
        findings.get(k, {}).get("op_count", 0) <= 1
        for k in ("input_layernorm", "act_fn")
    )
    status = "CONFIRMED" if fused_trivial else "NOT_REPRODUCED"
    detail = "; ".join(
        f"{k}: {f.get('op_count', '?')} ops (len={f.get('content_length', 0)})"
        if f.get("accessible") else f"{k}: FAILED"
        for k, f in findings.items()
    )
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "findings": findings,
    }


def run_hf(model_name):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)
    findings = {}

    modules_to_test = [
        ("input_layernorm", model.model.layers[0].input_layernorm),
        ("self_attn", model.model.layers[0].self_attn),
        ("act_fn", model.model.layers[0].mlp.act_fn),
        ("decoder_layer", model.model.layers[0]),
    ]
    for name, mod in modules_to_test:
        try:
            src = mod.source
            src_str = str(src)
            op_count = src_str.count("->")
            print(f"\n{name}.source ({op_count} ops):")
            print(src_str[:500])
            findings[name] = {"accessible": True, "content_length": len(src_str), "op_count": op_count}
        except Exception as e:
            print(f"{name}.source: ERROR -- {type(e).__name__}: {e}")
            findings[name] = {"accessible": False, "error": str(e)}

    # HF: source tracing should work with multiple ops on all modules
    all_accessible = all(f.get("accessible", False) for f in findings.values())
    hf_has_ops = all(f.get("op_count", 0) > 1 for f in findings.values() if f.get("accessible"))
    status = "NO_GAP" if all_accessible and hf_has_ops else "UNEXPECTED_FAILURE"
    detail = "; ".join(
        f"{k}: {f.get('op_count', '?')} ops" if f.get("accessible") else f"{k}: FAILED"
        for k, f in findings.items()
    )
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "findings": findings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--backend", choices=["vllm", "hf"], required=True)
    args = parser.parse_args()

    if args.backend == "vllm":
        result = run_vllm(args.model)
    else:
        result = run_hf(args.model)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
