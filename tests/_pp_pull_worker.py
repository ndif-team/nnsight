#!/usr/bin/env python
"""Worker subprocess for PP tests.

Usage:
    python _pp_pull_worker.py <scenario> --pp N --output path [--tp N] [--layer N] [--max_tokens N]
"""

import argparse
import json
import traceback


def make_model(pp_size, tp_size=1):
    from nnsight.modeling.vllm import VLLM
    kwargs = {
        "gpu_memory_utilization": 0.1,
        "dispatch": True,
    }
    if pp_size > 1:
        kwargs["pipeline_parallel_size"] = pp_size
    if tp_size > 1:
        kwargs["tensor_parallel_size"] = tp_size
    return VLLM("openai-community/gpt2", **kwargs)


PROMPT = "The Eiffel Tower is located in the city of"


def scenario_basic_trace(model, args):
    """Basic trace — just get logits, no interventions."""
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        logits = model.logits.output.save()
    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())
    return {
        "argmax": argmax,
        "top_token": model.tokenizer.decode(argmax),
    }


def scenario_logits(model, args):
    """Save logits and return top token."""
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        logits = model.logits.output.save()
    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())
    return {
        "argmax": argmax,
        "top_token": model.tokenizer.decode(argmax),
        "shape": list(logits_cpu.shape),
    }


def scenario_hidden(model, args):
    """Save hidden states from a specific layer."""
    layer = args.layer
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        hidden = model.transformer.h[layer].output[0].save()
    hidden_cpu = hidden.float().cpu()
    return {
        "shape": list(hidden_cpu.shape),
        "mean": float(hidden_cpu.mean().item()),
    }


def scenario_multigen(model, args):
    """Multi-token generation, save per-step logits."""
    max_tokens = args.max_tokens
    with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=max_tokens) as tracer:
        logit_list = list().save()
        for step in tracer.iter[0:max_tokens]:
            logit_list.append(model.logits.output)

    tokens = []
    argmaxes = []
    for logit in logit_list:
        am = int(logit.argmax(dim=-1).item())
        argmaxes.append(am)
        tokens.append(model.tokenizer.decode(am))

    return {
        "tokens": tokens,
        "argmaxes": argmaxes,
        "num_steps": len(logit_list),
    }


def scenario_cross_stage_read(model, args):
    """Cross-stage read: capture layer 0 (stage 0) output and save it from
    stage 1's perspective.

    GPT-2 with PP=2: layers 0-5 on stage 0, layers 6-11 on stage 1.
    The mediator runs on ALL ranks. On stage 0, layer 0 hook fires
    and the value is cloned into pp_hook_buffer. On stage 1, layer 0
    is PPMissing so the Envoy returns a LazyRemoteTensor. When we
    save it, it materializes via pull_from_remote.

    We also save layer 0 from a PP=1 baseline to compare.
    """
    import torch

    with model.trace(PROMPT, temperature=0.0, top_p=1):
        # This accesses layer 0 — on stage 0 it's real, on stage 1 it's
        # PPMissing. The .save() should trigger materialization on stage 1.
        h0 = model.transformer.h[0].output[0].save()
        logits = model.logits.output.save()

    h0_cpu = h0.float().cpu()
    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())

    return {
        "h0_shape": list(h0_cpu.shape),
        "h0_mean": float(h0_cpu.mean().item()),
        "h0_std": float(h0_cpu.std().item()),
        "argmax": argmax,
        "top_token": model.tokenizer.decode(argmax),
    }


def scenario_cross_stage_write(model, args):
    """Cross-stage write: read layer 2 (stage 0), write to layer 8 (stage 1).

    GPT-2 PP=2: layer 2 on stage 0, layer 8 on stage 1.
    The mediator captures h2 on stage 0 (real), then on stage 1
    writes h2 to layer 8's output (real module, in-place modification).
    """
    import torch

    with model.trace(PROMPT, temperature=0.0, top_p=1):
        h2 = model.transformer.h[2].output[0]
        model.transformer.h[8].output[0][:] = h2
        logits = model.logits.output.save()

    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())

    return {
        "argmax": argmax,
        "top_token": model.tokenizer.decode(argmax),
    }


def scenario_cross_stage_multigen(model, args):
    """Cross-stage read during multi-token generation.

    Each step: capture layer 0 (stage 0) and layer 11 (stage 1).
    """
    import torch
    max_tokens = args.max_tokens

    with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=max_tokens) as tracer:
        h0_list = list().save()
        h11_list = list().save()
        logit_list = list().save()
        for step in tracer.iter[0:max_tokens]:
            h0_list.append(model.transformer.h[0].output[0])
            h11_list.append(model.transformer.h[11].output[0])
            logit_list.append(model.logits.output)

    tokens = []
    for logit in logit_list:
        am = int(logit.argmax(dim=-1).item())
        tokens.append(model.tokenizer.decode(am))

    return {
        "tokens": tokens,
        "num_steps": len(logit_list),
        "h0_shapes": [list(h.shape) for h in h0_list],
        "h11_shapes": [list(h.shape) for h in h11_list],
    }


def scenario_save_all_layers(model, args):
    """Save hidden states from ALL 12 layers across both stages.

    Tests iteration tracker across the PP boundary — layers 0-5 are
    on stage 0, layers 6-11 on stage 1.
    """
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        hiddens = []
        for i in range(12):
            hiddens.append(model.transformer.h[i].output[0].save())
        logits = model.logits.output.save()

    return {
        "num_layers": len(hiddens),
        "shapes": [list(h.shape) for h in hiddens],
        "argmax": int(logits.argmax(dim=-1).item()),
        "top_token": model.tokenizer.decode(int(logits.argmax(dim=-1).item())),
    }


def scenario_cross_stage_clone_modify(model, args):
    """Clone a stage-0 tensor, modify it, write to stage-1.

    h2 (stage 0) → clone → multiply by 0.5 → write to h8 (stage 1).
    Tests that materialized LazyRemoteTensor works with torch ops.
    """
    import torch

    with model.trace(PROMPT, temperature=0.0, top_p=1):
        h2 = model.transformer.h[2].output[0].clone()
        modified = h2 * 0.5
        model.transformer.h[8].output[0][:] = modified
        logits = model.logits.output.save()

    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())
    return {
        "argmax": argmax,
        "top_token": model.tokenizer.decode(argmax),
    }


def scenario_ablation(model, args):
    """Zero out a specific layer's output.

    Zero layer 3 (stage 0) and check effect on logits (stage 1).
    Also zero layer 8 (stage 1) directly. Both should change output.
    """
    import torch

    # Baseline
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        baseline = model.logits.output.save()

    # Ablate layer 3 (stage 0)
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        model.transformer.h[3].output[0][:] = 0
        ablated_l3 = model.logits.output.save()

    # Ablate layer 8 (stage 1)
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        model.transformer.h[8].output[0][:] = 0
        ablated_l8 = model.logits.output.save()

    base_am = int(baseline.argmax(dim=-1).item())
    l3_am = int(ablated_l3.argmax(dim=-1).item())
    l8_am = int(ablated_l8.argmax(dim=-1).item())

    return {
        "baseline": model.tokenizer.decode(base_am),
        "ablated_l3": model.tokenizer.decode(l3_am),
        "ablated_l8": model.tokenizer.decode(l8_am),
        "l3_changed": base_am != l3_am,
        "l8_changed": base_am != l8_am,
    }


def scenario_steering(model, args):
    """Add a steering vector to a cross-stage layer.

    Capture layer 2 mean (stage 0), use it as a steering vector
    added to layer 8 (stage 1).
    """
    import torch

    with model.trace(PROMPT, temperature=0.0, top_p=1):
        h2_mean = model.transformer.h[2].output[0].mean(dim=0, keepdim=True)
        model.transformer.h[8].output[0][:] = model.transformer.h[8].output[0] + h2_mean * 0.1
        logits = model.logits.output.save()

    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())
    return {
        "argmax": argmax,
        "top_token": model.tokenizer.decode(argmax),
    }


def scenario_cross_compare(model, args):
    """Compare PP=2 hidden states against PP=1 reference.

    Saves layer 0, 5, 6, 11 and logits. These span both stages.
    The caller should compare against PP=1 results.
    """
    import torch

    with model.trace(PROMPT, temperature=0.0, top_p=1):
        h0 = model.transformer.h[0].output[0].save()
        h5 = model.transformer.h[5].output[0].save()
        h6 = model.transformer.h[6].output[0].save()
        h11 = model.transformer.h[11].output[0].save()
        logits = model.logits.output.save()

    return {
        "h0_mean": float(h0.float().mean().item()),
        "h5_mean": float(h5.float().mean().item()),
        "h6_mean": float(h6.float().mean().item()),
        "h11_mean": float(h11.float().mean().item()),
        "argmax": int(logits.argmax(dim=-1).item()),
        "top_token": model.tokenizer.decode(int(logits.argmax(dim=-1).item())),
    }


def scenario_multigen_cross_write(model, args):
    """Multi-token generation with cross-stage write per step.

    Each step: capture layer 2 (stage 0), add to layer 8 (stage 1).
    """
    import torch
    max_tokens = args.max_tokens

    with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=max_tokens) as tracer:
        logit_list = list().save()
        for step in tracer.iter[0:max_tokens]:
            h2 = model.transformer.h[2].output[0]
            model.transformer.h[8].output[0][:] = model.transformer.h[8].output[0] + h2 * 0.01
            logit_list.append(model.logits.output)

    tokens = []
    for logit in logit_list:
        am = int(logit.argmax(dim=-1).item())
        tokens.append(model.tokenizer.decode(am))

    return {
        "tokens": tokens,
        "num_steps": len(logit_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=[
        "basic_trace", "logits", "hidden", "multigen",
        "cross_stage_read", "cross_stage_write", "cross_stage_multigen",
        "save_all_layers", "cross_clone_modify", "ablation",
        "steering", "cross_compare", "multigen_cross_write",
    ])
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=3)
    args = parser.parse_args()

    try:
        model = make_model(args.pp, args.tp)

        scenarios = {
            "basic_trace": scenario_basic_trace,
            "logits": scenario_logits,
            "hidden": scenario_hidden,
            "multigen": scenario_multigen,
            "cross_stage_read": scenario_cross_stage_read,
            "cross_stage_write": scenario_cross_stage_write,
            "cross_stage_multigen": scenario_cross_stage_multigen,
            "save_all_layers": scenario_save_all_layers,
            "cross_clone_modify": scenario_cross_stage_clone_modify,
            "ablation": scenario_ablation,
            "steering": scenario_steering,
            "cross_compare": scenario_cross_compare,
            "multigen_cross_write": scenario_multigen_cross_write,
        }
        result = scenarios[args.scenario](model, args)
        result["status"] = "ok"

    except Exception as e:
        result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    with open(args.output, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
