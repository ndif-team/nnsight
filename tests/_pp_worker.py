#!/usr/bin/env python
"""
Worker subprocess for PP integration tests.

Each invocation creates a VLLM model with specified PP size, runs a
single test scenario, and writes JSON results to a temp file.

Usage:
    python _pp_worker.py <scenario> --pp <1|2> --prompt <text> --output <path> [--layer N] [--max_tokens N]

Scenarios:
    logits       - Basic logit comparison (save logits.output)
    hidden       - Hidden state extraction from a specific layer
    cross_stage  - Cross-stage write (layer 2 -> layer 8)
    multigen     - Multi-token generation
    hidden_only  - Only save hidden states (no logits access, avoids WrapperModule issue)
"""

import argparse
import json
import sys
import traceback

import torch


def make_model(pp_size):
    """Create a VLLM model with given pipeline parallel size."""
    from nnsight.modeling.vllm import VLLM

    kwargs = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.1,
        "dispatch": True,
    }
    if pp_size > 1:
        kwargs["pipeline_parallel_size"] = pp_size

    return VLLM("gpt2", **kwargs)


def scenario_logits(model, prompt):
    """Get logits for a prompt."""
    with model.trace(prompt, temperature=0.0, top_p=1):
        logits = model.logits.output.save()

    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())
    top_token = model.tokenizer.decode(argmax)

    return {
        "argmax": argmax,
        "top_token": top_token,
        "logits": logits_cpu.flatten().tolist(),
    }


def scenario_hidden(model, prompt, layer):
    """Get hidden states from a specific layer, plus logits."""
    layers = model.transformer.h

    with model.trace(prompt, temperature=0.0, top_p=1):
        hidden = layers[layer].output[0].save()
        logits = model.logits.output.save()

    hidden_cpu = hidden.float().cpu()
    logits_cpu = logits.float().cpu()
    return {
        "shape": list(hidden_cpu.shape),
        "hidden": hidden_cpu.flatten().tolist(),
        "argmax": int(logits_cpu.argmax(dim=-1).item()),
        "top_token": model.tokenizer.decode(int(logits_cpu.argmax(dim=-1).item())),
    }


def scenario_hidden_only(model, prompt, layer):
    """Get ONLY hidden states from a specific layer. No logits access.

    This avoids the WrapperModule issue where model.logits is not
    recognized as PPMissing on non-last ranks.
    """
    layers = model.transformer.h

    with model.trace(prompt, temperature=0.0, top_p=1):
        hidden = layers[layer].output[0].save()

    hidden_cpu = hidden.float().cpu()
    return {
        "shape": list(hidden_cpu.shape),
        "hidden": hidden_cpu.flatten().tolist(),
    }


def scenario_cross_stage(model, prompt):
    """Cross-stage write: read layer 2, write to layer 8, save logits."""
    with model.trace(prompt, temperature=0.0, top_p=1):
        h2 = model.transformer.h[2].output[0]
        model.transformer.h[8].output[0][:] = h2
        logits = model.logits.output.save()

    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())
    top_token = model.tokenizer.decode(argmax)

    return {
        "argmax": argmax,
        "top_token": top_token,
    }


def scenario_cross_stage_no_logits(model, prompt):
    """Cross-stage write: read layer 2, write to layer 8, save layer 11 hidden.

    Avoids logits WrapperModule issue.
    """
    with model.trace(prompt, temperature=0.0, top_p=1):
        h2 = model.transformer.h[2].output[0]
        model.transformer.h[8].output[0][:] = h2
        h11 = model.transformer.h[11].output[0].save()

    h11_cpu = h11.float().cpu()
    return {
        "shape": list(h11_cpu.shape),
        "hidden": h11_cpu.flatten().tolist(),
    }


def scenario_multigen(model, prompt, max_tokens):
    """Multi-token generation."""
    with model.trace(
        prompt, temperature=0.0, top_p=1, max_tokens=max_tokens
    ) as tracer:
        logit_list = list().save()
        with tracer.iter[0:max_tokens]:
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


def scenario_multigen_hidden(model, prompt, max_tokens, layer):
    """Multi-token generation saving hidden states instead of logits."""
    with model.trace(
        prompt, temperature=0.0, top_p=1, max_tokens=max_tokens
    ) as tracer:
        hidden_list = list().save()
        with tracer.iter[0:max_tokens]:
            hidden_list.append(model.transformer.h[layer].output[0])

    shapes = []
    hiddens = []
    for h in hidden_list:
        h_cpu = h.float().cpu()
        shapes.append(list(h_cpu.shape))
        hiddens.append(h_cpu.flatten().tolist())

    return {
        "shapes": shapes,
        "hiddens": hiddens,
        "num_steps": len(hidden_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=[
        "logits", "hidden", "hidden_only", "cross_stage", "cross_stage_no_logits",
        "multigen", "multigen_hidden",
    ])
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Path to write JSON result")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=3)
    args = parser.parse_args()

    try:
        model = make_model(args.pp)

        if args.scenario == "logits":
            result = scenario_logits(model, args.prompt)
        elif args.scenario == "hidden":
            result = scenario_hidden(model, args.prompt, args.layer)
        elif args.scenario == "hidden_only":
            result = scenario_hidden_only(model, args.prompt, args.layer)
        elif args.scenario == "cross_stage":
            result = scenario_cross_stage(model, args.prompt)
        elif args.scenario == "cross_stage_no_logits":
            result = scenario_cross_stage_no_logits(model, args.prompt)
        elif args.scenario == "multigen":
            result = scenario_multigen(model, args.prompt, args.max_tokens)
        elif args.scenario == "multigen_hidden":
            result = scenario_multigen_hidden(model, args.prompt, args.max_tokens, args.layer)
        else:
            result = {"error": f"Unknown scenario: {args.scenario}"}

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
