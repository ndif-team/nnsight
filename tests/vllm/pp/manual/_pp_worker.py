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
    cross_stage_replace - Cross-stage write via replacement (layer 2 -> layer 8)
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
        logits = model.logits.save()

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
        logits = model.logits.save()

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


def scenario_cross_stage_replace(model, prompt):
    """Cross-stage write via whole-output REPLACEMENT: read layer 2 (stage 0),
    set layer 8's hidden (stage 1) to (h8 + h2) by assigning the WHOLE `.output`.
    This is the documented-correct vLLM write — it sidesteps both the
    inference-tensor in-place error (which assigning `.output[0]` hits) and the
    tuple-reconstruction hang. Reports the top token so PP=1 (write applies on one
    GPU) and PP=2 (the cross-stage write) can be compared — they match exactly.
    """
    with model.trace(prompt, temperature=0.0, top_p=1):
        h2 = model.transformer.h[2].output[0]
        out8 = model.transformer.h[8].output
        is_tuple = isinstance(out8, tuple)
        hidden8 = out8[0] if is_tuple else out8
        new_hidden = hidden8 + h2                 # fresh tensor, not an in-place mutation
        # Assign the WHOLE .output (the nnbench-proven legal vLLM write); assigning
        # .output[0] instead is still an in-place inference-tensor write and is rejected.
        model.transformer.h[8].output = (new_hidden, *out8[1:]) if is_tuple else new_hidden
        logits = model.logits.save()

    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())
    return {"argmax": argmax, "top_token": model.tokenizer.decode(argmax)}


def scenario_downstream_read(model, prompt):
    """Read a DOWNSTREAM layer's output and MATERIALIZE it on the earlier
    (non-owning) rank — forces a backward stage1->stage0 pull, no write.

    Isolates the suspected cross-node-write hang mechanism: a raw lazy read
    + .save() is a no-op (works), but using the value (``out * 2``) triggers
    ``_materialize`` -> backward pull on stage 0 for a value stage 1 hasn't
    produced yet. The ``go_remote`` machinery is supposed to release stage 0's
    forward so the pull resolves; this probes whether it actually does.
    """
    with model.trace(prompt, temperature=0.0, top_p=1):
        out8 = model.transformer.h[8].output[0]   # downstream under pp=2
        used = (out8 * 2.0).sum().save()

    return {"used": float(used.float().cpu().item())}


def scenario_tuple_lazy(model, prompt):
    """Iterate a slice of a DOWNSTREAM lazy output: ``tuple(out[1:])``.

    On the non-owning (earlier) rank ``out`` is a LazyRemoteTensor with no
    ``__iter__`` and a ``__getitem__`` that never raises ``IndexError``, so
    ``tuple()`` spins forever -> that rank never finishes -> driver hangs.
    Faithful minimal repro of the ``s_cross`` hang (its ``+ tuple(out[1:])``
    term), without the gpt2 tuple-shape mismatch of cross_stage_replace.
    """
    with model.trace(prompt, temperature=0.0, top_p=1):
        out = model.transformer.h[8].output
        _ = tuple(out[1:])             # spun forever on the non-owning rank pre-fix
        logits = model.logits.save()

    # Reaching here at all is the regression signal: pre-fix the non-owning
    # rank never finished and the driver timed out.
    return {"argmax": int(logits.float().cpu().argmax(dim=-1).item())}


def scenario_multigen(model, prompt, max_tokens):
    """Multi-token generation."""
    with model.trace(
        prompt, temperature=0.0, top_p=1, max_tokens=max_tokens
    ) as tracer:
        logit_list = list().save()
        with tracer.iter[0:max_tokens]:
            logit_list.append(model.logits)

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


def scenario_multigen_ooo(model, prompt, max_tokens):
    """Out-of-forward-order access inside ``tracer.iter``.

    Each step reads a DOWNSTREAM module (``model.logits``, owned by the last
    stage) BEFORE a LOCAL early layer (``transformer.h[2]``, owned by stage 0) —
    i.e. a later-stage module before an earlier-stage one within one iteration.

    Single-GPU nnsight rejects this with ``OutOfOrderError``. Under PP it used to
    deadlock (stage 0's downstream access released the forward early, so the
    later local hook was missed and never re-fired). The PP path must surface the
    SAME ``OutOfOrderError`` promptly instead of hanging.
    """
    with model.trace(
        prompt, temperature=0.0, top_p=1, max_tokens=max_tokens
    ) as tracer:
        toks = list().save()
        early = list().save()
        for _ in tracer.iter[0:max_tokens]:
            toks.append(model.logits.argmax(dim=-1))      # downstream (last stage)
            early.append(model.transformer.h[2].output[0])  # local (stage 0)

    return {"num_steps": len(toks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=[
        "logits", "hidden", "hidden_only", "cross_stage_replace",
        "downstream_read", "tuple_lazy",
        "multigen", "multigen_hidden", "multigen_ooo",
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
        elif args.scenario == "cross_stage_replace":
            result = scenario_cross_stage_replace(model, args.prompt)
        elif args.scenario == "downstream_read":
            result = scenario_downstream_read(model, args.prompt)
        elif args.scenario == "tuple_lazy":
            result = scenario_tuple_lazy(model, args.prompt)
        elif args.scenario == "multigen":
            result = scenario_multigen(model, args.prompt, args.max_tokens)
        elif args.scenario == "multigen_hidden":
            result = scenario_multigen_hidden(model, args.prompt, args.max_tokens, args.layer)
        elif args.scenario == "multigen_ooo":
            result = scenario_multigen_ooo(model, args.prompt, args.max_tokens)
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
