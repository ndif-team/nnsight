#!/usr/bin/env python
"""Subprocess runner for the PP parity suite.

Each invocation boots one vLLM engine (PP=1 or PP=2), runs one scenario, and
writes JSON to ``--output``. The parity tests run the same scenario at both PP
sizes in separate subprocesses (two engines cannot share a process cleanly) and
compare the JSON in the parent.

Scenarios:
    logits      final-step logits for one prompt
    hidden      early-layer and late-layer hidden states plus the logits argmax
    write       cross-stage graft: read the early layer, rewrite the late
                layer's output, report the resulting argmax
    multigen    per-step sampled ids and late-layer hidden over several tokens
    concurrent  two invokes in one trace, per-invoke hidden states and argmax
"""

import argparse
import json
import os
import sys
import traceback

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"),
)

import torch

MODEL = "Qwen/Qwen2.5-0.5B"
# Qwen2.5-0.5B has 24 layers; at PP=2 stage 0 holds 0-11, stage 1 holds 12-23.
EARLY, LATE = 2, 20


def make_model(pp):
    from nnsight.modeling.vllm import VLLM

    kwargs = dict(
        tensor_parallel_size=1,
        gpu_memory_utilization=0.15,
        dispatch=True,
    )
    if pp > 1:
        kwargs["pipeline_parallel_size"] = pp
    return VLLM(MODEL, **kwargs)


def _flat(tensor):
    return tensor.float().cpu().flatten().tolist()


def _argmax(logits):
    return int(logits.float().cpu().argmax(dim=-1).flatten()[-1].item())


def scenario_logits(model, args):
    with model.trace(args.prompt, temperature=0.0, max_tokens=1):
        logits = model.logits.save()
    return {
        "argmax": _argmax(logits),
        "top_token": model.tokenizer.decode(_argmax(logits)),
        "logits": _flat(logits),
    }


def scenario_hidden(model, args):
    with model.trace(args.prompt, temperature=0.0, max_tokens=1):
        early = model.model.layers[EARLY].output[0].save()
        late = model.model.layers[LATE].output[0].save()
        logits = model.logits.save()
    return {
        "early": _flat(early),
        "early_shape": list(early.shape),
        "late": _flat(late),
        "late_shape": list(late.shape),
        "argmax": _argmax(logits),
    }


def scenario_write_local(model, args):
    # Zero the early layer's hidden state: read and write live on the same
    # stage, so the swap happens inside the layer's own hook on the owning
    # rank and is absorbed on the other. Replacement of the WHOLE output
    # tuple is the documented-legal vLLM write.
    with model.trace(args.prompt, temperature=0.0, max_tokens=1):
        early_out = model.model.layers[EARLY].output
        model.model.layers[EARLY].output = (early_out[0] * 0, *early_out[1:])
        logits = model.logits.save()
    return {"argmax": _argmax(logits), "logits": _flat(logits)}


def scenario_write_cross(model, args):
    # Read the early layer (stage 0), graft it into the late layer's output
    # (stage 1). On the owning rank the graft forces a cross-stage pull while
    # parked inside the late layer's hook.
    with model.trace(args.prompt, temperature=0.0, max_tokens=1):
        # Clone at read time: holding the live activation across the layers
        # between read and write lets vLLM's buffer reuse mutate it. The pull
        # path clones at publish, so without this snapshot PP=1 (mutated
        # buffer) and PP=2 (clean clone) graft different values.
        early_hidden = model.model.layers[EARLY].output[0].clone()
        late_out = model.model.layers[LATE].output
        late_hidden = late_out[0]
        # Scale to the residual's norm: on Qwen2 the layer tuple is
        # (hidden, residual) and the residual carries the stream, so a
        # perturbation at hidden's own scale barely moves the logits.
        graft = late_hidden + early_hidden * (
            late_out[1].norm() / early_hidden.norm()
        )
        model.model.layers[LATE].output = (graft, *late_out[1:])
        logits = model.logits.save()
    return {"argmax": _argmax(logits), "logits": _flat(logits)}


def scenario_multigen(model, args):
    import nnsight

    n = args.max_tokens
    with model.trace(args.prompt, temperature=0.0, max_tokens=n) as tracer:
        step_ids = nnsight.save([])
        step_late = nnsight.save([])
        for _ in tracer.iter[:n]:
            step_late.append(model.model.layers[LATE].output[0])
            step_ids.append(model.samples)
    return {
        "ids": [int(torch.as_tensor(s).flatten()[-1].item()) for s in step_ids],
        "late": [_flat(t) for t in step_late],
        "late_shapes": [list(t.shape) for t in step_late],
    }


def scenario_concurrent(model, args):
    with model.trace(temperature=0.0, max_tokens=1) as tracer:
        with tracer.invoke(args.prompt):
            early_a = model.model.layers[EARLY].output[0].save()
            late_a = model.model.layers[LATE].output[0].save()
            logits_a = model.logits.save()
        with tracer.invoke(args.prompt_b):
            early_b = model.model.layers[EARLY].output[0].save()
            late_b = model.model.layers[LATE].output[0].save()
            logits_b = model.logits.save()
    return {
        "first": {
            "prompt_tokens": len(model.tokenizer.encode(args.prompt)),
            "early": _flat(early_a),
            "early_shape": list(early_a.shape),
            "late": _flat(late_a),
            "late_shape": list(late_a.shape),
            "argmax": _argmax(logits_a),
        },
        "second": {
            "prompt_tokens": len(model.tokenizer.encode(args.prompt_b)),
            "early": _flat(early_b),
            "early_shape": list(early_b.shape),
            "late": _flat(late_b),
            "late_shape": list(late_b.shape),
            "argmax": _argmax(logits_b),
        },
    }


SCENARIOS = {
    "logits": scenario_logits,
    "hidden": scenario_hidden,
    "write_local": scenario_write_local,
    "write_cross": scenario_write_cross,
    "multigen": scenario_multigen,
    "concurrent": scenario_concurrent,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-b", dest="prompt_b", default=None)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        model = make_model(args.pp)
        with torch.no_grad():
            result = SCENARIOS[args.scenario](model, args)
        result["status"] = "ok"
    except Exception as exception:
        result = {
            "status": "error",
            "error": repr(exception),
            "traceback": traceback.format_exc(),
        }

    with open(args.output, "w") as f:
        json.dump(result, f)
    # Exit normally: vLLM's atexit hooks shut the spawned EngineCore down.
    # A hard exit here orphans it, and an orphan holds GPU memory (and any
    # inherited pipe) indefinitely.
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
