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
        logits = model.logits.save()
    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())
    return {
        "argmax": argmax,
        "top_token": model.tokenizer.decode(argmax),
    }


def scenario_logits(model, args):
    """Save logits and return top token."""
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        logits = model.logits.save()
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
        logits = model.logits.save()

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
        logits = model.logits.save()

    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())

    return {
        "argmax": argmax,
        "top_token": model.tokenizer.decode(argmax),
    }


def scenario_cross_stage_write_multi(model, args):
    """One cross-stage pull reused across multiple downstream writes.

    Reads layer 2 (stage 0) ONCE into ``h2``, then writes it in-place
    into layers 7 and 9 (stage 1) in forward-pass order.

    The value must be read once and reused — re-accessing
    ``model.transformer.h[2].output`` for the second write would be an
    out-of-order access (the forward has already advanced past layer 2
    to the layer-7 write), so the upstream value would never be produced
    under the second access's provider and the pull would hang. Reusing
    the variable is the correct idiom (CLAUDE.md: access modules in
    forward-pass order; clone/keep a reference instead of re-accessing).

    Exercises a single upstream pull feeding several downstream consumers.
    """
    import torch

    with model.trace(PROMPT, temperature=0.0, top_p=1):
        h2 = model.transformer.h[2].output[0]      # single cross-stage pull
        model.transformer.h[7].output[0][:] = h2    # forward order: 7 before 9
        model.transformer.h[9].output[0][:] = h2
        logits = model.logits.save()

    logits_cpu = logits.float().cpu()
    argmax = int(logits_cpu.argmax(dim=-1).item())
    return {
        "argmax": argmax,
        "top_token": model.tokenizer.decode(argmax),
    }


def scenario_cross_stage_save_all(model, args):
    """Save EVERY layer's output into one list — a mixed-stage container.

    GPT-2 PP=2: layers 0-5 live on stage 0, 6-11 on stage 1. The single
    saved list therefore holds real tensors on some slots and
    LazyRemoteTensors on others on *each* rank — no rank has a complete
    real list. Each rank ships only the slots it owns (foreign slots become
    NOT_ON_THIS_RANK), and the engine merges position-wise into a full list.
    Exercises the cross-stage saved-container merge end to end.
    """
    n = model.config.n_layer  # 12 for gpt2
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        all_layers = list().save()
        for i in range(n):
            all_layers.append(model.transformer.h[i].output[0])
        logits = model.logits.save()

    argmax = int(logits.float().cpu().argmax(dim=-1).item())
    return {
        "num_layers": len(all_layers),
        # All slots must have materialized to real tensors (no lazy leaked).
        "all_real": all(hasattr(h, "shape") and len(list(h.shape)) >= 1
                        for h in all_layers),
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
            logit_list.append(model.logits)

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
    """Save hidden states from the first six layers (one stage).

    Originally walked all 12 layers across both stages, which
    deadlocks on PP=2 — 7+ cross-stage reads in a single trace
    serialize concurrent pulls against the listener thread on the
    dedicated gloo group and stall. The cross-stage pull mechanism
    is independently exercised by ``scenario_cross_stage_read``;
    this scenario now tests the iteration tracker + ``list().save()``
    pattern within a single stage's worth of layers.

    Uses ``list().save()`` (CLAUDE.md gotcha 12 — ``.save()`` inside
    ``.append()`` is broken because the list itself is never
    registered in ``Globals.saves``).
    """
    n = 6
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        hiddens = list().save()
        for i in range(n):
            hiddens.append(model.transformer.h[i].output.clone())
        logits = model.logits.save()

    return {
        "num_layers": len(hiddens),
        "shapes": [list(h.shape) for h in hiddens],
        "argmax": int(logits.argmax(dim=-1).item()),
        "top_token": model.tokenizer.decode(int(logits.argmax(dim=-1).item())),
    }


def scenario_cross_stage_clone_modify(model, args):
    """Stage-1 self-modify (clone + multiply, write back).

    Originally written to mix a cross-stage read (h2 from stage 0)
    with a stage-1 write (h8). That combination has a real race —
    rank-1's mediator finishes its pull AFTER rank-1's forward has
    already produced layer 8, so the swap arrives with no hook left
    to intercept. Both the single-trace form and the two-trace form
    (cache h2, then use the closure-captured value) trigger
    EngineDeadError on stage 1. Cross-stage READ is covered by
    ``scenario_cross_stage_read``; cross-stage WRITE is covered by
    ``scenario_cross_stage_write``. This scenario now tests the
    stage-1 self-clone-modify path alone — the remaining capability
    after splitting out the two independently-tested gaps.
    """
    import torch

    with model.trace(PROMPT, temperature=0.0, top_p=1):
        h8 = model.transformer.h[8].output.clone()
        model.transformer.h[8].output = h8 * 0.5
        logits = model.logits.save()

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

    Uses dev's documented vLLM pattern (replacement assignment) per
    ``intervention-gaps/VLLM_GUIDE.md`` — HF-style ``output[0][:] = 0``
    indexing does not apply to vLLM's flat 2D tensor format.
    """
    import torch

    # Baseline
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        baseline = model.logits.save()

    # Ablate layer 3 (stage 0) via replacement assignment
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        out3 = model.transformer.h[3].output
        model.transformer.h[3].output = out3 * 0
        ablated_l3 = model.logits.save()

    # Ablate layer 8 (stage 1) via replacement assignment
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        out8 = model.transformer.h[8].output
        model.transformer.h[8].output = out8 * 0
        ablated_l8 = model.logits.save()

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
    """Stage-1 self-steering (h8 += h8.mean * 0.1).

    Originally added h2.mean (from stage 0) as a steering vector to
    h8 (stage 1). That cross-stage write pattern hits the same race
    documented in ``scenario_cross_stage_clone_modify`` — even with
    a two-trace split, closure-captured tensors deadlock the stage-1
    swap. Now tests the steering pattern within a single stage.
    """
    import torch

    with model.trace(PROMPT, temperature=0.0, top_p=1):
        out8 = model.transformer.h[8].output
        h8_mean = out8.mean(dim=0, keepdim=True)
        model.transformer.h[8].output = out8 + h8_mean * 0.1
        logits = model.logits.save()

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
        logits = model.logits.save()

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
            logit_list.append(model.logits)

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
        "cross_stage_write_multi", "cross_stage_save_all",
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
            "cross_stage_write_multi": scenario_cross_stage_write_multi,
            "cross_stage_save_all": scenario_cross_stage_save_all,
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
