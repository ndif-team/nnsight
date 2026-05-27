#!/usr/bin/env python
"""Single-case PP cross-stage-write repro worker (runs in its own process).

Cases (GPT-2 PP=2: layers 0-5 stage 0, 6-11 stage 1):
  inplace_xstage   : layer8.output[0][:] = h2      (h2 = layer2, cross-stage read)  -> control, expected PASS
  replace_xstage   : layer8.output = <rhs(h2)>      (replacement write, RHS needs pull) -> suspect
  replace_local    : layer8.output = layer8 * 0.5   (replacement, RHS all local)     -> control, expected PASS

Writes a JSON result (status / error / traceback) to --output.
Optional --debug enables a per-rank watchdog that dumps all thread stacks.
"""
import argparse, json, os, traceback

PROMPT = "The Eiffel Tower is located in the city of"


def make_model(pp=2):
    from nnsight.modeling.vllm import VLLM
    return VLLM("openai-community/gpt2", pipeline_parallel_size=pp,
                gpu_memory_utilization=0.1, dispatch=True)


def case_inplace_xstage(model):
    with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
        h2 = model.transformer.h[2].output[0]
        model.transformer.h[8].output[0][:] = h2
        logits = model.logits.save()
    am = int(logits.float().cpu().argmax(dim=-1).item())
    return {"argmax": am, "top_token": model.tokenizer.decode(am)}


def case_replace_xstage(model):
    # The deadlock case: rank 0 has a local upstream hook (h2 = layer 2) that
    # freezes its forward, THEN a downstream read (out8 = layer 8) whose pull
    # can't resolve until rank 0's forward finishes + sends. Tensor idiom
    # (layer output is flat 2D), matching the passing controls.
    with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
        h2 = model.transformer.h[2].output
        out8 = model.transformer.h[8].output
        model.transformer.h[8].output = out8 + h2 * 0.5
        logits = model.logits.save()
    am = int(logits.float().cpu().argmax(dim=-1).item())
    return {"argmax": am, "top_token": model.tokenizer.decode(am)}


def case_replace_local(model):
    # Downstream pull on rank 0 (out8 = layer8 is PPMissing there), but NO
    # upstream local hook -> rank 0's forward is never frozen, so it sends
    # its PP activation and rank 1 can produce layer 8. Tensor idiom (vLLM
    # layer output is a flat 2D tensor), matching the passing `ablation`.
    with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
        out8 = model.transformer.h[8].output
        model.transformer.h[8].output = out8 * 0.5
        logits = model.logits.save()
    am = int(logits.float().cpu().argmax(dim=-1).item())
    return {"argmax": am, "top_token": model.tokenizer.decode(am)}


CASES = {
    "inplace_xstage": case_inplace_xstage,
    "replace_xstage": case_replace_xstage,
    "replace_local": case_replace_local,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=list(CASES))
    p.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        model = make_model()
        result = CASES[args.case](model)
        result["status"] = "ok"
    except Exception as e:
        result = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
    with open(args.output, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
