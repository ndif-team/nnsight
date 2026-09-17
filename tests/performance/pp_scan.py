"""Synthetic cross-stage scan: one engine, a set of block shapes, median wall time each.

Run with the branch's src on PYTHONPATH:
    PYTHONPATH=<worktree>/src CUDA_VISIBLE_DEVICES=4,5 python pp_scan.py --pp 2 --out push.json
    PYTHONPATH=<worktree>/src CUDA_VISIBLE_DEVICES=4   python pp_scan.py --pp 1 --out ref.json
"""

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MODEL = os.environ.get("PP_SCAN_MODEL", "Qwen/Qwen2.5-0.5B")
PROMPT = "The Eiffel Tower is located in the city of"


def layers_of(model):
    return model.model.layers


def shape_read(model, indices, max_tokens=1):
    def run():
        with model.trace(PROMPT, temperature=0.0, max_tokens=max_tokens):
            sums = [float(layers_of(model)[i].output[0].float().sum()) for i in indices]
            out = list(sums).save()
        return out

    return run


def shape_save_all(model, n_layers):
    def run():
        with model.trace(PROMPT, temperature=0.0, max_tokens=1):
            kept = list().save()
            for i in range(n_layers):
                kept.append(layers_of(model)[i].output[0])
        return len(kept)

    return run


def shape_logits_steps(model, steps):
    def run():
        with model.trace(PROMPT, temperature=0.0, max_tokens=steps, ignore_eos=True) as tracer:
            toks = list().save()
            for _ in tracer.iter[:steps]:
                toks.append(int(model.logits[-1].argmax()))
        return len(toks)

    return run


def shape_write_each(model, n_layers):
    def run():
        with model.trace(PROMPT, temperature=0.0, max_tokens=1):
            for i in range(n_layers):
                out = layers_of(model)[i].output
                layers_of(model)[i].output = (out[0] * 1.0,) + tuple(out[1:])
            logits = model.logits.save()
        return logits.shape[-1]

    return run


def shape_head_lens(model, indices):
    def run():
        with model.trace(PROMPT, temperature=0.0, max_tokens=1):
            tops = list().save()
            weight = model.lm_head.param("weight")
            for i in indices:
                out = layers_of(model)[i].output
                normed, _ = model.model.norm(out[0][-1:], out[1][-1:])
                tops.append(int((normed @ weight.T).argmax()))
        return len(tops)

    return run


def shape_plain(model, steps):
    def run():
        with model.trace(PROMPT, temperature=0.0, max_tokens=steps, ignore_eos=True):
            logits = model.logits.save()
        return logits.shape[-1]

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    from nnsight.modeling.vllm import VLLM

    model = VLLM(MODEL, pipeline_parallel_size=args.pp, gpu_memory_utilization=0.25, dispatch=True)
    n_layers = len(layers_of(model))
    half = n_layers // 2
    shapes = {}
    for k in (1, 4, 8, 12):
        shapes[f"read_late_{k}"] = shape_read(model, list(range(half, half + k)))
        shapes[f"read_early_{k}"] = shape_read(model, list(range(k)))
    shapes["save_all"] = shape_save_all(model, n_layers)
    shapes["write_each_layer"] = shape_write_each(model, n_layers)
    shapes["head_lens_early_12"] = shape_head_lens(model, list(range(half)))
    for steps in (32, 128):
        shapes[f"plain_gen_{steps}"] = shape_plain(model, steps)
        shapes[f"logits_steps_{steps}"] = shape_logits_steps(model, steps)
    if args.only:
        wanted = args.only.split(",")
        shapes = {name: shape for name, shape in shapes.items() if name in wanted}

    results = {}
    for name, run in shapes.items():
        for _ in range(args.warmup):
            run()
        times = []
        for _ in range(args.trials):
            t0 = time.perf_counter()
            run()
            times.append(time.perf_counter() - t0)
        results[name] = {"median_s": statistics.median(times), "min_s": min(times), "trials": times}
        print(f"{name:24s} median {statistics.median(times)*1000:8.1f} ms  min {min(times)*1000:8.1f} ms", flush=True)

    with open(args.out, "w") as f:
        json.dump({"model": MODEL, "pp": args.pp, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
