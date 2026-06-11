"""Measure pp_hook_buffer growth to distinguish a real leak from benign
allocator caching, and measure the intra-request peak for long generation.

Relies on env-gated `[PPBUF rankN]` diagnostic lines emitted by
GPUModelRunner.collect_nnsight (set NNSIGHT_PP_BUFFER_DEBUG=1).

Two tests:
  (b) cross-request: run N sequential short traces; the per-trace pre-clear
      size should be ~constant and the post-clear size should return to ~0.
      A leak shows post-clear size climbing across traces.
  (a) intra-request peak: one long-generation trace accessing K layers/step;
      the pre-clear size is the peak number of retained clones for that
      single request (= O(K * max_tokens)).

Run:
  HF_HUB_CACHE=/disk/u/models CUDA_VISIBLE_DEVICES=3,4 \
    NNSIGHT_PP_BUFFER_DEBUG=1 VLLM_WORKER_MULTIPROC_METHOD=spawn MASTER_PORT=29920 \
    conda run --no-capture-output -n ndif-dev python tests/vllm/pp/manual/measure_buffer.py \
      --model Qwen/Qwen2.5-7B-Instruct --pp 2 --gpu-mem 0.45 \
      --bursts 40 --layers 8 --short-tokens 4 --long-tokens 256 --long-layers 12
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time

import torch


class GPUSampler:
    """Background thread sampling per-GPU used memory via nvidia-smi.

    Records the peak used-MB on each visible device over its lifetime so we
    can see whether GPU memory climbs *during* a long-generation request
    (pre-fix: O(layers x tokens) buffer clones accumulate on GPU) or stays
    flat (post-fix: clones migrate to CPU after each forward).
    """

    def __init__(self, devices, period=0.25):
        self._devices = [d.strip() for d in devices.split(",") if d.strip()]
        self._period = period
        self._stop = threading.Event()
        self._thread = None
        self.peak = {d: 0 for d in self._devices}
        self.baseline = {}

    def _sample(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"], text=True, timeout=5)
        except Exception:
            return {}
        m = {}
        for line in out.strip().splitlines():
            idx, used = (s.strip() for s in line.split(","))
            m[idx] = int(used)
        return m

    def _loop(self):
        self.baseline = self._sample()
        while not self._stop.is_set():
            m = self._sample()
            for d in self._devices:
                if d in m:
                    self.peak[d] = max(self.peak[d], m[d])
            self._stop.wait(self._period)

    def start(self):
        # CUDA_VISIBLE_DEVICES remaps indices; sample the physical ids it lists.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd:
            self._devices = [d.strip() for d in cvd.split(",") if d.strip()]
            self.peak = {d: 0 for d in self._devices}
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def report(self, tag):
        for d in self._devices:
            base = self.baseline.get(d, 0)
            print(f"[gpusample {tag}] GPU{d}: baseline={base}MB "
                  f"peak={self.peak[d]}MB delta={self.peak[d]-base}MB", flush=True)

PROMPT = "The Eiffel Tower is located in the city of"


def discover_layers(underlying):
    """Return (dotted path, n_layers) of the decoder ModuleList."""
    cands = []
    for name, mod in underlying.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 2:
            first = mod[0]
            kids = {n for n, _ in first.named_children()}
            if kids & {"self_attn", "attn", "attention", "self_attention"} and \
               kids & {"mlp", "feed_forward", "ffn", "block_sparse_moe"}:
                cands.append((name, len(mod)))
    if not cands:
        for name, mod in underlying.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 2:
                cands.append((name, len(mod)))
    name, n = max(cands, key=lambda kv: kv[1])
    return name, n


def resolve(envoy, dotted):
    cur = envoy
    for p in dotted.split("."):
        cur = getattr(cur, p)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--bursts", type=int, default=40)
    ap.add_argument("--layers", type=int, default=8, help="layers/step for the cross-request test")
    ap.add_argument("--short-tokens", type=int, default=4)
    ap.add_argument("--long-tokens", type=int, default=256)
    ap.add_argument("--long-layers", type=int, default=12)
    args = ap.parse_args()

    from nnsight.modeling.vllm import VLLM

    print(f"[measure] loading {args.model} pp={args.pp}", flush=True)
    t0 = time.time()
    model = VLLM(
        args.model,
        pipeline_parallel_size=args.pp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        dispatch=True,
    )
    print(f"[measure] loaded in {time.time()-t0:.1f}s", flush=True)

    layers_path, n_layers = discover_layers(model._model)
    layers = resolve(model, layers_path)
    print(f"[measure] layers_path={layers_path} n_layers={n_layers}", flush=True)

    # Even spread of layer indices for the access set.
    def spread(k):
        k = min(k, n_layers)
        step = max(1, n_layers // k)
        return sorted(set(min(n_layers - 1, i * step) for i in range(k)))

    # ----- Test (b): cross-request, N sequential short traces -----
    short_layers = spread(args.layers)
    print(f"\n[measure] === TEST (b) cross-request: {args.bursts} sequential traces, "
          f"layers={short_layers} tokens={args.short_tokens} ===", flush=True)
    print("[measure] watch the [PPBUF] post-clear size: flat => no leak, "
          "climbing => leak", flush=True)
    for i in range(args.bursts):
        with model.trace(PROMPT, temperature=0.0, top_p=1,
                         max_tokens=args.short_tokens) as tracer:
            saved = list().save()
            for _ in tracer.iter[0:args.short_tokens]:
                for L in short_layers:
                    saved.append(layers[L].output[0])
        if (i + 1) % 5 == 0:
            print(f"[measure] (b) completed {i+1}/{args.bursts} traces", flush=True)

    # ----- Test (a): intra-request peak, one long-generation trace -----
    long_layers = spread(args.long_layers)
    print(f"\n[measure] === TEST (a) intra-request peak: 1 trace, "
          f"layers={long_layers} tokens={args.long_tokens} ===", flush=True)
    print(f"[measure] expected peak clones ~= {len(long_layers)} layers x "
          f"{args.long_tokens} tokens = {len(long_layers)*args.long_tokens}", flush=True)
    sampler = GPUSampler(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    sampler.start()
    time.sleep(1.0)  # capture baseline before the request
    with model.trace(PROMPT, temperature=0.0, top_p=1,
                     max_tokens=args.long_tokens) as tracer:
        saved = list().save()
        for _ in tracer.iter[0:args.long_tokens]:
            for L in long_layers:
                saved.append(layers[L].output[0])
    sampler.stop()
    sampler.report("long-gen")
    print(f"[measure] (a) done — [PPBUF] pre-clear shows buffer device; "
          f"[gpusample] delta shows GPU growth during the request", flush=True)

    print("\n[measure] complete", flush=True)


if __name__ == "__main__":
    main()
